"""MEDDIAG inference pipeline.

Flow (SRS §10):
    1. Load image
    2. Preprocess -> MobileViT encode -> (B, N, C_vision)
    3. Projector -> (B, K, D_llm) in bf16
    4. Build a text "retrieval query" from visible priors + retrieve top-k
    5. Build chat messages via prompts.py (single source of truth)
    6. Tokenize text portions; splice visual embeddings at IMAGE_PLACEHOLDER
    7. Call model.generate with inputs_embeds + attention_mask ONLY
    8. Parse structured output
    9. Return diagnosis + evidence + reasoning

Explicit non-goals of this module:
    - No training. That lives under experiments/ (to be added in Stage 1/2 work).
    - No caching of anything to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .config import CONFIG, Config
from .llm import LoadedLLM, load_llm
from .output_parser import parse_output
from .projector import PerceiverResampler
from .prompts import IMAGE_PLACEHOLDER, build_chat_messages
from .retrieval import RetrievedSnippet, Retriever
from .vision import VisionEncoder


@dataclass
class DiagnosisResult:
    diagnosis: str                     # "NORMAL" | "ABNORMAL" | "UNPARSEABLE"
    evidence_used: list[int]           # 1-indexed evidence ids the model cited
    reasoning: str                     # free-text reasoning block
    retrieved: list[RetrievedSnippet]  # what we fed in (full snippets)
    raw_output: str                    # untouched model output, for debugging


class MeddiagPipeline:
    def __init__(
        self,
        config: Config = CONFIG,
        projector_weights: Path | None = None,
        lora_adapter_dir: Path | None = None,
    ):
        config.validate()
        self.config = config

        self.vision = VisionEncoder(config)
        self.llm: LoadedLLM = load_llm(config)

        # Hard assertion: LLM hidden dim must match config.
        if self.llm.hidden_dim != config.llm_hidden_dim:
            raise RuntimeError(
                f"LLM hidden dim {self.llm.hidden_dim} != config.llm_hidden_dim "
                f"{config.llm_hidden_dim}. Update config.py."
            )

        # Load LoRA adapter if available (Stage 2 trained weights)
        if lora_adapter_dir is not None:
            try:
                from peft import PeftModel
                self.llm.model = PeftModel.from_pretrained(
                    self.llm.model, str(lora_adapter_dir)
                )
                self.llm.model = self.llm.model.merge_and_unload()
                print(f"[pipeline] LoRA adapter merged from {lora_adapter_dir}")
            except ImportError:
                raise RuntimeError(
                    "peft is required to load LoRA adapter. Install with: pip install peft"
                )

        self.projector = PerceiverResampler(
            vision_dim=config.vision_hidden_dim,
            llm_dim=self.llm.hidden_dim,
            num_latents=config.num_visual_tokens,
            num_heads=config.projector_num_heads,
            num_layers=config.projector_num_layers,
        )
        if projector_weights is not None and projector_weights.exists():
            state = torch.load(projector_weights, map_location="cpu")
            self.projector.load_state_dict(state)
        # Even pre-training, move projector to the LLM's device so splicing works.
        self.projector.to(self.llm.device)
        self.vision.to(self.llm.device)
        self.projector.eval()

        # Retriever: load from disk. If missing, fail loudly — FAISS is mandatory.
        self.retriever = Retriever(config)
        self.retriever.load()  # raises FileNotFoundError with a helpful message

    # -------- helpers --------

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the LLM's input embeddings for token ids, in the LLM's dtype."""
        return self.llm.model.get_input_embeddings()(input_ids)

    def _build_retrieval_query(self) -> str:
        """Query text for FAISS.

        First pass: use a generic clinical prompt. A better approach (future work)
        is to caption the image with the base LLM first and use that caption as
        the query — but that adds a forward pass and complicates the RAG-off
        baseline comparison. Keeping it simple and honest for now.
        """
        return "Chest X-ray findings and clinical impression."

    def _splice_visual(
        self,
        prompt_text: str,
        visual_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompt_text and splice visual_embeds where IMAGE_PLACEHOLDER sits.

        Returns (inputs_embeds, attention_mask). We never return input_ids here —
        that's the whole point (SRS §5 forward pass rule).
        """
        tokenizer = self.llm.tokenizer
        device = self.llm.device

        if IMAGE_PLACEHOLDER not in prompt_text:
            raise ValueError(
                f"Prompt does not contain {IMAGE_PLACEHOLDER}; cannot splice image."
            )

        left_text, right_text = prompt_text.split(IMAGE_PLACEHOLDER, 1)

        # Tokenize without special tokens — apply_chat_template already added them.
        left_ids = tokenizer(left_text, add_special_tokens=False, return_tensors="pt").input_ids
        right_ids = tokenizer(right_text, add_special_tokens=False, return_tensors="pt").input_ids

        left_ids = left_ids.to(device)
        right_ids = right_ids.to(device)

        left_embeds = self._embed_tokens(left_ids)                    # (1, L_left, D)
        right_embeds = self._embed_tokens(right_ids)                  # (1, L_right, D)
        visual_embeds = visual_embeds.to(device=device, dtype=left_embeds.dtype)  # (1, K, D)

        inputs_embeds = torch.cat([left_embeds, visual_embeds, right_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )
        return inputs_embeds, attention_mask

    # -------- public API --------

    @torch.no_grad()
    def diagnose(self, image_path: str | Path) -> DiagnosisResult:
        image = Image.open(image_path)

        # 1. Vision
        pixel_values = self.vision.preprocess(image).to(self.llm.device)
        vision_tokens = self.vision(pixel_values)                      # (1, N, C_vision)

        # 2. Projector
        visual_embeds = self.projector(vision_tokens)                  # (1, K, D_llm)

        # 3. Retrieval
        query = self._build_retrieval_query()
        retrieved = self.retriever.query(query, k=self.config.retrieval_top_k)
        snippets = [r.text for r in retrieved]

        # 4. Build chat text via the tokenizer's own chat template — no hand-rolled
        #    special tokens. This renders system+user turns into a single string.
        messages = build_chat_messages(snippets)
        prompt_text = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 5. Splice visual embeddings
        inputs_embeds, attention_mask = self._splice_visual(prompt_text, visual_embeds)

        # 6. Generate — inputs_embeds + attention_mask ONLY (SRS §5)
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            pad_token_id=self.llm.tokenizer.pad_token_id,
            temperature=self.config.temperature if self.config.do_sample else None,
            top_p=None,
        )
        # Remove None-valued keys — transformers warns if invalid flags are passed
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        output_ids = self.llm.model.generate(**gen_kwargs)
        # When only inputs_embeds is passed, HF returns only the NEW tokens — no
        # prefix stripping needed. (Verified for Llama in transformers >=4.44.)
        raw_text = self.llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        parsed = parse_output(raw_text)
        return DiagnosisResult(
            diagnosis=parsed["diagnosis"],
            evidence_used=parsed["evidence_used"],
            reasoning=parsed["reasoning"],
            retrieved=retrieved,
            raw_output=raw_text,
        )