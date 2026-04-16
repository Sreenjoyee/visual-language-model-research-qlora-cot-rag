"""Streaming iterators with retry.

SRS §3: HF streaming may drop -> retry logic required.
SRS §4: streaming only; no local caching.

This module provides a thin retry wrapper. It does NOT implement balanced
sampling — that's a training concern (Stage 1/2) and will live alongside the
training loops when those are written, not here.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


def retrying_iter(
    source_factory: Callable[[], Iterator[T]],
    max_retries: int = 5,
    backoff_seconds: float = 2.0,
) -> Iterator[T]:
    """Wrap a stream factory with reconnect-on-failure.

    source_factory: callable returning a fresh iterator (so we can reopen the stream).
    """
    attempts = 0
    while True:
        try:
            for item in source_factory():
                yield item
            return  # normal completion
        except Exception as e:  # noqa: BLE001 — HF can raise many things on network blips
            attempts += 1
            if attempts > max_retries:
                raise
            sleep_for = backoff_seconds * (2 ** (attempts - 1))
            print(
                f"[retrying_iter] Stream error ({type(e).__name__}: {e}). "
                f"Retry {attempts}/{max_retries} in {sleep_for:.1f}s."
            )
            time.sleep(sleep_for)