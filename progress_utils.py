from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = False,
) -> Iterable[T]:
    """Wrap an iterable with tqdm when available, otherwise return it unchanged."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = False,
) -> Iterator[T]:
    wrapped = progress(iterable, total=total, desc=desc, leave=leave)
    for item in wrapped:
        yield item
