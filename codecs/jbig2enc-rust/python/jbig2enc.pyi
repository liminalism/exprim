"""Type stubs for the jbig2enc extension module."""

from collections.abc import Buffer, Sequence
from typing import Literal

__version__: str

Preset = Literal[
    "default",
    "text",
    "generic",
    "lossless",
    "text_lossless",
    "text_lossless_max_compression",
    "text_symbol_unify",
    "text_symbol_unify_safe",
]
PixelFormat = Literal["bytes", "packed1", "gray8"]
Polarity = Literal["one_is_black", "zero_is_black"]

def encode(
    pixels: Buffer,
    width: int,
    height: int,
    *,
    preset: Preset | None = None,
    symbol_mode: bool = True,
    refine: bool = False,
    lossless: bool = False,
    pixel_format: PixelFormat = "bytes",
    stride: int | None = None,
    polarity: Polarity = "one_is_black",
    threshold: int = 128,
) -> bytes:
    """Encode one bilevel image as a standalone JBIG2 file.

    By default `pixels` is one byte per pixel, row-major, and any non-zero
    value is black. `pixel_format="packed1"` takes MSB-first packed rows
    (`stride` bytes apart, default `ceil(width / 8)`) in the given
    `polarity`; `"gray8"` thresholds 8-bit grayscale, black below
    `threshold`. `preset` names a whole configuration and cannot be combined
    with `symbol_mode` or `lossless`.
    """

def encode_for_pdf(
    pixels: Buffer,
    width: int,
    height: int,
    *,
    preset: Preset | None = None,
    symbol_mode: bool = True,
    refine: bool = False,
    lossless: bool = False,
    pixel_format: PixelFormat = "bytes",
    stride: int | None = None,
    polarity: Polarity = "one_is_black",
    threshold: int = 128,
) -> tuple[bytes | None, bytes]:
    """Encode one image as a PDF fragment: `(globals, page_data)`."""

def encode_document(
    pages: Sequence[tuple[Buffer, int, int]],
    *,
    preset: Preset | None = None,
    symbol_mode: bool = True,
    refine: bool = False,
    lossless: bool = False,
    pixel_format: PixelFormat = "bytes",
    stride: int | None = None,
    polarity: Polarity = "one_is_black",
    threshold: int = 128,
) -> tuple[bytes | None, list[bytes]]:
    """Encode several pages against one shared symbol dictionary."""

def decode(data: Buffer) -> tuple[bytes, int, int]:
    """Decode a standalone JBIG2 file's first page.

    Returns `(pixels, width, height)`, one byte per pixel, 1 = black --
    the same layout the encode functions accept.
    """

def decode_document(data: Buffer) -> list[tuple[bytes, int, int]]:
    """Decode every page of a standalone JBIG2 file, in stream order."""

def decode_pdf_stream(
    page_data: Buffer, globals: Buffer | None = None
) -> tuple[bytes, int, int]:
    """Decode a PDF-embedded page stream against its optional globals."""

def pack_grayscale(
    pixels: Buffer, width: int, height: int, threshold: int = 128
) -> bytes:
    """Threshold 8-bit grayscale into encoder input, inverting polarity."""

def presets() -> list[str]:
    """The names `preset=` accepts."""

def version() -> str: ...
def build_info() -> str: ...
