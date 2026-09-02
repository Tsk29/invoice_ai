"""Optional PaddleOCR + text-LLM hybrid extraction path.

Architecture: PaddleOCR reads the raw text off the image locally, then a
text-only Groq call (no image tokens) structures that text into the
InvoiceData schema - instead of one vision-LLM call doing both jobs at once.

This is NOT wired into the default requirements.txt: paddlepaddle alone is a
~100MB wheel (~186MB on Linux), and the recognition/detection models
PaddleOCR downloads on first use add another ~250-400MB depending on which
language packs get pulled in. Install requirements-ocr.txt to use this path;
the app detects the import failure and disables the option in the UI rather
than crashing.

Spike findings this module is built on (see README's OCR Architecture
section): on this CPU-only dev machine, a single-image OCR pass ranged from
~9s (Tamil, warm model cache) to ~37s (English, first "server" detection
model load) - not a clean win over one vision-LLM call without a GPU. Tamil
recognition quality was genuinely good (0.9-1.0 confidence on most fields),
and critically it isolates a line like "TOTAL" / "Rs.1,370.00" as its own
text region, which should make it easier for the downstream text LLM to
report the literal printed total instead of recomputing it - the same bug
class the vision-only prompt already had to be patched for.
"""
import json
import tempfile
import time

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# App's language picker -> PaddleOCR language code. PaddleOCR's coverage of
# Spanish/French/German recognition is not as mature as its English/Chinese
# models; "Other" falls back to English since PaddleOCR needs a specific
# code, not a wildcard.
LANGUAGE_CODE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Tamil": "ta",
    "Other": "en",
}

_ocr_engines = {}  # cached per language code - model loading takes seconds


def get_ocr_engine(language: str):
    if not PADDLEOCR_AVAILABLE:
        raise RuntimeError(f"paddleocr is not installed ({_IMPORT_ERROR}). See requirements-ocr.txt.")
    lang_code = LANGUAGE_CODE_MAP.get(language, "en")
    if lang_code not in _ocr_engines:
        _ocr_engines[lang_code] = PaddleOCR(use_textline_orientation=True, lang=lang_code)
    return _ocr_engines[lang_code]


def extract_text_paddleocr(image_bytes: bytes, language: str = "English") -> tuple[str, float]:
    """Run PaddleOCR over raw image bytes. Returns (raw_text, elapsed_seconds).

    PaddleOCR's predict() takes a file path, not bytes, hence the temp file.
    Text lines come back in the detector's reading order, which is usually
    close to visual order for a simple invoice layout but isn't guaranteed -
    the downstream structuring prompt is told this explicitly.
    """
    ocr = get_ocr_engine(language)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
        f.write(image_bytes)
        f.flush()
        t0 = time.time()
        result = ocr.predict(f.name)
        elapsed = time.time() - t0

    lines = []
    for res in result:
        lines.extend(res.get("rec_texts", []))
    return "\n".join(lines), elapsed


STRUCTURING_PROMPT_TEMPLATE = """
You are an invoice structuring agent. Below is raw text extracted by OCR
from an invoice written in {language}. The OCR reading order may not
exactly match the original visual layout, and individual characters may
occasionally be misrecognized.

Structure this text into JSON matching this schema: {schema}.
Include a confidence score (0.0 to 1.0) per field in a separate
'confidence_scores' object - lower it for any field you had to infer rather
than read directly.
If a field is not present in the text, return it as null.
IMPORTANT: total_amount must be the literal total figure that appears in
the text (e.g. next to a line like "TOTAL" or "Grand Total"), not a value
you compute by adding subtotal and tax yourself - if they look inconsistent,
report the literal total anyway; that inconsistency is a real finding.
Return the result strictly in JSON format with 'data' and 'confidence_scores' keys.

OCR TEXT:
---
{raw_text}
---
"""
