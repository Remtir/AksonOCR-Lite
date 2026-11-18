"""AksonOCR-Lite: Thai document OCR with a grapheme-cluster CTC recognizer (layout, text, tables) on ONNX Runtime."""
from .pipeline import Config, Converter
from .schema import Block, Line, Page

__version__ = "1.0.0"
__all__ = ["load", "Converter", "Config", "Page", "Block", "Line", "__version__"]


def load(**kw) -> Converter:
    """Build a converter. Keyword arguments are Config fields, e.g. load(device="cpu", dewarp=True)."""
    from .models import touch_manifest

    touch_manifest()
    return Converter(Config(**kw))
