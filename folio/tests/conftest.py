"""Parsing the 200-page report takes ~40 s, so parsed documents are shared across test files."""
import os
import pytest

UP = os.environ.get("FOLIO_SAMPLES", "/mnt/user-data/uploads")
FILES = {"imac": "iMac_PER_Oct2024.pdf", "gpo": "GPO-Anual-report.pdf", "gep": "GEP-Jun-2026.pdf"}
_cache = {}


def parsed(name):
    from rag.ingest.parser import parse_pdf
    if name not in _cache:
        _cache[name] = parse_pdf(os.path.join(UP, FILES[name]))
    return _cache[name]


def sample(name):
    return os.path.join(UP, FILES[name])


needs_samples = pytest.mark.skipif(not os.path.exists(os.path.join(UP, FILES["imac"])),
                                   reason="sample PDFs not present")
