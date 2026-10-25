"""Background ingestion. One worker at a time: parsing and embedding are CPU-bound, and a
200-page report would otherwise starve every other request."""
import logging
import threading
import traceback

from django.db import close_old_connections

from rag.pipeline import index_document
from rag.store import get_index

from .models import Document

log = logging.getLogger(__name__)
_worker = threading.Semaphore(1)


def _set(doc_id, **fields):
    Document.objects.filter(pk=doc_id).update(**fields)


def _run(pk):
    close_old_connections()
    with _worker:
        doc = Document.objects.get(pk=pk)
        _set(pk, status="processing", stage="parsing", progress=2, message="Starting", error="")

        def progress(stage, pct, msg=""):
            _set(pk, stage=stage, progress=int(pct), message=msg[:300])

        try:
            meta = index_document(doc.file.path, doc.doc_id, profile=doc.profile_override or None, progress=progress)
            _set(pk, status="ready", stage="ready", progress=100, message="Ready", meta=meta,
                 profile=meta["profile"], title=(meta.get("title") or doc.filename)[:300])
        except Exception as exc:
            log.exception("ingest failed")
            _set(pk, status="error", stage="error", message=str(exc)[:300], error=traceback.format_exc()[-4000:])
        finally:
            close_old_connections()


def start_ingest(doc: Document, sync: bool = False):
    Document.objects.filter(pk=doc.pk).update(status="queued", stage="queued", progress=0, message="Waiting")
    if sync:
        _run(doc.pk)
    else:
        threading.Thread(target=_run, args=(doc.pk,), daemon=True).start()


def remove(doc: Document):
    get_index().delete(doc.doc_id)
    doc.file.delete(save=False)
    doc.delete()


def recover_interrupted():
    """Documents left mid-ingest by a server restart are marked so the user can re-index."""
    Document.objects.filter(status__in=["queued", "processing"]).update(
        status="error", stage="error", message="Indexing was interrupted by a restart. Re-index to continue.")
