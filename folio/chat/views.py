"""Function-based Django views. One HTML page; everything else is JSON."""
import json
import logging
import re
import uuid

import pymupdf
from django.conf import settings as dj
from django.http import HttpResponse, JsonResponse, Http404
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods

from rag.config import settings as rs
from rag.embeddings import embedder_info
from rag.ingest.profile import STRATEGIES
from rag.llm import status as llm_status
from rag.workflow import ask

from .models import Document, Message
from .services import start_ingest, remove

log = logging.getLogger(__name__)


def _err(msg, status=400):
    return JsonResponse({"error": msg}, status=status)


def _doc(doc_id) -> Document:
    try:
        return Document.objects.get(pk=uuid.UUID(hex=doc_id))
    except (ValueError, Document.DoesNotExist):
        raise Http404("No such document")


def _health():
    emb = embedder_info()
    return {"embedder": emb, "llm": llm_status(), "rerank": rs.rerank_enabled,
            "warnings": ([] if emb["semantic"] else
                         ["Embeddings are the offline hashed fallback (keyword-like, not semantic). "
                          "Allow the fastembed model download or set EMBEDDINGS_PROVIDER=gemini."]) +
                        ([] if llm_status()["answer_llm"] else
                         ["No GROQ_API_KEY: answers are extractive (top sentences) and routing uses rules."])}


_recovered = False


@ensure_csrf_cookie
def index(request):
    global _recovered
    if not _recovered:                      # first request after a restart
        from .services import recover_interrupted
        recover_interrupted()
        _recovered = True
    boot = {"documents": [d.as_dict() for d in Document.objects.all()],
            "strategies": {k: {"label": v["label"], "why": v["why"]} for k, v in STRATEGIES.items()}}
    return render(request, "chat/index.html", {"boot": json.dumps(boot)})


def api_health(request):
    return JsonResponse(_health())


@require_http_methods(["GET", "POST"])
def api_documents(request):
    if request.method == "GET":
        return JsonResponse({"documents": [d.as_dict() for d in Document.objects.all()]})
    f = request.FILES.get("file")
    if not f:
        return _err("Choose a PDF to upload.")
    if not f.name.lower().endswith(".pdf"):
        return _err("Only PDF files can be indexed.")
    if f.size > dj.MAX_PDF_BYTES:
        return _err(f"The file is larger than {dj.MAX_PDF_BYTES // 2**20} MB.")
    head = f.read(5)
    f.seek(0)
    if head != b"%PDF-":
        return _err("This file does not look like a PDF.")
    profile = request.POST.get("profile", "")
    doc = Document.objects.create(filename=f.name[:300], file=f, size=f.size,
                                  profile_override=profile if profile in STRATEGIES else "")
    start_ingest(doc)
    return JsonResponse({"document": doc.as_dict()}, status=201)


@require_http_methods(["GET", "DELETE", "POST"])
def api_document(request, doc_id):
    doc = _doc(doc_id)
    if request.method == "DELETE":
        remove(doc)
        return JsonResponse({"deleted": doc_id})
    if request.method == "POST":                       # re-index, optionally with another strategy
        body = json.loads(request.body or "{}")
        prof = body.get("profile", "")
        doc.profile_override = prof if prof in STRATEGIES else ""
        doc.save(update_fields=["profile_override"])
        start_ingest(doc)
        doc.refresh_from_db()
    return JsonResponse({"document": doc.as_dict(full=True)})


def api_page_image(request, doc_id, page):
    doc = _doc(doc_id)
    cache = rs.data_dir / "cache" / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{doc.doc_id}_{page}.png"
    if not f.exists():
        pdf = pymupdf.open(doc.file.path)
        if not 1 <= page <= len(pdf):
            raise Http404("No such page")
        f.write_bytes(pdf[page - 1].get_pixmap(dpi=96).tobytes("png"))
    resp = HttpResponse(f.read_bytes(), content_type="image/png")
    resp["Cache-Control"] = "max-age=86400"
    return resp


@require_http_methods(["POST"])
def api_ask(request):
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return _err("Send JSON.")
    q = (body.get("question") or "").strip()
    if not q:
        return _err("Type a question first.")
    if len(q) > 1000:
        return _err("Keep questions under 1,000 characters.")
    mode = body.get("mode", "advanced")
    if mode not in ("advanced", "naive", "compare"):
        return _err("Unknown mode.")
    ready = {d.doc_id for d in Document.objects.filter(status="ready")}
    ids = [d for d in body.get("doc_ids") or [] if d in ready] or sorted(ready)
    if not ids:
        return _err("Upload a document and wait until it is ready.")
    try:
        result = ask(q, ids, mode)
    except Exception as exc:
        log.exception("ask failed")
        return _err(f"The pipeline failed: {exc}", 500)
    msg = Message.objects.create(question=q, doc_ids=ids, mode=mode, result=result)
    return JsonResponse({"id": msg.pk, "question": q, "mode": mode, "doc_ids": ids, "result": result})


def api_history(request):
    return JsonResponse({"messages": [{"id": m.pk, "question": m.question, "mode": m.mode, "doc_ids": m.doc_ids,
                                       "result": m.result, "created": m.created.isoformat()}
                                      for m in Message.objects.all()[:30]][::-1]})


def api_samples(request):
    from rag.evaluate import load_questions
    out = {}
    qs = load_questions()
    for d in Document.objects.filter(status="ready"):
        mine = [q["question"] for q in qs if re.search(q["file_pattern"], d.filename, re.I)][:6]
        if not mine:      # unknown upload: questions built from its own outline
            names = [o["name"] for o in (d.meta or {}).get("outline", [])
                     if o["depth"] <= 1 and 2 <= len(o["name"].split()) <= 7 and not o["name"].isupper()
                     and o["chunks"] >= 3][:3]
            mine = [f"What does the section “{n}” say?" for n in names[:3]] + \
                   ["What are the main findings of this document?"]
        out[d.doc_id] = mine
    return JsonResponse({"samples": out})


@require_http_methods(["POST"])
def api_evaluate(request):
    from rag.evaluate import run_eval
    body = json.loads(request.body or "{}")
    docs = {d.doc_id: d.filename for d in Document.objects.filter(status="ready")}
    if body.get("doc_id"):
        docs = {k: v for k, v in docs.items() if k == body["doc_id"]}
    if not docs:
        return _err("No ready documents to evaluate.")
    return JsonResponse(run_eval(docs, modes=body.get("modes") or ["naive", "advanced"]))
