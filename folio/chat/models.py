import uuid

from django.db import models


class Document(models.Model):
    STATUS = [("queued", "Queued"), ("processing", "Processing"), ("ready", "Ready"), ("error", "Error")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=300, blank=True)
    filename = models.CharField(max_length=300)
    file = models.FileField(upload_to="pdfs/")
    size = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS, default="queued")
    stage = models.CharField(max_length=40, blank=True)
    progress = models.PositiveSmallIntegerField(default=0)
    message = models.CharField(max_length=300, blank=True)
    profile_override = models.CharField(max_length=30, blank=True)
    profile = models.CharField(max_length=30, blank=True)
    meta = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created"]

    @property
    def doc_id(self) -> str:
        return self.id.hex

    def as_dict(self, full=False):
        m = self.meta or {}
        d = {"id": self.doc_id, "title": self.title or self.filename, "filename": self.filename,
             "size": self.size, "status": self.status, "stage": self.stage, "progress": self.progress,
             "message": self.message, "profile": self.profile, "profile_override": self.profile_override,
             "pages": m.get("pages"), "stats": m.get("stats", {}), "reasons": m.get("reasons", []),
             "strategy_label": (m.get("strategy") or {}).get("label"), "why": (m.get("strategy") or {}).get("why"),
             "vision_note": m.get("vision_note"), "error": self.error, "created": self.created.isoformat()}
        if full:
            d["outline"] = m.get("outline", [])[:120]
            d["aliases"] = m.get("aliases", {})
            d["features"] = m.get("features", {})
            d["embedder"] = m.get("embedder", {})
        return d


class Message(models.Model):
    question = models.TextField()
    doc_ids = models.JSONField(default=list)
    mode = models.CharField(max_length=20, default="advanced")
    result = models.JSONField(default=dict)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]
