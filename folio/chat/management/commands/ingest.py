"""python manage.py ingest file.pdf [file2.pdf ...] [--profile visual]  (synchronous)"""
import os

from django.core.files import File
from django.core.management.base import BaseCommand

from chat.models import Document
from chat.services import start_ingest


class Command(BaseCommand):
    help = "Index one or more PDFs synchronously"

    def add_arguments(self, parser):
        parser.add_argument("paths", nargs="+")
        parser.add_argument("--profile", default="", choices=["", "narrative", "long_structured", "visual"])

    def handle(self, *paths, **opts):
        for p in opts["paths"]:
            name = os.path.basename(p)
            Document.objects.filter(filename=name).delete()
            with open(p, "rb") as fh:
                doc = Document.objects.create(filename=name, size=os.path.getsize(p),
                                              profile_override=opts["profile"])
                doc.file.save(name, File(fh), save=True)
            self.stdout.write(f"Indexing {name} ...")
            start_ingest(doc, sync=True)
            doc.refresh_from_db()
            s = doc.meta.get("stats", {})
            self.stdout.write(f"  {doc.status}: {doc.profile} · {s.get('chunks')} chunks · "
                              f"{s.get('tables')} tables · {s.get('graph_nodes')} graph nodes · {s.get('seconds')} s"
                              + (f" · {doc.message}" if doc.status != 'ready' else ""))
