"""python manage.py evaluate [--modes naive advanced] — scores every ready document that has
checked questions in eval/questions.json and writes eval/results.json + eval/results.md."""
from pathlib import Path

from django.core.management.base import BaseCommand

from chat.models import Document
from rag.evaluate import run_eval, to_markdown


class Command(BaseCommand):
    help = "Run the evaluation set (naive vs advanced)"

    def add_arguments(self, parser):
        parser.add_argument("--modes", nargs="+", default=["naive", "advanced"])

    def handle(self, *args, **o):
        import json
        docs = {d.doc_id: d.filename for d in Document.objects.filter(status="ready")}
        if not docs:
            self.stderr.write("No ready documents. Run `python manage.py ingest <pdf>` first.")
            return

        def progress(qid, row):
            marks = "  ".join(f"{m}: rank {row[m]['rank'] or '–'} {'✓' if row[m]['correct'] else '✗'}"
                              for m in o["modes"] if m in row)
            self.stdout.write(f"  {qid:<9} {marks}")

        res = run_eval(docs, o["modes"], progress)
        out = Path(__file__).resolve().parents[3] / "eval"
        (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        md = to_markdown(res)
        (out / "results.md").write_text(md)
        self.stdout.write("\n" + md.split("\n## ")[0])
