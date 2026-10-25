import os
import time

os.environ.setdefault("EMBEDDINGS_PROVIDER", "hash")
os.environ.setdefault("RERANK", "0")
os.environ.pop("GROQ_API_KEY", None)

from django.test import TransactionTestCase, Client

from chat.models import Document

SAMPLE = os.path.join(os.environ.get("FOLIO_SAMPLES", "/mnt/user-data/uploads"), "iMac_PER_Oct2024.pdf")


class FlowTest(TransactionTestCase):
    def setUp(self):
        self.c = Client()

    def _upload(self):
        with open(SAMPLE, "rb") as fh:
            r = self.c.post("/api/documents", {"file": fh})
        self.assertEqual(r.status_code, 201, r.content)
        return r.json()["document"]["id"]

    def _wait(self, doc_id, timeout=120):
        t = time.time()
        while time.time() - t < timeout:
            d = self.c.get(f"/api/documents/{doc_id}").json()["document"]
            if d["status"] in ("ready", "error"):
                return d
            time.sleep(0.5)
        self.fail("ingest timed out")

    def test_upload_ask_delete(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        doc_id = self._upload()
        d = self._wait(doc_id)
        self.assertEqual(d["status"], "ready", d.get("message"))
        self.assertEqual(d["profile"], "visual")
        self.assertGreater(d["stats"]["tables"], 0)

        r = self.c.post("/api/ask", {"question": "How much less energy than ENERGY STAR does iMac use?",
                                     "doc_ids": [doc_id]}, content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content)
        res = r.json()["result"]
        self.assertTrue(res["passages"])
        self.assertIn("Understand", [t["stage"] for t in res["trace"]])

        r = self.c.post("/api/ask", {"question": "What is the total product footprint?", "mode": "compare"},
                        content_type="application/json")
        self.assertEqual(set(r.json()["result"]), {"advanced", "naive"})

        img = self.c.get(f"/api/documents/{doc_id}/page/2.png")
        self.assertEqual(img["Content-Type"], "image/png")
        self.assertEqual(self.c.get("/api/health").status_code, 200)
        self.assertEqual(len(self.c.get("/api/history").json()["messages"]), 2)

        self.assertEqual(self.c.delete(f"/api/documents/{doc_id}").status_code, 200)
        self.assertFalse(Document.objects.exists())

    def test_validation(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        r = self.c.post("/api/documents", {"file": SimpleUploadedFile("x.pdf", b"hello")})
        self.assertEqual(r.json()["error"], "This file does not look like a PDF.")
        r = self.c.post("/api/ask", {"question": ""}, content_type="application/json")
        self.assertEqual(r.json()["error"], "Type a question first.")
        r = self.c.post("/api/ask", {"question": "hi"}, content_type="application/json")
        self.assertIn("Upload a document", r.json()["error"])
