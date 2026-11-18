"""
app_server.py
=============
Unified Web Server for JobShield AI.
Serves the exact `index.html` webpage (the same webpage shown when clicking "Go Live")
and attaches it to the SQLite User History Database (`jobshield_history.db`) and
the 4-Branch Hybrid Ensemble Model (`fraud_model.pkl`).

Works out of the box with Python's built-in `http.server` (even if FastAPI/Uvicorn
are not installed in your virtual environment) and automatically opens the webpage
in your browser when launched from the VS Code terminal.
"""

import os
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from jobshield_model import load_artifact
from document_parser import parse_uploaded_document
from database import (
    init_db,
    log_scan,
    get_history,
    delete_scan,
    clear_all_history,
    get_dashboard_stats,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_artifact(os.path.join(BASE_DIR, "fraud_model.pkl"))
vectorizer = load_artifact(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
init_db()


class JobShieldRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal output clean
        pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_path = os.path.join(BASE_DIR, "index.html")
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/stats":
            metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
            metrics = getattr(model, "metrics_summary_", {})
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            self._send_json({"db_stats": get_dashboard_stats(), "model_metrics": metrics})
            return

        if parsed.path == "/api/history":
            qs = parse_qs(parsed.query)
            q = qs.get("q", [""])[0]
            verdict = qs.get("verdict", ["ALL"])[0]
            source = qs.get("source", ["ALL"])[0]
            records = get_history(limit=200, search_query=q, verdict_filter=verdict, source_filter=source)
            self._send_json({"records": records})
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        if parsed.path == "/api/analyze-text":
            req = json.loads(raw_body.decode("utf-8"))
            text = req.get("text", "")
            title = req.get("title", "")
            company = req.get("company", "")
            res = model.analyze_posting(vectorizer, text=text, title=title)
            scan_id = log_scan(
                analysis_result=res,
                extracted_text=text,
                source_type=req.get("source_type", "Direct Text"),
                file_name=req.get("file_name", "Pasted Job Text"),
                file_size_kb=round(len(text.encode("utf-8")) / 1024.0, 2),
                extraction_method=req.get("extraction_method", "Direct Text Input"),
                job_title=title or "Job Posting Scan",
                company_name=company or "Not Specified",
            )
            self._send_json({"scan_id": scan_id, "analysis": res, "db_stats": get_dashboard_stats()})
            return

        if parsed.path == "/api/clear-history":
            clear_all_history()
            self._send_json({"cleared": True, "db_stats": get_dashboard_stats()})
            return

        self.send_error(404, "Not Found")

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/history/"):
            scan_id = int(parsed.path.split("/")[-1])
            delete_scan(scan_id)
            self._send_json({"deleted": scan_id, "db_stats": get_dashboard_stats()})
            return
        self.send_error(404, "Not Found")


def run_server(port: int = 8000, open_browser: bool = True):
    url = f"http://127.0.0.1:{port}"
    server = HTTPServer(("127.0.0.1", port), JobShieldRequestHandler)
    print("=" * 64)
    print(f"🛡️  JobShield AI Web Application Running at: {url}")
    print("   (Serving index.html + SQLite History Database jobshield_history.db)")
    print("   Press Ctrl+C in the terminal to stop the server.")
    print("=" * 64)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped JobShield AI server.")
        server.server_close()


if __name__ == "__main__":
    run_server(port=8000, open_browser=True)
