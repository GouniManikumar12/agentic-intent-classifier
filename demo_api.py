import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from combined_inference import classify_query


class DemoHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/classify":
            self._send_json(404, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
            return

        text = payload.get("text", "").strip()
        if not text:
            self._send_json(400, {"error": "text_required"})
            return

        self._send_json(200, classify_query(text))

    def log_message(self, format: str, *args):
        return


def main():
    server = HTTPServer(("127.0.0.1", 8008), DemoHandler)
    print("Serving demo API on http://127.0.0.1:8008/classify")
    server.serve_forever()


if __name__ == "__main__":
    main()
