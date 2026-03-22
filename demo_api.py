import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from combined_inference import classify_query
from config import DEFAULT_API_HOST, DEFAULT_API_PORT, HEAD_CONFIGS, PROJECT_VERSION
from model_runtime import get_head
from schemas import (
    SchemaValidationError,
    default_version_payload,
    validate_classify_request,
    validate_classify_response,
    validate_health_response,
    validate_version_response,
)


class DemoHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body or b"{}")
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                "invalid_json",
                [{"field": "body", "message": f"invalid JSON: {exc.msg}", "type": "parse_error"}],
            ) from exc

    def _handle_classify(self):
        try:
            request_payload = validate_classify_request(self._read_json_body())
            response_payload = validate_classify_response(classify_query(request_payload["text"]))
        except SchemaValidationError as exc:
            status_code = 400 if exc.code == "invalid_json" else 422 if exc.code == "request_validation_failed" else 500
            self._send_json(status_code, {"error": exc.code, "details": exc.details})
            return

        self._send_json(200, response_payload)

    def _handle_health(self):
        payload = {
            "status": "ok",
            "system_version": PROJECT_VERSION,
            "heads": [get_head(head_name).status() for head_name in HEAD_CONFIGS],
        }
        try:
            response_payload = validate_health_response(payload)
        except SchemaValidationError as exc:
            self._send_json(500, {"error": exc.code, "details": exc.details})
            return
        self._send_json(200, response_payload)

    def _handle_version(self):
        try:
            response_payload = validate_version_response(default_version_payload())
        except SchemaValidationError as exc:
            self._send_json(500, {"error": exc.code, "details": exc.details})
            return
        self._send_json(200, response_payload)

    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
            return
        if self.path == "/version":
            self._handle_version()
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send_json(404, {"error": "not_found"})
            return
        self._handle_classify()

    def log_message(self, format: str, *args):
        return


def main():
    server = HTTPServer((DEFAULT_API_HOST, DEFAULT_API_PORT), DemoHandler)
    print(f"Serving demo API on http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
