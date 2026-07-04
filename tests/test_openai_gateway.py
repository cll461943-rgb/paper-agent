from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import requests

from scholar_agent.infra.openai_gateway import GatewayConfig, create_gateway_server


class RecordingUpstreamHandler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.calls.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        payload = {
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {"total_tokens": 3},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return None


def serve(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def close(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


def test_gateway_returns_clear_503_without_api_keys():
    gateway = create_gateway_server(
        GatewayConfig(
            host="127.0.0.1",
            port=0,
            upstream_base_url="https://api.openai.example/v1",
            api_keys=(),
            timeout_seconds=2.0,
        )
    )
    serve(gateway)
    try:
        url = f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions"

        response = requests.post(url, json={"model": "gpt-test", "messages": []}, timeout=2)

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "upstream_api_key_missing"
    finally:
        close(gateway)


def test_gateway_forwards_chat_completions_with_pooled_api_keys():
    RecordingUpstreamHandler.calls = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordingUpstreamHandler)
    serve(upstream)
    gateway = create_gateway_server(
        GatewayConfig(
            host="127.0.0.1",
            port=0,
            upstream_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
            api_keys=("key-a", "key-b"),
            timeout_seconds=2.0,
        )
    )
    serve(gateway)
    try:
        url = f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions"

        first = requests.post(url, json={"model": "gpt-test", "messages": []}, timeout=2)
        second = requests.post(url, json={"model": "gpt-test", "messages": []}, timeout=2)

        assert first.status_code == 200
        assert second.status_code == 200
        assert [call["path"] for call in RecordingUpstreamHandler.calls] == [
            "/v1/chat/completions",
            "/v1/chat/completions",
        ]
        assert [call["authorization"] for call in RecordingUpstreamHandler.calls] == [
            "Bearer key-a",
            "Bearer key-b",
        ]
    finally:
        close(gateway)
        close(upstream)
