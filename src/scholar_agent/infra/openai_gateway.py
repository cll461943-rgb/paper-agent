from __future__ import annotations

import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests


def split_api_keys(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    normalized = raw_value.replace(";", ",").replace("\n", ",")
    return tuple(item.strip() for item in normalized.split(",") if item.strip())


@dataclass(frozen=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 3010
    upstream_base_url: str = "https://api.openai.com/v1"
    api_keys: tuple[str, ...] = ()
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        api_keys = split_api_keys(os.getenv("OPENAI_API_KEYS"))
        if not api_keys:
            api_keys = split_api_keys(os.getenv("OPENAI_API_KEY"))
        return cls(
            host=os.getenv("OPENAI_GATEWAY_HOST", "127.0.0.1"),
            port=int(os.getenv("OPENAI_GATEWAY_PORT", "3010")),
            upstream_base_url=os.getenv(
                "OPENAI_GATEWAY_UPSTREAM_BASE_URL",
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ),
            api_keys=api_keys,
            timeout_seconds=float(os.getenv("OPENAI_GATEWAY_TIMEOUT_SECONDS", "60")),
        )


class GatewayState:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._api_key_index = 0

    def next_api_key(self) -> str:
        if not self.config.api_keys:
            return ""
        key = self.config.api_keys[self._api_key_index % len(self.config.api_keys)]
        self._api_key_index += 1
        return key

    @property
    def upstream_chat_completions_url(self) -> str:
        return f"{self.config.upstream_base_url.rstrip('/')}/chat/completions"


class OpenAICompatibleGatewayHandler(BaseHTTPRequestHandler):
    state: GatewayState

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(
                200,
                {
                    "ok": True,
                    "upstream_configured": bool(self.state.config.api_keys),
                    "key_count": len(self.state.config.api_keys),
                },
            )
            return
        self._write_json(404, {"error": {"code": "not_found", "message": "Not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"code": "not_found", "message": "Not found"}})
            return

        api_key = self.state.next_api_key()
        if not api_key:
            self._discard_request_body()
            self._write_json(
                503,
                {
                    "error": {
                        "code": "upstream_api_key_missing",
                        "message": "Configure OPENAI_API_KEY or OPENAI_API_KEYS before using the gateway.",
                    }
                },
            )
            return

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._write_json(400, {"error": {"code": "invalid_json", "message": str(exc)}})
            return

        if payload.get("stream") is True:
            self._write_json(
                501,
                {
                    "error": {
                        "code": "streaming_not_supported",
                        "message": "This gateway slice supports non-streaming chat completions only.",
                    }
                },
            )
            return

        try:
            response = requests.post(
                self.state.upstream_chat_completions_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=self.state.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            self._write_json(
                502,
                {
                    "error": {
                        "code": "upstream_request_failed",
                        "message": str(exc),
                    }
                },
            )
            return

        content_type = response.headers.get("Content-Type", "application/json")
        self.send_response(response.status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response.content)))
        self.end_headers()
        self.wfile.write(response.content)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("empty request body")
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _discard_request_body(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 0:
            self.rfile.read(length)

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return None


def create_gateway_server(config: GatewayConfig) -> ThreadingHTTPServer:
    state = GatewayState(config)

    class Handler(OpenAICompatibleGatewayHandler):
        pass

    Handler.state = state
    return ThreadingHTTPServer((config.host, config.port), Handler)


def main() -> None:
    from scholar_agent.infra.config import _load_dotenv_if_present

    _load_dotenv_if_present()
    config = GatewayConfig.from_env()
    server = create_gateway_server(config)
    print(
        f"OpenAI-compatible gateway listening on http://{config.host}:{server.server_port}/v1 "
        f"with {len(config.api_keys)} configured upstream key(s)."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
