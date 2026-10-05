from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProviderOutput:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderConfigurationError(RuntimeError):
    """Raised before inference when a provider is not configured correctly."""


class ProviderRequestError(RuntimeError):
    """A sanitized API error safe to write into experiment artifacts."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def public_endpoint(value: str) -> str:
    """Remove query strings/userinfo before persisting an endpoint."""
    if "://" not in value:
        return value
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def image_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_openai_vision_payload(
    *,
    model_id: str,
    prompt: str,
    image_uri: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an image request without ever accepting or embedding a label."""
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": False,
    }
    for key in ("temperature", "top_p", "max_tokens", "seed"):
        if key in generation and generation[key] is not None:
            payload[key] = generation[key]
    return payload


def extract_openai_message_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderRequestError(
            "OpenAI-compatible response is missing choices[0].message.content",
            retryable=False,
        ) from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                pieces.append(str(block["text"]))
        if pieces:
            return "".join(pieces)
    raise ProviderRequestError(
        "OpenAI-compatible response content is not text",
        retryable=False,
    )


def _retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


class OpenAICompatibleVisionProvider:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        prompt: str,
        generation: Mapping[str, Any],
        timeout_seconds: float,
    ) -> None:
        self.model_id = str(config["model_id"])
        self.base_url = str(config["base_url"])
        self.endpoint = _chat_completions_url(self.base_url)
        self.prompt = prompt
        self.generation = dict(generation)
        self.timeout_seconds = timeout_seconds
        key_env = str(config.get("api_key_env", ""))
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        if bool(config.get("api_key_required", True)) and not self.api_key:
            raise ProviderConfigurationError(
                f"Missing required API key environment variable: {key_env}"
            )

    def predict(self, image_path: str | Path) -> ProviderOutput:
        payload = build_openai_vision_payload(
            model_id=self.model_id,
            prompt=self.prompt,
            image_uri=image_data_uri(image_path),
            generation=self.generation,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read()
                request_id = response.headers.get("x-request-id")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            status = int(exc.code)
            retryable = status in {408, 409, 425, 429} or status >= 500
            raise ProviderRequestError(
                f"HTTP {status} from {public_endpoint(self.endpoint)}: {body}",
                retryable=retryable,
                status_code=status,
                retry_after_seconds=_retry_after(exc.headers),
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderRequestError(
                f"Network error from {public_endpoint(self.endpoint)}: {exc}",
                retryable=True,
            ) from exc

        try:
            decoded = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderRequestError(
                "OpenAI-compatible endpoint returned invalid JSON",
                retryable=False,
            ) from exc

        text = extract_openai_message_text(decoded)
        metadata: dict[str, Any] = {
            "response_id": decoded.get("id"),
            "request_id": request_id,
        }
        if isinstance(decoded.get("usage"), Mapping):
            metadata["usage"] = dict(decoded["usage"])
        return ProviderOutput(text=text, metadata=metadata)

    def close(self) -> None:
        return None


class GOTGradioProvider:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> None:
        try:
            from gradio_client import Client
            try:
                from gradio_client import handle_file
            except ImportError:
                from gradio_client import file as handle_file
        except ImportError as exc:
            raise ProviderConfigurationError(
                "GOT-OCR2.0 Gradio API requires gradio_client; "
                "install requirements-api.txt"
            ) from exc

        self._handle_file = handle_file
        self.source = str(config["source"])
        self.api_name = str(config.get("api_name", "/run_GOT"))
        self.mode = str(config.get("mode", "plain texts OCR"))
        token_env = str(config.get("token_env", "HF_TOKEN"))
        token = os.environ.get(token_env) or None

        kwargs: dict[str, Any] = {}
        signature = inspect.signature(Client)
        if token:
            if "hf_token" in signature.parameters:
                kwargs["hf_token"] = token
            elif "token" in signature.parameters:
                kwargs["token"] = token
        if "httpx_kwargs" in signature.parameters:
            kwargs["httpx_kwargs"] = {"timeout": timeout_seconds}
        try:
            self.client = Client(self.source, verbose=False, **kwargs)
        except Exception as exc:
            raise ProviderConfigurationError(
                f"Failed to initialize GOT Gradio source {self.source}: {exc}"
            ) from exc

    def predict(self, image_path: str | Path) -> ProviderOutput:
        try:
            result = self.client.predict(
                self._handle_file(str(image_path)),
                self.mode,
                "",
                "",
                "",
                api_name=self.api_name,
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"GOT Gradio request failed: {exc}",
                retryable=True,
            ) from exc

        if isinstance(result, (list, tuple)):
            text = "" if not result else str(result[0] or "")
        else:
            text = str(result or "")
        if text.lstrip().lower().startswith("error:"):
            raise ProviderRequestError(text[:1000], retryable=True)
        return ProviderOutput(text=text, metadata={"mode": self.mode})

    def close(self) -> None:
        return None


def extract_paddle_text(result: Any) -> tuple[str, dict[str, Any]]:
    pages = getattr(result, "pages", None)
    if pages is None:
        raise ProviderRequestError(
            "PaddleOCR official API result is missing pages",
            retryable=False,
        )
    texts = [str(getattr(page, "markdown_text", "") or "") for page in pages]
    return "\n".join(texts), {
        "job_id": getattr(result, "job_id", None),
        "pages": len(texts),
    }


class PaddleOCROfficialProvider:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> None:
        try:
            from paddleocr import PaddleOCRClient, PaddleOCRVLOptions
        except ImportError as exc:
            raise ProviderConfigurationError(
                "PaddleOCR-VL official API requires paddleocr>=3.6.0; "
                "install requirements-api.txt"
            ) from exc

        token_env = str(config.get("token_env", "PADDLEOCR_ACCESS_TOKEN"))
        token = os.environ.get(token_env, "")
        if not token:
            raise ProviderConfigurationError(
                f"Missing required API token environment variable: {token_env}"
            )
        base_url_env = str(config.get("base_url_env", "PADDLEOCR_BASE_URL"))
        base_url = os.environ.get(base_url_env) or None
        self.model_id = str(config["model_id"])
        self.client = PaddleOCRClient(
            token=token,
            base_url=base_url,
            request_timeout=float(config.get("request_timeout_seconds", timeout_seconds)),
            poll_timeout=float(config.get("poll_timeout_seconds", 600.0)),
        )
        options_cfg = dict(config.get("options", {}))
        self.options = PaddleOCRVLOptions(**options_cfg)

    def predict(self, image_path: str | Path) -> ProviderOutput:
        try:
            result = self.client.parse_document(
                model=self.model_id,
                file_path=str(image_path),
                options=self.options,
            )
        except Exception as exc:
            name = type(exc).__name__
            non_retryable = name in {"AuthError", "InvalidRequestError"}
            raise ProviderRequestError(
                f"PaddleOCR official API {name}: {exc}",
                retryable=not non_retryable,
            ) from exc
        text, metadata = extract_paddle_text(result)
        return ProviderOutput(text=text, metadata=metadata)

    def close(self) -> None:
        self.client.close()


def create_provider(
    config: Mapping[str, Any],
    *,
    prompt: str,
    generation: Mapping[str, Any],
    timeout_seconds: float,
) -> Any:
    provider = str(config["provider"])
    if provider == "openai_compatible":
        return OpenAICompatibleVisionProvider(
            config,
            prompt=prompt,
            generation=generation,
            timeout_seconds=timeout_seconds,
        )
    if provider == "gradio_space":
        return GOTGradioProvider(config, timeout_seconds=timeout_seconds)
    if provider == "paddleocr_official":
        return PaddleOCROfficialProvider(
            config,
            timeout_seconds=timeout_seconds,
        )
    raise ProviderConfigurationError(f"Unsupported API provider: {provider}")


def validate_provider_environment(config: Mapping[str, Any]) -> list[str]:
    """Return preflight issues without making any network request."""
    issues: list[str] = []
    provider = str(config.get("provider", ""))
    if provider == "openai_compatible":
        key_env = str(config.get("api_key_env", ""))
        if bool(config.get("api_key_required", True)) and not os.environ.get(key_env):
            issues.append(f"missing environment variable {key_env}")
        if not config.get("base_url"):
            issues.append("missing base_url")
    elif provider == "gradio_space":
        try:
            import gradio_client  # noqa: F401
        except ImportError:
            issues.append("missing Python package gradio_client")
        if not config.get("source"):
            issues.append("missing Gradio source")
    elif provider == "paddleocr_official":
        token_env = str(config.get("token_env", "PADDLEOCR_ACCESS_TOKEN"))
        if not os.environ.get(token_env):
            issues.append(f"missing environment variable {token_env}")
        try:
            from paddleocr import PaddleOCRClient  # noqa: F401
        except (ImportError, AttributeError):
            issues.append("missing paddleocr>=3.6.0 with PaddleOCRClient")
    else:
        issues.append(f"unsupported provider {provider!r}")
    return issues
