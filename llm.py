"""Groq client with rate-limit waiting, model fallback and usage tracking."""
import json
import re
import time
from dataclasses import dataclass, field

import groq

from .config import AGENT_MODELS, FAST_MODELS, MAX_COMPLETION_TOKENS, groq_api_key

MAX_SHORT_WAIT = 12  # seconds worth waiting before switching model
MAX_LONG_WAIT = 65   # seconds worth waiting when every model is busy


class LLMError(Exception):
    """A problem the user can act on (missing key, quota, network)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    tokens: int = 0

    def as_message(self) -> dict:
        msg = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        return msg


class GroqLLM:
    def __init__(self, api_key: str | None = None):
        key = api_key or groq_api_key()
        if not key:
            raise LLMError("GROQ_API_KEY is missing. Add it to the .env file (see .env.example).")
        self.client = groq.Groq(api_key=key, max_retries=0, timeout=60)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, role: str = "agent",
             as_json: bool = False, max_tokens: int = MAX_COMPLETION_TOKENS) -> Reply:
        models = list(AGENT_MODELS if role == "agent" else FAST_MODELS)
        waits: dict[str, float] = {}
        self.last_error = ""
        for attempt in range(2):  # second pass only after waiting for a busy model
            for model in models:
                reply, wait = self._try_model(model, messages, tools, as_json, max_tokens)
                if reply:
                    return reply
                if wait is not None:
                    waits[model] = wait
            usable = {m: w for m, w in waits.items() if w <= MAX_LONG_WAIT}
            if attempt or not usable:
                break
            model = min(usable, key=usable.get)
            time.sleep(usable[model] + 0.5)
            models = [model]
        if self.last_error and not waits:
            raise LLMError(f"Groq rejected the request: {self.last_error}")
        raise LLMError("All Groq models are rate limited or unavailable right now. Wait a minute and try again.")

    def _try_model(self, model, messages, tools, as_json, max_tokens):
        """Return (reply, None), or (None, seconds to wait), or (None, None) to skip the model."""
        args = dict(model=model, messages=messages, temperature=0.2, max_completion_tokens=max_tokens)
        if "gpt-oss" in model:
            args |= dict(reasoning_effort="low", include_reasoning=False)
        elif "qwen" in model:
            args |= dict(reasoning_format="hidden")
        if tools:
            args |= dict(tools=tools, tool_choice="auto")
        if as_json:
            args |= dict(response_format={"type": "json_object"})
        bad_requests = 0
        waited = False
        while True:
            try:
                r = self.client.chat.completions.create(**args)
            except groq.RateLimitError as e:
                wait = _retry_after(e)
                if waited or wait > MAX_SHORT_WAIT:
                    return None, wait
                waited = True
                time.sleep(wait + 0.3)
                continue
            except groq.AuthenticationError:
                raise LLMError("Groq rejected the API key. Check GROQ_API_KEY.") from None
            except groq.BadRequestError as e:
                bad_requests += 1  # e.g. tool_use_failed / json_validate_failed: usually fine on retry
                if bad_requests >= 2:
                    self.last_error = _message(e)
                    return None, None
                args["temperature"] = 0.5
                continue
            except (groq.NotFoundError, groq.PermissionDeniedError, groq.InternalServerError):
                return None, None
            except groq.APIStatusError as e:
                if e.status_code in (413, 498, 503):
                    return None, None
                raise LLMError(f"Groq error {e.status_code}: {_message(e)}") from None
            except groq.APIConnectionError:
                raise LLMError("Could not reach Groq. Check the internet connection.") from None
            msg = r.choices[0].message
            calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in msg.tool_calls or []]
            return Reply(msg.content or "", calls, model, r.usage.total_tokens if r.usage else 0), None


def _retry_after(e: groq.RateLimitError) -> float:
    try:
        return float(e.response.headers.get("retry-after", ""))
    except ValueError:
        match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
        return int(match[1] or 0) * 60 + float(match[2]) if match else 20.0


def _message(e) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    return str(body.get("error", {}).get("message", e))[:300]


def parse_json(text: str) -> dict:
    """First JSON object in the text, or {}."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        value = json.loads(match[0]) if match else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
