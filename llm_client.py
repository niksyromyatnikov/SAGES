from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import hashlib
import json
import os
import re


_TRANSIENT_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _prompt_ref(prompt: str) -> Dict[str, Any]:
    """Return a compact, reproducible reference to a prompt without embedding it."""
    p = prompt if isinstance(prompt, str) else str(prompt)
    return {
        "sha256": _sha256_hex(p),
        "chars": len(p),
        "lines": p.count("\n") + 1 if p else 0,
    }


def build_request_debug(spec: "LLMCallSpec") -> Dict[str, Any]:
    """Build a sanitized debug view of the exact HTTP request parameters.

    - Includes resolved URL and JSON payload shape/values.
    - Redacts API keys and replaces prompt bodies with a stable hash reference.
    """

    provider = (spec.provider or "").strip().lower()
    pref = _prompt_ref(spec.prompt)
    prompt_placeholder = f"<PROMPT sha256={pref['sha256']} chars={pref['chars']}>"

    if provider == "openai":
        base = (spec.api_base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com").rstrip("/")
        url = f"{base}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt_placeholder}],
            "temperature": spec.temperature,
            "max_tokens": spec.max_output_tokens,
        }
        opts = spec.provider_options or {}
        if isinstance(opts, dict):
            if "reasoning_effort" in opts:
                payload["reasoning_effort"] = opts.get("reasoning_effort")
            if "max_completion_tokens" in opts and opts.get("max_completion_tokens") is not None:
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = opts.get("max_completion_tokens")
        if spec.force_json_object:
            payload["response_format"] = {"type": "json_object"}

        return {
            "provider": "openai",
            "url": url,
            "headers": {"Authorization": "Bearer ***", "Content-Type": "application/json"},
            "payload": payload,
            "prompt_ref": pref,
        }

    if provider == "gemini":
        base = (spec.api_base_url or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")
        # API key is passed via querystring in the real request; redact it here.
        url = f"{base}/v1beta/models/{spec.model}:generateContent?key=***"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt_placeholder}]}],
            "generationConfig": {
                "temperature": spec.temperature,
                "maxOutputTokens": spec.max_output_tokens,
            },
        }
        opts = spec.provider_options or {}
        if isinstance(opts, dict):
            thinking_cfg = opts.get("thinkingConfig")
            if isinstance(thinking_cfg, dict) and thinking_cfg:
                payload["generationConfig"]["thinkingConfig"] = thinking_cfg

        return {
            "provider": "gemini",
            "url": url,
            "headers": {"Content-Type": "application/json"},
            "payload": payload,
            "prompt_ref": pref,
        }

    if provider in {"claude", "anthropic"}:
        base = (spec.api_base_url or os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com").rstrip("/")
        url = f"{base}/v1/messages"
        headers: Dict[str, Any] = {
            "x-api-key": "***",
            "anthropic-version": os.getenv("ANTHROPIC_VERSION") or "2023-06-01",
            "content-type": "application/json",
        }
        beta = (os.getenv("ANTHROPIC_BETA") or "").strip()
        if beta:
            headers["anthropic-beta"] = beta

        payload = {
            "model": spec.model,
            "max_tokens": int(spec.max_output_tokens),
            "temperature": float(spec.temperature),
            "messages": [{"role": "user", "content": prompt_placeholder}],
        }

        opts = spec.provider_options or {}
        if isinstance(opts, dict) and isinstance(opts.get("thinking"), dict):
            payload["thinking"] = opts.get("thinking")
        else:
            thinking_budget = (os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS") or "").strip()
            if thinking_budget:
                try:
                    payload["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
                except Exception:
                    pass

        return {
            "provider": "claude",
            "url": url,
            "headers": headers,
            "payload": payload,
            "prompt_ref": pref,
        }

    return {
        "provider": provider,
        "error": f"Unsupported provider for request debug: {spec.provider!r}",
        "prompt_ref": pref,
    }


def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _sum_ints(*xs: Optional[int]) -> Optional[int]:
    vals = [x for x in xs if isinstance(x, int)]
    return sum(vals) if vals else None


def _normalize_usage(input_tokens: Any = None, output_tokens: Any = None, reasoning_tokens: Any = None, total_tokens: Any = None, provider_usage: Any = None) -> Dict[str, Any]:
    inp = _to_int_or_none(input_tokens)
    out = _to_int_or_none(output_tokens)
    rea = _to_int_or_none(reasoning_tokens)
    tot = _to_int_or_none(total_tokens)
    if tot is None:
        tot = _sum_ints(inp, out)
    usage: Dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "reasoning_tokens": rea,
        "total_tokens": tot,
    }
    if provider_usage is not None:
        usage["provider_usage"] = provider_usage
    return usage


@dataclass(frozen=True)
class LLMCallSpec:
    provider: str  # "openai" | "gemini" | "claude"
    model: str
    prompt: str
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_sec: int = 120

    # Retry policy (client-side)
    max_retries: int = 3
    retry_backoff_sec: float = 5.0

    # Prefer provider-side JSON mode when available
    force_json_object: bool = False

    # Optional provider-specific overrides
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    # Provider-specific request fields (non-secret). Used for "thinking" config, etc.
    provider_options: Dict[str, Any] = field(default_factory=dict)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output."""
    if not text or not isinstance(text, str):
        raise LLMError("Empty model output")

    s = text.strip()

    # Strip common markdown fences
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = (m.group(1) or "").strip()

    # Direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Heuristic: take substring from first { to last }
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        chunk = s[i : j + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            raise LLMError(f"Failed to parse JSON object from model output: {e}") from e

    raise LLMError("No JSON object found in model output")


def _require_api_key(provider: str, api_key: Optional[str], env_name: str) -> str:
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    v = (os.getenv(env_name) or "").strip()
    if not v:
        raise LLMError(f"Missing API key (set {env_name} or configure api_key) for provider={provider!r}")
    return v


def _is_transient_http(status_code: int) -> bool:
    return int(status_code) in _TRANSIENT_HTTP_STATUS


def _maybe_read_error_body(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:
        try:
            return (resp.text or "")[:4000]
        except Exception:
            return "<unavailable>"


def _post_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_sec: int, max_retries: int, retry_backoff_sec: float) -> Dict[str, Any]:
    try:
        import requests
    except Exception as e:
        raise LLMError("requests is required for LLM API calls. Install with: pip install requests") from e

    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if r.status_code >= 400:
                if _is_transient_http(r.status_code) and attempt < max_retries:
                    last_error = LLMError(f"Transient HTTP {r.status_code}: {_maybe_read_error_body(r)}")
                    import time

                    time.sleep(float(retry_backoff_sec) * attempt)
                    continue
                raise LLMError(f"LLM API error {r.status_code} for {url}: {_maybe_read_error_body(r)}")

            try:
                body = r.json()
            except Exception:
                raise LLMError(f"LLM API returned non-JSON body for {url}: {(r.text or '')[:4000]}")
            if not isinstance(body, dict):
                raise LLMError(f"Unexpected LLM API response shape (expected object): {type(body)}")
            return body

        except Exception as e:
            # Network/transient errors
            last_error = e if isinstance(e, Exception) else Exception(str(e))
            if attempt < max_retries:
                import time

                time.sleep(float(retry_backoff_sec) * attempt)
                continue
            break

    raise LLMError(f"LLM request failed after {max_retries} attempts: {last_error}")


def _extract_openai_text(resp: Dict[str, Any]) -> str:
    # Newer Responses API convenience field
    if isinstance(resp.get("output_text"), str) and resp.get("output_text"):
        return str(resp["output_text"])

    # Responses API structured output
    out = resp.get("output")
    if isinstance(out, list):
        parts = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if isinstance(c.get("text"), str):
                    parts.append(str(c.get("text")))
        if parts:
            return "\n".join(parts).strip()

    # Chat Completions shape
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return str(msg.get("content") or "")

    # Fallback: try generic fields
    for k in ("text", "content"):
        if isinstance(resp.get(k), str) and resp.get(k):
            return str(resp[k])

    raise LLMError("Unable to extract text from OpenAI response")


def _extract_openai_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return _normalize_usage(provider_usage=None)

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    reasoning_tokens = None
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        reasoning_tokens = details.get("reasoning_tokens")
    elif usage.get("reasoning_tokens") is not None:
        reasoning_tokens = usage.get("reasoning_tokens")

    norm = _normalize_usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        provider_usage=usage,
    )

    # OpenAI sometimes reports prompt caching as a subset of input/prompt tokens.
    # Chat Completions: usage.prompt_tokens_details.cached_tokens
    # Responses API: usage.input_tokens_details.cached_tokens
    cached_tokens = None
    ptd = usage.get("prompt_tokens_details")
    if not isinstance(ptd, dict):
        ptd = usage.get("input_tokens_details")
    if isinstance(ptd, dict) and ptd.get("cached_tokens") is not None:
        cached_tokens = ptd.get("cached_tokens")
    elif usage.get("cached_tokens") is not None:
        cached_tokens = usage.get("cached_tokens")
    cached_tokens_i = _to_int_or_none(cached_tokens)
    if cached_tokens_i is not None:
        norm["cached_input_tokens"] = cached_tokens_i

    return norm


def _call_openai(spec: LLMCallSpec) -> str:
    api_key = _require_api_key(provider="openai", api_key=spec.api_key, env_name="OPENAI_API_KEY")
    base = (spec.api_base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com").rstrip("/")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prefer Chat Completions for broad compatibility.
    chat_url = f"{base}/v1/chat/completions"

    base_payload: Dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "temperature": spec.temperature,
        "max_tokens": spec.max_output_tokens,
    }

    # Provider-specific options (e.g., reasoning_effort)
    opts = spec.provider_options or {}
    if isinstance(opts, dict):
        if "reasoning_effort" in opts:
            base_payload["reasoning_effort"] = opts.get("reasoning_effort")
        # Allow overriding token param naming if needed
        if "max_completion_tokens" in opts and opts.get("max_completion_tokens") is not None:
            base_payload.pop("max_tokens", None)
            base_payload["max_completion_tokens"] = opts.get("max_completion_tokens")

    # If JSON mode is supported, it improves reliability.
    if spec.force_json_object:
        base_payload["response_format"] = {"type": "json_object"}

    # Try with JSON mode first; if it 400s, retry without it.
    try:
        resp = _post_with_retries(
            url=chat_url,
            headers=headers,
            payload=base_payload,
            timeout_sec=spec.timeout_sec,
            max_retries=spec.max_retries,
            retry_backoff_sec=spec.retry_backoff_sec,
        )
        return _extract_openai_text(resp)
    except LLMError as e:
        if not spec.force_json_object:
            raise
        # Fallback: remove response_format for older models/endpoints.
        payload2 = dict(base_payload)
        payload2.pop("response_format", None)
        resp = _post_with_retries(
            url=chat_url,
            headers=headers,
            payload=payload2,
            timeout_sec=spec.timeout_sec,
            max_retries=spec.max_retries,
            retry_backoff_sec=spec.retry_backoff_sec,
        )
        return _extract_openai_text(resp)


def _call_openai_with_usage(spec: LLMCallSpec) -> Tuple[str, Dict[str, Any]]:
    api_key = _require_api_key(provider="openai", api_key=spec.api_key, env_name="OPENAI_API_KEY")
    base = (spec.api_base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com").rstrip("/")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    chat_url = f"{base}/v1/chat/completions"
    base_payload: Dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "temperature": spec.temperature,
        "max_tokens": spec.max_output_tokens,
    }

    opts = spec.provider_options or {}
    if isinstance(opts, dict):
        if "reasoning_effort" in opts:
            base_payload["reasoning_effort"] = opts.get("reasoning_effort")
        if "max_completion_tokens" in opts and opts.get("max_completion_tokens") is not None:
            base_payload.pop("max_tokens", None)
            base_payload["max_completion_tokens"] = opts.get("max_completion_tokens")

    if spec.force_json_object:
        base_payload["response_format"] = {"type": "json_object"}

    try:
        resp = _post_with_retries(
            url=chat_url,
            headers=headers,
            payload=base_payload,
            timeout_sec=spec.timeout_sec,
            max_retries=spec.max_retries,
            retry_backoff_sec=spec.retry_backoff_sec,
        )
        return _extract_openai_text(resp), _extract_openai_usage(resp)
    except LLMError:
        if not spec.force_json_object:
            raise
        payload2 = dict(base_payload)
        payload2.pop("response_format", None)
        resp = _post_with_retries(
            url=chat_url,
            headers=headers,
            payload=payload2,
            timeout_sec=spec.timeout_sec,
            max_retries=spec.max_retries,
            retry_backoff_sec=spec.retry_backoff_sec,
        )
        return _extract_openai_text(resp), _extract_openai_usage(resp)


def _extract_gemini_text(resp: Dict[str, Any]) -> str:
    cands = resp.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise LLMError("Gemini response missing candidates[]")
    content = cands[0].get("content") if isinstance(cands[0], dict) else None
    if not isinstance(content, dict):
        raise LLMError("Gemini response candidate missing content")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise LLMError("Gemini response content missing parts[]")
    texts = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            texts.append(p["text"])
    if not texts:
        raise LLMError("Gemini response had no text parts")
    return "\n".join(texts).strip()


def _extract_gemini_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    um = resp.get("usageMetadata")
    if not isinstance(um, dict):
        return _normalize_usage(provider_usage=None)
    return _normalize_usage(
        input_tokens=um.get("promptTokenCount"),
        output_tokens=um.get("candidatesTokenCount"),
        total_tokens=um.get("totalTokenCount"),
        provider_usage=um,
    )


def _call_gemini(spec: LLMCallSpec) -> str:
    api_key = _require_api_key(provider="gemini", api_key=spec.api_key, env_name="GEMINI_API_KEY")
    # Gemini Generative Language API
    base = (spec.api_base_url or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")

    # Models look like "gemini-1.5-pro" etc.
    url = f"{base}/v1beta/models/{spec.model}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": spec.prompt}]}],
        "generationConfig": {
            "temperature": spec.temperature,
            # Gemini uses maxOutputTokens
            "maxOutputTokens": spec.max_output_tokens,
        },
    }

    opts = spec.provider_options or {}
    if isinstance(opts, dict):
        thinking_cfg = opts.get("thinkingConfig")
        if isinstance(thinking_cfg, dict) and thinking_cfg:
            payload["generationConfig"]["thinkingConfig"] = thinking_cfg
    resp = _post_with_retries(
        url=url,
        headers=headers,
        payload=payload,
        timeout_sec=spec.timeout_sec,
        max_retries=spec.max_retries,
        retry_backoff_sec=spec.retry_backoff_sec,
    )
    return _extract_gemini_text(resp)


def _call_gemini_with_usage(spec: LLMCallSpec) -> Tuple[str, Dict[str, Any]]:
    api_key = _require_api_key(provider="gemini", api_key=spec.api_key, env_name="GEMINI_API_KEY")
    base = (spec.api_base_url or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base}/v1beta/models/{spec.model}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": spec.prompt}]}],
        "generationConfig": {
            "temperature": spec.temperature,
            "maxOutputTokens": spec.max_output_tokens,
        },
    }

    opts = spec.provider_options or {}
    if isinstance(opts, dict):
        thinking_cfg = opts.get("thinkingConfig")
        if isinstance(thinking_cfg, dict) and thinking_cfg:
            payload["generationConfig"]["thinkingConfig"] = thinking_cfg

    resp = _post_with_retries(
        url=url,
        headers=headers,
        payload=payload,
        timeout_sec=spec.timeout_sec,
        max_retries=spec.max_retries,
        retry_backoff_sec=spec.retry_backoff_sec,
    )
    return _extract_gemini_text(resp), _extract_gemini_usage(resp)


def _extract_claude_text(resp: Dict[str, Any]) -> str:
    # Anthropic Messages API: {"content": [{"type": "text", "text": "..."}, ...]}
    content = resp.get("content")
    if not isinstance(content, list) or not content:
        raise LLMError("Claude response missing content[]")
    texts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block.get("text") or "")
    if not texts:
        # If thinking/tool blocks exist but no text, surface types for debugging.
        types = [b.get("type") for b in content if isinstance(b, dict)]
        raise LLMError(f"Claude response had no text blocks (types={types})")
    return "\n".join(t.strip() for t in texts if t).strip()


def _extract_claude_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return _normalize_usage(provider_usage=None)

    # Anthropic usage typically includes input_tokens/output_tokens; may also include cache-related fields.
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    tot = None
    if inp is not None and out is not None:
        tot = _sum_ints(_to_int_or_none(inp), _to_int_or_none(out))

    norm = _normalize_usage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=tot,
        provider_usage=usage,
    )

    # Anthropic may report cache token counters for prompt caching.
    # cache_creation_input_tokens: tokens written into cache
    # cache_read_input_tokens: tokens served from cache (i.e., cache hits)
    ccreate = _to_int_or_none(usage.get("cache_creation_input_tokens"))
    cread = _to_int_or_none(usage.get("cache_read_input_tokens"))
    if ccreate is not None:
        norm["cache_creation_input_tokens"] = ccreate
    if cread is not None:
        norm["cache_read_input_tokens"] = cread
        norm.setdefault("cached_input_tokens", cread)

    return norm


def _call_claude(spec: LLMCallSpec) -> str:
    api_key = _require_api_key(provider="claude", api_key=spec.api_key, env_name="ANTHROPIC_API_KEY")
    base = (spec.api_base_url or os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com").rstrip("/")
    url = f"{base}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv("ANTHROPIC_VERSION") or "2023-06-01",
        "content-type": "application/json",
    }
    beta = (os.getenv("ANTHROPIC_BETA") or "").strip()
    if beta:
        headers["anthropic-beta"] = beta

    payload: Dict[str, Any] = {
        "model": spec.model,
        "max_tokens": int(spec.max_output_tokens),
        "temperature": float(spec.temperature),
        "messages": [{"role": "user", "content": spec.prompt}],
    }

    # Thinking can be set via config/provider_options, or env as fallback.
    opts = spec.provider_options or {}
    if isinstance(opts, dict) and isinstance(opts.get("thinking"), dict):
        payload["thinking"] = opts.get("thinking")
    else:
        thinking_budget = (os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS") or "").strip()
        if thinking_budget:
            try:
                payload["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
            except Exception:
                pass

    resp = _post_with_retries(
        url=url,
        headers=headers,
        payload=payload,
        timeout_sec=spec.timeout_sec,
        max_retries=spec.max_retries,
        retry_backoff_sec=spec.retry_backoff_sec,
    )
    return _extract_claude_text(resp)


def _call_claude_with_usage(spec: LLMCallSpec) -> Tuple[str, Dict[str, Any]]:
    api_key = _require_api_key(provider="claude", api_key=spec.api_key, env_name="ANTHROPIC_API_KEY")
    base = (spec.api_base_url or os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com").rstrip("/")
    url = f"{base}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv("ANTHROPIC_VERSION") or "2023-06-01",
        "content-type": "application/json",
    }
    beta = (os.getenv("ANTHROPIC_BETA") or "").strip()
    if beta:
        headers["anthropic-beta"] = beta

    payload: Dict[str, Any] = {
        "model": spec.model,
        "max_tokens": int(spec.max_output_tokens),
        "temperature": float(spec.temperature),
        "messages": [{"role": "user", "content": spec.prompt}],
    }

    opts = spec.provider_options or {}
    if isinstance(opts, dict) and isinstance(opts.get("thinking"), dict):
        payload["thinking"] = opts.get("thinking")
    else:
        thinking_budget = (os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS") or "").strip()
        if thinking_budget:
            try:
                payload["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
            except Exception:
                pass

    resp = _post_with_retries(
        url=url,
        headers=headers,
        payload=payload,
        timeout_sec=spec.timeout_sec,
        max_retries=spec.max_retries,
        retry_backoff_sec=spec.retry_backoff_sec,
    )
    return _extract_claude_text(resp), _extract_claude_usage(resp)


def call_llm_text(spec: LLMCallSpec) -> str:
    provider = (spec.provider or "").strip().lower()
    if provider == "openai":
        return _call_openai(spec)
    if provider == "gemini":
        return _call_gemini(spec)
    if provider in {"claude", "anthropic"}:
        return _call_claude(spec)
    raise LLMError(f"Unsupported LLM provider: {spec.provider!r}")


def call_llm_text_with_usage(spec: LLMCallSpec) -> Tuple[str, Dict[str, Any]]:
    """Call provider and return (text, usage).

    usage is normalized to keys: input_tokens, output_tokens, reasoning_tokens, total_tokens,
    and may include provider_usage for debugging.
    """
    provider = (spec.provider or "").strip().lower()
    if provider == "openai":
        return _call_openai_with_usage(spec)
    if provider == "gemini":
        return _call_gemini_with_usage(spec)
    if provider in {"claude", "anthropic"}:
        return _call_claude_with_usage(spec)
    raise LLMError(f"Unsupported LLM provider: {spec.provider!r}")


def call_llm_json(spec: LLMCallSpec) -> Dict[str, Any]:
    text = call_llm_text(spec)
    return _extract_first_json_object(text)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Public helper: extract a JSON object from raw model output text."""
    return _extract_first_json_object(text)


def llm_cache_key(spec: LLMCallSpec) -> str:
    # Note: do not include secrets.
    key_obj = {
        "provider": spec.provider,
        "model": spec.model,
        "temperature": spec.temperature,
        "max_output_tokens": spec.max_output_tokens,
        "timeout_sec": spec.timeout_sec,
        "force_json_object": spec.force_json_object,
        "api_base_url": spec.api_base_url,
        "provider_options": spec.provider_options,
        "prompt": spec.prompt,
    }
    return _sha256_hex(_stable_json(key_obj))
