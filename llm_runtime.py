from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os

from llm_client import (
    LLMCallSpec,
    LLMError,
    build_request_debug,
    call_llm_text_with_usage,
    extract_json_object,
    llm_cache_key,
)


def _read_text(path: Path, max_bytes: int = 512_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise LLMError("PyYAML is required to load llm config. Install with: pip install pyyaml") from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_llm_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    if not p.exists() or not p.is_file():
        return {}
    try:
        return _load_yaml(p)
    except Exception:
        return {}


def _provider_cfg(llm_cfg: Dict[str, Any], provider: str) -> Dict[str, Any]:
    providers = llm_cfg.get("providers") if isinstance(llm_cfg, dict) else None
    if not isinstance(providers, dict):
        return {}
    cfg = providers.get(provider)
    return cfg if isinstance(cfg, dict) else {}


def resolve_default_provider(*, cli_provider: str, llm_cfg: Dict[str, Any]) -> str:
    p = (cli_provider or "").strip().lower() or "openai"
    cfg_default = str(llm_cfg.get("default_provider") or "").strip().lower() if isinstance(llm_cfg, dict) else ""
    # If user didn't specify provider (or left it as default openai), allow config to pick one.
    if cfg_default and p == os.getenv("SAGES_LLM_PROVIDER", "openai").strip().lower():
        return cfg_default
    return p


def _resolve_api_key(provider_cfg: Dict[str, Any], *, fallback_env: str) -> str:
    api_key = str(provider_cfg.get("api_key") or "").strip()
    if api_key:
        return api_key
    env_name = str(provider_cfg.get("api_key_env") or "").strip() or fallback_env
    return (os.getenv(env_name) or "").strip()


def _has_any_api_key(provider: str, provider_cfg: Dict[str, Any]) -> bool:
    provider = (provider or "").strip().lower()
    if provider == "openai":
        return bool(_resolve_api_key(provider_cfg, fallback_env="OPENAI_API_KEY"))
    if provider == "gemini":
        return bool(_resolve_api_key(provider_cfg, fallback_env="GEMINI_API_KEY"))
    if provider in {"claude", "anthropic"}:
        return bool(_resolve_api_key(provider_cfg, fallback_env="ANTHROPIC_API_KEY"))
    return False


def _thinking_to_provider_options(provider: str, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate `thinking:` config block into provider-specific request fields."""
    thinking = provider_cfg.get("thinking")
    if not isinstance(thinking, dict):
        return {}

    enabled = thinking.get("enabled")
    if enabled is False:
        return {}

    provider = (provider or "").strip().lower()
    out: Dict[str, Any] = {}

    if provider == "openai":
        # Chat Completions supports reasoning_effort for certain models.
        eff = thinking.get("reasoning_effort")
        if eff is not None:
            out["reasoning_effort"] = eff

    elif provider == "gemini":
        # generationConfig.thinkingConfig.thinkingLevel
        level = thinking.get("thinking_level")
        if level is not None:
            out["thinkingConfig"] = {"thinkingLevel": level}

    elif provider in {"claude", "anthropic"}:
        # Messages API: thinking: {type: enabled, budget_tokens: N}
        budget = thinking.get("budget_tokens")
        if budget is not None:
            try:
                out["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
            except Exception:
                pass

    return out


def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if v is None:
            continue
        out[k] = v
    return out


@dataclass
class LLMRuntime:
    llm_cfg: Dict[str, Any]
    provider: str
    default_model: str
    activation_model: str = ""
    scoring_model: str = ""

    def provider_cfg(self) -> Dict[str, Any]:
        return _provider_cfg(self.llm_cfg, self.provider)

    def build_spec(self, *, phase: str, prompt: str) -> LLMCallSpec:
        pcfg = self.provider_cfg()

        # Select model
        phase = (phase or "").strip().lower()
        if phase == "activation":
            model = (self.activation_model or pcfg.get("activation_model") or self.default_model or pcfg.get("model") or "").strip()
        else:
            model = (self.scoring_model or pcfg.get("scoring_model") or self.default_model or pcfg.get("model") or "").strip()

        if not model:
            # Backwards-compatible/provider-default fallbacks when neither opts/env nor llm_config.yaml specify a model.
            p = (self.provider or "").strip().lower()
            if p == "openai":
                model = "gpt-5"
            elif p == "gemini":
                model = "gemini-1.5-pro"
            elif p in {"claude", "anthropic"}:
                model = "claude-3-7-sonnet-latest"

        if not model:
            raise LLMError("No model configured for LLM call")

        api_base_url = str(pcfg.get("api_base_url") or "").strip() or None

        api_key = ""
        if self.provider == "openai":
            api_key = _resolve_api_key(pcfg, fallback_env="OPENAI_API_KEY")
        elif self.provider == "gemini":
            api_key = _resolve_api_key(pcfg, fallback_env="GEMINI_API_KEY")
        elif self.provider in {"claude", "anthropic"}:
            api_key = _resolve_api_key(pcfg, fallback_env="ANTHROPIC_API_KEY")

        # Generic params
        temperature = float(pcfg.get("temperature", 0.0))
        max_output_tokens = int(pcfg.get("max_output_tokens", 4096))
        timeout_sec = int(pcfg.get("timeout_sec", 180 if phase != "activation" else 120))
        max_retries = int(pcfg.get("max_retries", 3))
        retry_backoff_sec = float(pcfg.get("retry_backoff_sec", 5.0))
        force_json_object = bool(pcfg.get("force_json_object", False))

        # Provider options
        provider_options: Dict[str, Any] = {}
        provider_options = _merge_dict(provider_options, _thinking_to_provider_options(self.provider, pcfg))
        # Allow raw passthrough options (advanced)
        raw_opts = pcfg.get("provider_options")
        if isinstance(raw_opts, dict):
            provider_options = _merge_dict(provider_options, raw_opts)

        if not (api_key or _has_any_api_key(self.provider, pcfg)):
            raise LLMError(
                "LLM API key missing for provider=%r (set api_key/api_key_env in llm_config.yaml or relevant env var)." % self.provider
            )

        return LLMCallSpec(
            provider=self.provider,
            model=str(model),
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            force_json_object=force_json_object,
            api_key=api_key or None,
            api_base_url=api_base_url,
            provider_options=provider_options,
        )

    def call_json_cached_with_usage(
        self,
        cache_dir: Path,
        kind: str,
        phase: str,
        prompt: str,
        outputs_dir: Optional[Path] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Call LLM (or load cached) and return (json_obj, usage).

        usage is best-effort; for cache hits created by older versions, usage may be empty.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        spec = self.build_spec(phase=phase, prompt=prompt)
        key = llm_cache_key(spec)
        short = key[:16]

        json_path = cache_dir / f"{kind}_{short}.json"
        raw_path = cache_dir / f"{kind}_{short}.txt"
        meta_path = cache_dir / f"{kind}_{short}.meta.json"

        request_debug = build_request_debug(spec)

        if json_path.exists():
            try:
                obj = json.loads(_read_text(json_path, max_bytes=50_000_000))
                if isinstance(obj, dict):
                    meta_obj = None
                    try:
                        meta_obj = json.loads(_read_text(meta_path, max_bytes=2_000_000)) if meta_path.exists() else None
                    except Exception:
                        meta_obj = None
                    usage = {}
                    if isinstance(meta_obj, dict) and isinstance(meta_obj.get("usage"), dict):
                        usage = dict(meta_obj.get("usage") or {})
                    # Mark cache hit for consumers
                    usage.setdefault("cache_hit", True)
                    usage.setdefault("provider", spec.provider)
                    usage.setdefault("model", spec.model)
                    usage.setdefault("phase", phase)
                    usage.setdefault("kind", kind)
                    usage.setdefault("cache_key", key)

                    # Always (re)write meta with current request params for audit/debug,
                    # even if this is a cache hit or older meta didn't exist.
                    try:
                        _write_json(
                            meta_path,
                            {
                                "provider": spec.provider,
                                "model": spec.model,
                                "temperature": spec.temperature,
                                "max_output_tokens": spec.max_output_tokens,
                                "timeout_sec": spec.timeout_sec,
                                "max_retries": spec.max_retries,
                                "retry_backoff_sec": spec.retry_backoff_sec,
                                "force_json_object": spec.force_json_object,
                                "api_base_url": spec.api_base_url,
                                "provider_options": spec.provider_options,
                                "cache_key": key,
                                "phase": phase,
                                "kind": kind,
                                "request": request_debug,
                                "usage": usage,
                                "cache_hit": True,
                            },
                        )
                    except Exception:
                        pass

                    if outputs_dir is not None:
                        try:
                            outputs_dir.mkdir(parents=True, exist_ok=True)
                            _write_json(outputs_dir / f"{kind}.request.json", request_debug)
                            _write_json(
                                outputs_dir / f"{kind}.meta.json",
                                {
                                    "provider": spec.provider,
                                    "model": spec.model,
                                    "temperature": spec.temperature,
                                    "max_output_tokens": spec.max_output_tokens,
                                    "timeout_sec": spec.timeout_sec,
                                    "max_retries": spec.max_retries,
                                    "retry_backoff_sec": spec.retry_backoff_sec,
                                    "force_json_object": spec.force_json_object,
                                    "api_base_url": spec.api_base_url,
                                    "provider_options": spec.provider_options,
                                    "cache_key": key,
                                    "phase": phase,
                                    "kind": kind,
                                    "request": request_debug,
                                    "usage": usage,
                                    "cache_hit": True,
                                },
                            )
                        except Exception:
                            pass
                    return obj, usage
            except Exception:
                pass

        text, usage = call_llm_text_with_usage(spec)
        usage = dict(usage or {})
        usage["cache_hit"] = False
        usage.setdefault("provider", spec.provider)
        usage.setdefault("model", spec.model)
        usage.setdefault("phase", phase)
        usage.setdefault("kind", kind)
        usage.setdefault("cache_key", key)
        _write_text(raw_path, text)
        obj = extract_json_object(text)
        _write_json(json_path, obj)
        _write_json(
            meta_path,
            {
                "provider": spec.provider,
                "model": spec.model,
                "temperature": spec.temperature,
                "max_output_tokens": spec.max_output_tokens,
                "timeout_sec": spec.timeout_sec,
                "max_retries": spec.max_retries,
                "retry_backoff_sec": spec.retry_backoff_sec,
                "force_json_object": spec.force_json_object,
                "api_base_url": spec.api_base_url,
                "provider_options": spec.provider_options,
                "cache_key": key,
                "phase": phase,
                "kind": kind,
                "request": request_debug,
                "usage": usage,
                "cache_hit": False,
            },
        )

        if outputs_dir is not None:
            outputs_dir.mkdir(parents=True, exist_ok=True)
            _write_json(outputs_dir / f"{kind}.json", obj)
            _write_json(outputs_dir / f"{kind}.request.json", request_debug)
            _write_json(
                outputs_dir / f"{kind}.meta.json",
                {
                    "provider": spec.provider,
                    "model": spec.model,
                    "temperature": spec.temperature,
                    "max_output_tokens": spec.max_output_tokens,
                    "timeout_sec": spec.timeout_sec,
                    "max_retries": spec.max_retries,
                    "retry_backoff_sec": spec.retry_backoff_sec,
                    "force_json_object": spec.force_json_object,
                    "api_base_url": spec.api_base_url,
                    "provider_options": spec.provider_options,
                    "cache_key": key,
                    "phase": phase,
                    "kind": kind,
                    "request": request_debug,
                    "usage": usage,
                    "cache_hit": False,
                },
            )

        return obj, usage

    def call_json_cached(self, *, cache_dir: Path, kind: str, phase: str, prompt: str, outputs_dir: Optional[Path] = None) -> Dict[str, Any]:
        obj, _usage = self.call_json_cached_with_usage(
            cache_dir=cache_dir,
            kind=kind,
            phase=phase,
            prompt=prompt,
            outputs_dir=outputs_dir,
        )
        return obj
