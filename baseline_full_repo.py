from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aggregation import (
    AggregationPolicy,
    GroupAggregationResult,
    GroupDefinition,
    LeafScore,
    ModuleAggregationResult,
    aggregate_final,
    aggregate_group,
    aggregate_module,
    build_report_payload,
)
from llm_runtime import LLMRuntime, load_llm_config, resolve_default_provider
from main import (
    _TokenUsageTracker,
    _clamp01,
    _coerce_string_list,
    _load_yaml,
    _looks_generated_or_minified_content,
    _read_text,
    _should_skip_retrieval_path,
    _to_float_or_none,
    _write_json,
    _write_text,
    activation_to_group_defs,
    eval_applies_if,
    load_rubric,
)
from repo_signals import detect_languages, infer_repo_signals


_BASELINE_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "css",
    "include",
    "vendor",
    "vendors",
    "libs",
    "lib",
    "third_party",
    "third-party",
    "site-packages",
    ".next",
    ".nuxt",
    ".angular",
    ".cache",
    "public",
    "static",
    "target",
    "out",
    ".gradle",
    ".husky",
    ".pnpm",
    "jspm_packages",
    "bower_components",
    "coverage",
    "htmlcov",
    ".tox",
    ".nox",
    ".terraform",
    ".serverless",
    ".aws-sam",
    ".eggs",
}

_BASELINE_IGNORED_FILE_NAMES = {
    # ".gitignore",
    # "cargo.lock",
    # "composer.lock",
    # "gemfile.lock",
    # "go.sum",
    # "go.work.sum",
    # "package-lock.json",
    # "pipfile.lock",
    # "pnpm-lock.yaml",
    # "poetry.lock",
    # "yarn.lock",
}

_BASELINE_IGNORED_FILE_EXTS = {
    # ".lock",
}

_BASELINE_BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}

_BASELINE_PREF_TEXT_EXTS = {
    ".py",
    ".go",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cs",
    ".php",
    ".rb",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".lua",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".tf",
    ".tfvars",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".rst",
    ".txt",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".less",
    ".svelte",
    ".vue",
}

_BASELINE_SKIP_PREFIXES = (
)

_ALLOWED_REPO_KINDS = {
    "backend",
    "web_frontend",
    "mobile_app",
    "infra",
    "library",
    "ml_project",
    "unknown",
}


@dataclass
class BaselineOptions:
    repo: str
    rubric: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_config: Optional[str] = None
    llm_model: Optional[str] = None
    scoring_model: str = ""
    out: str = "baseline_quality_report.json"
    review_scope: str = "full_repo"
    max_files_in_prompt: int = 180
    max_total_chars: int = 350_000
    max_chars_per_file: int = 8_000
    max_bytes_per_file: int = 96_000
    max_walk_files: int = 25_000


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _should_ignore_baseline_file(path_like: Path | str) -> bool:
    path = Path(path_like)
    name = path.name.lower()
    ext = path.suffix.lower()

    if name in _BASELINE_IGNORED_FILE_NAMES:
        return True
    if ext in _BASELINE_IGNORED_FILE_EXTS:
        return True
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return True
    return False


def _should_skip_baseline_path(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/").lstrip("./")
    if not normalized:
        return True
    if _should_skip_retrieval_path(normalized):
        return True
    if normalized.startswith(_BASELINE_SKIP_PREFIXES):
        return True
    parts = [part for part in normalized.split("/") if part]
    return any(part in _BASELINE_IGNORE_DIRS for part in parts[:-1])


def _is_probably_text_file(path: Path, max_sample_bytes: int = 2048) -> bool:
    if _should_ignore_baseline_file(path):
        return False

    ext = path.suffix.lower()
    if ext in _BASELINE_BINARY_EXTS:
        return False
    if ext in _BASELINE_PREF_TEXT_EXTS:
        return True

    try:
        with path.open("rb") as handle:
            chunk = handle.read(max_sample_bytes)
    except OSError:
        return False

    if not chunk:
        return True
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _iter_baseline_files(repo_root: Path, max_files: int) -> Iterable[Path]:
    count = 0
    for root, dirs, files in os.walk(repo_root):
        rel_root = ""
        try:
            rel_root = str(Path(root).resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except Exception:
            rel_root = ""

        filtered_dirs: List[str] = []
        for dir_name in dirs:
            rel_dir = f"{rel_root}/{dir_name}" if rel_root and rel_root != "." else dir_name
            rel_dir = rel_dir.replace("\\", "/").lstrip("./")
            if dir_name in _BASELINE_IGNORE_DIRS:
                continue
            if rel_dir.startswith(_BASELINE_SKIP_PREFIXES):
                continue
            if _should_skip_retrieval_path(rel_dir + "/"):
                continue
            filtered_dirs.append(dir_name)
        dirs[:] = filtered_dirs

        for file_name in files:
            path = Path(root) / file_name
            try:
                rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
            except Exception:
                continue
            if _should_ignore_baseline_file(rel_path):
                continue
            if count >= max_files:
                return
            count += 1
            yield path


def _looks_binary_text_sample(text: str) -> bool:
    if not text:
        return False
    if "\x00" in text:
        return True
    control_chars = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t\f\b")
    return (control_chars / max(1, len(text))) > 0.02


def _path_priority(rel_path: str) -> Tuple[int, int, str]:
    low = rel_path.lower()
    name = Path(rel_path).name.lower()
    parts = [p for p in low.split("/") if p]
    score = 100

    if len(parts) == 1:
        score -= 30
    if name in {
        "readme",
        "readme.md",
        "readme.rst",
        "license",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "go.mod",
        "go.sum",
        "cargo.toml",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
        "makefile",
    }:
        score -= 25
    if any(low.startswith(prefix) for prefix in ("src/", "app/", "lib/", "pkg/", "cmd/", "internal/")):
        score -= 18
    if any(low.startswith(prefix) for prefix in ("tests/", "test/", "docs/", "scripts/", ".github/")):
        score -= 10
    if any(low.startswith(prefix) for prefix in ("sample_runs/", "summarized_reports/", "runs_", "notebooks/")):
        score += 20
    if low.endswith((".md", ".txt", ".rst")):
        score -= 4
    if low.endswith((".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".php", ".rb", ".rs")):
        score -= 6
    return (score, len(parts), rel_path)


def _render_numbered_text(text: str, max_chars: int) -> Tuple[str, int, int, bool]:
    lines = text.splitlines()
    if not lines and text:
        lines = [text]
    if not lines:
        return "", 0, 0, False

    out: List[str] = []
    used = 0
    last_line = 0
    total_lines = len(lines)
    truncated = False
    for line_no, line in enumerate(lines, start=1):
        rendered = f"{line_no:04d}: {line}\n"
        if used + len(rendered) > max_chars:
            truncated = True
            if not out:
                clipped = rendered[:max_chars]
                out.append(clipped)
                last_line = line_no
            break
        out.append(rendered)
        used += len(rendered)
        last_line = line_no
    if last_line < total_lines:
        truncated = True
    return "".join(out).rstrip(), last_line, total_lines, truncated


def build_repository_dump(
    repo_root: Path,
    max_files_in_prompt: int,
    max_total_chars: int,
    max_chars_per_file: int,
    max_bytes_per_file: int,
    max_walk_files: int,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    skipped = {
        "generated_path": 0,
        "ignored_name_or_ext": 0,
        "non_text": 0,
        "empty": 0,
        "minified_or_generated_content": 0,
    }

    for path in _iter_baseline_files(repo_root, max_files=max_walk_files):
        try:
            rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
        except Exception:
            continue

        if _should_skip_baseline_path(rel_path):
            skipped["generated_path"] += 1
            continue
        if _should_ignore_baseline_file(rel_path):
            skipped["ignored_name_or_ext"] += 1
            continue
        if not _is_probably_text_file(path):
            skipped["non_text"] += 1
            continue

        text = _read_text(path, max_bytes=max_bytes_per_file)
        if not text or not text.strip():
            skipped["empty"] += 1
            continue
        if path.suffix.lower() not in _BASELINE_PREF_TEXT_EXTS and _looks_binary_text_sample(text):
            skipped["non_text"] += 1
            continue
        if _looks_generated_or_minified_content(text):
            skipped["minified_or_generated_content"] += 1
            continue

        candidates.append(
            {
                "path": rel_path,
                "priority": _path_priority(rel_path),
                "text": text,
                "bytes_read": len(text.encode("utf-8", errors="replace")),
            }
        )

    candidates.sort(key=lambda item: item["priority"])

    sections: List[str] = []
    selected_files: List[Dict[str, Any]] = []
    total_chars_used = 0
    omitted_due_to_limits = 0

    for item in candidates:
        if len(selected_files) >= max_files_in_prompt or total_chars_used >= max_total_chars:
            omitted_due_to_limits += 1
            continue

        rel_path = item["path"]
        rendered, last_line, total_lines, truncated = _render_numbered_text(item["text"], max_chars_per_file)
        if not rendered:
            omitted_due_to_limits += 1
            continue

        header = f"### FILE: {rel_path}"
        if truncated:
            header += f" (showing lines 1-{last_line} of {total_lines})"
        section = f"{header}\n{rendered}\n"

        if total_chars_used + len(section) > max_total_chars:
            remaining = max_total_chars - total_chars_used
            if remaining < 300:
                omitted_due_to_limits += 1
                continue
            allowed_body = max(120, remaining - len(header) - 5)
            rendered, last_line, total_lines, truncated = _render_numbered_text(item["text"], allowed_body)
            if not rendered:
                omitted_due_to_limits += 1
                continue
            header = f"### FILE: {rel_path}"
            if truncated:
                header += f" (showing lines 1-{last_line} of {total_lines})"
            section = f"{header}\n{rendered}\n"

        sections.append(section.rstrip())
        total_chars_used += len(section)
        selected_files.append(
            {
                "path": rel_path,
                "lines_shown": last_line,
                "total_lines": total_lines,
                "truncated": truncated,
                "chars_in_prompt": len(section),
                "bytes_read": item["bytes_read"],
            }
        )

    return {
        "repository_dump": "\n\n".join(sections),
        "selected_files": selected_files,
        "selected_file_count": len(selected_files),
        "candidate_file_count": len(candidates),
        "omitted_due_to_limits": omitted_due_to_limits,
        "total_chars_in_prompt": total_chars_used,
        "limits": {
            "max_files_in_prompt": max_files_in_prompt,
            "max_total_chars": max_total_chars,
            "max_chars_per_file": max_chars_per_file,
            "max_bytes_per_file": max_bytes_per_file,
            "max_walk_files": max_walk_files,
        },
        "skipped_counts": skipped,
    }


def build_compact_rubric_payload(modules: List[Any]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for module in modules:
        payload.append(
            {
                "module_id": module.module_id,
                "module_name": module.module_name,
                "module_weight": module.module_weight,
                "applies_if": module.applies_if,
                "groups": [
                    {
                        "group_id": group.group_id,
                        "group_name": group.group_name,
                        "group_weight": group.group_weight,
                        "leaves": [
                            {"leaf_id": leaf.leaf_id, "criterion": leaf.question}
                            for leaf in group.leaves
                        ],
                    }
                    for group in module.groups
                ],
            }
        )
    return payload


def build_baseline_prompt(
    repo_name: str,
    review_scope: str,
    compact_rubric: List[Dict[str, Any]],
    repository_dump: Dict[str, Any],
    fallback_languages: List[str],
    fallback_signals: Dict[str, Any],
) -> str:
    dump_manifest = {
        "repo_name": repo_name,
        "review_scope": review_scope,
        "selected_file_count": repository_dump["selected_file_count"],
        "candidate_file_count": repository_dump["candidate_file_count"],
        "omitted_due_to_limits": repository_dump["omitted_due_to_limits"],
        "limits": repository_dump["limits"],
        "selected_files": repository_dump["selected_files"],
        "skipped_counts": repository_dump["skipped_counts"],
        "fallback_languages": fallback_languages,
        "fallback_repo_signals": fallback_signals,
    }

    return f"""You are evaluating the quality of a software repository as a whole artifact.

Your task is to perform a repository-level quality assessment using ONLY:
1) the filtered repository content provided in this prompt;
2) the predefined rubric below.

Do NOT use external tools, external knowledge, or assumptions not grounded in the repository content.

The goal is to produce a structured repository assessment directly comparable to a staged rubric-based evaluation pipeline.

GENERAL RULES
- Treat this as a full-repository baseline: no tool output, no external scanners, no runtime execution.
- Use only evidence visible in the provided repository content.
- If a criterion cannot be judged confidently from the provided content, use \"insufficient_evidence\".
- Use \"na\" only when the criterion is truly not applicable to this repository.
- Do NOT use \"tool_failed\".
- High scores (4-5) require explicit positive evidence.
- Missing evidence is not proof of quality.
- Missing evidence is not proof of absence either.
- Be conservative.
- Prefer lower confidence when evidence is partial or indirect.
- Cite only repository files that appear in the provided input.
- Each citation must use the form \"path:start-end\" where possible.
- The repository dump is filtered and may be truncated; if information is absent from this prompt, treat it as unseen rather than absent from the real repository.

OUTPUT COMPLETENESS RULES
- Return one module result for every rubric module, in rubric order.
- Return one group result for every group in each module, in rubric order.
- For disabled groups, return an empty leaf_results array.
- For enabled groups, return exactly one leaf result for every predefined leaf in rubric order.
- Do not invent modules, groups, or leaves.
- Do not omit predefined modules, groups, or leaves.

SCORE SCALE
- 0 = very poor / strongly contradicting the criterion
- 1 = poor
- 2 = weak
- 3 = acceptable / mixed
- 4 = good
- 5 = strong / clearly supported

CONFIDENCE SCALE
- 0.0 to 1.0
- Use lower confidence when evidence is indirect, partial, or ambiguous.

ACTIVATION TASK
For each module and group in the rubric:
- decide whether it is enabled for this repository,
- use applies_if as a semantic applicability guide, not a hard-coded rule,
- provide a short reason and activation_confidence.

SCORING TASK
For each enabled group:
- score each predefined leaf criterion individually,
- output one leaf result per predefined leaf,
- do not invent new leaves,
- do not omit leaves from enabled groups.

IMPORTANT SCORING CONSTRAINTS
- The final repository score will be aggregated outside the model.
- Do NOT produce final_score, module_score, or group_score.
- Only produce activation decisions and leaf-level judgments.

SIGNAL VOCABULARY
- inferred_repo_signals.repo_kind should be one of: backend, web_frontend, mobile_app, infra, library, ml_project, unknown.
- Set boolean repo signals using best-effort inference from the provided repository content only.

FIELD RULES
- score must be null when status is na or insufficient_evidence.
- score must be an integer 0..5 when status is scored.
- citations may be empty only when status is na.
- remediation may be empty for strong scores, but should usually be present for weak or mixed results.
- needs_human_review should be true when evidence is ambiguous, conflicting, or heavily inferential.

OUTPUT JSON SCHEMA
{{
  \"normalized_languages\": [\"...\"],
  \"inferred_repo_signals\": {{
    \"uses_api\": bool,
    \"uses_db\": bool,
    \"uses_queue\": bool,
    \"uses_filesystem\": bool,
    \"uses_frontend_framework\": bool,
    \"uses_mobile_sdk\": bool,
    \"uses_ml_framework\": bool,
    \"has_container\": bool,
    \"has_k8s\": bool,
    \"has_iac\": bool,
    \"uses_concurrency\": bool,
    \"uses_async\": bool,
    \"repo_kind\": \"backend\" | \"web_frontend\" | \"mobile_app\" | \"infra\" | \"library\" | \"ml_project\" | \"unknown\"
  }},
  \"module_results\": [
    {{
      \"module_id\": \"string\",
      \"enabled\": true,
      \"reason\": \"short reason\",
      \"activation_confidence\": 0.0,
      \"group_results\": [
        {{
          \"group_id\": \"string\",
          \"enabled\": true,
          \"reason\": \"short reason\",
          \"activation_confidence\": 0.0,
          \"leaf_results\": [
            {{
              \"leaf_id\": \"string\",
              \"status\": \"scored\" | \"na\" | \"insufficient_evidence\",
              \"score\": 0,
              \"confidence\": 0.0,
              \"needs_human_review\": false,
              \"citations\": [\"path:start-end\"],
              \"justification\": \"1-3 short sentences\",
              \"remediation\": [\"short action\", \"short action\"]
            }}
          ]
        }}
      ]
    }}
  ],
  \"global_notes\": [\"short note\"]
}}

COMPACT RUBRIC
{_json(compact_rubric)}

FILTERED REPOSITORY DUMP MANIFEST
{_json(dump_manifest)}

FILTERED REPOSITORY CONTENT
{repository_dump['repository_dump']}

Return JSON only.
"""


def _normalize_languages(value: Any, fallback: List[str]) -> List[str]:
    if not isinstance(value, list):
        return list(fallback)
    seen = set()
    out: List[str] = []
    for item in value:
        s = str(item).strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out or list(fallback)


def _normalize_repo_kind(value: Any, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in _ALLOWED_REPO_KINDS:
        return raw
    aliases = {
        "frontend": "web_frontend",
        "web": "web_frontend",
        "mobile": "mobile_app",
        "ml": "ml_project",
        "data_science": "ml_project",
        "service": "backend",
        "server": "backend",
        "full_stack": "backend",
        "cli": "library",
    }
    mapped = aliases.get(raw)
    if mapped in _ALLOWED_REPO_KINDS:
        return mapped
    if fallback in _ALLOWED_REPO_KINDS:
        return fallback
    return "unknown"


def _normalize_repo_signals(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    out = dict(fallback)
    for key in (
        "uses_api",
        "uses_db",
        "uses_queue",
        "uses_filesystem",
        "uses_frontend_framework",
        "uses_mobile_sdk",
        "uses_ml_framework",
        "has_container",
        "has_k8s",
        "has_iac",
        "uses_concurrency",
        "uses_async",
    ):
        if key in raw:
            out[key] = bool(raw.get(key))
    out["repo_kind"] = _normalize_repo_kind(raw.get("repo_kind"), str(fallback.get("repo_kind", "unknown")))
    return out


def normalize_activation_from_baseline(
    modules: List[Any],
    baseline_obj: Dict[str, Any],
    fallback_languages: List[str],
    fallback_signals: Dict[str, Any],
) -> Dict[str, Any]:
    raw_modules = baseline_obj.get("module_results")
    if not isinstance(raw_modules, list):
        raise ValueError("baseline JSON missing module_results[]")

    raw_by_module = {
        str(item.get("module_id", "")).strip(): item
        for item in raw_modules
        if isinstance(item, dict) and str(item.get("module_id", "")).strip()
    }

    normalized_languages = _normalize_languages(baseline_obj.get("normalized_languages"), fallback_languages)
    inferred_repo_signals = _normalize_repo_signals(baseline_obj.get("inferred_repo_signals"), fallback_signals)
    global_notes = _coerce_string_list(baseline_obj.get("global_notes"), max_items=50)

    module_activation = []
    for module in modules:
        raw_module = raw_by_module.get(module.module_id, {}) if isinstance(raw_by_module.get(module.module_id), dict) else {}
        module_enabled = bool(raw_module.get("enabled", eval_applies_if(module.applies_if, inferred_repo_signals)))
        module_reason = str(raw_module.get("reason", "")).strip() or (
            "Enabled by baseline activation output." if module_enabled else "Disabled by baseline activation output."
        )
        module_conf = _clamp01(_to_float_or_none(raw_module.get("activation_confidence")) or (0.75 if module_enabled else 0.6))

        raw_groups = raw_module.get("group_results") if isinstance(raw_module, dict) else []
        raw_group_map = {
            str(item.get("group_id", "")).strip(): item
            for item in raw_groups
            if isinstance(item, dict) and str(item.get("group_id", "")).strip()
        }

        activated_groups = []
        for group in module.groups:
            raw_group = raw_group_map.get(group.group_id, {}) if isinstance(raw_group_map.get(group.group_id), dict) else {}
            group_enabled = bool(raw_group.get("enabled", module_enabled))
            group_reason = str(raw_group.get("reason", "")).strip() or (
                "Enabled by baseline activation output." if group_enabled else "Disabled by baseline activation output."
            )
            group_conf = _clamp01(_to_float_or_none(raw_group.get("activation_confidence")) or module_conf)
            activated_groups.append(
                {
                    "group_id": group.group_id,
                    "enabled": group_enabled,
                    "reason": group_reason,
                    "activation_confidence": group_conf,
                    "evidence_hints": [],
                }
            )

        module_activation.append(
            {
                "module_id": module.module_id,
                "enabled": module_enabled,
                "reason": module_reason,
                "activation_confidence": module_conf,
                "activated_groups": activated_groups,
            }
        )

    return {
        "normalized_languages": normalized_languages,
        "language_confidence": {},
        "inferred_repo_signals": inferred_repo_signals,
        "module_activation": module_activation,
        "global_notes": global_notes,
    }


def _index_baseline_groups(baseline_obj: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for raw_module in baseline_obj.get("module_results") or []:
        if not isinstance(raw_module, dict):
            continue
        module_id = str(raw_module.get("module_id", "")).strip()
        if not module_id:
            continue
        for raw_group in raw_module.get("group_results") or []:
            if not isinstance(raw_group, dict):
                continue
            group_id = str(raw_group.get("group_id", "")).strip()
            if not group_id:
                continue
            out[(module_id, group_id)] = raw_group
    return out


def parse_baseline_leaf_scores(group_def: GroupDefinition, raw_group: Dict[str, Any]) -> List[LeafScore]:
    raw_leaf_results = raw_group.get("leaf_results") if isinstance(raw_group, dict) else []
    raw_leaf_map = {
        str(item.get("leaf_id", "")).strip(): item
        for item in raw_leaf_results or []
        if isinstance(item, dict) and str(item.get("leaf_id", "")).strip()
    }

    out: List[LeafScore] = []
    valid_statuses = {"scored", "na", "insufficient_evidence"}
    for leaf_def in group_def.leaves:
        raw_leaf = raw_leaf_map.get(leaf_def.leaf_id)
        if not isinstance(raw_leaf, dict):
            out.append(
                LeafScore(
                    leaf_id=leaf_def.leaf_id,
                    status="insufficient_evidence",
                    score=None,
                    confidence=0.0,
                    needs_human_review=True,
                    justification="Leaf result missing from baseline model output.",
                )
            )
            continue

        raw_status = str(raw_leaf.get("status", "")).strip().lower()
        status = raw_status if raw_status in valid_statuses else "insufficient_evidence"
        score: Optional[float] = None
        if status == "scored":
            score_val = _to_float_or_none(raw_leaf.get("score"))
            if score_val is None:
                status = "insufficient_evidence"
            else:
                score = max(0.0, min(5.0, float(int(round(score_val)))))

        confidence_val = _to_float_or_none(raw_leaf.get("confidence"))
        confidence = _clamp01(confidence_val) if confidence_val is not None else None
        citations = _coerce_string_list(raw_leaf.get("citations"), max_items=40)
        remediation = _coerce_string_list(raw_leaf.get("remediation"), max_items=20)
        justification_raw = raw_leaf.get("justification", "")
        justification = str(justification_raw).strip() if isinstance(justification_raw, (str, int, float, bool)) else ""
        needs_human_review = bool(raw_leaf.get("needs_human_review", status != "scored"))

        out.append(
            LeafScore(
                leaf_id=leaf_def.leaf_id,
                status=status,  # type: ignore[arg-type]
                score=score,
                confidence=confidence,
                needs_human_review=needs_human_review,
                citations=citations,
                justification=justification,
                remediation=remediation,
            )
        )

    return out


def run_baseline(opts: BaselineOptions) -> Dict[str, Any]:
    if not opts.repo:
        raise ValueError("repo is required")

    llm_config_path = Path(opts.llm_config).resolve() if opts.llm_config else Path(__file__).with_name("llm_config.yaml")
    llm_cfg = load_llm_config(str(llm_config_path))
    provider = resolve_default_provider(
        cli_provider=str(opts.llm_provider or "openai"),
        llm_cfg=llm_cfg,
    )
    default_model = str(opts.llm_model or "")
    scoring_model = str(opts.scoring_model or "")

    repo_root = Path(opts.repo).resolve()
    rubric_path = Path(opts.rubric).resolve() if opts.rubric else Path(__file__).with_name("nodes.yaml").resolve()
    out_path = Path(opts.out).resolve()

    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"repo must be a directory: {repo_root}")
    if not rubric_path.exists():
        raise ValueError(f"Rubric YAML not found: {rubric_path}")

    modules = load_rubric(_load_yaml(rubric_path))
    fallback_languages = detect_languages(repo_root)
    fallback_signals = infer_repo_signals(repo_root)

    work_dir = out_path.parent / (out_path.stem + "_work")
    llm_cache_dir = work_dir / "llm_cache"
    llm_outputs_dir = work_dir / "llm_outputs"
    token_usage_path = work_dir / "token_usage.json"
    usage_tracker = _TokenUsageTracker(token_usage_path)

    repo_dump = build_repository_dump(
        repo_root,
        max_files_in_prompt=max(1, int(opts.max_files_in_prompt)),
        max_total_chars=max(10_000, int(opts.max_total_chars)),
        max_chars_per_file=max(500, int(opts.max_chars_per_file)),
        max_bytes_per_file=max(2_000, int(opts.max_bytes_per_file)),
        max_walk_files=max(1_000, int(opts.max_walk_files)),
    )
    compact_rubric = build_compact_rubric_payload(modules)
    prompt = build_baseline_prompt(
        repo_name=repo_root.name,
        review_scope=opts.review_scope,
        compact_rubric=compact_rubric,
        repository_dump=repo_dump,
        fallback_languages=fallback_languages,
        fallback_signals=fallback_signals,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    _write_text(work_dir / "baseline_prompt.txt", prompt)
    _write_json(work_dir / "baseline_repo_dump_manifest.json", {k: v for k, v in repo_dump.items() if k != "repository_dump"})

    runtime = LLMRuntime(
        llm_cfg=llm_cfg,
        provider=str(provider or "openai").strip().lower(),
        default_model=default_model,
        scoring_model=scoring_model,
    )

    baseline_obj, baseline_usage = runtime.call_json_cached_with_usage(
        cache_dir=llm_cache_dir,
        kind="baseline_full_repo",
        phase="baseline",
        prompt=prompt,
        outputs_dir=llm_outputs_dir,
    )
    usage_tracker.add(stage="baseline", usage=baseline_usage)
    usage_tracker.flush()
    _write_json(work_dir / "baseline_raw_output.json", baseline_obj)

    activation = normalize_activation_from_baseline(
        modules=modules,
        baseline_obj=baseline_obj,
        fallback_languages=fallback_languages,
        fallback_signals=fallback_signals,
    )
    group_defs = activation_to_group_defs(modules, activation)
    _write_json(work_dir / "activation_used.json", activation)
    _write_json(work_dir / "group_defs.json", [dataclasses.asdict(group_def) for group_def in group_defs])

    policy = AggregationPolicy(
        missing_evidence_mode="exclude",
        empty_score_mode="none",
        apply_evidence_penalty_to_score=False,
    )

    raw_group_map = _index_baseline_groups(baseline_obj)
    rubric_group_by_id = {}
    for module in modules:
        for group in module.groups:
            rubric_group_by_id[group.group_id] = group

    group_results: List[GroupAggregationResult] = []
    group_results_by_module: Dict[str, List[GroupAggregationResult]] = {}
    for group_def in group_defs:
        rubric_group = rubric_group_by_id.get(group_def.group_id)
        if rubric_group is None:
            continue

        if not group_def.enabled:
            leaf_scores = [
                LeafScore(
                    leaf_id=leaf.leaf_id,
                    status="na",
                    score=None,
                    confidence=1.0,
                    needs_human_review=False,
                    justification="Group disabled by baseline activation gating.",
                )
                for leaf in rubric_group.leaves
            ]
        else:
            raw_group = raw_group_map.get((group_def.module_id, group_def.group_id), {})
            leaf_scores = parse_baseline_leaf_scores(group_def, raw_group)

        group_result = aggregate_group(group_def=group_def, leaf_scores=leaf_scores, policy=policy)
        group_results.append(group_result)
        group_results_by_module.setdefault(group_def.module_id, []).append(group_result)

    module_results: List[ModuleAggregationResult] = []
    for module in modules:
        module_enabled = any(group_def.enabled for group_def in group_defs if group_def.module_id == module.module_id)
        module_group_defs = [group_def for group_def in group_defs if group_def.module_id == module.module_id]
        module_group_results = group_results_by_module.get(module.module_id, [])
        module_result = aggregate_module(
            module_id=module.module_id,
            module_name=module.module_name,
            module_weight=module.module_weight,
            group_defs=module_group_defs,
            group_results=module_group_results,
            policy=policy,
            module_enabled=module_enabled,
        )
        module_results.append(module_result)

    final = aggregate_final(module_results, policy)
    _write_json(work_dir / "group_results.json", [dataclasses.asdict(group_result) for group_result in group_results])

    languages = activation.get("normalized_languages") or fallback_languages
    signals = activation.get("inferred_repo_signals") or fallback_signals
    report = {
        "final": dataclasses.asdict(final),
        "report_payload": build_report_payload(final),
        "activation": activation,
        "simal_disabled": True,
        "token_usage_path": str(token_usage_path),
        "token_usage_total": dict(usage_tracker.data.get("total") or {}),
        "repo": {
            "path": str(repo_root),
            "name": repo_root.name,
            "detected_languages": list(languages),
            "signals": dict(signals),
        },
        "work_dir": str(work_dir),
    }
    _write_json(out_path, report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SAGES baseline full-repository evaluation")
    ap.add_argument("--repo", type=str, required=True, help="Path to repository to evaluate")
    ap.add_argument("--rubric", type=str, default=str(Path(__file__).with_name("nodes.yaml")), help="Rubric DAG YAML")
    ap.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        help="LLM provider for the single baseline assessment call: openai|gemini|claude",
    )
    ap.add_argument(
        "--llm-config",
        type=str,
        default=str(Path(__file__).with_name("llm_config.yaml")),
        help="Path to YAML config for LLM providers",
    )
    ap.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="Default LLM model name for the baseline call",
    )
    ap.add_argument(
        "--scoring-model",
        type=str,
        default="",
        help="Optional model override for the baseline evaluation call",
    )
    ap.add_argument("--out", type=str, default="baseline_quality_report.json", help="Output JSON report path")
    ap.add_argument("--review-scope", type=str, default="full_repo", help="Review scope string for prompts")
    ap.add_argument("--max-files-in-prompt", type=int, default=180, help="Maximum files included in the baseline prompt")
    ap.add_argument("--max-total-chars", type=int, default=350000, help="Maximum total prompt characters reserved for repository content")
    ap.add_argument("--max-chars-per-file", type=int, default=8000, help="Maximum prompt characters per individual file")
    ap.add_argument("--max-bytes-per-file", type=int, default=96000, help="Maximum bytes read per file before truncation")
    ap.add_argument("--max-walk-files", type=int, default=25000, help="Maximum repository files scanned while building the prompt")
    args = ap.parse_args(argv)

    run_baseline(
        BaselineOptions(
            repo=args.repo,
            rubric=args.rubric,
            llm_provider=args.llm_provider,
            llm_config=args.llm_config,
            llm_model=args.llm_model,
            scoring_model=args.scoring_model,
            out=args.out,
            review_scope=args.review_scope,
            max_files_in_prompt=args.max_files_in_prompt,
            max_total_chars=args.max_total_chars,
            max_chars_per_file=args.max_chars_per_file,
            max_bytes_per_file=args.max_bytes_per_file,
            max_walk_files=args.max_walk_files,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
