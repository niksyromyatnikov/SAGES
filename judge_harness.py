from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from autoeval_scoring import JudgePromptInputs, JudgeStrategy
from tool_summarizer import SummarizeBudgets, summarize_tool_report


CandidateStrategy = Literal["pipeline", "baseline"]

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".mypy_cache",
    ".next",
    ".nox",
    ".nuxt",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "bower_components",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "jspm_packages",
    "node_modules",
    "site-packages",
    "target",
    "venv",
    "vendor",
}

BINARY_EXTS = {
    ".7z",
    ".avi",
    ".bmp",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".flac",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".ogg",
    ".pdf",
    ".png",
    ".pyc",
    ".pyd",
    ".pyo",
    ".rar",
    ".so",
    ".svg",
    ".tar",
    ".tgz",
    ".wav",
    ".webm",
    ".webp",
    ".zip",
}

TEXT_EXTS = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".dockerfile",
    ".env",
    ".go",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".less",
    ".lua",
    ".md",
    ".mjs",
    ".php",
    ".properties",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".tf",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}


@dataclass
class RepoDumpOptions:
    max_files: Optional[int] = None
    max_chars_per_file: Optional[int] = None
    max_total_chars: Optional[int] = None
    include_line_numbers: bool = True


@dataclass
class ToolEvidenceLoadOptions:
    # These options control judge-facing normalization of cached tool artifacts.
    # The .result.json files are treated as execution manifests, not as complete raw tool output.
    include_full_findings: bool = True
    require_declared_artifacts: bool = True
    max_findings_per_report: int = 5000
    max_top_rules: int = 20
    max_top_files: int = 20
    max_excerpt_chars: int = 500
    max_stdout_chars: int = 2000
    max_stderr_chars: int = 4000


def _read_text(path: Path, *, max_bytes: Optional[int] = None) -> str:
    data = path.read_bytes()
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_metadata(path: Path) -> Dict[str, Any]:
    text = _read_text(path)
    return {
        "path": str(path),
        "chars": len(text),
        "lines": text.count("\n") + 1 if text else 0,
    }


def _is_probably_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTS:
        return False
    if suffix in TEXT_EXTS or path.name.lower() in {"dockerfile", "makefile"}:
        return True

    try:
        chunk = path.read_bytes()[:2048]
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


def _iter_repo_files(repo_root: Path, max_files: Optional[int]) -> Iterable[Path]:
    seen = 0
    for path in sorted(repo_root.rglob("*")):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(repo_root).parts
        if any(part in IGNORE_DIRS for part in rel_parts[:-1]):
            continue
        if not _is_probably_text_file(path):
            continue
        yield path
        seen += 1
        if max_files is not None and seen >= max_files:
            break


def _line_numbered(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{index:0{width}d}: {line}" for index, line in enumerate(lines, start=1))


def build_repo_source_dump(repo_root: Path | str, options: Optional[RepoDumpOptions] = None) -> str:
    repo_root = Path(repo_root)
    options = options or RepoDumpOptions()
    chunks: List[str] = []
    total_chars = 0

    for path in _iter_repo_files(repo_root, options.max_files):
        rel_path = path.relative_to(repo_root).as_posix()
        text = _read_text(path)
        truncated = False
        if options.max_chars_per_file is not None and len(text) > options.max_chars_per_file:
            text = text[: options.max_chars_per_file]
            truncated = True
        if options.include_line_numbers:
            text = _line_numbered(text)
        header = f"===== FILE: {rel_path} =====\n"
        if truncated:
            header += "[TRUNCATED]\n"
        chunk = header + text + "\n"

        if options.max_total_chars is not None and total_chars + len(chunk) > options.max_total_chars:
            remaining = options.max_total_chars - total_chars
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total_chars += remaining
            break

        chunks.append(chunk)
        total_chars += len(chunk)

    return "\n".join(chunks)


def _classification_for_citation(citation: str) -> str:
    citation = (citation or "").strip()
    if not citation:
        return "empty"
    if citation.startswith("retrieval."):
        return "retrieval"
    if citation.startswith("activation_hint_evidence."):
        return "activation_hint"
    if citation.startswith("simal"):
        return "simal"
    if citation.startswith("dynamic_tools") or citation.startswith("tool_summary"):
        return "tool"
    if ":" in citation and any(char.isdigit() for char in citation.rsplit(":", 1)[-1]):
        return "repo_line"
    return "other"


def _citation_stats(group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    total = 0
    for group in group_results:
        for leaf in group.get("leaf_results") or []:
            for citation in leaf.get("citations") or []:
                kind = _classification_for_citation(str(citation))
                counts[kind] = counts.get(kind, 0) + 1
                total += 1
    return {
        "total_citations": total,
        "by_kind": dict(sorted(counts.items())),
        "has_repo_line_citations": counts.get("repo_line", 0) > 0,
        "has_retrieval_citations": counts.get("retrieval", 0) > 0,
        "has_activation_hint_citations": counts.get("activation_hint", 0) > 0,
        "has_simal_citations": counts.get("simal", 0) > 0,
        "has_tool_citations": counts.get("tool", 0) > 0,
    }


def _compact_report_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "status": summary.get("status"),
        "tool": summary.get("tool"),
        "format": summary.get("format"),
    }
    if summary.get("status") == "ok":
        compact.update(
            {
                "findings_total": summary.get("findings_total"),
                "by_severity": summary.get("by_severity"),
                "top_rules": summary.get("top_rules"),
                "top_files": summary.get("top_files"),
                "sample_findings": summary.get("sample_findings"),
                "extra": summary.get("extra"),
            }
        )
    if summary.get("status") == "unavailable":
        compact["reason"] = summary.get("reason")
    return compact


def _compact_tool_result(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    tool_summary = tool_result.get("tool_summary") if isinstance(tool_result.get("tool_summary"), dict) else {}
    reports = tool_summary.get("reports") if isinstance(tool_summary.get("reports"), dict) else {}
    compact_reports = {
        name: _compact_report_summary(report)
        for name, report in sorted(reports.items())
        if isinstance(report, dict)
    }
    return {
        "tool": tool_result.get("tool"),
        "evidence_class": tool_result.get("evidence_class"),
        "status": tool_result.get("status"),
        "exit_code": tool_result.get("exit_code"),
        "duration_sec": tool_result.get("duration_sec"),
        "parsing_format": (tool_result.get("parsing") or {}).get("format"),
        "treat_nonzero_as_findings": tool_result.get("treat_nonzero_as_findings"),
        "reports": compact_reports,
    }


def _clamp_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _tool_evidence_json(repository_tool_evidence: List[Dict[str, Any]]) -> str:
    return json.dumps(repository_tool_evidence, ensure_ascii=False, indent=2)


def _resolve_tool_cache(path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    path = Path(path)
    if path.is_file():
        path = path.parent

    if path.name == "_cache" and path.parent.name == "tool_outputs":
        return path, path.parent.parent

    candidate = path / "results_work" / "tool_outputs" / "_cache"
    if candidate.exists():
        return candidate, path / "results_work"

    return None, None


def _resolve_work_file(work_dir: Path, relative_path: str) -> Path:
    return work_dir / Path(relative_path)


def _normalized_report_for_judge(
    tool: str,
    parsing_format: str,
    report_name: str,
    report_path: Path,
    options: ToolEvidenceLoadOptions,
) -> Dict[str, Any]:
    requested_limit = max(0, int(options.max_findings_per_report))
    budgets = SummarizeBudgets(
        max_findings=requested_limit,
        max_top_rules=options.max_top_rules,
        max_top_files=options.max_top_files,
        max_excerpt_chars=options.max_excerpt_chars,
    )
    summary = summarize_tool_report(
        tool=tool,
        format_name=parsing_format,
        report_file=report_path,
        budgets=budgets,
    )
    report: Dict[str, Any] = {
        "report_name": report_name,
        "status": summary.get("status"),
        "tool": summary.get("tool"),
        "format": summary.get("format"),
        "report_path": str(report_path).replace("\\", "/"),
    }
    if summary.get("status") == "ok":
        findings = list(summary.get("sample_findings") or [])
        findings_total = int(summary.get("findings_total") or 0)
        findings_included_count = len(findings) if options.include_full_findings else 0
        report.update(
            {
                "findings_total": findings_total,
                "by_severity": summary.get("by_severity"),
                "top_rules": summary.get("top_rules"),
                "top_files": summary.get("top_files"),
                "extra": summary.get("extra"),
                "findings": findings if options.include_full_findings else [],
                "findings_included_count": findings_included_count,
                "findings_limit": requested_limit,
                "findings_truncated": bool(options.include_full_findings) and findings_included_count < findings_total,
                "findings_complete": bool(options.include_full_findings) and findings_included_count >= findings_total,
            }
        )
    if summary.get("status") == "unavailable":
        report["reason"] = summary.get("reason")
    return report


def _load_tool_evidence(path: Path, options: Optional[ToolEvidenceLoadOptions] = None) -> List[Dict[str, Any]]:
    options = options or ToolEvidenceLoadOptions()
    cache_dir, work_dir = _resolve_tool_cache(path)
    if cache_dir is None or work_dir is None or not cache_dir.exists():
        return []

    out: List[Dict[str, Any]] = []
    for result_path in sorted(cache_dir.glob("*.result.json")):
        try:
            tool_result = _read_json(result_path)
        except Exception:
            continue

        if not isinstance(tool_result, dict):
            continue

        result_path_str = str(result_path).replace("\\", "/")

        # result.json provides execution metadata plus artifact locations.
        # The actual judge-visible evidence should come from the artifact files when present.
        parsing_format = str((tool_result.get("parsing") or {}).get("format") or "")
        artifacts = tool_result.get("artifacts") if isinstance(tool_result.get("artifacts"), dict) else {}
        reports: Dict[str, Any] = {}
        missing_artifacts: List[str] = []
        if parsing_format and artifacts:
            for report_name, report_rel_path in sorted(artifacts.items()):
                if not isinstance(report_rel_path, str) or not report_rel_path:
                    continue
                report_file = _resolve_work_file(work_dir, report_rel_path)
                if not report_file.exists():
                    missing_artifacts.append(str(report_file).replace("\\", "/"))
                    continue
                reports[report_name] = _normalized_report_for_judge(
                    tool=str(tool_result.get("tool") or ""),
                    parsing_format=parsing_format,
                    report_name=report_name,
                    report_path=report_file,
                    options=options,
                )

        if missing_artifacts and options.require_declared_artifacts:
            raise FileNotFoundError(
                "Declared tool artifact(s) missing for "
                f"{tool_result.get('tool')!r} in {result_path_str} : {missing_artifacts}"
            )

        if artifacts and not reports and options.require_declared_artifacts:
            raise FileNotFoundError(
                "No declared tool artifacts could be loaded for "
                f"{tool_result.get('tool')!r} in {result_path_str}"
            )

        if not reports:
            out.append(_compact_tool_result(tool_result))
            continue

        stdout_excerpt = ""
        stderr_excerpt = ""
        stdout_path = tool_result.get("stdout_path")
        stderr_path = tool_result.get("stderr_path")
        if isinstance(stdout_path, str) and stdout_path:
            stdout_file = _resolve_work_file(work_dir, stdout_path)
            if stdout_file.exists():
                stdout_excerpt = _clamp_text(_read_text(stdout_file, max_bytes=options.max_stdout_chars), options.max_stdout_chars)
        if isinstance(stderr_path, str) and stderr_path:
            stderr_file = _resolve_work_file(work_dir, stderr_path)
            if stderr_file.exists():
                stderr_excerpt = _clamp_text(_read_text(stderr_file, max_bytes=options.max_stderr_chars), options.max_stderr_chars)

        entry: Dict[str, Any] = {
            "tool": tool_result.get("tool"),
            "tool_description": tool_result.get("tool_description"),
            "evidence_class": tool_result.get("evidence_class"),
            "status": tool_result.get("status"),
            "exit_code": tool_result.get("exit_code"),
            "duration_sec": tool_result.get("duration_sec"),
            "parsing_format": parsing_format,
            "treat_nonzero_as_findings": tool_result.get("treat_nonzero_as_findings"),
            "treat_nonzero_as_execution_result": tool_result.get("treat_nonzero_as_execution_result"),
            "cmd": tool_result.get("cmd"),
            "declared_outputs": tool_result.get("declared_outputs") or [],
            "artifacts": {name: str(path) for name, path in sorted(artifacts.items())},
            "reports": reports,
        }
        if stdout_excerpt:
            entry["stdout_excerpt"] = stdout_excerpt
        if stderr_excerpt:
            entry["stderr_excerpt"] = stderr_excerpt
        out.append(entry)

    return out


def load_tool_evidence_from_run(
    run_dir: Path | str,
    options: Optional[ToolEvidenceLoadOptions] = None,
) -> List[Dict[str, Any]]:
    return _load_tool_evidence(Path(run_dir), options)


def load_tool_evidence_from_cache(
    cache_dir: Path | str,
    options: Optional[ToolEvidenceLoadOptions] = None,
) -> List[Dict[str, Any]]:
    return _load_tool_evidence(Path(cache_dir), options)


def summarize_tool_evidence_for_prompt(
    repository_tool_evidence: List[Dict[str, Any]],
    prompt_text: Optional[str] = None,
) -> Dict[str, Any]:
    report_paths: List[str] = []
    findings_total = 0
    findings_included_count = 0
    for entry in repository_tool_evidence:
        reports = entry.get("reports") if isinstance(entry, dict) else None
        if not isinstance(reports, dict):
            continue
        for report in reports.values():
            if not isinstance(report, dict):
                continue
            report_path = report.get("report_path")
            if isinstance(report_path, str) and report_path:
                report_paths.append(report_path)
            findings_total += int(report.get("findings_total") or 0)
            findings_included_count += int(report.get("findings_included_count") or 0)

    serialized = _tool_evidence_json(repository_tool_evidence)
    summary: Dict[str, Any] = {
        "tool_entry_count": len(repository_tool_evidence),
        "report_count": len(report_paths),
        "findings_total": findings_total,
        "findings_included_count": findings_included_count,
        "serialized_chars": len(serialized),
        "sample_report_paths": report_paths[:10],
    }
    if prompt_text is not None:
        summary["tool_evidence_json_in_prompt"] = serialized in prompt_text
        summary["missing_report_paths_in_prompt"] = [path for path in report_paths if path not in prompt_text][:10]
    return summary


def _manifest_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    selected_files = manifest.get("selected_files") if isinstance(manifest.get("selected_files"), list) else []
    truncated_files = sum(1 for item in selected_files if isinstance(item, dict) and item.get("truncated"))
    total_chars = sum(int(item.get("chars_in_prompt") or 0) for item in selected_files if isinstance(item, dict))
    return {
        "selected_file_count": len(selected_files),
        "truncated_file_count": truncated_files,
        "prompt_chars_from_repo_dump": total_chars,
        "sample_files": [item.get("path") for item in selected_files[:25] if isinstance(item, dict)],
    }


def _base_evidence_profile(
    strategy: JudgeStrategy,
    group_results: List[Dict[str, Any]],
    *,
    candidate_tool_evidence: Optional[List[Dict[str, Any]]] = None,
    source_visibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    citation_stats = _citation_stats(group_results)
    profile: Dict[str, Any] = {
        "strategy": strategy,
        "citation_stats": citation_stats,
        "group_result_count": len(group_results),
        "has_candidate_tool_evidence": bool(candidate_tool_evidence),
    }
    if source_visibility is not None:
        profile["source_visibility"] = source_visibility
    if candidate_tool_evidence:
        profile["candidate_tool_evidence_count"] = len(candidate_tool_evidence)
        profile["candidate_tool_evidence_classes"] = sorted(
            {
                str(item.get("evidence_class"))
                for item in candidate_tool_evidence
                if item.get("evidence_class")
            }
        )
    return profile


def _load_group_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_json(path)
    return data if isinstance(data, list) else []


def build_pipeline_candidate_bundle(run_dir: Path | str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    results = _read_json(run_dir / "results")
    group_results = _load_group_results(run_dir / "results_work" / "group_results.json")
    candidate_tool_evidence = _load_tool_evidence(run_dir)
    evidence_availability = _base_evidence_profile(
        "pipeline",
        group_results,
        candidate_tool_evidence=candidate_tool_evidence,
    )
    evidence_availability["simal_disabled"] = bool(results.get("simal_disabled"))

    candidate_assessment = {
        "repo": results.get("repo"),
        "final": results.get("final"),
        "report_payload": results.get("report_payload"),
        "activation": results.get("activation"),
        "group_results": group_results,
        "warnings": (results.get("final") or {}).get("warnings", []),
    }
    return {
        "strategy": "pipeline",
        "candidate_label": run_dir.name,
        "candidate_assessment": candidate_assessment,
        "evidence_availability": evidence_availability,
        "candidate_tool_evidence": candidate_tool_evidence,
    }


def build_baseline_candidate_bundle(run_dir: Path | str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    report = _read_json(run_dir / "target_repo_baseline_quality_report.json")
    group_results = _load_group_results(run_dir / "group_results.json")
    activation = _read_json(run_dir / "activation_used.json")
    prompt_meta = _prompt_metadata(run_dir / "baseline_prompt.txt") if (run_dir / "baseline_prompt.txt").exists() else {}
    manifest_path = run_dir / "baseline_repo_dump_manifest.json"
    source_visibility = _manifest_summary(_read_json(manifest_path)) if manifest_path.exists() else {}
    evidence_availability = _base_evidence_profile(
        "baseline",
        group_results,
        candidate_tool_evidence=[],
        source_visibility=source_visibility,
    )
    evidence_availability["baseline_prompt"] = prompt_meta
    evidence_availability["repo_dump_is_filtered"] = True

    candidate_assessment = {
        "repo": report.get("repo"),
        "final": report.get("final"),
        "report_payload": report.get("report_payload"),
        "activation": activation,
        "group_results": group_results,
        "warnings": (report.get("final") or {}).get("warnings", []),
        "source_visibility": source_visibility,
    }
    return {
        "strategy": "baseline",
        "candidate_label": run_dir.name,
        "candidate_assessment": candidate_assessment,
        "evidence_availability": evidence_availability,
        "candidate_tool_evidence": [],
    }


def build_candidate_bundle(run_dir: Path | str, strategy: CandidateStrategy) -> Dict[str, Any]:
    if strategy == "pipeline":
        return build_pipeline_candidate_bundle(run_dir)
    if strategy == "baseline":
        return build_baseline_candidate_bundle(run_dir)
    raise ValueError(f"Unsupported strategy: {strategy!r}")


def make_judge_prompt_inputs(
    repo_source_text: str,
    rubric_text: str,
    candidate_bundle: Dict[str, Any],
    repository_tool_evidence: Optional[List[Dict[str, Any]]] = None,
    extra_instructions: str = "",
) -> JudgePromptInputs:
    strategy = candidate_bundle.get("strategy")
    strategy = strategy if strategy in {"pipeline", "baseline", "unknown"} else "unknown"
    return JudgePromptInputs(
        repo_source_text=repo_source_text,
        rubric_text=rubric_text,
        candidate_assessment=candidate_bundle.get("candidate_assessment") or {},
        candidate_strategy=strategy,
        candidate_label=str(candidate_bundle.get("candidate_label") or "candidate_assessment"),
        repository_tool_evidence=repository_tool_evidence,
        extra_instructions=extra_instructions,
    )


def load_rubric_text(rubric_path: Path | str) -> str:
    return Path(rubric_path).read_text(encoding="utf-8")
