from __future__ import annotations

import json
import re
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable


# ============================================================
# Summary schema (V1)
# ============================================================

@dataclass
class SummarizeBudgets:
    max_findings: int = 25
    max_top_rules: int = 10
    max_top_files: int = 10
    max_excerpt_chars: int = 180
    max_bytes_json: int = 8_000_000
    max_bytes_text: int = 3_000_000
    max_items_inventory: int = 200


def _safe_read_bytes(path: Path, max_bytes: int) -> Optional[bytes]:
    if not path.exists() or not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data


def _safe_read_text(path: Path, max_bytes: int) -> str:
    b = _safe_read_bytes(path, max_bytes=max_bytes)
    if b is None:
        return ""
    return b.decode("utf-8", errors="replace")


def _safe_read_json(path: Path, max_bytes: int) -> Optional[Any]:
    b = _safe_read_bytes(path, max_bytes=max_bytes)
    if b is None:
        return None
    try:
        return json.loads(b.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _safe_read_json_lines(path: Path, max_bytes: int) -> List[Any]:
    """
    For tools that emit JSON stream (one JSON per line).
    We keep it deterministic: parse each line, ignore invalid lines.
    """
    txt = _safe_read_text(path, max_bytes=max_bytes)
    out: List[Any] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _clamp_str(s: str, n: int) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/").lstrip("./")


def _severity_rank(sev: str) -> int:
    sev = (sev or "").lower()
    order = {
        "critical": 5,
        "error": 4,
        "high": 4,
        "warning": 3,
        "medium": 3,
        "low": 2,
        "info": 1,
        "note": 1,
        "unknown": 0,
    }
    return order.get(sev, 0)


def _stable_top_k(counter: Dict[str, int], k: int) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _finding_sort_key(f: Dict[str, Any]) -> Tuple[int, str, int, str]:
    fp = f.get("file")
    if not isinstance(fp, str) or not fp:
        fp = f.get("path") if isinstance(f.get("path"), str) else ""
    return (
        -_severity_rank(str(f.get("severity", "unknown"))),
        _norm_path(str(fp or "")),
        int(f.get("line", 0) or 0),
        str(f.get("rule_id", "")),
    )


def _finalize_summary(
    tool: str,
    format_name: str,
    report_path: str,
    findings: List[Dict[str, Any]],
    by_severity: Dict[str, int],
    top_rules: Dict[str, int],
    top_files: Dict[str, int],
    extra: Dict[str, Any],
    budgets: SummarizeBudgets,
) -> Dict[str, Any]:
    # Repo-scope hygiene: many tools report issues in dependency/build folders.
    # We filter those out here so LLM-facing summaries focus on project code.
    excluded_dirs = {
        ".cache",
        ".mypy_cache",
        ".next",
        ".nuxt",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".yarn",
        "__pycache__",
        "bower_components",
        "build",
        "coverage",
        "dist",
        "jspm_packages",
        "node_modules",
        "target",
        "vendor",
        "venv",
    }

    def _extract_finding_path(f: Dict[str, Any]) -> str:
        loc = f.get("location")
        if isinstance(loc, dict):
            p = loc.get("path")
            if isinstance(p, str):
                return p
        for k in ("file", "path", "filename"):
            p = f.get(k)
            if isinstance(p, str):
                return p
        return ""

    def _is_excluded_path(path_str: str) -> bool:
        if not path_str:
            return False
        rp = (path_str or "").replace("\\", "/")
        parts = [p for p in rp.split("/") if p and p not in {".", ".."}]
        return any(p in excluded_dirs for p in parts)

    before = len(findings)
    findings = [f for f in findings if not _is_excluded_path(_extract_finding_path(f))]
    excluded_count = before - len(findings)

    # Recompute counters from filtered findings to keep totals consistent.
    by_severity = {}
    top_rules = {}
    top_files = {}
    for f in findings:
        sev = str(f.get("severity", "unknown") or "unknown").lower()
        by_severity[sev] = int(by_severity.get(sev, 0)) + 1

        rule = str(
            f.get("rule_id")
            or f.get("code")
            or f.get("check_id")
            or f.get("id")
            or ""
        ).strip()
        if rule:
            top_rules[rule] = int(top_rules.get(rule, 0)) + 1

        fp = _norm_path(_extract_finding_path(f))
        if fp:
            top_files[fp] = int(top_files.get(fp, 0)) + 1

    findings_sorted = sorted(findings, key=_finding_sort_key)[: budgets.max_findings]
    for f in findings_sorted:
        if "message" in f and isinstance(f["message"], str):
            f["message"] = _clamp_str(f["message"], budgets.max_excerpt_chars)
        if "excerpt" in f and isinstance(f["excerpt"], str):
            f["excerpt"] = _clamp_str(f["excerpt"], budgets.max_excerpt_chars)

    by_sev_sorted = dict(
        sorted(by_severity.items(), key=lambda kv: (-_severity_rank(kv[0]), kv[0]))
    )

    total = int(sum(by_severity.values())) if by_severity else len(findings)

    extra_out = dict(extra or {})
    if excluded_count:
        extra_out["repo_scope_filter"] = {
            "excluded_dirs": sorted(excluded_dirs),
            "excluded_findings": int(excluded_count),
        }

    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "ok",
        "tool": tool,
        "format": format_name,
        "report_path": report_path,
        "findings_total": total,
        "by_severity": by_sev_sorted,
        "top_rules": _stable_top_k(top_rules, budgets.max_top_rules),
        "top_files": _stable_top_k(top_files, budgets.max_top_files),
        "sample_findings": findings_sorted,
        "extra": extra_out,
    }


def _unavailable(tool: str, format_name: str, report_path: str, reason: str) -> Dict[str, Any]:
    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "unavailable",
        "tool": tool,
        "format": format_name,
        "report_path": report_path,
        "reason": reason,
    }


# ============================================================
# Generic helpers (used by many tools)
# ============================================================

def _generic_json_findings_from_common_fields(report: Any) -> List[Dict[str, Any]]:
    """
    Tries to extract findings from common shapes:
    - list of dicts with file/line/message/code/severity
    - dict with "results"/"issues"/"findings"/"warnings"/"vulnerabilities"
    This is a fallback, deterministic.
    """
    candidates: List[Any] = []

    if isinstance(report, list):
        candidates = report
    elif isinstance(report, dict):
        for key in ["results", "issues", "findings", "warnings", "diagnostics", "problems", "vulnerabilities"]:
            v = report.get(key)
            if isinstance(v, list):
                candidates = v
                break

    out: List[Dict[str, Any]] = []
    for it in candidates:
        if not isinstance(it, dict):
            continue
        file = _norm_path(str(it.get("file") or it.get("path") or it.get("filename") or it.get("filePath") or ""))
        line = int(it.get("line") or (it.get("start_line") or 0) or 0)
        rule = str(it.get("rule") or it.get("rule_id") or it.get("code") or it.get("check_id") or it.get("id") or "unknown")
        msg = str(it.get("message") or it.get("msg") or it.get("text") or it.get("description") or "Finding.")
        sev = str(it.get("severity") or it.get("level") or "unknown").lower()
        out.append({"severity": sev, "rule_id": rule, "file": file, "line": line, "message": msg})
    return out


def _summarize_findings_list(
    tool: str,
    format_name: str,
    report_path: str,
    findings: List[Dict[str, Any]],
    budgets: SummarizeBudgets,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    by_sev: Dict[str, int] = {}
    top_rules: Dict[str, int] = {}
    top_files: Dict[str, int] = {}

    for f in findings:
        sev = str(f.get("severity") or "unknown").lower()
        rid = str(f.get("rule_id") or "unknown")
        file = _norm_path(str(f.get("file") or ""))
        by_sev[sev] = by_sev.get(sev, 0) + 1
        top_rules[rid] = top_rules.get(rid, 0) + 1
        if file:
            top_files[file] = top_files.get(file, 0) + 1

    return _finalize_summary(
        tool=tool,
        format_name=format_name,
        report_path=report_path,
        findings=findings,
        by_severity=by_sev,
        top_rules=top_rules,
        top_files=top_files,
        extra=extra or {},
        budgets=budgets,
    )


def _with_glossary(extra: Optional[Dict[str, Any]], glossary: Dict[str, str]) -> Dict[str, Any]:
    """Merge a small glossary into summary.extra.

    Intended for LLM-facing summaries: keep definitions short and high-signal.
    """
    out: Dict[str, Any] = dict(extra or {})
    if not glossary:
        return out
    existing = out.get("glossary")
    merged: Dict[str, str] = dict(existing) if isinstance(existing, dict) else {}
    for k, v in glossary.items():
        if k and v and str(k) not in merged:
            merged[str(k)] = str(v)
    if merged:
        out["glossary"] = merged
    return out


# ============================================================
# Specific parsers (stable schemas)
# ============================================================

def summarize_gitleaks_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # gitleaks JSON is usually a list, sometimes { "Leaks": [...] }
    items = report.get("Leaks") if isinstance(report, dict) and isinstance(report.get("Leaks"), list) else report
    if not isinstance(items, list):
        items = []

    findings = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rule = str(it.get("RuleID") or it.get("rule") or "unknown")
        file = _norm_path(str(it.get("File") or it.get("file") or ""))
        line = int(it.get("StartLine") or it.get("Line") or it.get("line") or 0)
        sev = str(it.get("Severity") or it.get("severity") or "unknown").lower()
        msg = str(it.get("Description") or it.get("description") or "Secret-like pattern detected.")
        findings.append({"severity": sev, "rule_id": rule, "file": file, "line": line, "message": msg, "excerpt": "***"})

    extra_out = _with_glossary(
        None,
        {
            "RuleID": "Gitleaks rule identifier for the detected secret pattern.",
            "Severity": "Tool-reported severity level for the finding.",
        },
    )
    return _summarize_findings_list(
        tool="gitleaks",
        format_name="gitleaks_json",
        report_path=rp,
        findings=findings,
        budgets=budgets,
        extra=extra_out,
    )


def summarize_semgrep_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    results = report.get("results") if isinstance(report, dict) else None
    if not isinstance(results, list):
        results = []

    findings = []
    for r in results:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("check_id") or "unknown")
        file = _norm_path(str(r.get("path") or ""))
        start = r.get("start") or {}
        line = int(start.get("line") or 0) if isinstance(start, dict) else 0
        extra = r.get("extra") or {}
        msg = str(extra.get("message") or "Semgrep finding.") if isinstance(extra, dict) else "Semgrep finding."
        sev = str(extra.get("severity") or "unknown").lower() if isinstance(extra, dict) else "unknown"
        findings.append({"severity": sev, "rule_id": rid, "file": file, "line": line, "message": msg})

    extra_out = _with_glossary(
        {
            "errors": len(report.get("errors") or []) if isinstance(report, dict) and isinstance(report.get("errors"), list) else 0,
        },
        {
            "check_id": "Semgrep rule identifier.",
            "path": "File path where the finding occurred.",
            "severity": "Semgrep-reported severity for a finding.",
        },
    )
    return _summarize_findings_list(tool="semgrep", format_name="semgrep_json", report_path=rp, findings=findings, budgets=budgets, extra=extra_out)


def summarize_npm_audit_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    if not isinstance(report, dict):
        report = {}
    findings: List[Dict[str, Any]] = []

    vulns = report.get("vulnerabilities")
    if isinstance(vulns, dict):
        for pkg, meta in vulns.items():
            if not isinstance(meta, dict):
                continue
            sev = str(meta.get("severity") or "unknown").lower()
            via = meta.get("via")
            rid = "npm-audit"
            msg = "npm vulnerability"
            if isinstance(via, list) and via:
                first = via[0]
                if isinstance(first, dict):
                    rid = str(first.get("source") or first.get("name") or "advisory")
                    msg = str(first.get("title") or first.get("name") or "npm advisory")
                else:
                    rid = str(first)
            findings.append({"severity": sev, "rule_id": rid, "file": str(pkg), "line": 0, "message": msg})

    advisories = report.get("advisories")
    if isinstance(advisories, dict) and advisories:
        for _, adv in advisories.items():
            if not isinstance(adv, dict):
                continue
            sev = str(adv.get("severity") or "unknown").lower()
            pkg = str(adv.get("module_name") or adv.get("name") or "")
            rid = str(adv.get("cve") or adv.get("id") or adv.get("url") or "advisory")
            msg = str(adv.get("title") or adv.get("overview") or "npm advisory")
            findings.append({"severity": sev, "rule_id": rid, "file": pkg, "line": 0, "message": msg})

    extra = _with_glossary(
        {"metadata": report.get("metadata")},
        {
            "CVE": "Common Vulnerabilities and Exposures identifier.",
            "GHSA": "GitHub Security Advisory identifier.",
        },
    )
    return _summarize_findings_list(tool="npm-audit", format_name="npm_audit_json", report_path=rp, findings=findings, budgets=budgets, extra=extra)


def summarize_osv_scanner_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    if not isinstance(report, dict):
        report = {}
    results = report.get("results")
    if not isinstance(results, list):
        results = []

    findings: List[Dict[str, Any]] = []
    for res in results:
        if not isinstance(res, dict):
            continue
        source = _norm_path(str(res.get("source") or res.get("source_path") or res.get("path") or ""))
        vulns = res.get("vulnerabilities") or res.get("vulns") or []
        if not isinstance(vulns, list):
            continue
        for v in vulns:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("id") or v.get("ID") or "OSV")
            msg = str(v.get("summary") or v.get("details") or "Dependency vulnerability.")
            findings.append({"severity": "high", "rule_id": vid, "file": source, "line": 0, "message": msg})

    extra_out = _with_glossary(
        None,
        {
            "OSV": "Open Source Vulnerabilities; database and schema used by osv-scanner.",
            "CVE": "Common Vulnerabilities and Exposures identifier.",
            "GHSA": "GitHub Security Advisory identifier.",
        },
    )
    return _summarize_findings_list(tool="osv-scanner", format_name="osv_scanner_json", report_path=rp, findings=findings, budgets=budgets, extra=extra_out)


def summarize_pip_audit_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    items = report if isinstance(report, list) else report.get("dependencies") if isinstance(report, dict) else []
    if not isinstance(items, list):
        items = []

    findings: List[Dict[str, Any]] = []
    for dep in items:
        if not isinstance(dep, dict):
            continue
        name = str(dep.get("name") or dep.get("package") or "")
        vulns = dep.get("vulns") or dep.get("vulnerabilities") or []
        if not isinstance(vulns, list):
            continue
        for v in vulns:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("id") or v.get("cve") or v.get("alias") or "VULN")
            msg = str(v.get("description") or v.get("details") or "Python dependency vulnerability.")
            findings.append({"severity": "high", "rule_id": vid, "file": name, "line": 0, "message": msg})

    extra_out = _with_glossary(
        None,
        {
            "CVE": "Common Vulnerabilities and Exposures identifier.",
            "GHSA": "GitHub Security Advisory identifier.",
        },
    )
    return _summarize_findings_list(tool="pip-audit", format_name="pip_audit_json", report_path=rp, findings=findings, budgets=budgets, extra=extra_out)


def summarize_golangci_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    issues = report.get("Issues") if isinstance(report, dict) else None
    if not isinstance(issues, list):
        issues = []

    findings: List[Dict[str, Any]] = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        pos = it.get("Pos") or {}
        file = _norm_path(str(pos.get("Filename") or "")) if isinstance(pos, dict) else ""
        line = int(pos.get("Line") or 0) if isinstance(pos, dict) else 0
        rid = str(it.get("FromLinter") or "golangci")
        msg = str(it.get("Text") or "Go lint issue.")
        findings.append({"severity": "warning", "rule_id": rid, "file": file, "line": line, "message": msg})

    extra_out = _with_glossary(
        None,
        {
            "FromLinter": "Name of the underlying linter that reported the issue.",
        },
    )
    return _summarize_findings_list(tool="golangci-lint", format_name="golangci_json", report_path=rp, findings=findings, budgets=budgets, extra=extra_out)


def summarize_eslint_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    items = report if isinstance(report, list) else []
    findings: List[Dict[str, Any]] = []
    for fr in items:
        if not isinstance(fr, dict):
            continue
        file = _norm_path(str(fr.get("filePath") or ""))
        msgs = fr.get("messages") or []
        if not isinstance(msgs, list):
            continue
        for m in msgs:
            if not isinstance(m, dict):
                continue
            rid = str(m.get("ruleId") or "eslint")
            line = int(m.get("line") or 0)
            sev_num = int(m.get("severity") or 0)
            sev = "warning" if sev_num == 1 else "error" if sev_num == 2 else "info"
            msg = str(m.get("message") or "ESLint issue.")
            findings.append({"severity": sev, "rule_id": rid, "file": file, "line": line, "message": msg})

    extra_out = _with_glossary(
        None,
        {
            "ruleId": "ESLint rule identifier.",
        },
    )
    return _summarize_findings_list(tool="eslint", format_name="eslint_json", report_path=rp, findings=findings, budgets=budgets, extra=extra_out)


def summarize_ruff_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    items = report if isinstance(report, list) else []
    findings: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        file = _norm_path(str(it.get("filename") or it.get("file") or ""))
        rid = str(it.get("code") or "ruff")
        msg = str(it.get("message") or "Ruff issue.")
        loc = it.get("location") or {}
        line = int((loc.get("row") if isinstance(loc, dict) else 0) or 0)
        sev = str(it.get("severity") or "warning").lower()
        findings.append({"severity": sev, "rule_id": rid, "file": file, "line": line, "message": msg})

    extra_out = _with_glossary(
        None,
        {
            "code": "Ruff rule code (typically maps to underlying linter rule families).",
        },
    )
    return _summarize_findings_list(tool="ruff", format_name="ruff_json", report_path=rp, findings=findings, budgets=budgets, extra=extra_out)


# ============================================================
# Inventory / dependency list parsers
# ============================================================

def summarize_npm_ls_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # npm ls --json: huge tree, summarize counts
    pkgs: Dict[str, int] = {}
    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        deps = node.get("dependencies")
        if not isinstance(deps, dict):
            return
        for name, meta in deps.items():
            pkgs[str(name)] = pkgs.get(str(name), 0) + 1
            walk(meta)
    if isinstance(report, dict):
        walk(report)
    top = _stable_top_k(pkgs, budgets.max_top_files)
    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "ok",
        "tool": "npm-ls",
        "format": "npm_ls_json",
        "report_path": rp,
        "findings_total": 0,
        "by_severity": {},
        "top_rules": [],
        "top_files": top,
        "sample_findings": [],
        "extra": {"dependency_nodes_count": sum(pkgs.values()), "unique_packages_count": len(pkgs)},
    }


def summarize_pip_inspect_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # pip inspect: typically { "installed": [...] } or list
    items = report.get("installed") if isinstance(report, dict) else report if isinstance(report, list) else []
    if not isinstance(items, list):
        items = []
    pkgs = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or it.get("metadata", {}).get("name") or "")
        if name:
            pkgs[name] = pkgs.get(name, 0) + 1
    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "ok",
        "tool": "pip-inspect",
        "format": "pip_inspect_json",
        "report_path": rp,
        "findings_total": 0,
        "by_severity": {},
        "top_rules": [],
        "top_files": _stable_top_k(pkgs, budgets.max_top_files),
        "sample_findings": [],
        "extra": {"unique_packages_count": len(pkgs)},
    }


def summarize_cargo_metadata_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    pkgs = report.get("packages") if isinstance(report, dict) else None
    if not isinstance(pkgs, list):
        pkgs = []
    names = {}
    for p in pkgs:
        if isinstance(p, dict):
            n = str(p.get("name") or "")
            if n:
                names[n] = names.get(n, 0) + 1
    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "ok",
        "tool": "cargo-metadata",
        "format": "cargo_metadata_json",
        "report_path": rp,
        "findings_total": 0,
        "by_severity": {},
        "top_rules": [],
        "top_files": _stable_top_k(names, budgets.max_top_files),
        "sample_findings": [],
        "extra": {"packages_count": len(pkgs), "unique_names_count": len(names)},
    }


def summarize_composer_audit_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # composer audit json differs, fallback to generic extraction
    findings = _generic_json_findings_from_common_fields(report)
    if not findings and isinstance(report, dict):
        advisories = report.get("advisories")
        if isinstance(advisories, dict):
            for pkg, items in advisories.items():
                if not isinstance(items, list):
                    continue
                for a in items:
                    if not isinstance(a, dict):
                        continue
                    sev = str(a.get("severity") or "high").lower()
                    rid = str(a.get("cve") or a.get("id") or a.get("link") or "advisory")
                    msg = str(a.get("title") or a.get("summary") or "Composer advisory")
                    findings.append({"severity": sev, "rule_id": rid, "file": str(pkg), "line": 0, "message": msg})
    return _summarize_findings_list(tool="composer-audit", format_name="composer_audit_json", report_path=rp, findings=findings, budgets=budgets)


def summarize_bundler_audit_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # schema varies, generic fallback
    findings = _generic_json_findings_from_common_fields(report)
    return _summarize_findings_list(tool="bundler-audit", format_name="bundler_audit_json", report_path=rp, findings=findings, budgets=budgets)


def summarize_composer_show_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # list of packages
    pkgs = report.get("installed") if isinstance(report, dict) else None
    if not isinstance(pkgs, list):
        pkgs = report if isinstance(report, list) else []
    names = {}
    for p in pkgs:
        if isinstance(p, dict):
            n = str(p.get("name") or "")
            if n:
                names[n] = names.get(n, 0) + 1
    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "ok",
        "tool": "composer-show",
        "format": "composer_show_json",
        "report_path": rp,
        "findings_total": 0,
        "by_severity": {},
        "top_rules": [],
        "top_files": _stable_top_k(names, budgets.max_top_files),
        "sample_findings": [],
        "extra": {"unique_packages_count": len(names)},
    }


# ============================================================
# Text outputs
# ============================================================

def summarize_flake8_text(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    rx = re.compile(r"^(.*?):(\d+):(\d+):\s*([A-Za-z]\d+)\s+(.*)$")
    findings: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        findings.append({
            "severity": "warning",
            "rule_id": m.group(4),
            "file": _norm_path(m.group(1)),
            "line": int(m.group(2)),
            "message": m.group(5),
        })
    extra_out = _with_glossary(
        None,
        {
            "E###": "Flake8/pycodestyle error code family (style/formatting).",
            "W###": "Flake8/pycodestyle warning code family.",
            "F###": "Flake8/pyflakes code family (likely-bug / unused import / undefined name).",
        },
    )
    return _summarize_findings_list(
        tool="flake8",
        format_name="flake8_text",
        report_path=rp,
        findings=findings,
        budgets=budgets,
        extra=extra_out,
    )


def summarize_gocyclo_text(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # gocyclo lines typically contain complexity and function, we treat as "complexity_hotspot"
    findings = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        findings.append({"severity": "info", "rule_id": "cyclomatic", "file": "", "line": 0, "message": line})
    return _summarize_findings_list(tool="gocyclo", format_name="gocyclo_text", report_path=rp, findings=findings, budgets=budgets)


def summarize_dupl_plumbing_text(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # dupl plumbing is noisy, summarize by counting blocks and listing a few headers
    lines = [l for l in txt.splitlines() if l.strip()]
    findings = [{"severity": "info", "rule_id": "dup_block", "file": "", "line": 0, "message": l} for l in lines[: budgets.max_findings]]
    extra = {"lines_total": len(lines)}
    return _summarize_findings_list(tool="dupl", format_name="dupl_plumbing_text", report_path=rp, findings=findings, budgets=budgets, extra=extra)


def summarize_dart_analyze_machine(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # machine output: file|line|col|severity|message|code
    findings: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        file = _norm_path(parts[0])
        ln = int(parts[1] or 0)
        sev = str(parts[3] or "info").lower()
        msg = "|".join(parts[4:]).strip()
        findings.append({"severity": sev, "rule_id": "dart-analyze", "file": file, "line": ln, "message": msg})
    return _summarize_findings_list(tool="dart-analyze", format_name="dart_analyze_machine", report_path=rp, findings=findings, budgets=budgets)


def summarize_clang_tidy_text(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # clang-tidy output is plain text, parse "file:line:col: warning: message [check]"
    rx = re.compile(r"^(.*?):(\d+):(\d+):\s*(warning|error|note):\s*(.*?)(\s*\[(.*?)\])?$", re.IGNORECASE)
    findings: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        file = _norm_path(m.group(1))
        ln = int(m.group(2))
        sev = m.group(4).lower()
        msg = m.group(5)
        rid = m.group(7) or "clang-tidy"
        findings.append({"severity": sev, "rule_id": rid, "file": file, "line": ln, "message": msg})
    if not findings:
        # fallback: keep a few lines
        findings = [{"severity": "info", "rule_id": "clang-tidy", "file": "", "line": 0, "message": l} for l in txt.splitlines()[: budgets.max_findings]]
    return _summarize_findings_list(tool="clang-tidy", format_name="clang_tidy_text", report_path=rp, findings=findings, budgets=budgets)


def summarize_stdout_only(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # For mvn test / gradle test etc. We only provide minimal signals.
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    findings = [{"severity": "info", "rule_id": "stdout", "file": "", "line": 0, "message": l} for l in lines[: budgets.max_findings]]
    extra = {"lines_total": len(lines)}
    return _summarize_findings_list(tool="stdout", format_name="stdout_only", report_path=rp, findings=findings, budgets=budgets, extra=extra)


def summarize_generic_json(report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    findings = _generic_json_findings_from_common_fields(report)
    return _summarize_findings_list(tool="generic", format_name="generic_json", report_path=rp, findings=findings, budgets=budgets)


# ============================================================
# XML outputs
# ============================================================

def summarize_lizard_cppncss_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    """Summarize lizard's cppncss-style XML output produced by `lizard -X/--xml`.

    Shape (observed):
      <cppncss>
        <measure type="Function">
          <labels>Nr/NCSS/CCN</labels>
          <item name="func(...) at .\\path.py:123"><value>..</value><value>NCSS</value><value>CCN</value></item>
          <average label="NCSS" value="..."/>
          <average label="CCN" value="..."/>
        </measure>
      </cppncss>
    """
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("lizard", "lizard_cppncss_xml", rp, "invalid_xml")

    measure: Optional[ET.Element] = None
    for m in root.findall(".//measure"):
        if (m.attrib.get("type") or "").strip().lower() == "function":
            measure = m
            break
    if measure is None:
        return _unavailable("lizard", "lizard_cppncss_xml", rp, "missing_function_measure")

    avg_ccn: Optional[float] = None
    avg_ncss: Optional[float] = None
    for avg in measure.findall("average"):
        label = (avg.attrib.get("label") or "").strip().lower()
        raw_val = avg.attrib.get("value")
        try:
            val = float(raw_val) if raw_val is not None else None
        except Exception:
            val = None
        if label == "ccn":
            avg_ccn = val
        elif label == "ncss":
            avg_ncss = val

    rx_name = re.compile(r"^(?P<func>.+?)\s+at\s+(?P<path>.*?):(?P<line>\d+)\s*$")
    rows: List[Dict[str, Any]] = []
    for item in measure.findall("item"):
        name = item.attrib.get("name") or ""
        values = [v.text.strip() if v.text else "" for v in item.findall("value")]
        if len(values) < 3:
            continue

        try:
            ncss = int(float(values[1]))
        except Exception:
            ncss = 0
        try:
            ccn = int(float(values[2]))
        except Exception:
            ccn = 0

        func_name = name
        file_path = ""
        line_no = 0
        m = rx_name.match(name)
        if m:
            func_name = m.group("func").strip()
            file_path = _norm_path(m.group("path").strip())
            try:
                line_no = int(m.group("line"))
            except Exception:
                line_no = 0

        rows.append({"func": func_name, "file": file_path, "line": line_no, "ccn": ccn, "ncss": ncss})

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -int(r.get("ccn") or 0),
            -int(r.get("ncss") or 0),
            str(r.get("file") or ""),
            int(r.get("line") or 0),
        ),
    )
    findings: List[Dict[str, Any]] = []
    for r in rows_sorted[: budgets.max_findings]:
        findings.append({
            "severity": "info",
            "rule_id": "complexity_hotspot",
            "file": str(r.get("file") or ""),
            "line": int(r.get("line") or 0),
            "message": f"CCN={int(r.get('ccn') or 0)} NCSS={int(r.get('ncss') or 0)} {str(r.get('func') or '')}".strip(),
        })

    extra: Dict[str, Any] = {
        "functions_total": len(rows),
        "glossary": {
            "CCN": "Cyclomatic Complexity Number (McCabe). Roughly counts decision points + 1 per function.",
            "NCSS": "Non-Commenting Source Statements. Logical statement count excluding comments/blank lines.",
        },
    }
    if avg_ccn is not None:
        extra["avg_ccn"] = round(float(avg_ccn), 3)
    if avg_ncss is not None:
        extra["avg_ncss"] = round(float(avg_ncss), 3)

    return _summarize_findings_list(
        tool="lizard",
        format_name="lizard_cppncss_xml",
        report_path=rp,
        findings=findings,
        budgets=budgets,
        extra=extra,
    )

def summarize_junit_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("junit", "junit_xml", rp, "invalid_xml")

    total = failures = errors = skipped = 0
    failed_samples: List[Dict[str, Any]] = []

    suites: List[ET.Element] = []
    if root.tag.lower() == "testsuites":
        suites = list(root.findall("testsuite"))
    elif root.tag.lower() == "testsuite":
        suites = [root]

    for ts in suites:
        total += int(ts.attrib.get("tests", 0) or 0)
        failures += int(ts.attrib.get("failures", 0) or 0)
        errors += int(ts.attrib.get("errors", 0) or 0)
        skipped += int(ts.attrib.get("skipped", 0) or 0)

        for tc in ts.findall("testcase"):
            f = tc.find("failure")
            e = tc.find("error")
            if f is None and e is None:
                continue
            name = tc.attrib.get("name", "")
            classname = tc.attrib.get("classname", "")
            node = e if e is not None else f
            msg = (node.attrib.get("message", "") if node is not None else "") or (node.text or "" if node is not None else "")
            failed_samples.append({
                "severity": "error",
                "rule_id": "test_failure",
                "file": classname,
                "line": 0,
                "message": f"{classname}::{name} - {msg}".strip(),
            })

    by_sev: Dict[str, int] = {}
    if failures + errors:
        by_sev["error"] = failures + errors
    if skipped:
        by_sev["info"] = skipped

    extra_out = _with_glossary(
        {"tests": total, "failures": failures, "errors": errors, "skipped": skipped},
        {
            "tests": "Total number of executed tests.",
            "failures": "Assertion/test failures (test ran but failed).",
            "errors": "Unexpected errors during test execution (crashes/exceptions).",
            "skipped": "Tests that were intentionally skipped.",
        },
    )

    return _finalize_summary(
        tool="junit",
        format_name="junit_xml",
        report_path=rp,
        findings=failed_samples,
        by_severity=by_sev,
        top_rules={"test_failure": failures + errors} if (failures + errors) else {},
        top_files={str(f["file"]): 1 for f in failed_samples if f.get("file")},
        extra=extra_out,
        budgets=budgets,
    )


def summarize_trx_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # dotnet TRX: summarize outcomes
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("trx", "trx_xml", rp, "invalid_xml")

    # TRX uses namespaces often, handle by stripping
    def tag_endswith(el: ET.Element, suffix: str) -> bool:
        return el.tag.lower().endswith(suffix.lower())

    results = []
    for el in root.iter():
        if tag_endswith(el, "UnitTestResult"):
            results.append(el)

    total = len(results)
    failed = 0
    samples: List[Dict[str, Any]] = []
    for r in results:
        outcome = (r.attrib.get("outcome") or "").lower()
        test_name = r.attrib.get("testName") or ""
        if outcome == "failed":
            failed += 1
            samples.append({"severity": "error", "rule_id": "test_failure", "file": "", "line": 0, "message": test_name})

    by_sev = {"error": failed} if failed else {}

    extra_out = _with_glossary(
        {"tests": total, "failed": failed},
        {
            "TRX": "Visual Studio / dotnet test results XML format.",
        },
    )
    return _finalize_summary(
        tool="dotnet-test",
        format_name="trx_xml",
        report_path=rp,
        findings=samples,
        by_severity=by_sev,
        top_rules={"test_failure": failed} if failed else {},
        top_files={},
        extra=extra_out,
        budgets=budgets,
    )


def summarize_checkstyle_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("checkstyle", "checkstyle_xml", rp, "invalid_xml")

    findings: List[Dict[str, Any]] = []
    for file_el in root.findall(".//file"):
        fname = _norm_path(file_el.attrib.get("name", ""))
        for err in file_el.findall(".//error"):
            sev = (err.attrib.get("severity") or "warning").lower()
            ln = int(err.attrib.get("line") or 0)
            msg = err.attrib.get("message") or "Checkstyle issue."
            src = err.attrib.get("source") or "checkstyle"
            findings.append({"severity": sev, "rule_id": src, "file": fname, "line": ln, "message": msg})

    extra_out = _with_glossary(
        None,
        {
            "Checkstyle": "XML format commonly used for static analysis findings (files/errors).",
        },
    )
    return _summarize_findings_list(
        tool="checkstyle",
        format_name="checkstyle_xml",
        report_path=rp,
        findings=findings,
        budgets=budgets,
        extra=extra_out,
    )


def summarize_cppcheck_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("cppcheck", "cppcheck_xml", rp, "invalid_xml")

    findings: List[Dict[str, Any]] = []
    for err in root.findall(".//error"):
        sev = (err.attrib.get("severity") or "warning").lower()
        rid = err.attrib.get("id") or "cppcheck"
        msg = err.attrib.get("msg") or err.attrib.get("verbose") or "Cppcheck issue."
        loc = err.find("location")
        file = _norm_path(loc.attrib.get("file", "")) if loc is not None else ""
        ln = int(loc.attrib.get("line", 0)) if loc is not None else 0
        findings.append({"severity": sev, "rule_id": rid, "file": file, "line": ln, "message": msg})

    return _summarize_findings_list(tool="cppcheck", format_name="cppcheck_xml", report_path=rp, findings=findings, budgets=budgets)


def summarize_scalastyle_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("scalastyle", "scalastyle_xml", rp, "invalid_xml")

    findings: List[Dict[str, Any]] = []
    for msg in root.findall(".//message"):
        sev = (msg.attrib.get("severity") or "warning").lower()
        file = _norm_path(msg.attrib.get("file", ""))
        ln = int(msg.attrib.get("line", 0) or 0)
        rid = msg.attrib.get("key") or "scalastyle"
        text = msg.attrib.get("message") or (msg.text or "Scalastyle issue.")
        findings.append({"severity": sev, "rule_id": rid, "file": file, "line": ln, "message": text})

    return _summarize_findings_list(tool="scalastyle", format_name="scalastyle_xml", report_path=rp, findings=findings, budgets=budgets)


def summarize_phpcpd_pmd_xml(txt: str, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # PMD-like XML: duplicated blocks
    try:
        root = ET.fromstring(txt)
    except Exception:
        return _unavailable("phpcpd", "phpcpd_pmd_xml", rp, "invalid_xml")

    findings: List[Dict[str, Any]] = []
    for dup in root.findall(".//duplication"):
        lines = dup.attrib.get("lines") or ""
        for f in dup.findall(".//file"):
            file = _norm_path(f.attrib.get("path", "") or f.attrib.get("name", ""))
            ln = int(f.attrib.get("line", 0) or 0)
            findings.append({"severity": "info", "rule_id": "duplication", "file": file, "line": ln, "message": f"duplicated block lines={lines}"})

    return _summarize_findings_list(tool="phpcpd", format_name="phpcpd_pmd_xml", report_path=rp, findings=findings, budgets=budgets)


# ============================================================
# Many remaining JSON formats -> generic extraction (but stable)
# ============================================================

def summarize_generic_json_like(tool: str, format_name: str, report: Any, rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    findings = _generic_json_findings_from_common_fields(report)
    # If it looks like an inventory instead (list of packages), summarize packages
    if not findings and isinstance(report, list) and report and isinstance(report[0], dict):
        # try "name" field
        names = {}
        for it in report[: budgets.max_items_inventory]:
            n = str(it.get("name") or it.get("id") or "")
            if n:
                names[n] = names.get(n, 0) + 1
        if names:
            return {
                "schema": "ToolEvidenceSummaryV1",
                "status": "ok",
                "tool": tool,
                "format": format_name,
                "report_path": rp,
                "findings_total": 0,
                "by_severity": {},
                "top_rules": [],
                "top_files": _stable_top_k(names, budgets.max_top_files),
                "sample_findings": [],
                "extra": {"unique_items_count": len(names)},
            }
    return _summarize_findings_list(tool=tool, format_name=format_name, report_path=rp, findings=findings, budgets=budgets)


# ============================================================
# JSON stream formats
# ============================================================

def summarize_go_test_json_stream(items: List[Any], rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # go test -json: events with Action: pass/fail, Package/Test
    fails = []
    total_fail = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        if str(it.get("Action") or "").lower() == "fail":
            total_fail += 1
            pkg = str(it.get("Package") or "")
            test = str(it.get("Test") or "")
            out = str(it.get("Output") or "").strip()
            msg = f"{pkg}::{test} {out}".strip()
            fails.append({"severity": "error", "rule_id": "test_fail", "file": pkg, "line": 0, "message": msg})
    by_sev = {"error": total_fail} if total_fail else {}
    extra_out = _with_glossary(
        None,
        {
            "Action": "Go test event type (e.g., run, pass, fail, output).",
            "Package": "Go package import path under test.",
            "Test": "Test name (if present) for the event.",
        },
    )
    return _finalize_summary(
        tool="go-test",
        format_name="go_test_json_stream",
        report_path=rp,
        findings=fails,
        by_severity=by_sev,
        top_rules={"test_fail": total_fail} if total_fail else {},
        top_files={f.get("file",""): 1 for f in fails if f.get("file")},
        extra=extra_out,
        budgets=budgets,
    )


def summarize_govulncheck_json_stream(items: List[Any], rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # govulncheck -json emits events; we count vulnerability findings heuristically
    findings = []
    for it in items:
        if not isinstance(it, dict):
            continue
        # Heuristic: any event with "Vuln" / "OSV" keys
        if "Vuln" in it or "vuln" in it or "OSV" in it or "osv" in it:
            vid = str(it.get("Vuln") or it.get("vuln") or it.get("OSV") or it.get("osv") or "govuln")
            findings.append({"severity": "high", "rule_id": vid, "file": "", "line": 0, "message": "Go vulnerability reported."})
    return _summarize_findings_list(tool="govulncheck", format_name="govulncheck_json_stream", report_path=rp, findings=findings, budgets=budgets)


def summarize_go_list_modules_json_stream(items: List[Any], rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    names = {}
    for it in items:
        if isinstance(it, dict):
            p = str(it.get("Path") or "")
            if p:
                names[p] = names.get(p, 0) + 1
    return {
        "schema": "ToolEvidenceSummaryV1",
        "status": "ok",
        "tool": "go-list-modules",
        "format": "go_list_modules_json_stream",
        "report_path": rp,
        "findings_total": 0,
        "by_severity": {},
        "top_rules": [],
        "top_files": _stable_top_k(names, budgets.max_top_files),
        "sample_findings": [],
        "extra": {"unique_modules_count": len(names)},
    }


def summarize_cargo_messages_json(items: List[Any], rp: str, budgets: SummarizeBudgets) -> Dict[str, Any]:
    # cargo clippy/test --message-format=json outputs compiler messages
    findings = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("reason") != "compiler-message":
            continue
        msg = it.get("message") or {}
        if not isinstance(msg, dict):
            continue
        level = str(msg.get("level") or "warning").lower()
        code = msg.get("code") or {}
        rid = str(code.get("code") or "rustc") if isinstance(code, dict) else "rustc"
        message = str(msg.get("message") or "Rust compiler message.")
        spans = msg.get("spans") or []
        file = ""
        line = 0
        if isinstance(spans, list) and spans:
            sp0 = spans[0]
            if isinstance(sp0, dict):
                file = _norm_path(str(sp0.get("file_name") or ""))
                line = int(sp0.get("line_start") or 0)
        sev = "error" if level == "error" else "warning"
        findings.append({"severity": sev, "rule_id": rid, "file": file, "line": line, "message": message})
    return _summarize_findings_list(tool="cargo", format_name="cargo_messages_json", report_path=rp, findings=findings, budgets=budgets)


# ============================================================
# Dispatcher for ALL formats in registry
# ============================================================

# Each entry: format_name -> (kind, parser_fn)
# kind in {"json", "text", "xml_text", "json_lines"}
FORMAT_DISPATCH: Dict[str, Tuple[str, Callable[..., Dict[str, Any]]]] = {
    # core
    "gitleaks_json": ("json", summarize_gitleaks_json),
    "semgrep_json": ("json", summarize_semgrep_json),
    "osv_scanner_json": ("json", summarize_osv_scanner_json),
    "pip_audit_json": ("json", summarize_pip_audit_json),
    "npm_audit_json": ("json", summarize_npm_audit_json),
    "golangci_json": ("json", summarize_golangci_json),
    "eslint_json": ("json", summarize_eslint_json),
    "ruff_json": ("json", summarize_ruff_json),
    "flake8_text": ("text", summarize_flake8_text),
    "junit_xml": ("xml_text", summarize_junit_xml),

    # inventory
    "pip_inspect_json": ("json", lambda obj, rp, b: summarize_pip_inspect_json(obj, rp, b)),
    "npm_ls_json": ("json", lambda obj, rp, b: summarize_npm_ls_json(obj, rp, b)),
    "cargo_metadata_json": ("json", lambda obj, rp, b: summarize_cargo_metadata_json(obj, rp, b)),
    "composer_show_json": ("json", lambda obj, rp, b: summarize_composer_show_json(obj, rp, b)),

    # extra dependency scanners
    "composer_audit_json": ("json", lambda obj, rp, b: summarize_composer_audit_json(obj, rp, b)),
    "bundler_audit_json": ("json", lambda obj, rp, b: summarize_bundler_audit_json(obj, rp, b)),

    # JSON stream formats
    "go_test_json_stream": ("json_lines", lambda items, rp, b: summarize_go_test_json_stream(items, rp, b)),
    "govulncheck_json_stream": ("json_lines", lambda items, rp, b: summarize_govulncheck_json_stream(items, rp, b)),
    "go_list_modules_json_stream": ("json_lines", lambda items, rp, b: summarize_go_list_modules_json_stream(items, rp, b)),
    "cargo_messages_json": ("json_lines", lambda items, rp, b: summarize_cargo_messages_json(items, rp, b)),

    # XML formats
    "trx_xml": ("xml_text", summarize_trx_xml),
    "checkstyle_xml": ("xml_text", summarize_checkstyle_xml),
    "cppcheck_xml": ("xml_text", summarize_cppcheck_xml),
    "scalastyle_xml": ("xml_text", summarize_scalastyle_xml),
    "phpcpd_pmd_xml": ("xml_text", summarize_phpcpd_pmd_xml),
    "lizard_cppncss_xml": ("xml_text", summarize_lizard_cppncss_xml),

    # text formats
    "gocyclo_text": ("text", summarize_gocyclo_text),
    "dupl_plumbing_text": ("text", summarize_dupl_plumbing_text),
    "dart_analyze_machine": ("text", summarize_dart_analyze_machine),
    "clang_tidy_text": ("text", summarize_clang_tidy_text),
    "stdout_only": ("text", summarize_stdout_only),

    # generic fallbacks
    "generic_json": ("json", lambda obj, rp, b: summarize_generic_json(obj, rp, b)),

    # remaining JSON formats from registry -> generic_json_like
    "hadolint_json": ("json", lambda obj, rp, b: summarize_generic_json_like("hadolint", "hadolint_json", obj, rp, b)),
    "kube_linter_json": ("json", lambda obj, rp, b: summarize_generic_json_like("kube-linter", "kube_linter_json", obj, rp, b)),
    "kube_score_json": ("json", lambda obj, rp, b: summarize_generic_json_like("kube-score", "kube_score_json", obj, rp, b)),
    "trivy_config_json": ("json", lambda obj, rp, b: summarize_generic_json_like("trivy", "trivy_config_json", obj, rp, b)),
    "tfsec_json": ("json", lambda obj, rp, b: summarize_generic_json_like("tfsec", "tfsec_json", obj, rp, b)),
    "lizard_json": ("json", lambda obj, rp, b: summarize_generic_json_like("lizard", "lizard_json", obj, rp, b)),
    "jscpd_json": ("json", lambda obj, rp, b: summarize_generic_json_like("jscpd", "jscpd_json", obj, rp, b)),
    "phpstan_json": ("json", lambda obj, rp, b: summarize_generic_json_like("phpstan", "phpstan_json", obj, rp, b)),
    "phpcs_json": ("json", lambda obj, rp, b: summarize_generic_json_like("phpcs", "phpcs_json", obj, rp, b)),
    "psalm_json": ("json", lambda obj, rp, b: summarize_generic_json_like("psalm", "psalm_json", obj, rp, b)),
    "rubocop_json": ("json", lambda obj, rp, b: summarize_generic_json_like("rubocop", "rubocop_json", obj, rp, b)),
    "ktlint_json": ("json", lambda obj, rp, b: summarize_generic_json_like("ktlint", "ktlint_json", obj, rp, b)),
    "detekt_json": ("json", lambda obj, rp, b: summarize_generic_json_like("detekt", "detekt_json", obj, rp, b)),
    "swiftlint_json": ("json", lambda obj, rp, b: summarize_generic_json_like("swiftlint", "swiftlint_json", obj, rp, b)),
    "dotnet_format_json": ("json", lambda obj, rp, b: summarize_generic_json_like("dotnet-format", "dotnet_format_json", obj, rp, b)),
    "jest_json": ("json", lambda obj, rp, b: summarize_generic_json_like("jest", "jest_json", obj, rp, b)),
    "rspec_json": ("json", lambda obj, rp, b: summarize_generic_json_like("rspec", "rspec_json", obj, rp, b)),
}


# ============================================================
# Main entrypoints
# ============================================================

def summarize_tool_report(
    tool: str,
    format_name: str,
    report_file: Path,
    budgets: Optional[SummarizeBudgets] = None,
) -> Dict[str, Any]:
    budgets = budgets or SummarizeBudgets()
    rp = str(report_file).replace("\\", "/")

    dispatch = FORMAT_DISPATCH.get(format_name)
    if dispatch is None:
        return _unavailable(tool, format_name, rp, f"no_parser_for_format:{format_name}")

    kind, fn = dispatch

    if kind == "json":
        obj = _safe_read_json(report_file, max_bytes=budgets.max_bytes_json)
        if obj is None:
            return _unavailable(tool, format_name, rp, "missing_or_invalid_json")
        return fn(obj, rp, budgets)  # type: ignore[misc]

    if kind == "json_lines":
        items = _safe_read_json_lines(report_file, max_bytes=budgets.max_bytes_json)
        return fn(items, rp, budgets)  # type: ignore[misc]

    if kind == "text":
        txt = _safe_read_text(report_file, max_bytes=budgets.max_bytes_text)
        if not txt:
            return _unavailable(tool, format_name, rp, "missing_or_empty_text")
        return fn(txt, rp, budgets)  # type: ignore[misc]

    if kind == "xml_text":
        txt = _safe_read_text(report_file, max_bytes=budgets.max_bytes_text)
        if not txt:
            return _unavailable(tool, format_name, rp, "missing_or_empty_xml")
        return fn(txt, rp, budgets)  # type: ignore[misc]

    return _unavailable(tool, format_name, rp, f"unsupported_dispatch_kind:{kind}")


def summarize_dynamic_tool_execution(
    repo_root: Path,
    tool_execution_meta: Dict[str, Any],
    parsing_format: str,
    declared_outputs: List[str],
    budgets: Optional[SummarizeBudgets] = None,
) -> Dict[str, Any]:
    budgets = budgets or SummarizeBudgets()

    tool_name = str(tool_execution_meta.get("tool") or "")
    status = str(tool_execution_meta.get("status") or "")
    cmd = str(tool_execution_meta.get("cmd") or "")
    exit_code = tool_execution_meta.get("exit_code")

    # Resolve report file deterministically: first declared output that exists.
    report_path: Optional[Path] = None
    for rel in declared_outputs or []:
        p = repo_root / rel
        if p.exists():
            report_path = p
            break

    if status != "completed":
        summary = _unavailable(tool_name, parsing_format, str(report_path or ""), f"tool_status:{status}")
    else:
        if report_path is None:
            summary = _unavailable(tool_name, parsing_format, "", "declared_output_missing")
        else:
            summary = summarize_tool_report(
                tool=tool_name,
                format_name=parsing_format,
                report_file=report_path,
                budgets=budgets,
            )

    return {
        "tool_run": {
            "tool": tool_name,
            "status": status,
            "exit_code": exit_code,
            "cmd": cmd,
            "declared_outputs": declared_outputs,
        },
        "summary": summary,
    }


# ============================================================
# Pipeline integration shim (matches `main.py` expectations)
# ============================================================


def _env_int(name: str, default: int) -> int:
    try:
        v = (os.getenv(name) or "").strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _guess_format_for_file(path: Path) -> str:
    # Conservative fallback.
    s = path.suffix.lower()
    if s == ".json":
        return "generic_json"
    if s == ".xml":
        return "checkstyle_xml" if "checkstyle" in path.name.lower() else "junit_xml" if "junit" in path.name.lower() else "trx_xml" if "trx" in path.name.lower() else "generic_json"
    return "stdout_only"


def summarize_tool_result(tool_result: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Attach deterministic structured summaries to a completed tool run.

    This is a glue layer for the pipeline:
    - `run_dynamic_tool()` moves declared outputs into a tool work dir and records
      their relative paths in `tool_result['artifacts']`.
    - This wrapper resolves those moved artifacts and summarizes them using the
      registry's `parsing.format` when available.

    Raw embedding behavior:
    - If an artifact is small, it's safe to embed full content.
    - Controlled by `SAGES_TOOL_SUMMARY_EMBED_RAW_UNDER_BYTES`.
    """

    if not isinstance(tool_result, dict):
        return tool_result

    # We attach summaries for both successful and failed tool runs.
    # Failed runs often contain the most important context in stderr.
    status = str(tool_result.get("status") or "")
    if status not in {"completed", "tool_failed"}:
        return tool_result

    budgets = SummarizeBudgets()
    tool_name = str(tool_result.get("tool") or "")
    parsing = tool_result.get("parsing") if isinstance(tool_result.get("parsing"), dict) else {}
    parsing_format = str(parsing.get("format") or "").strip()

    declared_outputs = tool_result.get("declared_outputs")
    declared_outputs = list(declared_outputs) if isinstance(declared_outputs, list) else []

    artifacts = tool_result.get("artifacts") if isinstance(tool_result.get("artifacts"), dict) else {}
    # Deterministic ordering when we need to pick a default.
    artifact_items = sorted([(str(k), v) for k, v in artifacts.items() if isinstance(v, str) and v], key=lambda kv: kv[0])

    # LLM prompt hygiene: only embed raw tool output when it is genuinely small.
    # This is still overrideable via env var.
    embed_raw_under = _env_int("SAGES_TOOL_SUMMARY_EMBED_RAW_UNDER_BYTES", 10_000)

    reports: Dict[str, Any] = {}

    def _stderr_hint() -> str:
        sp = tool_result.get("stderr_path")
        if not isinstance(sp, str) or not sp:
            return ""
        p = (work_dir / sp).resolve()
        if not p.exists() or not p.is_file():
            return ""
        txt = _safe_read_text(p, max_bytes=24_000)
        if not txt:
            return ""
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            return ""
        # Drop very noisy, low-signal prefixes.
        filtered: List[str] = []
        for ln in lines:
            if ln.lower().startswith("deprecation notice"):
                continue
            if ln.lower().startswith("deprecated"):
                continue
            filtered.append(ln)
        if not filtered:
            filtered = lines
        return _clamp_str(filtered[-1], 220)

    def _resolve_report_path(report_rel: str, output_key: str) -> Optional[Path]:
        """Resolve a report path from a tool_result artifact reference.

        In some cached runs, the runner records artifacts under a synthetic
        `<run>.artifacts/` directory but only the bare output (e.g. `semgrep.json`)
        is kept under the `_cache/` folder. This tries a few deterministic fallbacks.
        """

        candidates: List[Path] = []
        rel = Path(report_rel)

        # 1) The recorded artifact path (normal case).
        candidates.append((work_dir / rel).resolve())

        # 2) The declared output key as a file in work_dir.
        if output_key:
            candidates.append((work_dir / output_key).resolve())

        # 3) If the recorded path is under a *.artifacts directory, try the parent folder.
        try:
            if rel.parent.name.endswith(".artifacts"):
                candidates.append((work_dir / rel.parent.parent / rel.name).resolve())
        except Exception:
            pass

        # 4) As a last resort, try work_dir / basename.
        candidates.append((work_dir / rel.name).resolve())

        # 5) Common cache layout: work_dir/tool_outputs/_cache/<basename>
        # (useful when declared_outputs are basenames and artifacts map is empty).
        candidates.append((work_dir / "tool_outputs" / "_cache" / rel.name).resolve())

        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    return c
            except Exception:
                continue
        return None

    def _add_report(key: str, report_rel: str, fmt: str) -> None:
        report_path = _resolve_report_path(report_rel, key)
        if report_path is None:
            missing_path = str((work_dir / report_rel).resolve()).replace("\\", "/")
            s = _unavailable(tool_name, fmt or "unknown", missing_path, "artifact_missing")
            hint = _stderr_hint()
            if hint:
                s["reason"] = _clamp_str(f"{s.get('reason')}; stderr_hint:{hint}", 380)
            reports[key] = s
            return
        # Summarize
        chosen_fmt = fmt or _guess_format_for_file(report_path)
        s = summarize_tool_report(tool=tool_name, format_name=chosen_fmt, report_file=report_path, budgets=budgets)

        if isinstance(s, dict) and str(s.get("status")) == "unavailable":
            hint = _stderr_hint()
            if hint and isinstance(s.get("reason"), str):
                s["reason"] = _clamp_str(f"{s.get('reason')}; stderr_hint:{hint}", 380)
        # Optionally embed raw for small artifacts
        try:
            size = report_path.stat().st_size
            if embed_raw_under and size <= embed_raw_under:
                s["raw_included"] = True
                s["raw_text"] = _safe_read_text(report_path, max_bytes=embed_raw_under)
            else:
                s["raw_included"] = False
        except Exception:
            pass
        reports[key] = s

    # Prefer summarizing declared outputs in order.
    for out_rel in declared_outputs:
        if not isinstance(out_rel, str) or not out_rel:
            continue
        # In tool_result['artifacts'], keys are the declared output strings.
        moved_rel = artifacts.get(out_rel)
        if isinstance(moved_rel, str) and moved_rel:
            _add_report(out_rel, moved_rel, parsing_format)
        else:
            # Even when artifact tracking is missing, attempt to summarize the output
            # by its declared name (or emit an explicit unavailable entry).
            _add_report(out_rel, out_rel, parsing_format)

    # Fallback: if no declared outputs matched, summarize whatever artifact exists.
    if not reports and artifact_items:
        k0, rel0 = artifact_items[0]
        _add_report(k0, rel0, parsing_format)

    # Preserve failure context: include stderr/stdout summaries when the tool failed
    # OR when we produced any unavailable report (often due to missing artifacts).
    include_streams = status == "tool_failed" or any(
        isinstance(v, dict) and str(v.get("status")) == "unavailable" for v in reports.values()
    )
    if include_streams or not reports:
        sp = tool_result.get("stderr_path")
        if isinstance(sp, str) and sp and "stderr" not in reports:
            _add_report("stderr", sp, "stdout_only")
        op = tool_result.get("stdout_path")
        if isinstance(op, str) and op and "stdout" not in reports:
            _add_report("stdout", op, "stdout_only")

    tool_result["tool_summary"] = {
        "schema": "ToolExecutionSummaryV1",
        "tool": tool_name,
        "format": parsing_format or None,
        "reports": reports,
        "notes": {
            "status": status,
            "declared_outputs": declared_outputs,
            "artifacts_count": len(artifact_items),
            "raw_embed_under_bytes": embed_raw_under,
        },
    }

    return tool_result
