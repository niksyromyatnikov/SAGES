from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aggregation import (
	AggregationPolicy,
	GroupAggregationResult,
	GroupDefinition,
	LeafDefinition,
	LeafScore,
	ModuleAggregationResult,
	aggregate_final,
	aggregate_group,
	aggregate_module,
	build_report_payload,
)
from node_prompt import ActivationPromptInputs, build_node_activation_prompt
from scoring_prompt import ScoringPromptInputs, build_compact_evidence_bundle, build_group_scoring_prompt
from tool_registry import get_tool_registry

from tool_summarizer import summarize_tool_result

from llm_client import (
	LLMError,
)

from llm_runtime import LLMRuntime, load_llm_config, resolve_default_provider

from sast_ruleset import SAST_FOCUS_MAP
from repo_signals import detect_languages, infer_repo_signals


# -----------------------------------------------------------------------------
# YAML loading (PyYAML)
# -----------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as e:  # pragma: no cover
		raise RuntimeError(
			"PyYAML is required to run this pipeline. Install with: pip install pyyaml"
		) from e

	with path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Expected YAML mapping at root: {path}")
	return data


def _yaml_dump(obj: Any) -> str:
	import yaml  # type: ignore

	return yaml.safe_dump(obj, sort_keys=False)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _read_text(path: Path, max_bytes: int = 512_000) -> str:
	if not path.exists():
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


def _stable_json(obj: Any) -> str:
	return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clamp01(x: float) -> float:
	return max(0.0, min(1.0, x))


def _to_int0(x: Any) -> int:
	try:
		if x is None:
			return 0
		return int(x)
	except Exception:
		return 0


def _to_float_or_none(x: Any) -> Optional[float]:
	try:
		if x is None or isinstance(x, bool):
			return None
		return float(x)
	except Exception:
		return None


def _coerce_string_list(value: Any, max_items: int = 50) -> List[str]:
	if not isinstance(value, list):
		return []
	out: List[str] = []
	for item in value:
		if len(out) >= max_items:
			break
		if isinstance(item, (str, int, float, bool)):
			s = str(item).strip()
			if s:
				out.append(s)
	return out


class _TokenUsageTracker:
	def __init__(self, out_path: Path):
		self.out_path = out_path
		self.data: Dict[str, Any] = {
			"stages": {},
			"total": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
			"calls": [],
		}

	def add(self, *, stage: str, usage: Optional[Dict[str, Any]]) -> None:
		u = dict(usage or {})
		stage_key = (stage or "unknown").strip().lower() or "unknown"
		stages = self.data.setdefault("stages", {})
		if stage_key not in stages:
			stages[stage_key] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
		stage_sum = stages[stage_key]
		total_sum = self.data["total"]

		inp = _to_int0(u.get("input_tokens"))
		out = _to_int0(u.get("output_tokens"))
		rea = _to_int0(u.get("reasoning_tokens"))
		tot = _to_int0(u.get("total_tokens"))
		if not tot:
			tot = inp + out

		for d in (stage_sum, total_sum):
			d["input_tokens"] += inp
			d["output_tokens"] += out
			d["reasoning_tokens"] += rea
			d["total_tokens"] += tot

		self.data["calls"].append(
			{
				"stage": stage_key,
				"provider": u.get("provider"),
				"model": u.get("model"),
				"phase": u.get("phase"),
				"kind": u.get("kind"),
				"cache_hit": bool(u.get("cache_hit")) if "cache_hit" in u else None,
				"cache_key": u.get("cache_key"),
				"input_tokens": inp,
				"output_tokens": out,
				"reasoning_tokens": rea if rea else None,
				"total_tokens": tot,
			},
		)

	def flush(self) -> None:
		_write_json(self.out_path, self.data)





def _iter_files(repo_root: Path, max_files: int = 50_000) -> Iterable[Path]:
	# Skip heavy / irrelevant dirs to keep scans bounded.
	skip_dirs = {
		".git",
		".hg",
		".svn",
		"node_modules",
		"dist",
		"build",
		".venv",
		"venv",
		"__pycache__",
		".mypy_cache",
		".pytest_cache",
		".ruff_cache",
		".idea",
		".vscode",
	}
	count = 0
	for p in repo_root.rglob("*"):
		if count >= max_files:
			break
		# prune directories
		if p.is_dir():
			if p.name in skip_dirs:
				# rglob can't be pruned perfectly, but skipping children by continue is OK.
				continue
			continue
		if any(part in skip_dirs for part in p.parts):
			continue
		count += 1
		yield p


# -----------------------------------------------------------------------------
# Applies-if expression evaluation (tiny safe parser)
# -----------------------------------------------------------------------------


Token = Tuple[str, str]  # (kind, value)


def _tokenize(expr: str) -> List[Token]:
	expr = expr.strip()
	if not expr:
		return []
	# Normalize common operators
	expr = expr.replace("||", " OR ").replace("&&", " AND ")
	pat = re.compile(
		r"\s*(?:(\()|(\))|(==)|\b(AND|OR)\b|\b(true|false)\b|([A-Za-z_][A-Za-z0-9_\-]*)|\"([^\"]*)\"|\'([^\']*)\')",
		re.IGNORECASE,
	)
	out: List[Token] = []
	pos = 0
	while pos < len(expr):
		m = pat.match(expr, pos)
		if not m:
			raise ValueError(f"Unrecognized token near: {expr[pos:pos+30]!r}")
		pos = m.end()
		if m.group(1):
			out.append(("LPAREN", "("))
		elif m.group(2):
			out.append(("RPAREN", ")"))
		elif m.group(3):
			out.append(("EQ", "=="))
		elif m.group(4):
			out.append((m.group(4).upper(), m.group(4).upper()))
		elif m.group(5):
			out.append(("BOOL", m.group(5).lower()))
		elif m.group(6):
			out.append(("IDENT", m.group(6)))
		elif m.group(7) is not None:
			out.append(("STRING", m.group(7)))
		elif m.group(8) is not None:
			out.append(("STRING", m.group(8)))
	return out


class _Parser:
	def __init__(self, tokens: List[Token], signals: Dict[str, Any]):
		self.tokens = tokens
		self.i = 0
		self.signals = signals

	def _peek(self) -> Optional[Token]:
		return self.tokens[self.i] if self.i < len(self.tokens) else None

	def _eat(self, kind: str) -> Token:
		tok = self._peek()
		if tok is None or tok[0] != kind:
			raise ValueError(f"Expected {kind}, got {tok}")
		self.i += 1
		return tok

	def parse(self) -> bool:
		if not self.tokens:
			return False
		v = self._parse_or()
		if self._peek() is not None:
			raise ValueError(f"Unexpected trailing tokens: {self.tokens[self.i:]}")
		return bool(v)

	def _parse_or(self) -> bool:
		v = self._parse_and()
		while True:
			tok = self._peek()
			if tok and tok[0] == "OR":
				self._eat("OR")
				rhs = self._parse_and()
				v = v or rhs
			else:
				break
		return v

	def _parse_and(self) -> bool:
		v = self._parse_term()
		while True:
			tok = self._peek()
			if tok and tok[0] == "AND":
				self._eat("AND")
				rhs = self._parse_term()
				v = v and rhs
			else:
				break
		return v

	def _parse_term(self) -> bool:
		tok = self._peek()
		if tok is None:
			return False
		if tok[0] == "LPAREN":
			self._eat("LPAREN")
			v = self._parse_or()
			self._eat("RPAREN")
			return v
		if tok[0] == "BOOL":
			self._eat("BOOL")
			return tok[1] == "true"
		if tok[0] == "IDENT":
			# IDENT [== literal]?
			left = self._eat("IDENT")[1]
			nxt = self._peek()
			if nxt and nxt[0] == "EQ":
				self._eat("EQ")
				rhs = self._parse_literal()
				lhs_val = self.signals.get(left)
				return str(lhs_val).lower() == str(rhs).lower()
			# bare identifier means "truthy" signal
			return bool(self.signals.get(left))
		raise ValueError(f"Unexpected token: {tok}")

	def _parse_literal(self) -> Any:
		tok = self._peek()
		if tok is None:
			raise ValueError("Expected literal")
		if tok[0] == "BOOL":
			self._eat("BOOL")
			return tok[1] == "true"
		if tok[0] == "IDENT":
			return self._eat("IDENT")[1]
		if tok[0] == "STRING":
			return self._eat("STRING")[1]
		raise ValueError(f"Expected literal, got {tok}")


def eval_applies_if(expr: str, signals: Dict[str, Any]) -> bool:
	expr = (expr or "").strip()
	if expr.lower() in ("always", "true"):
		return True
	if expr.lower() in ("never", "false", ""):
		return False
	tokens = _tokenize(expr)
	return _Parser(tokens, signals).parse()


def _parse_optional_bool(value: Any) -> Optional[bool]:
	if value is None:
		return None
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		s = value.strip().lower()
		if s in {"true", "1", "yes", "y", "on"}:
			return True
		if s in {"false", "0", "no", "n", "off"}:
			return False
	return None


# -----------------------------------------------------------------------------
# Rubric loading + normalization
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricGroup:
	module_id: str
	module_name: str
	module_weight: float
	module_applies_if: str

	group_id: str
	group_name: str
	group_weight: float
	evidence_profile: Optional[str]
	leaves: List[LeafDefinition]
	include_simal_schema_in_scoring: Optional[bool] = None



@dataclass(frozen=True)
class RubricModule:
	module_id: str
	module_name: str
	module_weight: float
	applies_if: str
	groups: List[RubricGroup]


def load_rubric(nodes_yaml: Dict[str, Any]) -> List[RubricModule]:
	modules_raw = nodes_yaml.get("modules")
	if not isinstance(modules_raw, list):
		raise ValueError("nodes.yaml missing 'modules' list")

	modules: List[RubricModule] = []
	for m in modules_raw:
		module_id = str(m.get("id"))
		module_name = str(m.get("name"))
		module_weight = float(m.get("module_weight", 0.0))
		applies_if = str(m.get("applies_if", "always"))
		groups_raw = m.get("groups") or []

		groups: List[RubricGroup] = []
		for g in groups_raw:
			group_id = str(g.get("id"))
			group_name = str(g.get("name"))
			group_weight = float(g.get("group_weight", 0.0))
			evidence_profile = g.get("evidence_profile")
			include_simal_schema_in_scoring = _parse_optional_bool(g.get("include_simal_schema_in_scoring"))
			leaves_def: List[LeafDefinition] = []
			for item in g.get("leaves") or []:
				# leaves are [{"A1.1": "..."}, ...]
				if isinstance(item, dict) and len(item) == 1:
					(leaf_id, question), = item.items()
					leaves_def.append(LeafDefinition(str(leaf_id), str(question), weight=None))
				else:
					raise ValueError(f"Unexpected leaf shape in {module_id}.{group_id}: {item!r}")

			groups.append(
				RubricGroup(
					module_id=module_id,
					module_name=module_name,
					module_weight=module_weight,
					module_applies_if=applies_if,
					group_id=group_id,
					group_name=group_name,
					group_weight=group_weight,
					evidence_profile=str(evidence_profile) if evidence_profile is not None else None,
					include_simal_schema_in_scoring=include_simal_schema_in_scoring,
					leaves=leaves_def,
				)
			)

		modules.append(
			RubricModule(
				module_id=module_id,
				module_name=module_name,
				module_weight=module_weight,
				applies_if=applies_if,
				groups=groups,
			)
		)
	return modules


def build_activation_subset_yaml(modules: List[RubricModule]) -> str:
	subset = {
		"modules": [
			{
				"id": m.module_id,
				"name": m.module_name,
				"applies_if": m.applies_if,
				"groups": [
					{
						"id": g.group_id,
						"name": g.group_name,
						"evidence_profile": g.evidence_profile,
					}
					for g in m.groups
				],
			}
			for m in modules
		]
	}
	return _yaml_dump(subset)


# -----------------------------------------------------------------------------
# Evidence profiles loading
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceProfile:
	name: str
	grading_mode: str
	evidence: List[Dict[str, Any]]


def load_profiles(profiles_yaml: Dict[str, Any]) -> Dict[str, EvidenceProfile]:
	eps = profiles_yaml.get("evidence_profiles")
	if not isinstance(eps, dict):
		raise ValueError("profiles.yaml missing 'evidence_profiles' mapping")
	out: Dict[str, EvidenceProfile] = {}
	for name, body in eps.items():
		if not isinstance(body, dict):
			continue
		out[name] = EvidenceProfile(
			name=str(name),
			grading_mode=str(body.get("grading_mode", "llm_assisted")),
			evidence=list(body.get("evidence") or []),
		)
	return out


# -----------------------------------------------------------------------------
# Evidence collection
# -----------------------------------------------------------------------------


def _glob_any(repo_root: Path, pattern: str) -> List[str]:
	# Python's glob is already recursive with **.
	hits = [str(p.relative_to(repo_root)).replace("\\", "/") for p in repo_root.glob(pattern)]
	# Remove directories
	hits = [h for h in hits if not (repo_root / h).is_dir()]
	return sorted(set(hits))


def collect_file_match(repo_root: Path, patterns: List[str], max_matches: int = 200) -> Dict[str, Any]:
	matched: Dict[str, List[str]] = {}
	for pat in patterns:
		hits = _glob_any(repo_root, pat)
		if hits:
			matched[pat] = hits[:max_matches]
	return {
		"matched": matched,
		"unmatched_patterns": [p for p in patterns if p not in matched],
	}


def collect_repo_scan(repo_root: Path, checks: List[str], max_top_entries: int = 200) -> Dict[str, Any]:
	out: Dict[str, Any] = {}
	top_entries = sorted([p.name for p in repo_root.iterdir()])[:max_top_entries]

	if "file_tree_top" in checks or "top_level_layout" in checks:
		out["top_level_entries"] = top_entries
		out["has_src"] = (repo_root / "src").exists()
		out["has_tests"] = (repo_root / "tests").exists() or any(repo_root.glob("**/*test*.*"))
		out["has_docs"] = (repo_root / "docs").exists()
		out["has_github"] = (repo_root / ".github").exists()

	if "languages_detected" in checks:
		out["detected_languages"] = detect_languages(repo_root)

	if "main_entry_candidates" in checks:
		candidates = []
		for rel in ["main.py", "app.py", "cmd", "bin", "src/main.py", "src/index.ts", "src/index.js"]:
			p = repo_root / rel
			if p.exists():
				candidates.append(rel)
		out["main_entry_candidates"] = candidates

	if "gitignore_quality" in checks:
		gi = repo_root / ".gitignore"
		txt = _read_text(gi)
		out["gitignore_present"] = gi.exists()
		out["gitignore_has_common_patterns"] = {
			"node_modules": "node_modules" in txt,
			"dist": re.search(r"(^|\n)dist/?($|\n)", txt) is not None,
			"build": re.search(r"(^|\n)build/?($|\n)", txt) is not None,
			"__pycache__": "__pycache__" in txt,
			".venv": ".venv" in txt or "venv" in txt,
		}

	if "binary_blobs" in checks:
		# Heuristic only
		large = []
		for p in _iter_files(repo_root, max_files=25_000):
			try:
				if p.stat().st_size >= 5 * 1024 * 1024:
					large.append(str(p.relative_to(repo_root)).replace("\\", "/"))
			except Exception:
				continue
		out["large_files_over_5mb"] = large[:100]

	if "generated_artifacts" in checks:
		out["generated_artifact_dirs_present"] = {
			"node_modules": (repo_root / "node_modules").exists(),
			"dist": (repo_root / "dist").exists(),
			"build": (repo_root / "build").exists(),
			"target": (repo_root / "target").exists(),
			"__pycache__": any(p.name == "__pycache__" for p in repo_root.rglob("__pycache__")),
		}

	return out


def _should_skip_retrieval_path(rel_path: str) -> bool:
	"""Heuristic filter for generated/bundled/minified outputs.

	Used by code_search and activation-hint file loading to keep evidence
	high-signal and prompt-friendly.
	"""
	rp = (rel_path or "").replace("\\", "/").lstrip("./")
	low = rp.lower()
	parts = [p for p in low.split("/") if p]
	generated_dirs = {
		"dist",
		"build",
		"target",
		"out",
		"coverage",
		".next",
		".nuxt",
		".cache",
		"node_modules",
		"vendor",
		"bower_components",
		"jspm_packages",
	}
	if any(p in generated_dirs for p in parts):
		return True
	# Common public build outputs.
	if low.startswith("public/build/") or low.startswith("public/dist/"):
		return True
	# Sourcemaps and minified/bundled assets.
	if low.endswith(".map"):
		return True
	if re.search(r"\.(min|bundle|chunk)\.(js|css)$", low):
		return True
	# Common framework bundle filenames.
	if low in {
		"public/js/app.js",
		"public/js/vendor.js",
		"public/js/manifest.js",
		"public/css/app.css",
	}:
		return True
	return False


def _normalize_retrieval_hint_value(value: str) -> str:
	return (value or "").replace("\\", "/").lstrip("./").strip()


def _extract_preferred_retrieval_hints(evidence_hints: Optional[List[Dict[str, Any]]]) -> Dict[str, List[str]]:
	preferred_paths: List[str] = []
	preferred_globs: List[str] = []
	for hint in evidence_hints or []:
		if not isinstance(hint, dict):
			continue
		hint_type = str(hint.get("type", "")).strip().lower()
		value = _normalize_retrieval_hint_value(str(hint.get("value", "")))
		if not value:
			continue
		if hint_type == "path":
			preferred_paths.append(value)
		elif hint_type == "glob":
			preferred_globs.append(value)
	return {
		"paths": preferred_paths,
		"globs": preferred_globs,
	}


def _path_matches_preferred_hint(rel_path: str, preferred_hints: Optional[Dict[str, List[str]]]) -> bool:
	if not preferred_hints:
		return False
	normalized = _normalize_retrieval_hint_value(rel_path)
	for path_hint in preferred_hints.get("paths") or []:
		if normalized == path_hint or normalized.startswith(path_hint.rstrip("/") + "/"):
			return True
	for glob_hint in preferred_hints.get("globs") or []:
		try:
			if Path(normalized).match(glob_hint):
				return True
		except Exception:
			continue
	return False


def _normalize_snippet_fingerprint(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip().lower())


def _looks_generated_or_minified_content(text: str) -> bool:
	"""Heuristic filter for minified/obfuscated/generated text blobs.

	Targets files where line structure is not meaningful for snippet retrieval.
	"""
	if not text:
		return False
	text_len = len(text)
	lines = text.splitlines()
	if not lines:
		return False

	line_lengths = [len(line) for line in lines]
	max_line_length = max(line_lengths)
	avg_line_length = sum(line_lengths) / max(1, len(line_lengths))
	total_lines = len(lines)
	longest_line_share = max_line_length / max(1, text_len)

	if max_line_length >= 3000:
		return True
	if max_line_length >= 1500 and avg_line_length >= 400:
		return True
	if total_lines <= 5 and avg_line_length >= 500:
		return True
	if total_lines <= 12 and avg_line_length >= 350 and longest_line_share >= 0.6:
		return True
	if longest_line_share >= 0.85 and max_line_length >= 1200:
		return True
	return False


def _snippet_sort_key(snippet: Dict[str, Any]) -> Tuple[int, int, int, int, str, int]:
	return (
		1 if snippet.get("has_exact_phrase") else 0,
		int(snippet.get("distinct_query_count", 0)),
		1 if snippet.get("preferred_hint_match") else 0,
		int(snippet.get("match_count", 0)),
		str(snippet.get("file", "")),
		-int(snippet.get("snippet_start_line", 0)),
	)


def _cap_snippet_window(
	*,
	start_line: int,
	end_line: int,
	match_lines: List[int],
	file_line_count: int,
	max_snippet_lines: int,
) -> Tuple[int, int]:
	if max_snippet_lines <= 0:
		return start_line, end_line
	current_span = end_line - start_line + 1
	if current_span <= max_snippet_lines:
		return start_line, end_line
	focus_line = match_lines[len(match_lines) // 2] if match_lines else start_line
	half_window = max_snippet_lines // 2
	new_start = max(1, focus_line - half_window)
	new_end = new_start + max_snippet_lines - 1
	if new_end > file_line_count:
		new_end = file_line_count
		new_start = max(1, new_end - max_snippet_lines + 1)
	return new_start, new_end


def _collect_file_code_search_candidates(
	*,
	repo_root: Path,
	file_path: Path,
	queries: List[str],
	compiled: List[Tuple[str, re.Pattern[str]]],
	max_snippet_lines: int,
	context_before: int,
	context_after: int,
	merge_distance_lines: int,
	preferred_hints: Optional[Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
	try:
		rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
	except Exception:
		return []
	if _should_skip_retrieval_path(rel):
		return []
	text = _read_text(file_path, max_bytes=256_000)
	if not text:
		return []
	if _looks_generated_or_minified_content(text):
		return []
	lines = text.splitlines()
	if not lines:
		return []

	occurrences: List[Dict[str, Any]] = []
	for q, rx in compiled:
		for idx, line in enumerate(lines, start=1):
			if not rx.search(line):
				continue
			occurrences.append(
				{
					"query": q,
					"match_line": idx,
					"exact_phrase": q.lower() in line.lower(),
				}
			)

	if not occurrences:
		return []

	occurrences.sort(key=lambda item: (int(item.get("match_line", 0)), str(item.get("query", ""))))
	merged: List[Dict[str, Any]] = []
	for occ in occurrences:
		match_line = int(occ.get("match_line", 0))
		start_line = max(1, match_line - context_before)
		end_line = min(len(lines), match_line + context_after)
		if merged and start_line <= int(merged[-1]["snippet_end_line"]) + merge_distance_lines:
			prev = merged[-1]
			prev["snippet_end_line"] = max(int(prev["snippet_end_line"]), end_line)
			prev_occ = prev.setdefault("occurrences", [])
			prev_occ.append(occ)
			continue
		merged.append(
			{
				"file": rel,
				"snippet_start_line": start_line,
				"snippet_end_line": end_line,
				"occurrences": [occ],
			}
		)

	query_order = {q: i for i, q in enumerate(queries)}
	candidates: List[Dict[str, Any]] = []
	for item in merged:
		all_match_lines = sorted(
			{int(occ.get("match_line", 0)) for occ in (item.get("occurrences") or []) if occ.get("match_line")}
		)
		start_line, end_line = _cap_snippet_window(
			start_line=int(item["snippet_start_line"]),
			end_line=int(item["snippet_end_line"]),
			match_lines=all_match_lines,
			file_line_count=len(lines),
			max_snippet_lines=max_snippet_lines,
		)
		snippet_text = "\n".join(lines[start_line - 1 : end_line])
		occ_list = item.get("occurrences") or []
		queries_hit = sorted({str(occ.get("query", "")) for occ in occ_list if occ.get("query")}, key=lambda q: query_order.get(q, len(query_order)))
		match_lines = all_match_lines[:20]
		has_exact_phrase = any(bool(occ.get("exact_phrase")) for occ in occ_list)
		candidates.append(
			{
				"file": rel,
				"query": queries_hit[0] if queries_hit else "",
				"queries_hit": queries_hit,
				"distinct_query_count": len(queries_hit),
				"match_count": len(occ_list),
				"match_line": match_lines[0] if match_lines else start_line,
				"match_lines": match_lines,
				"match_lines_truncated": len(all_match_lines) > len(match_lines),
				"snippet_start_line": start_line,
				"snippet_end_line": end_line,
				"snippet": snippet_text,
				"has_exact_phrase": has_exact_phrase,
				"preferred_hint_match": _path_matches_preferred_hint(rel, preferred_hints),
			}
		)
	return candidates


def collect_code_search(
	repo_root: Path,
	queries: List[str],
	max_files: int,
	max_snippet_lines: int,
	max_snippets_total: int = 30,
	max_snippets_per_query: int = 5,
	max_snippets_per_file: int = 5,
	context_before: Optional[int] = None,
	context_after: Optional[int] = None,
	merge_distance_lines: int = 10,
	preferred_hints: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
	default_context = max(0, int(max_snippet_lines) // 2) if max_snippet_lines else 25
	context_before = default_context if context_before is None else max(0, int(context_before))
	context_after = default_context if context_after is None else max(0, int(context_after))
	max_files = max(1, int(max_files or 15))
	max_snippets_total = max(1, int(max_snippets_total or 30))
	max_snippets_per_query = max(1, int(max_snippets_per_query or 5))
	max_snippets_per_file = max(1, int(max_snippets_per_file or 5))
	merge_distance_lines = max(0, int(merge_distance_lines or 10))
	candidates: List[Dict[str, Any]] = []
	compiled: List[Tuple[str, re.Pattern[str]]] = []
	for q in queries:
		try:
			compiled.append((q, re.compile(q)))
		except re.error:
			compiled.append((q, re.compile(re.escape(q))))

	for p in _iter_files(repo_root, max_files=35_000):
		# quick extension filter
		if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll"}:
			continue
		candidates.extend(
			_collect_file_code_search_candidates(
				repo_root=repo_root,
				file_path=p,
				queries=queries,
				compiled=compiled,
				max_snippet_lines=max_snippet_lines,
				context_before=context_before,
				context_after=context_after,
				merge_distance_lines=merge_distance_lines,
				preferred_hints=preferred_hints,
			)
		)

	candidates.sort(key=_snippet_sort_key, reverse=True)
	selected: List[Dict[str, Any]] = []
	selected_files: Dict[str, int] = {}
	selected_queries: Dict[str, int] = {}
	seen_fingerprints = set()
	truncated = False

	for candidate in candidates:
		if len(selected) >= max_snippets_total:
			truncated = True
			break
		candidate_file = str(candidate.get("file", ""))
		candidate_queries = [str(q) for q in candidate.get("queries_hit") or [] if q]
		if not candidate_file or not candidate_queries:
			continue
		if candidate_file not in selected_files and len(selected_files) >= max_files:
			truncated = True
			continue
		if selected_files.get(candidate_file, 0) >= max_snippets_per_file:
			truncated = True
			continue
		if any(selected_queries.get(q, 0) >= max_snippets_per_query for q in candidate_queries):
			truncated = True
			continue
		fingerprint = (candidate_file, _normalize_snippet_fingerprint(str(candidate.get("snippet", ""))))
		if fingerprint in seen_fingerprints:
			truncated = True
			continue
		selected.append(candidate)
		selected_files[candidate_file] = selected_files.get(candidate_file, 0) + 1
		for q in candidate_queries:
			selected_queries[q] = selected_queries.get(q, 0) + 1
		seen_fingerprints.add(fingerprint)

	return {
		"hits": selected,
		"limits": {
			"max_files": max_files,
			"max_snippets_total": max_snippets_total,
			"max_snippets_per_query": max_snippets_per_query,
			"max_snippets_per_file": max_snippets_per_file,
			"context_before": context_before,
			"context_after": context_after,
			"merge_distance_lines": merge_distance_lines,
		},
		"hits_by_query": selected_queries,
		"hits_by_file": selected_files,
		"candidate_snippets": len(candidates),
		"returned_snippets": len(selected),
		"truncated": truncated,
	}


def collect_simal_search(
	simal_schema_path: Optional[Path],
	queries: List[str],
	max_hits: int = 20,
	context_lines: int = 50,
	max_total_hits: int = 25,
	max_total_chars: int = 75_000,
	max_bytes: int = 2_000_000,
) -> Dict[str, Any]:
	"""Search a single SiMAL schema file and return merged line-window snippets.

	Unlike repo-wide code search, this collects multiple occurrences from one file,
	then merges overlapping or adjacent snippets into a single evidence item.
	Cross-query overlap is also deduplicated so broad structural queries do not
	re-embed the same schema region many times.
	"""
	if simal_schema_path is None:
		return {"status": "unavailable", "reason": "simal_schema_missing", "hits": []}
	if not simal_schema_path.exists() or not simal_schema_path.is_file():
		return {
			"status": "unavailable",
			"reason": "simal_schema_not_found",
			"file": str(simal_schema_path),
			"hits": [],
		}

	text = _read_text(simal_schema_path, max_bytes=max_bytes)
	if not text:
		return {
			"status": "completed",
			"file": simal_schema_path.name,
			"hits": [],
			"total_occurrences": 0,
			"total_snippets": 0,
		}

	compiled: List[Tuple[str, re.Pattern[str]]] = []
	for q in queries:
		try:
			compiled.append((q, re.compile(q)))
		except re.error:
			compiled.append((q, re.compile(re.escape(q))))

	lines = text.splitlines()
	occurrences_by_query: Dict[str, List[Dict[str, Any]]] = {}
	for q, rx in compiled:
		query_occurrences: List[Dict[str, Any]] = []
		for idx, line in enumerate(lines, start=1):
			if rx.search(line):
				query_occurrences.append(
					{
						"query": q,
						"match_line": idx,
						"line": _truncate_text(line, 500),
					}
				)
		if query_occurrences:
			occurrences_by_query[q] = query_occurrences

	if not occurrences_by_query:
		return {
			"status": "completed",
			"file": simal_schema_path.name,
			"hits": [],
			"hits_by_query": {},
			"total_occurrences": 0,
			"total_snippets": 0,
		}

	context_lines = max(0, int(context_lines))
	hits_by_query: Dict[str, List[Dict[str, Any]]] = {}
	local_hits_by_query: Dict[str, List[Dict[str, Any]]] = {}
	flattened_hits: List[Dict[str, Any]] = []
	total_occurrences = 0
	total_snippets_before_dedup = 0
	truncated_queries: List[str] = []

	for q in queries:
		query_occurrences = list(occurrences_by_query.get(q) or [])
		if not query_occurrences:
			continue
		total_occurrences += len(query_occurrences)
		query_occurrences.sort(key=lambda item: int(item.get("match_line", 0)))
		merged_hits: List[Dict[str, Any]] = []

		for occ in query_occurrences:
			match_line = int(occ.get("match_line", 0))
			start_line = max(1, match_line - context_lines)
			end_line = min(len(lines), match_line + context_lines)
			if merged_hits and start_line <= int(merged_hits[-1]["snippet_end_line"]) + 1:
				prev = merged_hits[-1]
				prev["snippet_end_line"] = max(int(prev["snippet_end_line"]), end_line)
				prev_occ = prev.setdefault("occurrences", [])
				if not any(int(existing.get("match_line", 0)) == match_line for existing in prev_occ):
					prev_occ.append(occ)
				continue

			merged_hits.append(
				{
					"query": q,
					"file": simal_schema_path.name,
					"snippet_start_line": start_line,
					"snippet_end_line": end_line,
					"occurrences": [occ],
				}
			)

		for hit in merged_hits:
			start_line = int(hit["snippet_start_line"])
			end_line = int(hit["snippet_end_line"])
			snippet_lines = lines[start_line - 1 : end_line]
			hit["snippet"] = "\n".join(snippet_lines)

		total_snippets_before_dedup += len(merged_hits)
		if len(merged_hits) > max_hits:
			truncated_queries.append(q)
		limited_hits = merged_hits[:max_hits]
		local_hits_by_query[q] = limited_hits
		flattened_hits.extend(limited_hits)

	flattened_hits.sort(key=lambda item: (int(item.get("snippet_start_line", 0)), int(item.get("snippet_end_line", 0))))
	merged_global_hits: List[Dict[str, Any]] = []
	for hit in flattened_hits:
		start_line = int(hit.get("snippet_start_line", 0))
		end_line = int(hit.get("snippet_end_line", 0))
		query_name = str(hit.get("query", ""))
		occurrences = list(hit.get("occurrences") or [])
		if merged_global_hits and start_line <= int(merged_global_hits[-1]["snippet_end_line"]) + 1:
			prev = merged_global_hits[-1]
			prev["snippet_end_line"] = max(int(prev["snippet_end_line"]), end_line)
			queries_hit = prev.setdefault("queries_hit", [])
			if query_name and query_name not in queries_hit:
				queries_hit.append(query_name)
			prev_occ = prev.setdefault("occurrences", [])
			seen_occ = {(str(existing.get("query", "")), int(existing.get("match_line", 0))) for existing in prev_occ}
			for occ in occurrences:
				occ_key = (str(occ.get("query", "")), int(occ.get("match_line", 0)))
				if occ_key not in seen_occ:
					prev_occ.append(occ)
					seen_occ.add(occ_key)
			continue

		merged_global_hits.append(
			{
				"file": simal_schema_path.name,
				"snippet_start_line": start_line,
				"snippet_end_line": end_line,
				"queries_hit": [query_name] if query_name else [],
				"occurrences": occurrences,
			}
		)

	for index, hit in enumerate(merged_global_hits, start=1):
		start_line = int(hit["snippet_start_line"])
		end_line = int(hit["snippet_end_line"])
		snippet_lines = lines[start_line - 1 : end_line]
		hit["hit_id"] = f"simal_{index}"
		hit["snippet"] = "\n".join(snippet_lines)
		queries_hit = sorted(dict.fromkeys(str(q) for q in (hit.get("queries_hit") or []) if q))
		hit["queries_hit"] = queries_hit
		hit["distinct_query_count"] = len(queries_hit)
		hit["query"] = queries_hit[0] if queries_hit else ""
		occurrences = sorted(
			list(hit.get("occurrences") or []),
			key=lambda occ: (int(occ.get("match_line", 0)), str(occ.get("query", ""))),
		)
		hit["occurrences"] = occurrences
		hit["occurrence_count"] = len(occurrences)
		match_lines = [int(occ.get("match_line", 0)) for occ in occurrences if int(occ.get("match_line", 0)) > 0]
		hit["match_lines"] = match_lines[:20]
		if len(match_lines) > 20:
			hit["match_lines_truncated"] = True

	returned_hits: List[Dict[str, Any]] = []
	returned_chars = 0
	global_truncated_by_hits = False
	global_truncated_by_chars = False
	for hit in merged_global_hits:
		if len(returned_hits) >= max_total_hits:
			global_truncated_by_hits = True
			break
		hit_chars = len(str(hit.get("snippet") or ""))
		if returned_hits and (returned_chars + hit_chars) > max_total_chars:
			global_truncated_by_chars = True
			break
		returned_hits.append(hit)
		returned_chars += hit_chars

	returned_hit_ids = {str(hit.get("hit_id", "")) for hit in returned_hits}
	for q in queries:
		query_hits = list(local_hits_by_query.get(q) or [])
		if not query_hits:
			continue
		query_summaries: List[Dict[str, Any]] = []
		for hit in query_hits:
			start_line = int(hit.get("snippet_start_line", 0))
			end_line = int(hit.get("snippet_end_line", 0))
			occurrences = list(hit.get("occurrences") or [])
			match_lines = [int(occ.get("match_line", 0)) for occ in occurrences if int(occ.get("match_line", 0)) > 0]
			hit_ids = [
				str(global_hit.get("hit_id", ""))
				for global_hit in returned_hits
				if start_line <= int(global_hit.get("snippet_end_line", 0)) and end_line >= int(global_hit.get("snippet_start_line", 0))
			]
			query_summary: Dict[str, Any] = {
				"snippet_start_line": start_line,
				"snippet_end_line": end_line,
				"occurrence_count": len(occurrences),
				"match_lines": match_lines[:20],
				"hit_ids": [hit_id for hit_id in hit_ids if hit_id in returned_hit_ids],
			}
			if len(match_lines) > 20:
				query_summary["match_lines_truncated"] = True
			query_summaries.append(query_summary)
		hits_by_query[q] = query_summaries

	return {
		"status": "completed",
		"file": simal_schema_path.name,
		"hits": returned_hits,
		"hits_by_query": hits_by_query,
		"total_occurrences": total_occurrences,
		"total_snippets": len(merged_global_hits),
		"total_snippets_before_dedup": total_snippets_before_dedup,
		"returned_snippets": len(returned_hits),
		"returned_chars": returned_chars,
		"truncated": bool(truncated_queries or global_truncated_by_hits or global_truncated_by_chars),
		"truncated_queries": truncated_queries,
		"max_hits_per_query": max_hits,
		"max_total_hits": max_total_hits,
		"max_total_chars": max_total_chars,
		"truncated_by_total_hits": global_truncated_by_hits,
		"truncated_by_total_chars": global_truncated_by_chars,
		"context_lines": context_lines,
	}


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
	out = dict(a)
	for k, v in b.items():
		if k in out and isinstance(out[k], dict) and isinstance(v, dict):
			out[k] = _merge_dicts(out[k], v)
		else:
			out[k] = v
	return out


_DYNAMIC_TOOL_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}


def _pick_declared_output_paths(spec: Dict[str, Any]) -> List[str]:
	outs = spec.get("outputs", [])
	if not outs:
		return []
	if isinstance(outs, list):
		return [str(x) for x in outs if x]
	return []


def _resolve_tool_candidates(
	registry: Dict[str, Any],
	evidence_class: str,
	languages: List[str],
) -> List[Dict[str, Any]]:
	cls = registry.get(evidence_class)
	if not isinstance(cls, dict):
		return []

	candidates: List[Dict[str, Any]] = []
	for lang in languages:
		for spec in cls.get(lang, []) or []:
			if isinstance(spec, dict):
				candidates.append(spec)
	for spec in cls.get("any", []) or []:
		if isinstance(spec, dict):
			candidates.append(spec)

	candidates.sort(key=lambda s: int(s.get("priority", 0)), reverse=True)
	return candidates


def _load_json_if_possible(path: Path) -> Optional[Dict[str, Any]]:
	try:
		if not path.exists() or not path.is_file():
			return None
		obj = json.loads(_read_text(path, max_bytes=10_000_000))
		return obj if isinstance(obj, dict) else None
	except Exception:
		return None


def _compact_tool_result_for_prompt(result: Any) -> Any:
	"""Return a prompt-friendly view of a tool result.

	We keep raw outputs on disk for debugging (stdout/stderr files + moved artifacts),
	while the scoring prompt should consume deterministic summaries.
	"""
	if not isinstance(result, dict):
		return result

	# Minimal metadata + summary fields.
	out: Dict[str, Any] = {}
	for k in [
		"status",
		"tool",
		"tool_description",
		"evidence_class",
		"cmd",
		"params",
		"languages",
		"exit_code",
		"duration_sec",
		"cache_hit",
		"cache_key",
		"treat_nonzero_as_findings",
		"treat_nonzero_as_execution_result",
		"declared_outputs",
		"missing_artifacts",
		"artifacts",
		"parsing",
		"tool_summary",
		"summary",
	]:
		if k in result:
			out[k] = result.get(k)

	return out


def _semgrep_extract_findings(semgrep_json: Dict[str, Any]) -> List[Dict[str, Any]]:
	results = semgrep_json.get("results")
	if not isinstance(results, list):
		return []
	out: List[Dict[str, Any]] = []
	for r in results:
		if not isinstance(r, dict):
			continue
		check_id = str(r.get("check_id", ""))
		path = str(r.get("path", ""))
		start = (r.get("start") or {}) if isinstance(r.get("start"), dict) else {}
		end = (r.get("end") or {}) if isinstance(r.get("end"), dict) else {}
		extra = (r.get("extra") or {}) if isinstance(r.get("extra"), dict) else {}
		severity = str(extra.get("severity", ""))
		message = str(extra.get("message") or ((extra.get("metadata") or {}) if isinstance(extra.get("metadata"), dict) else {}).get("message") or "")
		out.append(
			{
				"check_id": check_id,
				"path": path,
				"start": {"line": start.get("line"), "col": start.get("col")},
				"end": {"line": end.get("line"), "col": end.get("col")},
				"severity": severity,
				"message": message,
			}
		)
	return out


def _contains_any(haystack: str, needles: List[str]) -> bool:
	h = (haystack or "").lower()
	for n in needles:
		if (n or "").lower() in h:
			return True
	return False


def postprocess_semgrep_focus(
	*,
	raw_tool_result: Dict[str, Any],
	ruleset: str,
	work_dir: Path,
) -> Dict[str, Any]:
	focus = SAST_FOCUS_MAP.get(ruleset)
	if not focus:
		return {"status": "tool_failed", "reason": f"Unknown ruleset: {ruleset}"}

	# Find a semgrep JSON artifact path
	json_rel: Optional[str] = None
	artifacts = raw_tool_result.get("artifacts") or {}
	if isinstance(artifacts, dict):
		for src_name, rel in artifacts.items():
			if not isinstance(rel, str):
				continue
			if str(src_name).lower().endswith(".json") and "semgrep" in str(src_name).lower():
				json_rel = rel
				break

	semgrep_obj: Optional[Dict[str, Any]] = None
	if json_rel:
		semgrep_obj = _load_json_if_possible(work_dir / json_rel)
	if semgrep_obj is None:
		# Fallback: attempt to parse captured stdout file if it looks like JSON
		try:
			stdout_rel = raw_tool_result.get("stdout_path")
			if isinstance(stdout_rel, str) and stdout_rel:
				semgrep_obj = _load_json_if_possible(work_dir / stdout_rel)
		except Exception:
			semgrep_obj = None

	if semgrep_obj is None:
		return {"status": "tool_failed", "reason": "semgrep_json_unavailable"}

	findings = _semgrep_extract_findings(semgrep_obj)
	pf = (((focus.get("semgrep") or {}) if isinstance(focus.get("semgrep"), dict) else {}).get("post_filter") or {})
	pf = pf if isinstance(pf, dict) else {}

	rule_id_contains_any = list(pf.get("rule_id_contains_any") or [])
	message_contains_any = list(pf.get("message_contains_any") or [])
	severity_any = [str(x).upper() for x in (pf.get("severity_any") or [])]
	path_ext_any = [str(x).lower() for x in (pf.get("path_ext_any") or [])]

	filtered: List[Dict[str, Any]] = []
	for f in findings:
		check_id = str(f.get("check_id", ""))
		path = str(f.get("path", ""))
		severity = str(f.get("severity", ""))
		message = str(f.get("message", ""))

		if rule_id_contains_any and not _contains_any(check_id, rule_id_contains_any):
			continue
		if message_contains_any and not _contains_any(message, message_contains_any):
			continue
		if severity_any and str(severity).upper() not in severity_any:
			continue
		if path_ext_any:
			ext = Path(path).suffix.lower()
			if ext and ext not in path_ext_any:
				continue
		filtered.append(f)

	by_severity: Dict[str, int] = {}
	for f in filtered:
		s = str(f.get("severity") or "").upper() or "UNKNOWN"
		by_severity[s] = by_severity.get(s, 0) + 1

	return {
		"status": "completed",
		"ruleset": ruleset,
		"description": str(focus.get("description", "")),
		"findings_count": len(filtered),
		"by_severity": by_severity,
		"findings_preview": filtered[:25],
		"source": {
			"tool": raw_tool_result.get("tool"),
			"cache_key": raw_tool_result.get("cache_key"),
			"semgrep_json": json_rel,
		},
	}


def _classify_dynamic_tool_status(
	*,
	exit_code: Any,
	treat_nonzero_as_findings: bool,
	treat_nonzero_as_execution_result: bool,
	declared_outputs: List[str],
	artifacts: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
	rc = _to_float_or_none(exit_code)
	if rc is None or int(rc) != rc:
		return "tool_failed", "invalid_exit_code"
	rc_int = int(rc)
	if rc_int == 0:
		return "completed", None

	if rc_int == 127:
		return "tool_failed", "tool_not_found_or_not_executable"
	if rc_int == 126:
		return "tool_failed", "tool_not_executable"
	if rc_int < 0:
		return "tool_failed", f"terminated_by_signal:{-rc_int}"

	# Some tools use non-zero to signal findings or a meaningful execution result.
	if treat_nonzero_as_execution_result:
		return "completed", None

	has_artifacts = any(isinstance(v, str) and v for v in (artifacts or {}).values())
	if treat_nonzero_as_findings:
		if has_artifacts or not declared_outputs:
			return "completed", None
		return "tool_failed", "nonzero_exit_and_no_artifacts"

	return "tool_failed", f"nonzero_exit_code:{rc_int}"


def _normalize_dynamic_tool_result_status(
	*,
	result: Dict[str, Any],
	treat_nonzero_as_findings: bool,
	treat_nonzero_as_execution_result: bool,
	declared_outputs_default: List[str],
) -> Dict[str, Any]:
	if not isinstance(result, dict):
		return result
	if str(result.get("status") or "") != "completed":
		return result

	declared_outputs_raw = result.get("declared_outputs")
	declared_outputs = (
		[str(x) for x in declared_outputs_raw if x]
		if isinstance(declared_outputs_raw, list)
		else list(declared_outputs_default)
	)
	artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
	status, reason = _classify_dynamic_tool_status(
		exit_code=result.get("exit_code"),
		treat_nonzero_as_findings=treat_nonzero_as_findings,
		treat_nonzero_as_execution_result=treat_nonzero_as_execution_result,
		declared_outputs=declared_outputs,
		artifacts=artifacts,
	)
	result["status"] = status
	if reason:
		result["reason"] = reason
	else:
		result.pop("reason", None)
	return result


def run_dynamic_tool(
	repo_root: Path,
	evidence_class: str,
	languages: List[str],
	tool_out_dir: Path,
	run_tools: bool,
	params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	params = dict(params or {})
	registry = get_tool_registry()
	candidates = _resolve_tool_candidates(registry, evidence_class, languages)
	if not candidates:
		if evidence_class not in registry:
			return {"status": "tool_failed", "reason": f"Unknown evidence_class: {evidence_class}"}
		return {"status": "tool_failed", "reason": f"No tools mapped for evidence_class={evidence_class}"}

	# When tool execution is disabled, return the top usable candidate (no side effects).
	if not run_tools:
		spec = next((s for s in candidates if str(s.get("cmd", "")).strip()), candidates[0])
		cmd = str(spec.get("cmd", "")).strip()
		tool_name = str(spec.get("tool", ""))
		tool_description = str(spec.get("description", "") or "")
		parsing = spec.get("parsing") if isinstance(spec.get("parsing"), dict) else {}
		treat_nonzero_as_findings = bool(spec.get("treat_nonzero_as_findings"))
		treat_nonzero_as_execution_result = bool(spec.get("treat_nonzero_as_execution_result"))
		declared_outputs = _pick_declared_output_paths(spec)

		cache_params = params
		if evidence_class == "sast_scanner" and "ruleset" in cache_params:
			cache_params = {k: v for k, v in cache_params.items() if k != "ruleset"}
		cache_key_obj = {
			"repo": str(repo_root),
			"evidence_class": evidence_class,
			"languages": sorted(set(languages)),
			"tool": tool_name,
			"cmd": cmd,
			"params": cache_params,
		}
		cache_key = _sha256_hex(_stable_json(cache_key_obj))
		return {
			"status": "skipped",
			"tool": tool_name,
			"tool_description": tool_description,
			"cmd": cmd,
			"params": params,
			"evidence_class": evidence_class,
			"languages": sorted(set(languages)),
			"parsing": parsing,
			"treat_nonzero_as_findings": treat_nonzero_as_findings,
			"treat_nonzero_as_execution_result": treat_nonzero_as_execution_result,
			"cache_key": cache_key,
			"reason": "Dynamic tool execution disabled (pass --run-tools to enable).",
			"outputs": declared_outputs,
		}

	last_result: Optional[Dict[str, Any]] = None

	for spec in candidates:
		cmd = str(spec.get("cmd", "")).strip()
		tool_name = str(spec.get("tool", ""))
		tool_description = str(spec.get("description", "") or "")
		parsing = spec.get("parsing") if isinstance(spec.get("parsing"), dict) else {}
		treat_nonzero_as_findings = bool(spec.get("treat_nonzero_as_findings"))
		treat_nonzero_as_execution_result = bool(spec.get("treat_nonzero_as_execution_result"))
		declared_outputs = _pick_declared_output_paths(spec)
		if not cmd:
			continue

		cache_params = params
		if evidence_class == "sast_scanner" and "ruleset" in cache_params:
			cache_params = {k: v for k, v in cache_params.items() if k != "ruleset"}
		cache_key_obj = {
			"repo": str(repo_root),
			"evidence_class": evidence_class,
			"languages": sorted(set(languages)),
			"tool": tool_name,
			"cmd": cmd,
			"params": cache_params,
		}
		cache_key = _sha256_hex(_stable_json(cache_key_obj))
		cache_short = cache_key[:16]

		cache_dir = tool_out_dir / "_cache"
		cache_dir.mkdir(parents=True, exist_ok=True)
		meta_path = cache_dir / f"{evidence_class}_{tool_name}_{cache_short}.result.json"
		stdout_path = cache_dir / f"{evidence_class}_{tool_name}_{cache_short}.stdout.txt"
		stderr_path = cache_dir / f"{evidence_class}_{tool_name}_{cache_short}.stderr.txt"
		artifacts_dir = cache_dir / f"{evidence_class}_{tool_name}_{cache_short}.artifacts"

		# In-run memory cache
		if cache_key in _DYNAMIC_TOOL_RESULT_CACHE:
			cached = dict(_DYNAMIC_TOOL_RESULT_CACHE[cache_key])
			cached["cache_hit"] = True
			cached.setdefault("evidence_class", evidence_class)
			cached.setdefault("languages", sorted(set(languages)))
			cached.setdefault("parsing", parsing)
			cached.setdefault("tool_description", tool_description)
			cached.setdefault("treat_nonzero_as_findings", treat_nonzero_as_findings)
			cached.setdefault("treat_nonzero_as_execution_result", treat_nonzero_as_execution_result)
			cached.setdefault("declared_outputs", declared_outputs)
			cached = _normalize_dynamic_tool_result_status(
				result=cached,
				treat_nonzero_as_findings=treat_nonzero_as_findings,
				treat_nonzero_as_execution_result=treat_nonzero_as_execution_result,
				declared_outputs_default=declared_outputs,
			)
			cached = summarize_tool_result(tool_result=cached, work_dir=tool_out_dir.parent)
			_DYNAMIC_TOOL_RESULT_CACHE[cache_key] = dict(cached)
			if str(cached.get("status") or "") == "completed":
				return cached
			last_result = cached
			continue

		# On-disk cache
		if meta_path.exists():
			cached_obj = _load_json_if_possible(meta_path)
			if cached_obj:
				cached_obj["cache_hit"] = True
				cached_obj.setdefault("evidence_class", evidence_class)
				cached_obj.setdefault("languages", sorted(set(languages)))
				cached_obj.setdefault("parsing", parsing)
				cached_obj.setdefault("tool_description", tool_description)
				cached_obj.setdefault("treat_nonzero_as_findings", treat_nonzero_as_findings)
				cached_obj.setdefault("treat_nonzero_as_execution_result", treat_nonzero_as_execution_result)
				cached_obj.setdefault("declared_outputs", declared_outputs)
				cached_obj = _normalize_dynamic_tool_result_status(
					result=cached_obj,
					treat_nonzero_as_findings=treat_nonzero_as_findings,
					treat_nonzero_as_execution_result=treat_nonzero_as_execution_result,
					declared_outputs_default=declared_outputs,
				)
				cached_obj = summarize_tool_result(tool_result=cached_obj, work_dir=tool_out_dir.parent)
				try:
					_write_json(meta_path, cached_obj)
				except Exception:
					pass
				_DYNAMIC_TOOL_RESULT_CACHE[cache_key] = cached_obj
				if str(cached_obj.get("status") or "") == "completed":
					return cached_obj
				last_result = cached_obj
				continue

		# Note: many cmds rely on shell redirection; use shell=True.
		try:
			t0 = time.time()
			proc = subprocess.run(
				cmd,
				cwd=str(repo_root),
				shell=True,
				capture_output=True,
				text=True,
				timeout=60 * 20,
			)
			stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
			stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")

			artifacts: Dict[str, str] = {}
			missing_artifacts: List[str] = []
			if declared_outputs:
				artifacts_dir.mkdir(parents=True, exist_ok=True)
				for out_rel in declared_outputs:
					src = (repo_root / out_rel).resolve()
					try:
						src.relative_to(repo_root)
					except Exception:
						missing_artifacts.append(out_rel)
						continue
					if not src.exists():
						missing_artifacts.append(out_rel)
						continue

					dst = artifacts_dir / out_rel
					dst.parent.mkdir(parents=True, exist_ok=True)
					shutil.copy2(src, dst)
					rel_for_report = str(dst.relative_to(tool_out_dir.parent)).replace("\\", "/")
					artifacts[out_rel] = rel_for_report

			status, failure_reason = _classify_dynamic_tool_status(
				exit_code=proc.returncode,
				treat_nonzero_as_findings=treat_nonzero_as_findings,
				treat_nonzero_as_execution_result=treat_nonzero_as_execution_result,
				declared_outputs=declared_outputs,
				artifacts=artifacts,
			)
			result: Dict[str, Any] = {
				"status": status,
				"tool": tool_name,
				"tool_description": tool_description,
				"cmd": cmd,
				"params": params,
				"evidence_class": evidence_class,
				"languages": sorted(set(languages)),
				"parsing": parsing,
				"treat_nonzero_as_findings": treat_nonzero_as_findings,
				"treat_nonzero_as_execution_result": treat_nonzero_as_execution_result,
				"cache_key": cache_key,
				"cache_hit": False,
				"exit_code": proc.returncode,
				"duration_sec": round(time.time() - t0, 3),
				"stdout_path": str(stdout_path.relative_to(tool_out_dir.parent)).replace("\\", "/"),
				"stderr_path": str(stderr_path.relative_to(tool_out_dir.parent)).replace("\\", "/"),
				"artifacts": artifacts,
				"missing_artifacts": missing_artifacts,
				"declared_outputs": declared_outputs,
			}
			if failure_reason:
				result["reason"] = failure_reason
			result = summarize_tool_result(tool_result=result, work_dir=tool_out_dir.parent)
			try:
				_write_json(meta_path, result)
			except Exception:
				pass
			_DYNAMIC_TOOL_RESULT_CACHE[cache_key] = result
			if str(result.get("status") or "") == "completed":
				return result
			last_result = result
			continue
		except subprocess.TimeoutExpired:
			last_result = {
				"status": "tool_failed",
				"tool": tool_name,
				"tool_description": tool_description,
				"cmd": cmd,
				"reason": "timed_out",
			}
			continue
		except FileNotFoundError:
			last_result = {
				"status": "tool_failed",
				"tool": tool_name,
				"tool_description": tool_description,
				"cmd": cmd,
				"reason": "tool_not_found",
			}
			continue
		except Exception as e:
			last_result = {
				"status": "tool_failed",
				"tool": tool_name,
				"tool_description": tool_description,
				"cmd": cmd,
				"reason": f"exception: {e}",
			}
			continue

	return last_result or {
		"status": "tool_failed",
		"reason": f"All tool candidates failed for evidence_class={evidence_class}",
	}


def collect_evidence_bundle(
	repo_root: Path,
	profile: EvidenceProfile,
	languages: List[str],
	tool_out_dir: Path,
	run_tools: bool,
	simal_evidence: Optional[Dict[str, Any]] = None,
	simal_schema_path: Optional[Path] = None,
	evidence_hints: Optional[List[Dict[str, Any]]] = None,
	allow_simal: bool = True,
) -> Dict[str, Any]:
	file_matches: Dict[str, Any] = {}
	repo_scan: Dict[str, Any] = {}
	code_search: Dict[str, Any] = {}
	dynamic_tools: Dict[str, Any] = {}
	simal: Dict[str, Any] = dict(simal_evidence or {}) if allow_simal else {}
	preferred_hints = _extract_preferred_retrieval_hints(evidence_hints)
	evidence_meta: Dict[str, Any] = {
		"schema": "EvidenceBundleMetaV1",
		"profile": {
			"name": profile.name,
			"grading_mode": profile.grading_mode,
		},
		"steps": [],
		"notes": [
			"dynamic_tool: non-zero exit codes may indicate findings, not failure (see treat_nonzero_as_findings).",
			"code_search: regex-or-literal match over text files; collects multiple merged line-window snippets with global/query/file caps.",
			"simal_search: regex-or-literal match over one SiMAL schema file; hits include merged line-window snippets around all matching occurrences.",
			"file_match: glob-based existence/coverage check (does not read full contents).",
			"repo_scan: lightweight heuristics over repo layout and common files.",
		],
	}
	if not allow_simal:
		evidence_meta["notes"].append("simal: disabled by pipeline option; SiMAL inputs and simal_* evidence steps are ignored.")

	for step in profile.evidence:
		if not isinstance(step, dict):
			continue
		strat = str(step.get("strategy", "")).strip()
		if strat == "file_match":
			patterns = list(step.get("patterns") or [])
			evidence_meta["steps"].append({"strategy": "file_match", "patterns": patterns})
			file_matches = _merge_dicts(file_matches, collect_file_match(repo_root, patterns))
		elif strat == "repo_scan":
			checks = list(step.get("checks") or [])
			evidence_meta["steps"].append({"strategy": "repo_scan", "checks": checks})
			repo_scan = _merge_dicts(repo_scan, collect_repo_scan(repo_root, checks))
		elif strat == "code_search":
			queries = list(step.get("queries") or [])
			max_files = int(step.get("max_files", 15))
			max_snip = int(step.get("max_snippet_lines", 50))
			max_snippets_total = int(step.get("max_snippets_total", 30))
			max_snippets_per_query = int(step.get("max_snippets_per_query", 5))
			max_snippets_per_file = int(step.get("max_snippets_per_file", 5))
			context_before = step.get("context_before")
			context_after = step.get("context_after")
			merge_distance_lines = int(step.get("merge_distance_lines", 10))
			evidence_meta["steps"].append(
				{
					"strategy": "code_search",
					"queries": queries,
					"max_files": max_files,
					"max_snippet_lines": max_snip,
					"max_snippets_total": max_snippets_total,
					"max_snippets_per_query": max_snippets_per_query,
					"max_snippets_per_file": max_snippets_per_file,
					"context_before": context_before,
					"context_after": context_after,
					"merge_distance_lines": merge_distance_lines,
				}
			)
			code_search = _merge_dicts(
				code_search,
				collect_code_search(
					repo_root=repo_root,
					queries=queries,
					max_files=max_files,
					max_snippet_lines=max_snip,
					max_snippets_total=max_snippets_total,
					max_snippets_per_query=max_snippets_per_query,
					max_snippets_per_file=max_snippets_per_file,
					context_before=context_before,
					context_after=context_after,
					merge_distance_lines=merge_distance_lines,
					preferred_hints=preferred_hints,
				),
			)
		elif strat == "simal_search":
			queries = list(step.get("queries") or [])
			max_hits = int(step.get("max_hits", 10))
			context_lines = int(step.get("context_lines", max(1, int(step.get("max_snippet_lines", 101)) // 2)))
			max_total_hits = int(step.get("max_total_hits", max_hits))
			max_total_chars = int(step.get("max_total_chars", 75_000))
			if not allow_simal:
				evidence_meta["steps"].append(
					{
						"strategy": "simal_search",
						"queries": queries,
						"max_hits": max_hits,
						"max_total_hits": max_total_hits,
						"max_total_chars": max_total_chars,
						"context_lines": context_lines,
						"skipped": True,
						"reason": "simal_disabled",
					}
				)
				continue
			evidence_meta["steps"].append(
				{
					"strategy": "simal_search",
					"queries": queries,
					"max_hits": max_hits,
					"max_total_hits": max_total_hits,
					"max_total_chars": max_total_chars,
					"context_lines": context_lines,
				}
			)
			simal_search = collect_simal_search(
				simal_schema_path=simal_schema_path,
				queries=queries,
				max_hits=max_hits,
				context_lines=context_lines,
				max_total_hits=max_total_hits,
				max_total_chars=max_total_chars,
			)
			simal = _merge_dicts(simal, {"search": simal_search})
		elif strat == "dynamic_tool":
			dyn_evidence_class = str(step.get("evidence_class", "")).strip()
			dyn_params = step.get("params")
			dyn_params = dict(dyn_params) if isinstance(dyn_params, dict) else {}
			if dyn_evidence_class:
				step_entry: Dict[str, Any] = {
					"strategy": "dynamic_tool",
					"evidence_class": dyn_evidence_class,
					"params": dyn_params,
				}
				# For semgrep-based SAST, we cache the RAW scan once, then do focus-specific
				# post-filtering per ruleset without rerunning semgrep.
				exec_params = {} if dyn_evidence_class == "sast_scanner" else dyn_params
				raw = run_dynamic_tool(
					repo_root=repo_root,
					evidence_class=dyn_evidence_class,
					languages=languages,
					tool_out_dir=tool_out_dir,
					run_tools=run_tools,
					params=exec_params,
				)
				step_entry["tool"] = raw.get("tool")
				if raw.get("tool_description"):
					step_entry["tool_description"] = raw.get("tool_description")
				step_entry["treat_nonzero_as_findings"] = bool(raw.get("treat_nonzero_as_findings"))
				step_entry["treat_nonzero_as_execution_result"] = bool(raw.get("treat_nonzero_as_execution_result"))
				evidence_meta["steps"].append(step_entry)

				raw_for_prompt = _compact_tool_result_for_prompt(raw)
				# Store raw result under base key (first writer wins)
				dynamic_tools.setdefault(dyn_evidence_class, raw_for_prompt)
				# If tool is parameterized (e.g., SAST ruleset), also store a keyed variant.
				if dyn_evidence_class == "sast_scanner":
					ruleset = str(dyn_params.get("ruleset", "")).strip()
					if ruleset:
						focused = postprocess_semgrep_focus(
							raw_tool_result=raw,
							ruleset=ruleset,
							work_dir=tool_out_dir.parent,
						)
						focused = _compact_tool_result_for_prompt(focused)
						dynamic_tools[f"{dyn_evidence_class}:{ruleset}"] = focused
				elif dyn_params:
					param_key = _sha256_hex(_stable_json(dyn_params))[:10]
					dynamic_tools[f"{dyn_evidence_class}:{param_key}"] = raw_for_prompt
		elif strat == "simal_query":
			# Legacy alias retained for backwards compatibility.
			queries = list(step.get("queries") or [])
			max_hits = int(step.get("max_hits", 10))
			context_lines = int(step.get("context_lines", max(1, int(step.get("max_snippet_lines", 101)) // 2)))
			max_total_hits = int(step.get("max_total_hits", max_hits))
			max_total_chars = int(step.get("max_total_chars", 75_000))
			if not allow_simal:
				evidence_meta["steps"].append(
					{
						"strategy": "simal_search",
						"queries": queries,
						"max_hits": max_hits,
						"max_total_hits": max_total_hits,
						"max_total_chars": max_total_chars,
						"context_lines": context_lines,
						"legacy_alias": "simal_query",
						"skipped": True,
						"reason": "simal_disabled",
					}
				)
				continue
			evidence_meta["steps"].append(
				{
					"strategy": "simal_search",
					"queries": queries,
					"max_hits": max_hits,
					"max_total_hits": max_total_hits,
					"max_total_chars": max_total_chars,
					"context_lines": context_lines,
					"legacy_alias": "simal_query",
				}
			)
			simal_search = collect_simal_search(
				simal_schema_path=simal_schema_path,
				queries=queries,
				max_hits=max_hits,
				context_lines=context_lines,
				max_total_hits=max_total_hits,
				max_total_chars=max_total_chars,
			)
			simal = _merge_dicts(simal, {"search": simal_search})
			continue
		else:
			continue

	bundle = build_compact_evidence_bundle(
		simal_evidence=simal,
		retrieval_evidence={"code_search": code_search},
		tool_outputs=dynamic_tools,
		file_matches=file_matches,
		repo_scan=repo_scan,
		evidence_meta=evidence_meta,
	)
	return bundle


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------


def _file_match_has(file_matches: Dict[str, Any], token: str) -> bool:
	matched = file_matches.get("matched") or {}
	for pat, hits in matched.items():
		if token.lower() in str(pat).lower():
			if hits:
				return True
	# also check raw hits names
	for _, hits in matched.items():
		for h in hits:
			if token.lower() in str(h).lower():
				return True
	return False


def deterministic_group_score(
	group: RubricGroup,
	profile_name: Optional[str],
	evidence_bundle: Dict[str, Any],
) -> List[LeafScore]:
	file_matches = evidence_bundle.get("file_matches") or {}
	repo_scan = evidence_bundle.get("repo_scan") or {}
	dynamic_tools = evidence_bundle.get("dynamic_tools") or {}
	code_search = ((evidence_bundle.get("retrieval") or {}).get("code_search") or {}).get("hits")
	code_search = code_search or []

	leaf_scores: List[LeafScore] = []
	for leaf in group.leaves:
		q = leaf.question.lower()
		status: str = "insufficient_evidence"
		score: Optional[float] = None
		conf: Optional[float] = 0.0
		needs_human = True
		cites: List[str] = []
		justification = ""

		# Documentation basics
		if profile_name in {"doc_files_basic", "doc_files_plus_templates_and_changelog"}:
			needs_human = False
			if "license" in q or leaf.leaf_id.endswith(".1"):
				ok = _file_match_has(file_matches, "LICENSE") or _file_match_has(file_matches, "COPYING")
				status, score, conf = ("scored", 5.0, 0.9) if ok else ("scored", 0.0, 0.9)
				justification = "License file present." if ok else "No license file matched by patterns."
			elif "contributing" in q:
				ok = _file_match_has(file_matches, "CONTRIBUTING")
				status, score, conf = ("scored", 5.0, 0.85) if ok else ("scored", 1.0, 0.85)
				justification = "CONTRIBUTING guidelines found." if ok else "No CONTRIBUTING file matched by patterns."
			elif "code of conduct" in q:
				ok = _file_match_has(file_matches, "CODE_OF_CONDUCT")
				status, score, conf = ("scored", 4.0, 0.8) if ok else ("scored", 2.0, 0.8)
				justification = "Code of Conduct found." if ok else "No Code of Conduct file matched."
			elif "changelog" in q or "release" in q:
				ok = (
					_file_match_has(file_matches, "CHANGELOG")
					or _file_match_has(file_matches, "RELEASE_NOTES")
					or _file_match_has(file_matches, "HISTORY")
					or _file_match_has(file_matches, "NEWS")
					or _file_match_has(file_matches, "release")
				)
				status, score, conf = ("scored", 4.0, 0.8) if ok else ("scored", 2.0, 0.8)
				justification = "Changelog/release notes found." if ok else "No changelog/release notes matched."
			elif "template" in q or "issue/pr" in q:
				ok = (
					_file_match_has(file_matches, "ISSUE_TEMPLATE")
					or _file_match_has(file_matches, "PULL_REQUEST_TEMPLATE")
					or _file_match_has(file_matches, "issue_template")
					or _file_match_has(file_matches, "pull_request_template")
				)
				status, score, conf = ("scored", 4.0, 0.75) if ok else ("scored", 2.0, 0.75)
				justification = "Issue/PR templates found." if ok else "No issue/PR templates matched."
			else:
				# Ownership/maintainers/contact is fuzzy; lightly score if README exists.
				ok = _file_match_has(file_matches, "README")
				status, score, conf = ("scored", 3.0, 0.6) if ok else ("scored", 1.0, 0.6)
				justification = "README present; ownership/governance not deeply analyzed." if ok else "No README matched."

		# Repo structure scan
		elif profile_name == "repo_structure_scan":
			needs_human = False
			has_src = bool(repo_scan.get("has_src"))
			has_tests = bool(repo_scan.get("has_tests"))
			has_docs = bool(repo_scan.get("has_docs"))
			gi_ok = bool((repo_scan.get("gitignore_has_common_patterns") or {}).get("node_modules") or (repo_scan.get("gitignore_has_common_patterns") or {}).get("__pycache__"))
			if "structure" in q or "layout" in q:
				good = sum([has_src, has_tests, has_docs]) >= 2
				status, score, conf = ("scored", 4.0, 0.75) if good else ("scored", 2.0, 0.75)
				justification = "Top-level layout indicates standard structure." if good else "Top-level layout missing common dirs (src/tests/docs)."
			elif "entrypoints" in q:
				cands = repo_scan.get("main_entry_candidates") or []
				status, score, conf = ("scored", 4.0, 0.6) if cands else ("scored", 2.0, 0.6)
				justification = "Main entry candidates detected." if cands else "No obvious entrypoint candidates found by heuristic."
			elif "gitignore" in q or "generated" in q:
				status, score, conf = ("scored", 4.0, 0.7) if gi_ok else ("scored", 2.0, 0.7)
				justification = "Gitignore includes common patterns." if gi_ok else "Gitignore missing common patterns or not present."
			elif "binary" in q or "blob" in q:
				large = repo_scan.get("large_files_over_5mb") or []
				status, score, conf = ("scored", 4.0, 0.6) if not large else ("scored", 2.0, 0.6)
				justification = "No obvious large blobs detected." if not large else f"Large files detected: {large[:5]}"
			else:
				status, score, conf = "scored", 3.0, 0.5
				justification = "Repo structure heuristic scoring."

		# Dependency locking
		elif profile_name == "dependency_locking_and_sca":
			needs_human = True
			lock_present = any(
				_file_match_has(file_matches, t)
				for t in ["package-lock", "pnpm-lock", "yarn.lock", "poetry.lock", "Pipfile.lock", "Cargo.lock", "go.sum", "composer.lock"]
			)
			if "pinned" in q or "locked" in q:
				status, score, conf = ("scored", 4.0, 0.7) if lock_present else ("scored", 1.0, 0.7)
				needs_human = False
				justification = "Lockfile detected." if lock_present else "No lockfile matched; dependency pinning unclear."
			elif "vulnerability" in q or "scan" in q:
				tool = dynamic_tools.get("dependency_scanner")
				if tool and tool.get("status") == "completed":
					status, score, conf = "scored", 3.0, 0.6
					needs_human = True
					cites.append(f"tool:dependency_scanner:{tool.get('tool')}")
					justification = "Dependency scanner executed; interpret results manually (parsing not implemented here)."
				else:
					status, score, conf = "insufficient_evidence", None, 0.0
					justification = "Dependency scanner not executed; pass --run-tools or provide tool outputs."
			else:
				status, score, conf = "insufficient_evidence", None, 0.0
				justification = "Deterministic dependency sub-leaf requires deeper analysis or tool output."

		# Environment definition
		elif profile_name == "env_definition_files":
			needs_human = False
			has_env = any(
				_file_match_has(file_matches, t)
				for t in ["devcontainer", "Dockerfile", "docker-compose", "environment.yml", "poetry.lock", "requirements", "flake.nix", ".python-version"]
			)
			status, score, conf = ("scored", 4.0, 0.7) if has_env else ("scored", 2.0, 0.7)
			justification = "Environment definition files detected." if has_env else "No clear environment definition files detected."

		# Linters / style
		elif profile_name == "lint_and_style":
			needs_human = False
			has_cfg = any(
				_file_match_has(file_matches, t)
				for t in [".editorconfig", "prettier", "eslint", "pyproject.toml", "setup.cfg", "golangci"]
			)
			lint_tool = dynamic_tools.get("standard_linter")
			if lint_tool and lint_tool.get("status") == "completed":
				status, score, conf = "scored", 3.0, 0.6
				cites.append(f"tool:standard_linter:{lint_tool.get('tool')}")
				justification = "Linter executed; interpret findings count manually."
			else:
				status, score, conf = ("scored", 4.0, 0.7) if has_cfg else ("scored", 2.0, 0.7)
				justification = "Lint/format config files detected." if has_cfg else "No clear lint/format config files detected."

		# Tests / CI
		elif profile_name == "tests_and_ci_probe":
			needs_human = False
			has_ci = _file_match_has(file_matches, "workflows") or _file_match_has(file_matches, ".gitlab-ci") or _file_match_has(file_matches, "Jenkinsfile")
			has_tests = _file_match_has(file_matches, "tests/") or _file_match_has(file_matches, "__tests__") or _file_match_has(file_matches, "pytest.ini")
			if "unit tests" in q or "tests exist" in q:
				status, score, conf = ("scored", 4.0, 0.7) if has_tests else ("scored", 1.0, 0.7)
				justification = "Tests detected." if has_tests else "No tests detected by file patterns."
			elif "ci" in q:
				status, score, conf = ("scored", 4.0, 0.7) if has_ci else ("scored", 2.0, 0.7)
				justification = "CI config detected." if has_ci else "No CI config detected by patterns."
			else:
				status, score, conf = "scored", 3.0, 0.55
				justification = "Tests/CI heuristic scoring."

		# Secrets scan
		elif profile_name == "secrets_scan":
			tool = dynamic_tools.get("secrets_scanner")
			if tool and tool.get("status") == "completed":
				status, score, conf = "scored", 3.0, 0.6
				needs_human = True
				cites.append(f"tool:secrets_scanner:{tool.get('tool')}")
				justification = "Secrets scanner executed; interpret findings manually (parsing not implemented here)."
			else:
				status, score, conf = "insufficient_evidence", None, 0.0
				needs_human = True
				justification = "Secrets scanner not executed; pass --run-tools or provide tool outputs."

		else:
			status, score, conf = "insufficient_evidence", None, 0.0
			needs_human = True
			justification = "No deterministic scorer for this evidence profile; use LLM scoring output."

		leaf_scores.append(
			LeafScore(
				leaf_id=leaf.leaf_id,
				status=status,  # type: ignore[arg-type]
				score=score,
				confidence=conf,
				needs_human_review=bool(needs_human),
				citations=cites,
				justification=justification,
				remediation=[],
				evidence_summary={},
			)
		)

	return leaf_scores


def parse_group_scoring_json(obj: Dict[str, Any]) -> List[LeafScore]:
	results = obj.get("group_scoring_results")
	if not isinstance(results, list):
		raise ValueError("group scoring JSON missing group_scoring_results[]")
	valid_statuses = {"scored", "na", "insufficient_evidence", "tool_failed"}
	out: List[LeafScore] = []
	for r in results:
		if not isinstance(r, dict):
			continue
		leaf_id = str(r.get("leaf_id", "")).strip()
		if not leaf_id:
			continue
		raw_status = str(r.get("status", "")).strip().lower()
		status = raw_status if raw_status in valid_statuses else "insufficient_evidence"
		score: Optional[float] = None
		if status == "scored":
			score_val = _to_float_or_none(r.get("score"))
			if score_val is None:
				status = "insufficient_evidence"
			else:
				score = max(0.0, min(5.0, float(score_val)))
		confidence_val = _to_float_or_none(r.get("confidence"))
		confidence = _clamp01(confidence_val) if confidence_val is not None else None
		citations = _coerce_string_list(r.get("citations"), max_items=40)
		remediation = _coerce_string_list(r.get("remediation"), max_items=20)
		justification_raw = r.get("justification", "")
		justification = str(justification_raw).strip() if isinstance(justification_raw, (str, int, float, bool)) else ""
		evidence_summary_raw = r.get("evidence_summary")
		evidence_summary = dict(evidence_summary_raw) if isinstance(evidence_summary_raw, dict) else {}
		out.append(
			LeafScore(
				leaf_id=leaf_id,
				status=status,  # type: ignore[arg-type]
				score=score,
				confidence=confidence,
				needs_human_review=bool(r.get("needs_human_review", status != "scored")),
				citations=citations,
				justification=justification,
				remediation=remediation,
				evidence_summary=evidence_summary,
			)
		)
	if not out:
		raise ValueError("group scoring JSON contains no usable leaf results")
	return out


# -----------------------------------------------------------------------------
# Activation
# -----------------------------------------------------------------------------


def heuristic_activation(modules: List[RubricModule], signals: Dict[str, Any]) -> Dict[str, Any]:
	module_activation = []
	for m in modules:
		enabled = eval_applies_if(m.applies_if, signals)
		reason = "applies_if satisfied by inferred repo signals" if enabled else "applies_if not satisfied by inferred repo signals"
		module_activation.append(
			{
				"module_id": m.module_id,
				"enabled": bool(enabled),
				"reason": reason,
				"activation_confidence": 0.65 if m.applies_if.lower() != "always" else 0.9,
				"activated_groups": [
					{
						"group_id": g.group_id,
						"enabled": bool(enabled),
						"reason": "module enabled" if enabled else "module disabled",
						"activation_confidence": 0.65 if m.applies_if.lower() != "always" else 0.9,
						"evidence_hints": [],
					}
					for g in m.groups
				]
				if enabled
				else [],
			}
		)
	return {
		"normalized_languages": [],
		"language_confidence": {},
		"inferred_repo_signals": signals,
		"module_activation": module_activation,
		"global_notes": ["Heuristic activation used (no LLM activation output provided)."],
	}


def activation_to_group_defs(modules: List[RubricModule], activation: Dict[str, Any]) -> List[GroupDefinition]:
	act_mods = {m["module_id"]: m for m in activation.get("module_activation") or [] if isinstance(m, dict)}
	group_defs: List[GroupDefinition] = []
	for m in modules:
		m_act = act_mods.get(m.module_id, {"enabled": True, "activation_confidence": 0.5, "reason": "default"})
		module_enabled = bool(m_act.get("enabled", True))
		groups_act = {g["group_id"]: g for g in m_act.get("activated_groups") or [] if isinstance(g, dict)}
		for g in m.groups:
			g_act = groups_act.get(g.group_id, {"enabled": module_enabled, "activation_confidence": 0.5, "reason": "default"})
			enabled = bool(g_act.get("enabled", module_enabled))
			evidence_hints = normalize_evidence_hints(g_act.get("evidence_hints"))
			group_defs.append(
				GroupDefinition(
					module_id=g.module_id,
					module_name=g.module_name,
					group_id=g.group_id,
					group_name=g.group_name,
					module_weight=g.module_weight,
					group_weight=g.group_weight,
					leaves=g.leaves,
					enabled=enabled,
					activation_confidence=float(g_act.get("activation_confidence", m_act.get("activation_confidence", 0.5))),
					activation_reason=str(g_act.get("reason", m_act.get("reason", ""))),
					evidence_hints=evidence_hints,
				)
			)
	return group_defs


def build_group_activation_context(group_def: GroupDefinition, activation: Dict[str, Any]) -> Dict[str, Any]:
	act_mods = {m["module_id"]: m for m in activation.get("module_activation") or [] if isinstance(m, dict)}
	module_activation = act_mods.get(group_def.module_id) or {}
	group_activation = {}
	for item in module_activation.get("activated_groups") or []:
		if isinstance(item, dict) and str(item.get("group_id")) == group_def.group_id:
			group_activation = item
			break

	return {
		"inferred_repo_signals": activation.get("inferred_repo_signals"),
		"module_activation": {
			"module_id": group_def.module_id,
			"module_name": group_def.module_name,
			"enabled": bool(module_activation.get("enabled", group_def.enabled)),
			"reason": str(module_activation.get("reason", group_def.activation_reason)),
			"activation_confidence": float(module_activation.get("activation_confidence", group_def.activation_confidence)),
		},
		"group_activation": {
			"group_id": group_def.group_id,
			"group_name": group_def.group_name,
			"enabled": bool(group_activation.get("enabled", group_def.enabled)),
			"reason": str(group_activation.get("reason", group_def.activation_reason)),
			"activation_confidence": float(group_activation.get("activation_confidence", group_def.activation_confidence)),
			"evidence_hints": list(group_activation.get("evidence_hints") or group_def.evidence_hints or []),
		},
	}


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
	try:
		v = int(x)
	except Exception:
		return default
	return max(lo, min(hi, v))


def normalize_evidence_hints(hints: Any) -> List[Dict[str, Any]]:
	"""Normalize/validate evidence hints from activation output.

	We keep this permissive: unknown/invalid items are dropped.
	"""
	if not isinstance(hints, list):
		return []
	out: List[Dict[str, Any]] = []
	for h in hints:
		if not isinstance(h, dict):
			continue
		h_type = str(h.get("type", "")).strip().lower()
		if h_type not in {"path", "glob", "component", "keyword"}:
			continue
		value = str(h.get("value", "")).strip()
		if not value:
			continue
		priority = _clamp_int(h.get("priority", 5), 1, 10, 5)
		note_raw = h.get("note")
		note = str(note_raw).strip() if isinstance(note_raw, (str, int, float, bool)) else ""
		out.append({"type": h_type, "value": value, "priority": priority, "note": note})

	# Prefer higher-priority hints first.
	out.sort(key=lambda x: int(x.get("priority", 0)), reverse=True)
	return out


def _truncate_text(s: str, max_chars: int) -> str:
	if not s:
		return s
	if len(s) <= max_chars:
		return s
	return s[:max_chars] + "\n…(truncated)…"


def _resolve_repo_relative_path(repo_root: Path, value: str) -> Optional[Path]:
	"""Resolve a repo-relative path safely.

	Returns an absolute Path if it exists and is within repo_root, else None.
	"""
	rel = (value or "").strip().replace("\\", "/")
	while rel.startswith("./"):
		rel = rel[2:]
	rel = rel.lstrip("/")
	if not rel:
		return None
	try:
		candidate = (repo_root / rel).resolve()
	except Exception:
		return None
	try:
		candidate.relative_to(repo_root.resolve())
	except Exception:
		return None
	return candidate


def _iter_files_in_dir(root_dir: Path, max_files: int) -> Iterable[Path]:
	# Mirror the repo-wide skip list used by _iter_files().
	skip_dirs = {
		".git",
		".hg",
		".svn",
		"node_modules",
		"dist",
		"build",
		".venv",
		"venv",
		"__pycache__",
		".mypy_cache",
		".pytest_cache",
		".ruff_cache",
		".idea",
		".vscode",
	}
	count = 0
	for p in root_dir.rglob("*"):
		if count >= max_files:
			break
		try:
			if p.is_dir():
				# Skip ignored dirs
				if p.name in skip_dirs:
					continue
				continue
			# Skip files under ignored dirs
			if any(part in skip_dirs for part in p.parts):
				continue
			if not p.is_file():
				continue
		except Exception:
			continue
		count += 1
		yield p


def collect_evidence_from_activation_hints(
	*,
	repo_root: Path,
	evidence_hints: List[Dict[str, Any]],
	max_files_total: int = 20,
	max_files_per_dir_hint: int = 15,
	max_bytes_per_file: int = 30_000,
	max_chars_per_file: int = 15_000,
	max_search_files: int = 20,
	max_snippet_lines: int = 25,
) -> Dict[str, Any]:
	"""Retrieve evidence indicated by activation-time evidence_hints.

	- path: read the file, or all files under the directory (bounded)
	- glob: expand pattern to files and read them (bounded)
	- component/keyword: code search hits (bounded)
	"""
	if not evidence_hints:
		return {"status": "empty", "hints_used": []}

	files_out: List[Dict[str, Any]] = []
	missing: List[Dict[str, Any]] = []
	globs_expanded: List[Dict[str, Any]] = []
	dirs_expanded: List[Dict[str, Any]] = []

	# Deduplicate by normalized key while preserving priority ordering.
	seen_hint_keys = set()
	hints_used: List[Dict[str, Any]] = []
	for h in evidence_hints:
		k = (str(h.get("type")), str(h.get("value")).replace("\\", "/").strip())
		if k in seen_hint_keys:
			continue
		seen_hint_keys.add(k)
		hints_used.append(h)

	remaining_budget = max_files_total

	def _add_file(p: Path, *, hint: Dict[str, Any]) -> None:
		nonlocal remaining_budget
		if remaining_budget <= 0:
			return
		# quick extension filter
		if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll"}:
			return
		try:
			rel = str(p.relative_to(repo_root)).replace("\\", "/")
		except Exception:
			return
		if _should_skip_retrieval_path(rel):
			return
		content = _read_text(p, max_bytes=max_bytes_per_file)
		if not content:
			return
		files_out.append(
			{
				"path": rel,
				"content": _truncate_text(content, max_chars_per_file),
				"from_hint": {"type": hint.get("type"), "value": hint.get("value"), "priority": hint.get("priority")},
			}
		)
		remaining_budget -= 1

	# 1) path/glob-driven file loading
	for h in hints_used:
		h_type = str(h.get("type"))
		val = str(h.get("value", "")).strip()
		if remaining_budget <= 0:
			break
		if h_type == "path":
			p = _resolve_repo_relative_path(repo_root, val)
			if p is None or not p.exists():
				missing.append({"hint": h, "reason": "path_not_found_or_outside_repo"})
				continue
			if p.is_dir():
				dir_rel = str(p.relative_to(repo_root)).replace("\\", "/")
				loaded = 0
				for fp in _iter_files_in_dir(p, max_files=min(max_files_per_dir_hint, remaining_budget)):
					_add_file(fp, hint=h)
					loaded += 1
					if remaining_budget <= 0:
						break
				dirs_expanded.append({"dir": dir_rel + ("/" if not dir_rel.endswith("/") else ""), "loaded_files": loaded})
			else:
				_add_file(p, hint=h)
		elif h_type == "glob":
			pat = val.replace("\\", "/").lstrip("/")
			matched: List[str] = []
			try:
				for fp in repo_root.glob(pat):
					if remaining_budget <= 0:
						break
					try:
						if not fp.is_file():
							continue
						fp.relative_to(repo_root)
					except Exception:
						continue
					_add_file(fp, hint=h)
					matched.append(str(fp.relative_to(repo_root)).replace("\\", "/"))
			except Exception as e:
				missing.append({"hint": h, "reason": f"glob_failed:{e}"})
				continue
			globs_expanded.append({"pattern": pat, "matched_files": matched[:200]})

	# 2) content search for component/keyword
	queries: List[str] = []
	for h in hints_used:
		h_type = str(h.get("type"))
		if h_type in {"component", "keyword"}:
			queries.append(str(h.get("value", "")).strip())
	queries = [q for q in queries if q]
	# Deduplicate but preserve order
	seen_q = set()
	queries = [q for q in queries if not (q in seen_q or seen_q.add(q))]

	search = {"queries": queries, "hits": []}
	if queries:
		try:
			search = collect_code_search(
				repo_root=repo_root,
				queries=queries,
				max_files=max_search_files,
				max_snippet_lines=max_snippet_lines,
				preferred_hints=_extract_preferred_retrieval_hints(hints_used),
			)
			search["queries"] = queries
		except Exception as e:
			search = {"queries": queries, "error": str(e), "hits": []}

	return {
		"status": "completed",
		"hints_used": hints_used,
		"limits": {
			"max_files_total": max_files_total,
			"max_files_per_dir_hint": max_files_per_dir_hint,
			"max_bytes_per_file": max_bytes_per_file,
			"max_search_files": max_search_files,
			"max_snippet_lines": max_snippet_lines,
		},
		"loaded_files": files_out,
		"expanded_dirs": dirs_expanded,
		"expanded_globs": globs_expanded,
		"missing": missing,
		"search": search,
	}


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


@dataclass
class PipelineOptions:
	"""Configuration for running the pipeline programmatically.

	This mirrors the CLI flags in :func:`main` but is designed for import/use in
	Python (e.g., Jupyter notebooks).
	"""

	repo: str
	rubric: Optional[str] = None
	profiles: Optional[str] = None
	disable_simal: bool = False
	simal_schema: Optional[str] = None
	simal_json: Optional[str] = None
	activation_json: Optional[str] = None
	heuristic_activation: bool = False
	group_scores_dir: Optional[str] = None
	emit_prompts_dir: Optional[str] = None
	llm_provider: Optional[str] = None
	llm_config: Optional[str] = None
	llm_model: Optional[str] = None
	activation_model: str = ""
	scoring_model: str = ""
	run_tools: bool = False
	stop_after: str = ""
	out: str = "quality_report.json"
	review_scope: str = "full_repo"


def run_pipeline(opts: PipelineOptions) -> Dict[str, Any]:
	"""Run the pipeline and return the report dict (or a partial result).

	Returns a dict. For full runs, it matches the JSON written to `opts.out`.
	For early-stops (activation/tools/scoring), it returns a small status dict
	containing `work_dir` and any artifacts written so far.
	"""
	if not opts.repo:
		raise ValueError("repo is required")

	llm_config_path = Path(opts.llm_config).resolve() if opts.llm_config else Path(__file__).with_name("llm_config.yaml")
	llm_cfg = load_llm_config(str(llm_config_path))
	provider = resolve_default_provider(
		cli_provider=str(opts.llm_provider or os.getenv("SAGES_LLM_PROVIDER", "openai")),
		llm_cfg=llm_cfg,
	)
	# Model selection precedence should be:
	# 1) Explicit PipelineOptions.llm_model (or env SAGES_LLM_MODEL)
	# 2) llm_config.yaml provider defaults (providers.<name>.model / activation_model / scoring_model)
	# 3) Provider fallback (handled in LLMRuntime)
	default_model = str(opts.llm_model or os.getenv("SAGES_LLM_MODEL", ""))
	activation_model = str(opts.activation_model or os.getenv("SAGES_ACTIVATION_MODEL", ""))
	scoring_model = str(opts.scoring_model or os.getenv("SAGES_SCORING_MODEL", ""))

	repo_root = Path(opts.repo).resolve()
	rubric_path = Path(opts.rubric).resolve() if opts.rubric else Path(__file__).with_name("nodes.yaml").resolve()
	profiles_path = Path(opts.profiles).resolve() if opts.profiles else Path(__file__).with_name("profiles.yaml").resolve()
	out_path = Path(opts.out).resolve()

	if not repo_root.exists() or not repo_root.is_dir():
		raise ValueError(f"repo must be a directory: {repo_root}")
	if not rubric_path.exists():
		raise ValueError(f"Rubric YAML not found: {rubric_path}")
	if not profiles_path.exists():
		raise ValueError(f"Profiles YAML not found: {profiles_path}")

	nodes_yaml = _load_yaml(rubric_path)
	profiles_yaml = _load_yaml(profiles_path)
	modules = load_rubric(nodes_yaml)
	profiles = load_profiles(profiles_yaml)

	# Evidence + scoring output directory (used for prompts + tool cache regardless of activation mode)
	work_dir = out_path.parent / (out_path.stem + "_work")
	tool_out_dir = work_dir / "tool_outputs"
	group_evidence_dir = work_dir / "group_evidence"
	llm_cache_dir = work_dir / "llm_cache"
	llm_outputs_dir = work_dir / "llm_outputs"
	group_scores_dir = Path(opts.group_scores_dir).resolve() if opts.group_scores_dir else None
	token_usage_path = work_dir / "token_usage.json"
	usage_tracker = _TokenUsageTracker(token_usage_path)
	simal_disabled = bool(opts.disable_simal)

	simal_schema_text = ""
	simal_schema_path: Optional[Path] = None
	if not simal_disabled and opts.simal_schema is not None and str(opts.simal_schema).strip() != "":
		simal_schema_path = Path(str(opts.simal_schema)).expanduser().resolve()
		if not simal_schema_path.exists() or not simal_schema_path.is_file():
			raise ValueError(
				f"SiMAL schema file not found (or not a file): {simal_schema_path} (from opts.simal_schema={opts.simal_schema!r})"
			)
		simal_schema_text = _read_text(simal_schema_path)
	simal_evidence = None
	if not simal_disabled and opts.simal_json:
		simal_evidence = json.loads(_read_text(Path(opts.simal_json).resolve()))

	activation_subset_yaml = build_activation_subset_yaml(modules)

	# Default behavior: activation JSON is expected to come from an LLM using this prompt.
	# Heuristic activation remains available only via heuristic_activation.
	heuristic_hints: Dict[str, Any] = {}
	activation: Dict[str, Any]
	activation_languages: List[str] = []
	activation_signals: Dict[str, Any] = {}

	if opts.activation_json:
		activation = json.loads(_read_text(Path(opts.activation_json).resolve()))
		activation_languages = (
			list(activation.get("normalized_languages") or [])
			if isinstance(activation.get("normalized_languages"), list)
			else []
		)
		activation_signals = (
			dict(activation.get("inferred_repo_signals") or {})
			if isinstance(activation.get("inferred_repo_signals"), dict)
			else {}
		)
	else:
		if not opts.heuristic_activation and not simal_disabled:
			# Activation will be computed by LLM (if configured) after building the prompt.
			activation = {}
		else:
			activation_languages = detect_languages(repo_root)
			activation_signals = infer_repo_signals(repo_root)
			heuristic_hints = {"detected_languages": activation_languages, **activation_signals}
			activation = heuristic_activation(modules, activation_signals)

	activation_prompt = build_node_activation_prompt(
		ActivationPromptInputs(
			simal_schema_text=simal_schema_text,
			rubric_activation_yaml=activation_subset_yaml,
			repo_name=repo_root.name,
			review_scope=opts.review_scope,
			heuristic_hints=heuristic_hints,
		)
	)
	# Always persist the activation prompt for audit/debug.
	work_dir.mkdir(parents=True, exist_ok=True)
	_write_text(work_dir / "activation_prompt.txt", activation_prompt)
	group_evidence_dir.mkdir(parents=True, exist_ok=True)

	prompts_dir = Path(opts.emit_prompts_dir).resolve() if opts.emit_prompts_dir else None
	if prompts_dir:
		prompts_dir.mkdir(parents=True, exist_ok=True)
		_write_text(prompts_dir / "activation_prompt.txt", activation_prompt)
		if activation:
			_write_json(prompts_dir / "activation_used.json", activation)
		else:
			_write_json(
				prompts_dir / "activation_placeholder.json",
				{"status": "missing", "note": "Provide activation_json or set heuristic_activation."},
			)

	# If activation wasn't provided, attempt to call the configured LLM.
	if not activation and not opts.heuristic_activation and not simal_disabled:
		rt = LLMRuntime(
			llm_cfg=llm_cfg,
			provider=str(provider or "openai").strip().lower(),
			default_model=str(default_model or "").strip(),
			activation_model=str(activation_model or "").strip(),
			scoring_model=str(scoring_model or "").strip(),
		)
		try:
			activation, act_usage = rt.call_json_cached_with_usage(
				cache_dir=llm_cache_dir,
				kind="activation",
				phase="activation",
				prompt=activation_prompt,
				outputs_dir=llm_outputs_dir,
			)
			usage_tracker.add(stage="activation", usage=act_usage)
			usage_tracker.flush()
		except Exception as e:
			raise RuntimeError(
				"LLM activation not available: "
				+ str(e)
				+ "\nConfigure llm_config.yaml and/or API keys, or run your LLM using the prompt at: "
				+ str(work_dir / "activation_prompt.txt")
				+ " and re-run with activation_json=<path>. "
				+ "(Or set heuristic_activation=True to use fallback heuristics.)"
			) from e

	# If activation came from LLM, pick out normalized languages/signals.
	if activation and not activation_languages:
		if isinstance(activation.get("normalized_languages"), list):
			activation_languages = [str(x) for x in activation.get("normalized_languages") or []]
		if isinstance(activation.get("inferred_repo_signals"), dict):
			activation_signals = dict(activation.get("inferred_repo_signals") or {})

	# Prefer LLM-provided normalized languages/signals; fall back only if absent.
	languages = activation_languages or detect_languages(repo_root)
	signals = activation_signals or infer_repo_signals(repo_root)

	group_defs = activation_to_group_defs(modules, activation)
	# Persist activation-derived planning artifacts for partial execution / auditing.
	_write_json(work_dir / "activation_used.json", activation)
	_write_json(work_dir / "group_defs.json", [dataclasses.asdict(gd) for gd in group_defs])
	usage_tracker.flush()

	stop_after = (opts.stop_after or "").strip().lower()
	if stop_after in {"activation", "activate"}:
		return {
			"status": "stopped_after_activation",
			"work_dir": str(work_dir),
			"activation_prompt_path": str(work_dir / "activation_prompt.txt"),
			"activation": activation,
		}

	policy = AggregationPolicy(
		missing_evidence_mode="exclude",
		empty_score_mode="none",
		apply_evidence_penalty_to_score=False,
	)

	group_results: List[GroupAggregationResult] = []
	group_results_by_module: Dict[str, List[GroupAggregationResult]] = {}

	# Reuse a runtime for scoring calls (if needed).
	scoring_rt = LLMRuntime(
		llm_cfg=llm_cfg,
		provider=str(provider or "openai").strip().lower(),
		default_model=str(default_model or "").strip(),
		activation_model=str(activation_model or "").strip(),
		scoring_model=str(scoring_model or "").strip(),
	)

	# Build mapping from group_id -> rubric group (to get evidence_profile)
	rubric_group_by_id: Dict[str, RubricGroup] = {}
	for m in modules:
		for g in m.groups:
			rubric_group_by_id[g.group_id] = g

	for gd in group_defs:
		# Disabled groups should be ignored: no evidence collection, no tool runs.
		if not gd.enabled:
			rubric_g = rubric_group_by_id.get(gd.group_id)
			if rubric_g is None:
				continue
			_write_json(
				group_evidence_dir / f"{gd.module_id}_{gd.group_id}_evidence.json",
				{"status": "skipped", "reason": "group_disabled", "group_id": gd.group_id, "module_id": gd.module_id},
			)
			leaf_scores = [
				LeafScore(
					leaf_id=l.leaf_id,
					status="na",
					score=None,
					confidence=1.0,
					needs_human_review=False,
					justification="Group disabled by activation gating.",
				)
				for l in rubric_g.leaves
			]
			gr = aggregate_group(group_def=gd, leaf_scores=leaf_scores, policy=policy)
			group_results.append(gr)
			group_results_by_module.setdefault(gd.module_id, []).append(gr)
			continue

		rubric_g = rubric_group_by_id.get(gd.group_id)
		if rubric_g is None:
			continue

		profile_name = rubric_g.evidence_profile
		profile = profiles.get(profile_name) if profile_name else None
		evidence_bundle: Dict[str, Any] = {}
		if profile:
			evidence_bundle = collect_evidence_bundle(
				repo_root=repo_root,
				profile=profile,
				languages=languages,
				tool_out_dir=tool_out_dir,
				run_tools=bool(opts.run_tools),
				simal_evidence=simal_evidence,
				simal_schema_path=simal_schema_path,
				evidence_hints=list(gd.evidence_hints or []),
				allow_simal=not simal_disabled,
			)

		# Inject activation-driven evidence hints (if any).
		hint_evidence = collect_evidence_from_activation_hints(repo_root=repo_root, evidence_hints=list(gd.evidence_hints or []))
		if hint_evidence.get("status") != "empty":
			evidence_bundle = _merge_dicts(evidence_bundle, {"activation_hint_evidence": hint_evidence})

		# Persist evidence for audit/debug
		_write_json(group_evidence_dir / f"{gd.module_id}_{gd.group_id}_evidence.json", evidence_bundle)

		if stop_after in {"tools", "evidence", "tool"}:
			# Stop after evidence collection + (optional) dynamic tool execution.
			continue

		# Create scoring prompt (even if deterministic; useful for audit)
		leaves = [{"leaf_id": l.leaf_id, "question": l.question} for l in rubric_g.leaves]
		include_simal_for_scoring = bool(rubric_g.include_simal_schema_in_scoring) and not simal_disabled
		scoring_simal_schema_text = simal_schema_text if include_simal_for_scoring else ""
		scoring_prompt = build_group_scoring_prompt(
			ScoringPromptInputs(
				repo_name=repo_root.name,
				review_scope=opts.review_scope,
				module_id=gd.module_id,
				module_name=gd.module_name,
				group_id=gd.group_id,
				group_name=gd.group_name,
				simal_schema_text=scoring_simal_schema_text,
				leaves=leaves,
				evidence_profile_name=profile_name,
				grading_mode=profile.grading_mode if profile else "llm_assisted",
				evidence_bundle=evidence_bundle,
					activation_context=build_group_activation_context(gd, activation),
			)
		)
		if prompts_dir:
			_write_text(prompts_dir / f"scoring_prompt_{gd.module_id}_{gd.group_id}.txt", scoring_prompt)

		# Determine leaf scores
		leaf_scores: List[LeafScore]
		external_score_path = (group_scores_dir / f"{gd.module_id}_{gd.group_id}.json") if group_scores_dir else None
		if external_score_path and external_score_path.exists():
			leaf_scores = parse_group_scoring_json(json.loads(_read_text(external_score_path)))
		else:
			if profile and profile.grading_mode == "deterministic":
				leaf_scores = deterministic_group_score(rubric_g, profile_name, evidence_bundle)
			else:
				try:
					score_obj, score_usage = scoring_rt.call_json_cached_with_usage(
						cache_dir=llm_cache_dir,
						kind=f"group_{gd.module_id}_{gd.group_id}",
						phase="scoring",
						prompt=scoring_prompt,
						outputs_dir=llm_outputs_dir,
					)
					usage_tracker.add(stage="scoring", usage=score_usage)
					usage_tracker.flush()
					llm_outputs_dir.mkdir(parents=True, exist_ok=True)
					_write_json(llm_outputs_dir / f"group_score_{gd.module_id}_{gd.group_id}.json", score_obj)
					leaf_scores = parse_group_scoring_json(score_obj)
				except Exception as e:
					leaf_scores = [
						LeafScore(
							leaf_id=l.leaf_id,
							status="tool_failed",
							score=None,
							confidence=0.0,
							needs_human_review=True,
							justification=f"LLM scoring failed: {e}",
						)
						for l in rubric_g.leaves
					]

		gr = aggregate_group(
			group_def=gd,
			leaf_scores=leaf_scores,
			policy=policy,
		)
		group_results.append(gr)
		group_results_by_module.setdefault(gd.module_id, []).append(gr)

	# If we're stopping after tools, we haven't produced scores/aggregation.
	if stop_after in {"tools", "evidence", "tool"}:
		usage_tracker.flush()
		return {
			"status": "stopped_after_tools",
			"work_dir": str(work_dir),
			"activation": activation,
			"group_evidence_dir": str(group_evidence_dir),
		}

	if stop_after in {"scoring", "score"}:
		# Persist per-group aggregation results for convenience, but skip module/final report.
		try:
			_write_json(work_dir / "group_results.json", [dataclasses.asdict(x) for x in group_results])
		except Exception:
			# If the dataclasses contain non-serializable fields, fall back to repr.
			_write_json(work_dir / "group_results.json", [str(x) for x in group_results])
		usage_tracker.flush()
		return {
			"status": "stopped_after_scoring",
			"work_dir": str(work_dir),
			"activation": activation,
			"group_results_path": str(work_dir / "group_results.json"),
		}

	# Module aggregation
	module_results: List[ModuleAggregationResult] = []
	for m in modules:
		# module enabled if any group definition says enabled (coarse) OR activation says module enabled
		module_enabled = any(gd.enabled for gd in group_defs if gd.module_id == m.module_id)
		module_group_defs = [gd for gd in group_defs if gd.module_id == m.module_id]
		module_group_results = group_results_by_module.get(m.module_id, [])
		mr = aggregate_module(
			module_id=m.module_id,
			module_name=m.module_name,
			module_weight=m.module_weight,
			group_defs=module_group_defs,
			group_results=module_group_results,
			policy=policy,
			module_enabled=module_enabled,
		)
		module_results.append(mr)

	final = aggregate_final(module_results, policy)
	report = {
		"final": dataclasses.asdict(final),
		"report_payload": build_report_payload(final),
		"activation": activation,
		"simal_disabled": simal_disabled,
		"token_usage_path": str(token_usage_path),
		"token_usage_total": dict(usage_tracker.data.get("total") or {}),
		"repo": {
			"path": str(repo_root),
			"name": repo_root.name,
			"detected_languages": languages,
			"signals": signals,
		},
		"work_dir": str(work_dir),
	}
	usage_tracker.flush()
	_write_json(out_path, report)
	return report


def main(argv: Optional[List[str]] = None) -> int:
	ap = argparse.ArgumentParser(description="SAGES: software quality evaluation pipeline")
	ap.add_argument("--repo", type=str, required=True, help="Path to repository to evaluate")
	ap.add_argument("--rubric", type=str, default=str(Path(__file__).with_name("nodes.yaml")), help="Rubric DAG YAML")
	ap.add_argument("--profiles", type=str, default=str(Path(__file__).with_name("profiles.yaml")), help="Evidence profiles YAML")
	ap.add_argument(
		"--disable-simal",
		action="store_true",
		help="Universal no-SiMAL mode: ignore SiMAL schema/JSON inputs, force heuristic activation unless activation_json is supplied, skip simal_* evidence retrieval, and omit SiMAL schema from scoring prompts.",
	)
	ap.add_argument("--simal-schema", type=str, default=None, help="Optional SiMAL schema text file")
	ap.add_argument("--simal-json", type=str, default=None, help="Optional precomputed SiMAL evidence JSON")
	ap.add_argument("--activation-json", type=str, default=None, help="Optional activation JSON output (LLM or manual)")
	ap.add_argument(
		"--heuristic-activation",
		action="store_true",
		help="Fallback activation only: infer languages+repo_signals from repo files and evaluate applies_if heuristics.",
	)
	ap.add_argument("--group-scores-dir", type=str, default=None, help="Optional directory containing group scoring JSONs")
	ap.add_argument("--emit-prompts-dir", type=str, default=None, help="If set, write activation+scoring prompts here")
	ap.add_argument(
		"--llm-provider",
		type=str,
		default=os.getenv("SAGES_LLM_PROVIDER", "openai"),
		help="LLM provider for activation/scoring: openai|gemini|claude (default: openai)",
	)
	ap.add_argument(
		"--llm-config",
		type=str,
		default=str(Path(__file__).with_name("llm_config.yaml")),
		help="Path to YAML config for LLM providers (default: llm_config.yaml next to main.py)",
	)
	ap.add_argument(
		"--llm-model",
		type=str,
		default=os.getenv("SAGES_LLM_MODEL", ""),
		help="Default LLM model name (used for activation+scoring unless overridden)",
	)
	ap.add_argument(
		"--activation-model",
		type=str,
		default=os.getenv("SAGES_ACTIVATION_MODEL", ""),
		help="Optional model override for activation call",
	)
	ap.add_argument(
		"--scoring-model",
		type=str,
		default=os.getenv("SAGES_SCORING_MODEL", ""),
		help="Optional model override for group scoring calls",
	)
	ap.add_argument("--run-tools", action="store_true", help="Execute dynamic tools (SAST/SCA/secrets/etc.)")
	ap.add_argument(
		"--stop-after",
		type=str,
		default="",
		help="Stop pipeline after stage: activation|tools|scoring (writes intermediate outputs into *_work/)",
	)
	ap.add_argument("--out", type=str, default="quality_report.json", help="Output JSON report path")
	ap.add_argument("--review-scope", type=str, default="full_repo", help="Review scope string for prompts")
	args = ap.parse_args(argv)
	try:
		run_pipeline(
			PipelineOptions(
				repo=args.repo,
				rubric=args.rubric,
				profiles=args.profiles,
				disable_simal=bool(args.disable_simal),
				simal_schema=args.simal_schema,
				simal_json=args.simal_json,
				activation_json=args.activation_json,
				heuristic_activation=bool(args.heuristic_activation),
				group_scores_dir=args.group_scores_dir,
				emit_prompts_dir=args.emit_prompts_dir,
				llm_provider=args.llm_provider,
				llm_config=args.llm_config,
				llm_model=args.llm_model,
				activation_model=args.activation_model,
				scoring_model=args.scoring_model,
				run_tools=bool(args.run_tools),
				stop_after=args.stop_after,
				out=args.out,
				review_scope=args.review_scope,
			)
		)
		return 0
	except Exception as e:
		raise SystemExit(str(e))


if __name__ == "__main__":
	raise SystemExit(main())
