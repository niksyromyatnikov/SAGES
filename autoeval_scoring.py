from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import re
import textwrap
from typing import Any, Dict, List, Literal, Mapping, Optional


JudgeStrategy = Literal["pipeline", "baseline", "unknown"]

SCORE_KEYS = (
    "final_score_correctness_score",
    "structural_score_correctness_score",
    "issue_correctness_score",
    "issue_coverage_score",
    "confidence_calibration_score",
    "interpretability_grounding_score",
    "activation_scope_correctness_score",
)

COUNT_KEYS = (
    "major_score_mismatches",
    "minor_score_mismatches",
    "missing_relevant_groups",
    "irrelevant_groups_included",
    "status_misclassification_errors",
    "major_missed_issues",
    "minor_missed_issues",
    "false_or_overstated_issues",
    "major_confidence_errors",
    "minor_confidence_errors",
    "unsupported_strong_claims",
)

SUBSCORE_WEIGHTS: Dict[str, int] = {
    "final_score_correctness_score": 15,
    "structural_score_correctness_score": 25,
    "issue_correctness_score": 15,
    "issue_coverage_score": 15,
    "confidence_calibration_score": 10,
    "interpretability_grounding_score": 10,
    "activation_scope_correctness_score": 10,
}

PENALTY_WEIGHTS: Dict[str, int] = {
    "major_score_mismatches": 6,
    "minor_score_mismatches": 2,
    "missing_relevant_groups": 7,
    "irrelevant_groups_included": 3,
    "status_misclassification_errors": 2,
    "major_missed_issues": 6,
    "minor_missed_issues": 2,
    "false_or_overstated_issues": 4,
    "major_confidence_errors": 4,
    "minor_confidence_errors": 1,
    "unsupported_strong_claims": 3,
}

VERDICT_THRESHOLDS = (
    (90, "excellent"),
    (75, "good"),
    (55, "fair"),
    (30, "poor"),
    (0, "unusable"),
)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class JudgePromptInputs:
    # Full judge-visible repository source content or equivalent full repo dump.
    # This is the primary source of truth for repository semantics.
    repo_source_text: str
    rubric_text: str
    candidate_assessment: Dict[str, Any]
    candidate_strategy: JudgeStrategy = "unknown"
    candidate_label: str = "candidate_assessment"
    repository_tool_evidence: Optional[Dict[str, Any] | List[Any]] = None
    extra_instructions: str = ""


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_notes(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def verdict_for_score(score: int) -> str:
    for threshold, verdict in VERDICT_THRESHOLDS:
        if score >= threshold:
            return verdict
    return "unusable"


def build_judge_json_schema() -> Dict[str, Any]:
    score_properties = {
        key: {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
        }
        for key in SCORE_KEYS
    }
    count_properties = {
        key: {
            "type": "integer",
            "minimum": 0,
        }
        for key in COUNT_KEYS
    }

    return {
        "type": "object",
        "properties": {
            "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "verdict": {
                "type": "string",
                "enum": ["excellent", "good", "fair", "poor", "unusable"],
            },
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": list(SCORE_KEYS),
                "additionalProperties": False,
            },
            "counts": {
                "type": "object",
                "properties": count_properties,
                "required": list(COUNT_KEYS),
                "additionalProperties": False,
            },
            "notes": {
                "type": "object",
                "properties": {
                    "major_issues": {"type": "array", "items": {"type": "string"}},
                    "minor_issues": {"type": "array", "items": {"type": "string"}},
                    "comparisons": {"type": "array", "items": {"type": "string"}},
                    "out_of_rubric_important_issues": {"type": "array", "items": {"type": "string"}},
                    "suggested_improvements": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "major_issues",
                    "minor_issues",
                    "comparisons",
                    "out_of_rubric_important_issues",
                    "suggested_improvements",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["overall_score", "verdict", "scores", "counts", "notes"],
        "additionalProperties": False,
    }


def compute_weighted_base(scores: Mapping[str, int]) -> int:
    total = 0.0
    for key, weight in SUBSCORE_WEIGHTS.items():
        total += (clamp(_coerce_int(scores.get(key), default=0), 0, 5) / 5.0) * weight
    return round_half_up(total)


def compute_penalty(counts: Mapping[str, int]) -> int:
    total = 0
    for key, weight in PENALTY_WEIGHTS.items():
        total += max(0, _coerce_int(counts.get(key), default=0)) * weight
    return total


def apply_cap_rules(scores: Dict[str, int], counts: Mapping[str, int]) -> Dict[str, int]:
    capped = {key: clamp(_coerce_int(value), 0, 5) for key, value in scores.items()}

    if _coerce_int(counts.get("missing_relevant_groups"), default=0) > 0:
        capped["activation_scope_correctness_score"] = min(capped["activation_scope_correctness_score"], 4)
        capped["issue_coverage_score"] = min(capped["issue_coverage_score"], 4)
    if _coerce_int(counts.get("missing_relevant_groups"), default=0) >= 2:
        capped["activation_scope_correctness_score"] = min(capped["activation_scope_correctness_score"], 2)
        capped["issue_coverage_score"] = min(capped["issue_coverage_score"], 2)
    if _coerce_int(counts.get("major_score_mismatches"), default=0) > 0:
        capped["structural_score_correctness_score"] = min(capped["structural_score_correctness_score"], 4)
    if _coerce_int(counts.get("major_score_mismatches"), default=0) >= 2:
        capped["structural_score_correctness_score"] = min(capped["structural_score_correctness_score"], 2)
    if _coerce_int(counts.get("false_or_overstated_issues"), default=0) > 0:
        capped["issue_correctness_score"] = min(capped["issue_correctness_score"], 4)
    if _coerce_int(counts.get("major_confidence_errors"), default=0) > 0:
        capped["confidence_calibration_score"] = min(capped["confidence_calibration_score"], 3)
    if _coerce_int(counts.get("unsupported_strong_claims"), default=0) > 0:
        capped["interpretability_grounding_score"] = min(capped["interpretability_grounding_score"], 4)
    return capped


def recompute_overall_score(
    scores: Mapping[str, int],
    counts: Mapping[str, int],
    apply_caps_first: bool = True,
) -> Dict[str, Any]:
    normalized_scores = {
        key: clamp(_coerce_int(scores.get(key), default=0), 0, 5)
        for key in SCORE_KEYS
    }
    normalized_counts = {
        key: max(0, _coerce_int(counts.get(key), default=0))
        for key in COUNT_KEYS
    }
    if apply_caps_first:
        normalized_scores = apply_cap_rules(normalized_scores, normalized_counts)

    weighted_base = compute_weighted_base(normalized_scores)
    penalty = compute_penalty(normalized_counts)
    overall_score = clamp(weighted_base - penalty, 0, 100)

    return {
        "scores": normalized_scores,
        "counts": normalized_counts,
        "weighted_base": weighted_base,
        "penalty": penalty,
        "overall_score": overall_score,
        "verdict": verdict_for_score(overall_score),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty judge output")

    cleaned = text.strip()
    fenced = JSON_FENCE_RE.search(cleaned)
    if fenced:
        cleaned = (fenced.group(1) or "").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in judge output")


def normalize_judge_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    notes = payload.get("notes") if isinstance(payload.get("notes"), dict) else {}

    normalized_scores = {
        key: clamp(_coerce_int(scores.get(key), default=0), 0, 5)
        for key in SCORE_KEYS
    }
    normalized_counts = {
        key: max(0, _coerce_int(counts.get(key), default=0))
        for key in COUNT_KEYS
    }

    return {
        "overall_score": clamp(_coerce_int(payload.get("overall_score"), default=0), 0, 100),
        "verdict": str(payload.get("verdict") or "unusable"),
        "scores": normalized_scores,
        "counts": normalized_counts,
        "notes": {
            "major_issues": _coerce_notes(notes.get("major_issues")),
            "minor_issues": _coerce_notes(notes.get("minor_issues")),
            "comparisons": _coerce_notes(notes.get("comparisons")),
            "out_of_rubric_important_issues": _coerce_notes(notes.get("out_of_rubric_important_issues")),
            "suggested_improvements": _coerce_notes(notes.get("suggested_improvements")),
        },
    }


def decode_and_rescore_judge_output(
    raw_output: str | Dict[str, Any],
    apply_caps_first: bool = True,
) -> Dict[str, Any]:
    parsed = extract_json_object(raw_output) if isinstance(raw_output, str) else raw_output
    normalized = normalize_judge_payload(parsed)
    recomputed = recompute_overall_score(
        normalized["scores"],
        normalized["counts"],
        apply_caps_first=apply_caps_first,
    )

    validation_issues: List[str] = []
    if normalized["overall_score"] != recomputed["overall_score"]:
        validation_issues.append(
            f"Reported overall_score={normalized['overall_score']} does not match recomputed overall_score={recomputed['overall_score']}."
        )
    if normalized["verdict"] != recomputed["verdict"]:
        validation_issues.append(
            f"Reported verdict={normalized['verdict']!r} does not match recomputed verdict={recomputed['verdict']!r}."
        )
    if normalized["scores"] != recomputed["scores"]:
        validation_issues.append("One or more subscores violated cap rules and were adjusted during recomputation.")
    comparison_count = len(normalized["notes"]["comparisons"])
    minimum_comparisons = 5 if recomputed["overall_score"] >= 90 else 3
    if comparison_count < minimum_comparisons:
        validation_issues.append(
            f"notes.comparisons has {comparison_count} entries but at least {minimum_comparisons} are required for this overall_score band."
        )

    repaired_payload = {
        "overall_score": recomputed["overall_score"],
        "verdict": recomputed["verdict"],
        "scores": recomputed["scores"],
        "counts": recomputed["counts"],
        "notes": normalized["notes"],
    }

    return {
        "reported_payload": normalized,
        "repaired_payload": repaired_payload,
        "weighted_base": recomputed["weighted_base"],
        "penalty": recomputed["penalty"],
        "validation_issues": validation_issues,
    }


def build_judge_prompt(inputs: JudgePromptInputs) -> str:
    repository_tool_evidence = inputs.repository_tool_evidence if inputs.repository_tool_evidence is not None else []

    prompt = textwrap.dedent(
        f"""
        You are an expert judge for repository-level software quality evaluation.

        You will receive:
        1. The full repository source content.
        2. The evaluation rubric.
        3. A normalized candidate assessment bundle produced by an automated evaluation strategy.
        4. Optional repository tool evidence summaries.

        Your task is to evaluate how good the candidate assessment is.

        The full repository source content is the primary source of truth.
        The rubric defines the primary evaluation scope and scoring dimensions.
        The repository content is the source of truth for what is actually present in the codebase.
        Repository tool evidence is also source-of-truth evidence for tool-observable properties when present.
        The candidate assessment is an object to be judged, not the source of truth.

        Judge the candidate assessment primarily with respect to the rubric-defined task.
        Do not strongly penalize the candidate for failing to assess issues that are clearly outside the rubric's intended scope.
        However, if the repository contains major and obvious quality issues that fall outside the rubric but materially affect the credibility of the candidate assessment, record them separately as out_of_rubric_important_issues.
        Treat such issues as secondary observations unless they directly contradict strong claims made by the candidate assessment.

        CORE JUDGMENT AXES
        - Did the candidate activate the right modules/groups for this repository?
        - Are the scores directionally and structurally reasonable given the repository evidence?
        - Are the identified issues valid and appropriately scoped?
        - Does the candidate cover the important weaknesses and strengths?
        - Is confidence calibrated to evidence quality and completeness?
        - Are claims grounded in the evidence formats actually available to the candidate?

        EVIDENCE INTERPRETATION RULES
        - Direct repository citations like "path:start-end" are the strongest support for direct implementation claims.
        - Candidate citations may include retrieval hits, activation-hint evidence, SiMAL anchors, tool references, or other internal evidence handles. Treat these as pointers or summaries, not as source of truth by themselves.
        - Normalized tool summaries are authoritative for tool-observable properties. Do not require raw tool logs if the summary is complete enough.
        - Tool failure reflects unavailable evidence, not repository weakness by itself.
        - Missing evidence in the candidate assessment is not proof of absence.
        - Absence in the provided full repository content may be treated as absence for repository-owned artifacts unless the prompt explicitly states that the repository content is partial.

        EVIDENCE FORMAT GUIDE
        - Source code citations: references such as "path:start-end" point directly to repository files and lines. These are the strongest non-tool evidence because they directly expose repository content.
        - Retrieval citations: retrieval hits, code search references, and similar handles are indexing aids that point toward repository content. Treat them as a compact route to source code, not as independent proof.
        - Activation-hint evidence: loaded files, hint-derived searches, and similar activation artifacts are targeted excerpts selected before scoring. They are derived from repository content and are useful pointers, but they are still secondary to the underlying repository files themselves.
        - SiMAL citations or schema references: SiMAL is a structural summary derived from repository content. It is useful for repository shape, component relationships, or likely subsystem boundaries, but it is not by itself proof of semantic implementation quality when direct repository evidence is missing or contradictory.
        - Tool citations: normalized tool evidence is different from the other pointer types. For tool-observable properties, treat normalized tool output as source-of-truth evidence even when the raw logs are omitted.

        CONCRETE CITATION EXAMPLES
        - Direct source citation example: "starlette/applications.py:1-42"
        - Retrieval citation examples: "retrieval.code_search.hits:file=docs/config.md", "retrieval.code_search.hits_by_file:starlette/routing.py"
        - Activation-hint citation examples: "activation_hint_evidence.loaded_files:starlette/applications.py", "activation_hint_evidence.search:query=UserService"
        - SiMAL citation examples: "simal.search hits_by_query.\"@PATH(\"", "simal.components:auth-service", "simal.configs:path=k8s/deployment.yaml"
        - Tool citation examples: "dynamic_tools.secrets_scanner", "dynamic_tools.secrets_scanner.tool_summary", "tool_summary.reports.ruff.json"
        - File-match / repo-scan reference examples: "file_matches.matched:*.env*", "repo_scan.detected_languages"

        HOW TO VERIFY CANDIDATE EVIDENCE USE
        - Except for tool citations, do not spend effort trying to reconstruct the internal provenance chain of retrieval hits, activation hints, or SiMAL references.
        - Instead, check whether the candidate's claim is actually supported by the repository source content and whether the cited evidence was interpreted accurately.
        - If a candidate cites SiMAL, retrieval, or activation hints correctly but the underlying repository content does not support the claim, treat the claim as unsupported or overstated.
        - If a candidate cites SiMAL, retrieval, or activation hints imprecisely but the underlying repository content clearly supports the same conclusion, prefer a minor grounding issue rather than a major correctness failure.
        - For tool citations, verify that the candidate interpreted the normalized tool evidence correctly and did not exaggerate the scope of the tool finding beyond what the tool actually establishes.

        HOW TO USE REPOSITORY TOOL EVIDENCE
        - Use repository tool evidence to verify real repository issues, especially scanner-detectable findings such as leaked secrets, lint failures, dependency vulnerabilities, test failures, or duplication hotspots.
        - If repository tool evidence reveals a real issue that the candidate missed, count that as an issue-coverage or issue-correctness failure as appropriate.
        - Do not inflate penalties when the tool evidence is weak, unavailable, or only loosely connected to the rubric leaf being judged.

        COUNTING AND PENALTY RULES
        - Count distinct errors, not every downstream symptom.
        - Do not double-count the same root problem across multiple counters unless the harms are clearly separate.
        - If a relevant group is missing, do not also count every omitted leaf inside that group as separate missed issues.
        - If one unsupported strong claim drives both an issue error and a grounding error, count the main issue once and use unsupported_strong_claims only for the separate grounding failure.
        - Borderline activation calls for ambiguous repository types should usually be minor, not major, when the reasoning is plausible.
        - Use status_misclassification_errors for status misuse such as using na where a group or leaf is actually relevant, using insufficient_evidence where strong evidence clearly exists, or treating tool_failed as repository weakness.

        PRACTICAL TOLERANCE GUIDANCE ON THE 0-5 QUALITY SCALE
        - A local score drift up to about 0.5 is often acceptable or at most minor when evidence is mixed or incomplete.
        - A drift between about 0.6 and 1.0 should be judged in context: treat it as minor when evidence is genuinely ambiguous, and major when the repository evidence is clear or the drift is systematic.
        - A drift greater than about 1.0 is usually major unless the candidate explicitly used insufficient_evidence or low confidence because the evidence was materially incomplete.
        - Final-score correctness should be judged against repository reality as a whole, not by comparing it mechanically to another strategy's score.

        MAJOR VS MINOR ERROR GUIDANCE
        Minor errors:
        - local score drift under ambiguous evidence,
        - a plausible but debatable activation choice,
        - a modest missed issue,
        - mild confidence miscalibration,
        - somewhat weak but still relevant citation support.

        Major errors:
        - materially missing relevant groups,
        - clearly irrelevant groups that meaningfully affect the assessment,
        - repository-important issues missed entirely,
        - false important issue claims,
        - strong claims that outrun the available evidence,
        - major overconfidence,
        - misreading normalized tool evidence,
        - serious status misuse that distorts scope, confidence, or structural interpretation.

        EVALUATION PROCEDURE
        1. Determine which modules and groups are materially relevant for the repository.
        2. Form a grounded view of the repository's main strengths, weaknesses, and uncertainty areas from the source.
        3. Compare the candidate against that grounded view.
        4. Assign subscores and counted errors.
        5. Apply cap rules.
        6. Compute overall_score deterministically.

        LEAF STATUS INTERPRETATION
        - "scored" = criterion was applicable and scored numerically.
        - "na" = criterion is genuinely not applicable.
        - "insufficient_evidence" = criterion may be relevant, but evidence is too weak or narrow.
        - "tool_failed" = required evidence acquisition failed materially; this reflects unavailable evidence, not repository weakness by itself.

        ============================================================
        SUBSCORES (all integers 0–5)

        - final_score_correctness_score
        - structural_score_correctness_score
        - issue_correctness_score
        - issue_coverage_score
        - confidence_calibration_score
        - interpretability_grounding_score
        - activation_scope_correctness_score

        0-5 rubric:
        5 = excellent
        4 = good
        3 = mixed / acceptable
        2 = weak
        1 = largely incorrect
        0 = unusable

        SUBSCORE DEFINITIONS
        - final_score_correctness_score: how well the final overall score matches repository evidence as a whole.
        - structural_score_correctness_score: how well module/group/leaf scores align with repository evidence and rubric criteria.
        - issue_correctness_score: whether identified issues are actually valid.
        - issue_coverage_score: whether important repository issues are captured.
        - confidence_calibration_score: whether confidence values are appropriate for evidence strength and completeness.
        - interpretability_grounding_score: whether the candidate assessment is understandable, well-supported, and traceable to evidence.
        - activation_scope_correctness_score: whether relevant groups are included and irrelevant groups are excluded.

        COUNTED ERRORS (non-negative integers)
        - major_score_mismatches
        - minor_score_mismatches
        - missing_relevant_groups
        - irrelevant_groups_included
        - status_misclassification_errors
        - major_missed_issues
        - minor_missed_issues
        - false_or_overstated_issues
        - major_confidence_errors
        - minor_confidence_errors
        - unsupported_strong_claims

        CAP RULES
        - If missing_relevant_groups > 0, activation_scope_correctness_score must be <= 4.
        - If missing_relevant_groups > 0, issue_coverage_score must be <= 4.
        - If missing_relevant_groups >= 2, activation_scope_correctness_score must be <= 2.
        - If missing_relevant_groups >= 2, issue_coverage_score must be <= 2.
        - If major_score_mismatches > 0, structural_score_correctness_score must be <= 4.
        - If major_score_mismatches >= 2, structural_score_correctness_score must be <= 2.
        - If false_or_overstated_issues > 0, issue_correctness_score must be <= 4.
        - If major_confidence_errors > 0, confidence_calibration_score must be <= 3.
        - If unsupported_strong_claims > 0, interpretability_grounding_score must be <= 4.

        DETERMINISTIC OVERALL SCORE
        weighted_base = round_half_up(
          (final_score_correctness_score / 5) * 15 +
          (structural_score_correctness_score / 5) * 25 +
          (issue_correctness_score / 5) * 15 +
          (issue_coverage_score / 5) * 15 +
          (confidence_calibration_score / 5) * 10 +
          (interpretability_grounding_score / 5) * 10 +
          (activation_scope_correctness_score / 5) * 10
        )

        penalty =
            6 * major_score_mismatches +
            2 * minor_score_mismatches +
            7 * missing_relevant_groups +
            3 * irrelevant_groups_included +
            2 * status_misclassification_errors +
            6 * major_missed_issues +
            2 * minor_missed_issues +
            4 * false_or_overstated_issues +
            4 * major_confidence_errors +
            1 * minor_confidence_errors +
            3 * unsupported_strong_claims

        overall_score = clamp(weighted_base - penalty, 0, 100)

        Verdict thresholds:
        - 90-100 = "excellent"
        - 75-89 = "good"
        - 55-74 = "fair"
        - 30-54 = "poor"
        - 0-29 = "unusable"

        OUTPUT FORMAT (STRICT JSON ONLY)
        {json.dumps(build_judge_json_schema(), ensure_ascii=False, indent=2)}

        OUTPUT CONSTRAINTS
        - Output JSON only.
        - Do not include markdown fences or explanatory text.
        - All scores must be integers.
        - All counts must be non-negative integers.
        - overall_score must match the deterministic formula after cap rules.
        - verdict must match the threshold mapping exactly.
        - notes.comparisons must contain at least 3 concrete comparisons, or at least 5 when overall_score >= 90.
        - notes arrays may be empty but must be present.

        INPUTS

        CANDIDATE LABEL
        {inputs.candidate_label}

        REPOSITORY TOOL EVIDENCE
        {_json(repository_tool_evidence)}

        CANDIDATE ASSESSMENT
        {_json(inputs.candidate_assessment)}

        RUBRIC
        {inputs.rubric_text}

        FULL REPOSITORY SOURCE CONTENT
        {inputs.repo_source_text}
        """
    ).strip()

    if inputs.extra_instructions.strip():
        prompt += "\n\nADDITIONAL INSTRUCTIONS\n" + inputs.extra_instructions.strip()
    prompt += "\n\nNow evaluate the candidate assessment."
    return prompt


DEFAULT_JUDGE_PROMPT = build_judge_prompt(
    JudgePromptInputs(
        repo_source_text="<REPO_SOURCE_CONTENT>",
        rubric_text="<RUBRIC_TEXT>",
        candidate_assessment={"placeholder": True},
        repository_tool_evidence=[],
    )
)
