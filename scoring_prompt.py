from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import textwrap


@dataclass
class ScoringPromptInputs:
    # Identity / context
    repo_name: Optional[str]
    review_scope: str  # e.g. "full_repo", "backend_only", "src/auth/*"

    # Activated DAG context (single group scoring unit)
    module_id: str
    module_name: str
    group_id: str
    group_name: str

    # Predefined leaves to score (fixed pool only)
    # Example item:
    # {"leaf_id": "H1.1", "question": "No hardcoded secrets/keys/tokens in repository."}
    leaves: List[Dict[str, str]]

    # Optional: full SiMAL schema text (raw) for this repo.
    # If provided, it will be embedded verbatim in the scoring prompt.
    simal_schema_text: str = ""

    # Evidence profile / strategy metadata (optional but useful for prompting)
    evidence_profile_name: Optional[str] = None
    grading_mode: str = "llm_assisted"  # deterministic | llm_assisted

    # Evidence bundles (already retrieved/executed by your pipeline)
    # Keep these compact / truncated before passing in.
    # Actual compact structure emitted by the pipeline:
    # {
    #   "simal": {
    #       ...optional structured SiMAL evidence..., 
    #       "search": {"status": "completed", "hits": [...], "hits_by_query": {...}}
    #   },
    #   "retrieval": {
    #       "code_search": {
    #           "hits": [...],
    #           "limits": {...},
    #           "hits_by_query": {...},
    #           "hits_by_file": {...}
    #       }
    #   },
    #   "dynamic_tools": {...},
    #   "file_matches": {...},
    #   "repo_scan": {...},
    #   "evidence_meta": {"schema": "EvidenceBundleMetaV1", "profile": {...}, "steps": [...]},
    #   "activation_hint_evidence": {
    #       "hints_used": [...],
    #       "loaded_files": [{"path": "...", "content": "..."}],
    #       "search": {"queries": [...], "hits": [...]}
    #   }
    # }
    evidence_bundle: Dict[str, Any] = field(default_factory=dict)

    # Optional upstream activation trace (helps scoring consistency)
    activation_context: Optional[Dict[str, Any]] = None

    # Scoring policy knobs
    score_min: int = 0
    score_max: int = 5
    use_integer_scores_only: bool = True
    require_citations: bool = True
    require_remediation: bool = True
    require_confidence: bool = True
    require_needs_human_review: bool = True

    # Behavior controls
    disallow_guessing: bool = True
    allow_na: bool = True  # N/A for truly non-applicable leafs within an active group
    require_evidence_for_high_scores: bool = True  # 4-5 require positive evidence
    strict_output_json_only: bool = True

    # Token control
    max_global_notes: int = 6
    max_leaf_justification_sentences: int = 3
    max_leaf_remediation_bullets: int = 3
    max_simal_schema_chars: int = 40_000


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _truncate_middle(text: str, max_chars: int, *, head_ratio: float = 0.7) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max(1, int(max_chars * head_ratio))
    tail_chars = max(1, max_chars - head_chars)
    if head_chars + tail_chars >= len(text):
        return text
    return (
        text[:head_chars].rstrip()
        + "\n\n... [truncated for prompt size] ...\n\n"
        + text[-tail_chars:].lstrip()
    )


def _nullable_schema(schema: Dict[str, Any], *, null_description: str) -> Dict[str, Any]:
    return {
        "anyOf": [
            schema,
            {
                "type": "null",
                "description": null_description,
            },
        ]
    }


def _has_meaningful_simal_evidence(evidence_bundle: Dict[str, Any]) -> bool:
    simal = evidence_bundle.get("simal")
    if isinstance(simal, dict):
        return any(value not in (None, "", [], {}) for value in simal.values())
    return bool(simal)


def build_group_scoring_json_schema(inputs: ScoringPromptInputs) -> Dict[str, Any]:
    """Build a strict JSON Schema wrapper for in-prompt group scoring output guidance."""

    schema_name = "".join(
        ch if ch.isalnum() else "_"
        for ch in f"group_scoring_{inputs.module_id}_{inputs.group_id}".lower()
    ).strip("_") or "group_scoring"

    leaf_ids = [str(leaf.get("leaf_id", "")).strip() for leaf in inputs.leaves if str(leaf.get("leaf_id", "")).strip()]
    score_value_schema: Dict[str, Any] = {
        "type": "integer" if inputs.use_integer_scores_only else "number",
        "minimum": inputs.score_min,
        "maximum": inputs.score_max,
        "description": (
            f"Numeric quality score for this leaf when status is 'scored'. "
            f"Must be between {inputs.score_min} and {inputs.score_max}."
        ),
    }

    leaf_properties: Dict[str, Any] = {
        "leaf_id": {
            "type": "string",
            "description": "Exact leaf ID copied from the predefined leaf input. Do not invent, rename, or reorder leaf IDs.",
        },
        "status": {
            "type": "string",
            "enum": ["scored", "na", "insufficient_evidence", "tool_failed"],
            "description": (
                "Leaf scoring disposition. Use 'scored' when evidence supports a score, 'na' when the leaf is truly not applicable, "
                "'insufficient_evidence' when the leaf is applicable but evidence is too weak or narrow, and 'tool_failed' when a required "
                "evidence source materially failed. This field controls how 'score' should be interpreted."
            ),
        },
        "score": _nullable_schema(
            score_value_schema,
            null_description="Use null when status is not 'scored'. Do not emit numeric placeholders for non-scored leaves.",
        ),
        "justification": {
            "type": "string",
            "description": (
                f"Short evidence-grounded justification for the leaf result. Keep this to at most {inputs.max_leaf_justification_sentences} "
                "sentences and avoid claims not supported by the bundle. Summarize why the chosen status and score follow from the cited evidence."
            ),
        },
    }
    leaf_required = ["leaf_id", "status", "score", "justification"]

    if leaf_ids:
        leaf_properties["leaf_id"]["enum"] = leaf_ids

    if inputs.require_citations:
        leaf_properties["citations"] = {
            "type": "array",
            "description": (
                "Internal evidence references actually used for this leaf, such as tool identifiers, retrieval hit anchors, file paths with line "
                "ranges, activation-hint evidence references, or SiMAL path/component anchors. Include only references that materially support "
                "the leaf judgment."
            ),
            "items": {
                "type": "string",
                "description": "One internal evidence reference string.",
            },
            "maxItems": 40,
        }
        leaf_required.append("citations")

    if inputs.require_remediation:
        leaf_properties["remediation"] = {
            "type": "array",
            "description": (
                "Short actionable remediation suggestions tailored to this leaf. Use concrete next steps rather than generic best-practice slogans. "
                "If the leaf is already strong or not applicable, this may be an empty array."
            ),
            "items": {
                "type": "string",
                "description": "One actionable remediation suggestion.",
            },
            "maxItems": inputs.max_leaf_remediation_bullets,
        }
        leaf_required.append("remediation")

    if inputs.require_confidence:
        leaf_properties["confidence"] = _nullable_schema(
            {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model confidence for this leaf result on a 0.0 to 1.0 scale. Lower this when evidence is partial, conflicting, indirect, or runtime-sensitive.",
            },
            null_description="Use null only if confidence truly cannot be estimated from the available evidence.",
        )
        leaf_required.append("confidence")

    if inputs.require_needs_human_review:
        leaf_properties["needs_human_review"] = {
            "type": "boolean",
            "description": "Set to true when the leaf result should be reviewed by a human due to uncertainty, conflicting evidence, runtime-only concerns, or missing validation evidence.",
        }
        leaf_required.append("needs_human_review")

    leaf_properties["evidence_summary"] = {
        "type": "object",
        "description": "Compact breakdown of the evidence driving the leaf result. Use this to separate supporting evidence, weakening evidence, and notable evidence gaps.",
        "properties": {
            "positive_signals": {
                "type": "array",
                "description": "Positive evidence that supports a stronger score or more confident result.",
                "items": {"type": "string", "description": "One positive evidence signal."},
            },
            "negative_signals": {
                "type": "array",
                "description": "Negative evidence that lowers the score or confidence.",
                "items": {"type": "string", "description": "One negative evidence signal."},
            },
            "missing_evidence": {
                "type": "array",
                "description": "Evidence that would have been useful but is missing, truncated, unavailable, or too weak to justify a stronger judgment.",
                "items": {"type": "string", "description": "One missing-evidence note."},
            },
        },
        "required": ["positive_signals", "negative_signals", "missing_evidence"],
        "additionalProperties": False,
    }
    leaf_required.append("evidence_summary")

    schema: Dict[str, Any] = {
        "type": "object",
        "description": "Strict JSON response contract for scoring one activated rubric group. Follow the required keys, field descriptions, enums, and additionalProperties=false constraints exactly.",
        "properties": {
            "module_id": {
                "type": "string",
                "description": "Exact module ID for the activated scoring unit. Copy from the provided context without modification.",
                "enum": [inputs.module_id],
            },
            "group_id": {
                "type": "string",
                "description": "Exact group ID for the activated scoring unit. Copy from the provided context without modification.",
                "enum": [inputs.group_id],
            },
            "score_scale": {
                "type": "object",
                "description": "Score scale metadata copied from the scoring policy so downstream consumers can interpret score values consistently. This is metadata, not a scored result.",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "Minimum allowed score value in this scoring run.",
                        "enum": [inputs.score_min],
                    },
                    "max": {
                        "type": "integer",
                        "description": "Maximum allowed score value in this scoring run.",
                        "enum": [inputs.score_max],
                    },
                    "type": {
                        "type": "string",
                        "description": "Whether score values must be integers or may be generic numbers.",
                        "enum": ["integer" if inputs.use_integer_scores_only else "number"],
                    },
                },
                "required": ["min", "max", "type"],
                "additionalProperties": False,
            },
            "group_scoring_results": {
                "type": "array",
                "description": "One leaf result per predefined input leaf, in the same order as provided. Every predefined leaf must appear exactly once.",
                "items": {
                    "type": "object",
                    "properties": leaf_properties,
                    "required": leaf_required,
                    "additionalProperties": False,
                },
                "minItems": len(inputs.leaves),
                "maxItems": len(inputs.leaves),
            },
            "group_level_summary": {
                "type": "object",
                "description": "Summary derived strictly from the leaf-level results for this group. Do not introduce claims that are not already supported by the leaf outputs.",
                "properties": {
                    "scored_leaf_count": {
                        "type": "integer",
                        "description": "Number of leaf results with status 'scored'.",
                        "minimum": 0,
                    },
                    "na_leaf_count": {
                        "type": "integer",
                        "description": "Number of leaf results with status 'na'.",
                        "minimum": 0,
                    },
                    "insufficient_evidence_count": {
                        "type": "integer",
                        "description": "Number of leaf results with status 'insufficient_evidence'.",
                        "minimum": 0,
                    },
                    "tool_failed_count": {
                        "type": "integer",
                        "description": "Number of leaf results with status 'tool_failed'.",
                        "minimum": 0,
                    },
                    "overall_group_confidence": {
                        "type": "number",
                        "description": "Overall confidence for the group summary on a 0.0 to 1.0 scale. This should reflect the combined strength and consistency of the leaf-level evidence.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "quality_risks": {
                        "type": "array",
                        "description": "Group-level risks already supported by the leaf results. Keep these concise and evidence-grounded.",
                        "items": {"type": "string", "description": "One group-level quality risk."},
                        "maxItems": inputs.max_global_notes,
                    },
                    "strengths": {
                        "type": "array",
                        "description": "Group-level strengths already supported by the leaf results. Keep these concise and evidence-grounded.",
                        "items": {"type": "string", "description": "One group-level strength."},
                        "maxItems": inputs.max_global_notes,
                    },
                    "notes": {
                        "type": "array",
                        "description": "Short summary notes about ambiguity, evidence coverage, confidence limits, or evidence conflicts for the group.",
                        "items": {"type": "string", "description": "One group summary note."},
                        "maxItems": inputs.max_global_notes,
                    },
                },
                "required": [
                    "scored_leaf_count",
                    "na_leaf_count",
                    "insufficient_evidence_count",
                    "tool_failed_count",
                    "overall_group_confidence",
                    "quality_risks",
                    "strengths",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["module_id", "group_id", "score_scale", "group_scoring_results", "group_level_summary"],
        "additionalProperties": False,
    }

    return {
        "name": schema_name,
        "strict": True,
        "schema": schema,
    }


def build_group_scoring_prompt(inputs: ScoringPromptInputs) -> str:
    """
    Build a prompt for scoring a single activated group (bottom-up leaf scoring).
    The model must score ONLY predefined leaves and return strict JSON.
    """

    include_simal_schema_text = bool((inputs.simal_schema_text or "").strip())
    include_simal_guidance = include_simal_schema_text or _has_meaningful_simal_evidence(inputs.evidence_bundle)
    score_type_text = "integer" if inputs.use_integer_scores_only else "number"

    high_score_rule = ""
    if inputs.require_evidence_for_high_scores:
        high_score_rule = textwrap.dedent(f"""
        - Scores {max(4, inputs.score_max-1)}..{inputs.score_max} require explicit positive evidence in the provided bundle.
        - If evidence is missing/weak, do NOT assign a high score. Use lower confidence and/or needs_human_review.
        """).strip()

    na_rule = ""
    if inputs.allow_na:
        na_rule = textwrap.dedent("""
        - You may use status="na" only if the leaf is genuinely not applicable to the current repository scope/group context.
        - Do NOT use N/A merely because evidence is missing.
        - If applicable but evidence is insufficient, use status="insufficient_evidence".
        """).strip()
    else:
        na_rule = "- Do not use N/A. If uncertain, use insufficient_evidence with conservative scoring behavior."

    guessing_rule = ""
    if inputs.disallow_guessing:
        guessing_rule = "- Do not guess implementation details that are not supported by the evidence."

    output_schema = _json(build_group_scoring_json_schema(inputs))

    output_schema_usage_notes = textwrap.dedent("""
    SCHEMA USAGE RULES:
    - The output schema below is valid JSON and is the exact output contract.
    - Do not add comments, // annotations, trailing commas, markdown fences, or explanatory text outside the final JSON response.
    - Use each field's description as normative interpretation guidance.
    - Obey enum values, required fields, nullability, array limits, and additionalProperties=false exactly.
    """).strip()

    SIMAL_CHEAT_SHEET = textwrap.dedent("""
SiMAL QUICK REFERENCE (System Modeling and Annotation Language)

What it is:
- SiMAL is a compact DSL that describes a repository/system structure (services, components, configs, APIs).
- Treat it as a structured summary of the repo, not executable code or code-level facts unless supported by snippets or tool findings.

What matters most for scoring:
1) @PATH(...) annotations and well-known config paths (Dockerfile, k8s/, helm/, terraform/)
2) api types (http/graphql/grpc/event) and endpoint keywords (route/controller/handler)
3) component kinds (database/queue/cache/job/topic)

Notes:
- The schema text can be noisy/incomplete; prefer concrete path anchors and explicit api/component fields.
""").strip()

    evidence_sources_text = (
        "SiMAL excerpts, retrieval snippets, file matches, repo scan summaries, tool outputs, and activation-hint retrieved evidence when present"
        if include_simal_guidance
        else "retrieval snippets, file matches, repo scan summaries, tool outputs, and activation-hint retrieved evidence when present"
    )

    evidence_bundle_quick_map_lines = [
        "- retrieval.code_search: repo code-search results with hits plus limits, hits_by_query, hits_by_file, and truncation metadata.",
        "- activation_hint_evidence.loaded_files: file contents loaded from activation-time evidence_hints (may be truncated).",
        "- activation_hint_evidence.search: extra code-search results driven by activation-time component/keyword hints.",
        "- file_matches: glob/path pattern coverage collected by evidence profiles, typically with matched and unmatched_patterns.",
        "- repo_scan: lightweight repo metadata and heuristics (top-level layout, detected languages, entrypoint candidates, gitignore quality, etc.).",
        "- dynamic_tools: compact scanner/tool execution summaries keyed by evidence_class or evidence_class:variant.",
        "- evidence_meta: description of which evidence steps were run, with queries/patterns/checks/tool classes and interpretation notes.",
    ]
    if include_simal_guidance:
        evidence_bundle_quick_map_lines.insert(
            0,
            "- simal: structured system/repo summary if available; may also include simal.search with query-driven snippets from the SiMAL schema file.",
        )

    simal_understanding_section = ""
    if include_simal_guidance:
        simal_understanding_section = f"""

SIMAL UNDERSTANDING:
{SIMAL_CHEAT_SHEET}
"""

    simal_schema_section = ""
    if include_simal_schema_text:
        simal_schema_text = _truncate_middle(
            inputs.simal_schema_text,
            inputs.max_simal_schema_chars,
        )
        simal_schema_section = f"""

SIMAL SCHEMA (raw, complete if provided):
```text
{simal_schema_text}
```
"""

    evidence_precedence_section = textwrap.dedent("""
    EVIDENCE PRECEDENCE:
    - Prefer tool findings and retrieved code/config snippets over summaries.
    - Prefer direct file/path/snippet evidence over SiMAL summaries.
    - Treat SiMAL as structural context, not as proof of implementation quality unless backed by direct evidence.
    - If evidence sources conflict, explain the conflict and lower confidence.
    """).strip()

    absence_handling_section = textwrap.dedent("""
    ABSENCE HANDLING:
    - Do not conclude that a capability is absent solely because no matching retrieval snippet was returned.
    - Retrieval is sampled and may be incomplete.
    - Use "insufficient_evidence" when the available evidence is too narrow to support a confident negative judgment.
    - Assign low scores only when there is affirmative negative evidence, or when expected artifacts are clearly missing from file_matches/tool outputs/docs.
    """).strip()

    runtime_verifiability_section = textwrap.dedent("""
    RUNTIME VERIFIABILITY RULE:
    - For runtime-sensitive properties (performance, graceful shutdown, retries, degradation, test speed, resource behavior), static evidence alone usually cannot justify a top score.
    - Cap such leaves at 3 unless strong supporting evidence exists in tests, CI outputs, docs, or tool findings.
    - Prefer lower confidence and needs_human_review=true in these cases.
    """).strip()

    group_summary_rules_section = textwrap.dedent("""
    GROUP SUMMARY RULES:
    - group_level_summary must be derived from leaf results only.
    - Do not add new claims not already supported by leaf-level evidence.
    - If many leaves are insufficient_evidence, state that explicitly and keep confidence low.
    """).strip()

    leaf_to_evidence_matching_section = textwrap.dedent("""
    LEAF-TO-EVIDENCE MATCHING:
    - Use only the evidence that is relevant to the current leaf.
    - Do not penalize a leaf because an unrelated evidence source is absent.
    - Example: file_matches may support documentation/config presence leaves, while tool outputs may support style/security findings, and code snippets may support semantic interpretation.
    """).strip()

    high_score_rules_section = textwrap.dedent("""
    HIGH-SCORE RULE:
    - Score 5 only when the evidence is strong, direct, and consistent.
    - Score 4 only when positive evidence clearly outweighs uncertainty.
    - If evidence is partial, sampled, or indirect, prefer 2-3.
    """).strip()

    simal_usage_rules_section = ""
    if include_simal_guidance:
        simal_usage_rules_section = textwrap.dedent("""
        SIMAL USAGE RULES:
        - Treat SiMAL as repository structure/context, not as proof of runtime behavior or implementation quality.
        - Use SiMAL primarily for:
          - locating relevant paths/components
          - understanding service/component relationships
          - identifying likely API/config/runtime areas
        - Do not assign high scores based on SiMAL alone when code snippets/tool outputs are absent.
        - If raw SiMAL conflicts with direct code or tool evidence, prefer the direct evidence.
        """).strip()

    optional_simal_usage_rules_text = f"\n\n{simal_usage_rules_section}" if simal_usage_rules_section else ""

    prompt = f"""
You are performing BOTTOM-UP SCORING for a predefined software-quality rubric group.

TASK:
- Score ONLY the provided predefined leaf questions for ONE activated group.
- Use ONLY the provided evidence bundle ({evidence_sources_text}).
- Return strict JSON.
- Do NOT generate new leaves, do not rewrite leaf text, do not modify weights.

SCORING PRINCIPLES:
1) Evidence-grounded scoring only.
2) Conservative under uncertainty.
3) Interpretability over verbosity.
4) Leaf-level scoring first (bottom-up); do not collapse directly into a vague group judgment.
5) Distinguish "not applicable" from "insufficient evidence".

STATUS DEFINITIONS:
- "scored": leaf is applicable and evidence is sufficient to assign a score.
- "na": leaf is genuinely not applicable to this repository/scope/group context.
- "insufficient_evidence": leaf seems applicable but provided evidence is not enough for confident scoring.
- "tool_failed": required evidence source failed (e.g., scanner execution/parsing failed) and this materially affects scoring.

SCORE SCALE:
- {inputs.score_min} = very poor / serious issue / absent where expected
- 1 = poor
- 2 = weak
- 3 = acceptable / mixed
- 4 = good
- {inputs.score_max} = strong / well-implemented / clearly evidenced

IMPORTANT RULES:
{guessing_rule}
{high_score_rule}
{na_rule}
- If a tool reports findings, that does not automatically imply score=0; assess severity, scope, and compensating evidence.
- If a tool is unavailable but alternative evidence exists (code snippets/docs/configs), you may still score with lower confidence.
- If a leaf asks about a property that usually requires runtime validation but only static evidence is provided, reduce confidence and consider needs_human_review=true.
- Keep justifications concise (max {inputs.max_leaf_justification_sentences} sentences).
- Remediation should be specific and actionable, not generic best-practice slogans.

CITATION RULES (internal evidence references):
- Cite the evidence items you actually used for each leaf.
- Prefer precise references (file paths, line ranges if present, tool identifiers, retrieval hit fields).
- If no usable evidence exists, cite nothing and mark insufficient_evidence (or tool_failed if appropriate).

{evidence_precedence_section}

{absence_handling_section}

{runtime_verifiability_section}

{group_summary_rules_section}

{leaf_to_evidence_matching_section}

{high_score_rules_section}{optional_simal_usage_rules_text}

{output_schema_usage_notes}

EVIDENCE BUNDLE QUICK MAP (how to interpret common sections):
{chr(10).join(evidence_bundle_quick_map_lines)}{simal_understanding_section}

INTERPRETATION HELPERS:
- evidence_meta: describes what evidence collection steps were executed for this group (queries/patterns/checks/tool classes) and includes notes about semantics.
- evidence_meta.steps[].tool_description / dynamic_tools.<key>.tool_description: short human-readable explanation of what the tool does.
- dynamic_tools.<key>.tool_summary: when present, this is a normalized summary object (schema like ToolEvidenceSummaryV1) with counts, severities, sample findings, and extra fields.
- dynamic_tools.<key>.treat_nonzero_as_findings:
    - true => a non-zero exit_code often means "findings present" rather than "tool failed".
    - false => a non-zero exit_code more likely means execution failure.

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- Include one result per input leaf (same order).
- Do not omit leaves.
- Use exact leaf IDs.
- Score must be {score_type_text} in range [{inputs.score_min}, {inputs.score_max}] when status="scored".
- If status != "scored", set score to null.
- Treat the JSON Schema below as the source of truth for keys, types, and field interpretations.
{"- JSON only, no markdown fences." if inputs.strict_output_json_only else ""}

CONTEXT:
{_json({
    "repo_name": inputs.repo_name,
    "review_scope": inputs.review_scope,
    "module": {"id": inputs.module_id, "name": inputs.module_name},
    "group": {"id": inputs.group_id, "name": inputs.group_name},
    "evidence_profile_name": inputs.evidence_profile_name,
    "grading_mode": inputs.grading_mode,
    "activation_context": inputs.activation_context,
    "scoring_policy": {
        "score_min": inputs.score_min,
        "score_max": inputs.score_max,
        "use_integer_scores_only": inputs.use_integer_scores_only,
        "max_global_notes": inputs.max_global_notes,
        "max_leaf_justification_sentences": inputs.max_leaf_justification_sentences,
        "max_leaf_remediation_bullets": inputs.max_leaf_remediation_bullets,
    },
})}{simal_schema_section}

PREDEFINED LEAVES TO SCORE (source of truth):
{_json(inputs.leaves)}

EVIDENCE BUNDLE (source of truth):
{_json(inputs.evidence_bundle)}

OUTPUT JSON SCHEMA (source of truth):
{output_schema}

FINAL REMINDER:
- Score predefined leaves only.
- Be conservative and evidence-grounded.
- Distinguish NA vs insufficient evidence.
- Return JSON only.
""".strip()

    return prompt


# ---------------------------------------------------------------------
# Optional: helper to build a compact evidence bundle before prompting
# ---------------------------------------------------------------------

def build_compact_evidence_bundle(
    simal_evidence: Optional[Dict[str, Any]] = None,
    retrieval_evidence: Optional[Dict[str, Any]] = None,
    tool_outputs: Optional[Dict[str, Any]] = None,
    file_matches: Optional[Dict[str, Any]] = None,
    repo_scan: Optional[Dict[str, Any]] = None,
    evidence_meta: Optional[Dict[str, Any]] = None,
    activation_hint_evidence: Optional[Dict[str, Any]] = None,
    max_items_per_section: int = 20,
) -> Dict[str, Any]:
    """
    Simple truncation helper to keep prompt size under control.
    Returns the same top-level bundle shape the scoring prompt documents.
    You should do smarter compression in production.
    """
    def _truncate(obj: Any) -> Any:
        if isinstance(obj, list):
            return obj[:max_items_per_section]
        if isinstance(obj, dict):
            out = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_items_per_section:
                    out["__truncated__"] = True
                    break
                out[k] = _truncate(v)
            return out
        return obj

    return {
        "simal": _truncate(simal_evidence or {}),
        "retrieval": _truncate(retrieval_evidence or {}),
        "dynamic_tools": _truncate(tool_outputs or {}),
        "file_matches": _truncate(file_matches or {}),
        "repo_scan": _truncate(repo_scan or {}),
        "evidence_meta": _truncate(evidence_meta or {}),
        "activation_hint_evidence": _truncate(activation_hint_evidence or {}),
    }


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    leaves = [
        {"leaf_id": "H1.1", "question": "No hardcoded secrets/keys/tokens in repository (or history is mitigated)."},
        {"leaf_id": "H1.2", "question": "Sensitive values are not logged (or are consistently redacted)."},
        {"leaf_id": "H1.3", "question": "Secure secret injection mechanism used (env/secret store), not plaintext config."},
    ]

    evidence_bundle = {
        "simal": {
            "components": [
                {"name": "auth-service", "path": "services/auth"},
                {"name": "api-gateway", "path": "services/api"},
            ],
            "configs": [
                {"kind": "k8s", "path": "k8s/deployment.yaml"},
                {"kind": "env_example", "path": ".env.example"},
            ],
            "search": {
                "status": "completed",
                "file": "schema.simal",
                "hits": [
                    {
                        "query": "auth-service",
                        "file": "schema.simal",
                        "snippet_start_line": 12,
                        "snippet_end_line": 24,
                        "snippet": "component auth-service @PATH(services/auth)",
                    }
                ],
            },
        },
        "retrieval": {
            "code_search": {
                "hits": [
                    {
                        "query": "password|secret|token",
                        "queries_hit": ["password|secret|token"],
                        "file": "services/auth/config.py",
                        "match_line": 14,
                        "match_lines": [14],
                        "snippet_start_line": 10,
                        "snippet_end_line": 18,
                        "snippet": "TOKEN_TTL = int(os.getenv('TOKEN_TTL', '3600'))",
                    },
                    {
                        "query": "logger",
                        "queries_hit": ["logger"],
                        "file": "services/auth/login.py",
                        "match_line": 74,
                        "match_lines": [74],
                        "snippet_start_line": 70,
                        "snippet_end_line": 78,
                        "snippet": "logger.info('login attempt', extra={'user': email})",
                    },
                ],
                "limits": {"max_files": 15, "max_snippets_total": 30},
                "hits_by_query": {"password|secret|token": 1, "logger": 1},
                "hits_by_file": {"services/auth/config.py": 1, "services/auth/login.py": 1},
                "candidate_snippets": 4,
                "returned_snippets": 2,
                "truncated": False,
            },
        },
        "dynamic_tools": {
            "secrets_scanner": {
                "tool": "gitleaks",
                "tool_description": "Scans repository content for hardcoded secrets.",
                "status": "completed",
                "exit_code": 1,
                "treat_nonzero_as_findings": True,
                "tool_summary": {
                    "findings_count": 2,
                    "findings_preview": [
                        {"file": "scripts/dev_seed.py", "rule": "generic-api-key", "line": 14},
                        {"file": "tests/fixtures/sample.env", "rule": "password", "line": 3},
                    ],
                },
            }
        },
        "file_matches": {
            "matched": {
                "*.env*": [".env.example"],
                "k8s/*.yaml": ["k8s/deployment.yaml"],
            },
            "unmatched_patterns": ["secrets.yaml"],
        },
        "repo_scan": {
            "detected_languages": ["python"],
            "top_level_entries": ["README.md", "k8s", "pyproject.toml", "services", "tests"],
            "has_tests": True,
        },
        "evidence_meta": {
            "schema": "EvidenceBundleMetaV1",
            "profile": {"name": "secrets_scan", "grading_mode": "llm_assisted"},
            "steps": [
                {"strategy": "file_match", "patterns": ["*.env*", "k8s/*.yaml"]},
                {"strategy": "code_search", "queries": ["password|secret|token", "logger"]},
                {"strategy": "dynamic_tool", "evidence_class": "secrets_scanner", "tool": "gitleaks"},
            ],
        },
        "activation_hint_evidence": {
            "status": "completed",
            "hints_used": [{"type": "path", "value": "services/auth", "priority": 9}],
            "loaded_files": [{"path": "services/auth/config.py", "content": "TOKEN_TTL = ..."}],
            "search": {"queries": ["UserService"], "hits": []},
        },
    }

    prompt = build_group_scoring_prompt(
        ScoringPromptInputs(
            repo_name="example-repo",
            review_scope="full_repo",
            module_id="H",
            module_name="Security & supply chain",
            group_id="H1",
            group_name="Secrets & sensitive data",
            leaves=leaves,
            evidence_profile_name="secrets_scan",
            grading_mode="llm_assisted",
            evidence_bundle=evidence_bundle,
            activation_context={
                "module_activation_confidence": 0.97,
                "group_activation_confidence": 0.95,
                "activation_reason": "Schema and files indicate deployable backend with configs and auth paths.",
            },
        )
    )

    print(prompt)
