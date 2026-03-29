from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Tuple


LeafStatus = Literal["scored", "na", "insufficient_evidence", "tool_failed"]


# ============================================================
# Data models (runtime-normalized)
# ============================================================

@dataclass
class LeafDefinition:
    leaf_id: str
    question: str
    # Absolute weight inside its group (recommended to precompute before scoring)
    # If None, caller can provide equal-weight leaves and normalize later.
    weight: Optional[float] = None


@dataclass
class LeafScore:
    leaf_id: str
    status: LeafStatus
    score: Optional[float]  # expected 0..5 when status="scored"
    confidence: Optional[float] = None  # 0..1
    needs_human_review: bool = False
    citations: List[str] = field(default_factory=list)
    justification: str = ""
    remediation: List[str] = field(default_factory=list)

    # Optional structured evidence summary for report stats
    evidence_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupDefinition:
    module_id: str
    module_name: str
    group_id: str
    group_name: str
    module_weight: float
    group_weight: float
    leaves: List[LeafDefinition]

    # Activation information (from top-down gating)
    enabled: bool = True
    activation_confidence: float = 1.0
    activation_reason: str = ""

    # Optional per-group evidence hints from activation output
    # (Used to drive targeted retrieval for group scoring prompts)
    evidence_hints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GroupAggregationResult:
    module_id: str
    group_id: str
    enabled: bool

    # Scores are normalized to score_scale_max domain (e.g. 0..5)
    score: Optional[float]
    confidence: float

    # Diagnostics
    scored_leaf_count: int
    na_leaf_count: int
    insufficient_evidence_count: int
    tool_failed_count: int
    needs_human_review_count: int

    effective_weight_sum: float
    coverage_ratio: float  # scored/applicable(non-na)
    evidence_sufficiency_ratio: float  # scored/total leaves

    leaf_results: List[LeafScore]
    notes: List[str] = field(default_factory=list)


@dataclass
class ModuleAggregationResult:
    module_id: str
    module_name: str
    enabled: bool
    score: Optional[float]
    confidence: float

    module_weight: float
    effective_group_weight_sum: float
    coverage_ratio: float
    evidence_sufficiency_ratio: float

    group_results: List[GroupAggregationResult]
    notes: List[str] = field(default_factory=list)


@dataclass
class FinalAggregationResult:
    score: Optional[float]
    confidence: float

    score_scale_min: float
    score_scale_max: float

    effective_module_weight_sum: float
    module_results: List[ModuleAggregationResult]

    summary: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


# ============================================================
# Aggregation policy
# ============================================================

@dataclass
class AggregationPolicy:
    score_scale_min: float = 0.0
    score_scale_max: float = 5.0

    # How to treat leafs with insufficient evidence/tool failures
    # "exclude" => remove from numerator/denominator like unavailable observations
    # "penalize_neutral" => include with neutral score and low confidence
    # "penalize_low" => include with fixed low score (conservative)
    missing_evidence_mode: Literal["exclude", "penalize_neutral", "penalize_low"] = "exclude"

    neutral_score: float = 2.5
    low_penalty_score: float = 1.5

    # Confidence fallback if missing in leaf result
    default_leaf_confidence_if_missing: float = 0.5

    # Group/module confidence penalties
    apply_coverage_penalty_to_confidence: bool = True
    apply_human_review_penalty_to_confidence: bool = True
    human_review_penalty_factor: float = 0.15  # multiplied by fraction of leaves needing review

    # Optional score penalty for low evidence sufficiency
    apply_evidence_penalty_to_score: bool = False
    max_evidence_penalty_points: float = 0.5  # up to this many points on 0..5 scale

    # If a group/module has no scored evidence, emit None score or fallback
    empty_score_mode: Literal["none", "neutral"] = "none"

    # Minimum confidence floor after penalties
    confidence_floor: float = 0.05

    # Numeric sanity
    epsilon: float = 1e-9


# ============================================================
# Core aggregation helpers
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _weighted_mean(items: List[Tuple[float, float]], eps: float = 1e-9) -> Optional[float]:
    """
    items: list of (value, weight)
    Returns None if no positive total weight.
    """
    total_w = 0.0
    total_v = 0.0
    for v, w in items:
        if w is None or w <= 0:
            continue
        total_w += w
        total_v += v * w
    if total_w <= eps:
        return None
    return total_v / total_w


def _normalize_leaf_weights(leaves: List[LeafDefinition]) -> Dict[str, float]:
    """
    If leaf weights are missing, use equal weights.
    If provided, normalize among all leaves in the group to sum to 1.
    """
    if not leaves:
        return {}

    if all(l.weight is None for l in leaves):
        w = 1.0 / len(leaves)
        return {l.leaf_id: w for l in leaves}

    weights = []
    for l in leaves:
        raw = 0.0 if l.weight is None else float(l.weight)
        weights.append((l.leaf_id, max(0.0, raw)))

    s = sum(w for _, w in weights)
    if s <= 0:
        # fallback equal
        w = 1.0 / len(leaves)
        return {l.leaf_id: w for l in leaves}
    return {leaf_id: w / s for leaf_id, w in weights}


def _resolve_missing_leaf_for_scoring(
    leaf: LeafScore,
    policy: AggregationPolicy,
) -> Optional[Tuple[float, float]]:
    """
    Returns (score, confidence) surrogate if missing evidence mode injects a value,
    otherwise None to exclude from scoring.
    """
    if leaf.status == "scored" and leaf.score is not None:
        c = leaf.confidence if leaf.confidence is not None else policy.default_leaf_confidence_if_missing
        return (float(leaf.score), _clamp(float(c), 0.0, 1.0))

    if leaf.status in ("insufficient_evidence", "tool_failed"):
        if policy.missing_evidence_mode == "exclude":
            return None
        if policy.missing_evidence_mode == "penalize_neutral":
            return (policy.neutral_score, 0.25)
        if policy.missing_evidence_mode == "penalize_low":
            return (policy.low_penalty_score, 0.25)

    # na is always excluded from scoring
    return None


# ============================================================
# Group aggregation (leaf -> group)
# ============================================================

def aggregate_group(
    group_def: GroupDefinition,
    leaf_scores: List[LeafScore],
    policy: AggregationPolicy,
) -> GroupAggregationResult:
    leaf_score_map = {ls.leaf_id: ls for ls in leaf_scores}
    leaf_weights = _normalize_leaf_weights(group_def.leaves)

    scored_leaf_count = 0
    na_leaf_count = 0
    insufficient_evidence_count = 0
    tool_failed_count = 0
    needs_human_review_count = 0
    notes: List[str] = []

    # Applicable leaves = all non-NA leaves
    applicable_leafs = 0

    score_terms: List[Tuple[float, float]] = []  # (score, effective_weight)
    conf_terms: List[Tuple[float, float]] = []  # (leaf_confidence, effective_weight)

    missing_leaf_ids = []

    for leaf_def in group_def.leaves:
        ls = leaf_score_map.get(leaf_def.leaf_id)
        if ls is None:
            missing_leaf_ids.append(leaf_def.leaf_id)
            # treat missing result as tool_failed-like gap
            ls = LeafScore(
                leaf_id=leaf_def.leaf_id,
                status="insufficient_evidence",
                score=None,
                confidence=0.0,
                needs_human_review=True,
                justification="Leaf score missing from scorer output.",
            )

        if ls.needs_human_review:
            needs_human_review_count += 1

        if ls.status == "na":
            na_leaf_count += 1
        else:
            applicable_leafs += 1

        if ls.status == "scored":
            scored_leaf_count += 1
        elif ls.status == "insufficient_evidence":
            insufficient_evidence_count += 1
        elif ls.status == "tool_failed":
            tool_failed_count += 1

        surrogate = _resolve_missing_leaf_for_scoring(ls, policy)
        if surrogate is None:
            continue

        value, conf = surrogate
        # clamp score and confidence
        value = _clamp(value, policy.score_scale_min, policy.score_scale_max)
        conf = _clamp(conf, 0.0, 1.0)

        w = leaf_weights.get(leaf_def.leaf_id, 0.0)
        score_terms.append((value, w))
        conf_terms.append((conf, w))

    # Renormalized weighted mean over included leaves
    group_score = _weighted_mean(score_terms, eps=policy.epsilon)
    base_group_conf = _weighted_mean(conf_terms, eps=policy.epsilon)

    if group_score is None:
        if policy.empty_score_mode == "neutral":
            group_score = policy.neutral_score
            base_group_conf = min(base_group_conf or 0.0, 0.2)
            notes.append("No scored evidence; assigned neutral fallback due to policy.empty_score_mode=neutral.")
        else:
            notes.append("No usable scored evidence for group; score=None.")

    if base_group_conf is None:
        base_group_conf = 0.0

    total_leafs = len(group_def.leaves)
    coverage_ratio = (scored_leaf_count / applicable_leafs) if applicable_leafs > 0 else 0.0
    evidence_suff_ratio = (scored_leaf_count / total_leafs) if total_leafs > 0 else 0.0

    # Confidence penalties
    group_conf = base_group_conf

    if policy.apply_coverage_penalty_to_confidence:
        # Coverage penalty uses applicable leaves only (NA excluded)
        group_conf *= (0.5 + 0.5 * coverage_ratio) if applicable_leafs > 0 else 0.3

    if policy.apply_human_review_penalty_to_confidence and total_leafs > 0:
        frac_human = needs_human_review_count / total_leafs
        group_conf *= max(0.0, 1.0 - policy.human_review_penalty_factor * (frac_human / max(policy.epsilon, 1.0)))

    group_conf = _clamp(group_conf, policy.confidence_floor, 1.0) if group_def.enabled else 0.0

    # Optional score penalty for weak evidence sufficiency
    if group_score is not None and policy.apply_evidence_penalty_to_score:
        penalty_strength = 1.0 - evidence_suff_ratio  # more missing evidence => larger penalty
        penalty = penalty_strength * policy.max_evidence_penalty_points
        group_score = _clamp(group_score - penalty, policy.score_scale_min, policy.score_scale_max)
        notes.append(f"Applied evidence sufficiency score penalty: -{penalty:.3f}.")

    if missing_leaf_ids:
        notes.append(f"Missing scorer outputs for leaves: {missing_leaf_ids}")

    effective_weight_sum = sum(w for _, w in score_terms)  # before renorm helper, diagnostic only

    return GroupAggregationResult(
        module_id=group_def.module_id,
        group_id=group_def.group_id,
        enabled=group_def.enabled,
        score=group_score if group_def.enabled else None,
        confidence=group_conf,
        scored_leaf_count=scored_leaf_count,
        na_leaf_count=na_leaf_count,
        insufficient_evidence_count=insufficient_evidence_count,
        tool_failed_count=tool_failed_count,
        needs_human_review_count=needs_human_review_count,
        effective_weight_sum=effective_weight_sum,
        coverage_ratio=_clamp(coverage_ratio, 0.0, 1.0),
        evidence_sufficiency_ratio=_clamp(evidence_suff_ratio, 0.0, 1.0),
        leaf_results=[leaf_score_map.get(ld.leaf_id, LeafScore(ld.leaf_id, "insufficient_evidence", None)) for ld in group_def.leaves],
        notes=notes,
    )


# ============================================================
# Module aggregation (group -> module)
# ============================================================

def aggregate_module(
    module_id: str,
    module_name: str,
    module_weight: float,
    group_defs: List[GroupDefinition],
    group_results: List[GroupAggregationResult],
    policy: AggregationPolicy,
    module_enabled: bool = True,
) -> ModuleAggregationResult:
    group_result_map = {g.group_id: g for g in group_results}

    score_terms: List[Tuple[float, float]] = []
    conf_terms: List[Tuple[float, float]] = []
    notes: List[str] = []

    total_group_weight_enabled = 0.0
    scored_group_weight = 0.0

    coverage_terms = []
    evidence_terms = []

    for gd in group_defs:
        gr = group_result_map.get(gd.group_id)
        if gr is None:
            notes.append(f"Missing aggregation result for group {gd.group_id}; skipped.")
            continue
        if not gd.enabled or not gr.enabled:
            continue

        gw = max(0.0, float(gd.group_weight))
        total_group_weight_enabled += gw

        coverage_terms.append((gr.coverage_ratio, gw))
        evidence_terms.append((gr.evidence_sufficiency_ratio, gw))

        if gr.score is None:
            continue

        scored_group_weight += gw
        score_terms.append((gr.score, gw))
        conf_terms.append((gr.confidence, gw))

    # Renormalize among groups that have scores
    module_score = _weighted_mean(score_terms, eps=policy.epsilon)
    base_module_conf = _weighted_mean(conf_terms, eps=policy.epsilon)
    if base_module_conf is None:
        base_module_conf = 0.0

    if module_score is None:
        if policy.empty_score_mode == "neutral" and module_enabled:
            module_score = policy.neutral_score
            base_module_conf = min(base_module_conf, 0.2)
            notes.append("No scored groups; neutral fallback assigned.")
        else:
            notes.append("No scored groups; module score=None.")

    # Coverage and evidence ratios (weighted over enabled groups)
    module_coverage = _weighted_mean(coverage_terms, eps=policy.epsilon) if coverage_terms else 0.0
    module_evidence = _weighted_mean(evidence_terms, eps=policy.epsilon) if evidence_terms else 0.0
    module_coverage = 0.0 if module_coverage is None else _clamp(module_coverage, 0.0, 1.0)
    module_evidence = 0.0 if module_evidence is None else _clamp(module_evidence, 0.0, 1.0)

    module_conf = base_module_conf
    if policy.apply_coverage_penalty_to_confidence:
        module_conf *= (0.5 + 0.5 * module_coverage)
    module_conf = _clamp(module_conf, policy.confidence_floor, 1.0) if module_enabled else 0.0

    return ModuleAggregationResult(
        module_id=module_id,
        module_name=module_name,
        enabled=module_enabled,
        score=module_score if module_enabled else None,
        confidence=module_conf,
        module_weight=module_weight,
        effective_group_weight_sum=scored_group_weight,
        coverage_ratio=module_coverage,
        evidence_sufficiency_ratio=module_evidence,
        group_results=group_results,
        notes=notes,
    )


# ============================================================
# Final aggregation (module -> final repo score)
# ============================================================

def aggregate_final(
    module_results: List[ModuleAggregationResult],
    policy: AggregationPolicy,
) -> FinalAggregationResult:
    score_terms: List[Tuple[float, float]] = []
    conf_terms: List[Tuple[float, float]] = []
    warnings: List[str] = []

    total_enabled_module_weight = 0.0
    scored_enabled_module_weight = 0.0

    total_groups = 0
    total_leaves_scored = 0
    total_leaves_na = 0
    total_leaves_insufficient = 0
    total_leaves_tool_failed = 0
    total_leaves_human = 0

    for mr in module_results:
        if not mr.enabled:
            continue

        mw = max(0.0, float(mr.module_weight))
        total_enabled_module_weight += mw

        # stats
        for gr in mr.group_results:
            if not gr.enabled:
                continue
            total_groups += 1
            total_leaves_scored += gr.scored_leaf_count
            total_leaves_na += gr.na_leaf_count
            total_leaves_insufficient += gr.insufficient_evidence_count
            total_leaves_tool_failed += gr.tool_failed_count
            total_leaves_human += gr.needs_human_review_count

        if mr.score is None:
            continue

        scored_enabled_module_weight += mw
        score_terms.append((mr.score, mw))
        conf_terms.append((mr.confidence, mw))

    final_score = _weighted_mean(score_terms, eps=policy.epsilon)
    final_conf = _weighted_mean(conf_terms, eps=policy.epsilon)
    if final_conf is None:
        final_conf = 0.0

    if final_score is None:
        if policy.empty_score_mode == "neutral":
            final_score = policy.neutral_score
            final_conf = min(final_conf, 0.2)
            warnings.append("No scored modules; final neutral fallback assigned.")
        else:
            warnings.append("No scored modules; final score=None.")

    effective_module_weight_sum = scored_enabled_module_weight

    if total_enabled_module_weight > policy.epsilon and scored_enabled_module_weight < total_enabled_module_weight:
        warnings.append(
            f"Some enabled modules had no score and were excluded from final aggregation "
            f"(weighted coverage {scored_enabled_module_weight:.3f}/{total_enabled_module_weight:.3f})."
        )

    summary = {
        "enabled_module_count": sum(1 for m in module_results if m.enabled),
        "scored_module_count": sum(1 for m in module_results if m.enabled and m.score is not None),
        "total_enabled_group_count": total_groups,
        "leaf_counts": {
            "scored": total_leaves_scored,
            "na": total_leaves_na,
            "insufficient_evidence": total_leaves_insufficient,
            "tool_failed": total_leaves_tool_failed,
            "needs_human_review": total_leaves_human,
        },
        "weighted_module_coverage_ratio": (
            scored_enabled_module_weight / total_enabled_module_weight
            if total_enabled_module_weight > policy.epsilon else 0.0
        ),
    }

    return FinalAggregationResult(
        score=final_score,
        confidence=_clamp(final_conf, policy.confidence_floor, 1.0) if final_score is not None else 0.0,
        score_scale_min=policy.score_scale_min,
        score_scale_max=policy.score_scale_max,
        effective_module_weight_sum=effective_module_weight_sum,
        module_results=module_results,
        summary=summary,
        warnings=warnings,
    )


# ============================================================
# Optional: route gating helper (module-level hard gate)
# ============================================================

def apply_module_gates(
    module_results: List[ModuleAggregationResult],
    hard_disable_module_ids: List[str],
) -> List[ModuleAggregationResult]:
    """
    Returns updated module results with selected modules disabled (e.g., security route disabled by user config).
    """
    disabled = set(hard_disable_module_ids)
    out = []
    for mr in module_results:
        if mr.module_id in disabled:
            out.append(ModuleAggregationResult(
                module_id=mr.module_id,
                module_name=mr.module_name,
                enabled=False,
                score=None,
                confidence=0.0,
                module_weight=mr.module_weight,
                effective_group_weight_sum=0.0,
                coverage_ratio=0.0,
                evidence_sufficiency_ratio=0.0,
                group_results=[
                    GroupAggregationResult(
                        module_id=gr.module_id,
                        group_id=gr.group_id,
                        enabled=False,
                        score=None,
                        confidence=0.0,
                        scored_leaf_count=gr.scored_leaf_count,
                        na_leaf_count=gr.na_leaf_count,
                        insufficient_evidence_count=gr.insufficient_evidence_count,
                        tool_failed_count=gr.tool_failed_count,
                        needs_human_review_count=gr.needs_human_review_count,
                        effective_weight_sum=0.0,
                        coverage_ratio=gr.coverage_ratio,
                        evidence_sufficiency_ratio=gr.evidence_sufficiency_ratio,
                        leaf_results=gr.leaf_results,
                        notes=gr.notes + ["Disabled by hard module gate."],
                    )
                    for gr in mr.group_results
                ],
                notes=mr.notes + ["Module disabled by hard gate before final aggregation."],
            ))
        else:
            out.append(mr)
    return out


# ============================================================
# Optional: deterministic report payload for later LLM narration
# ============================================================

def build_report_payload(final_result: FinalAggregationResult) -> Dict[str, Any]:
    """
    Compact deterministic summary that can be fed to an LLM to generate prose report.
    Keep math separate from narration.
    """
    modules_out = []
    for m in final_result.module_results:
        modules_out.append({
            "module_id": m.module_id,
            "module_name": m.module_name,
            "enabled": m.enabled,
            "score": m.score,
            "confidence": m.confidence,
            "module_weight": m.module_weight,
            "coverage_ratio": m.coverage_ratio,
            "evidence_sufficiency_ratio": m.evidence_sufficiency_ratio,
            "notes": m.notes,
            "groups": [
                {
                    "group_id": g.group_id,
                    "enabled": g.enabled,
                    "score": g.score,
                    "confidence": g.confidence,
                    "coverage_ratio": g.coverage_ratio,
                    "evidence_sufficiency_ratio": g.evidence_sufficiency_ratio,
                    "counts": {
                        "scored": g.scored_leaf_count,
                        "na": g.na_leaf_count,
                        "insufficient_evidence": g.insufficient_evidence_count,
                        "tool_failed": g.tool_failed_count,
                        "needs_human_review": g.needs_human_review_count,
                    },
                    "notes": g.notes,
                }
                for g in m.group_results
            ],
        })

    return {
        "final_score": final_result.score,
        "final_confidence": final_result.confidence,
        "score_scale": {"min": final_result.score_scale_min, "max": final_result.score_scale_max},
        "summary": final_result.summary,
        "warnings": final_result.warnings,
        "modules": modules_out,
    }


# ============================================================
# Example usage (minimal)
# ============================================================

if __name__ == "__main__":
    policy = AggregationPolicy(
        missing_evidence_mode="exclude",
        empty_score_mode="none",
        apply_evidence_penalty_to_score=False,
    )

    group_def = GroupDefinition(
        module_id="H",
        module_name="Security & supply chain",
        group_id="H1",
        group_name="Secrets & sensitive data",
        module_weight=0.15,
        group_weight=0.25,
        leaves=[
            LeafDefinition("H1.1", "No hardcoded secrets...", weight=1.0),
            LeafDefinition("H1.2", "Sensitive values not logged...", weight=1.0),
            LeafDefinition("H1.3", "Secure secret injection...", weight=1.0),
        ],
        enabled=True,
        activation_confidence=0.95,
    )

    leaf_scores = [
        LeafScore("H1.1", "scored", 2.0, confidence=0.95, needs_human_review=False),
        LeafScore("H1.2", "insufficient_evidence", None, confidence=0.30, needs_human_review=True),
        LeafScore("H1.3", "scored", 4.0, confidence=0.75, needs_human_review=False),
    ]

    gr = aggregate_group(group_def, leaf_scores, policy)
    mr = aggregate_module(
        module_id="H",
        module_name="Security & supply chain",
        module_weight=0.15,
        group_defs=[group_def],
        group_results=[gr],
        policy=policy,
        module_enabled=True,
    )
    fr = aggregate_final([mr], policy)

    print("Group score:", gr.score, "conf:", gr.confidence)
    print("Module score:", mr.score, "conf:", mr.confidence)
    print("Final score:", fr.score, "conf:", fr.confidence)
    print("Summary:", fr.summary)
