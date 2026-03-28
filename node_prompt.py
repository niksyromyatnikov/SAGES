from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import textwrap


# IMPORTANT:
# Keep aligned with TOOL_REGISTRY keys.
TOOL_REGISTRY_LANG_KEYS: List[str] = [
    "python",
    "go",
    "typescript",
    "javascript",
    "php",
]


@dataclass
class ActivationPromptInputs:
    # Required
    simal_schema_text: str
    rubric_activation_yaml: str  # Activation subset only: module/group names + applies_if_hint

    # Optional context
    repo_name: Optional[str] = None
    review_scope: Optional[str] = None  # e.g. full_repo / backend_only / selected_paths
    heuristic_hints: Dict[str, Any] = field(default_factory=dict)

    # Prompt behavior
    ask_for_normalized_languages: bool = True
    ask_for_repo_signals: bool = True
    require_reasons: bool = True
    require_confidence: bool = True

    # Soft limits (not hard constraints)
    max_activated_modules: Optional[int] = None
    max_activated_groups_per_module: Optional[int] = None


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_node_activation_prompt(inputs: ActivationPromptInputs) -> str:
    """
    Builds a prompt for LLM-based semantic node activation (top-down gating).

    Key design:
    - applies_if is treated as a semantic applicability hint, NOT a strict predicate.
    - LLM resolves applicability from SiMAL schema + heuristic hints.
    - Output is strict JSON for downstream pipeline.
    """

    output_fields = []

    if inputs.ask_for_normalized_languages:
        output_fields.append(textwrap.dedent("""
        "normalized_languages": [
          // exact values ONLY from the allowed set
        ]""").strip())

    if inputs.ask_for_normalized_languages and inputs.require_confidence:
        output_fields.append(textwrap.dedent("""
        "language_confidence": {
          "<language_key>": 0.0
        }""").strip())

    if inputs.ask_for_repo_signals:
        output_fields.append(textwrap.dedent("""
        "inferred_repo_signals": {
          "uses_api": bool,
          "uses_db": bool,
          "uses_queue": bool,
          "uses_filesystem": bool,
          "uses_frontend_framework": bool,
          "uses_mobile_sdk": bool,
          "uses_ml_framework": bool,
          "has_container": bool,
          "has_k8s": bool,
          "has_iac": bool,
          "uses_concurrency": bool,
          "uses_async": bool,
          "repo_kind": "backend" | "web_frontend" | "mobile_app" | "infra" | "library" | etc.
        }""").strip())

    group_fields = ['"group_id": str', '"enabled": bool']
    if inputs.require_reasons:
        group_fields.append('"reason": str (Short schema-grounded reason)')
    if inputs.require_confidence:
        group_fields.append('"activation_confidence": float between 0.0 and 1.0')
    group_fields.append(textwrap.dedent("""
    "evidence_hints": [
      {"type":"path","value":"src/api/","priority":8},
      {"type":"component","value":"UserService","priority":7},
    ]""").strip())

    module_fields = ['"module_id": str', '"enabled": bool']
    if inputs.require_reasons:
        module_fields.append('"reason": str (Short schema-grounded reason)')
    if inputs.require_confidence:
        module_fields.append('"activation_confidence": float between 0.0 and 1.0')
    module_fields.append('"activated_groups": [ { ' + ", ".join(group_fields) + " } ]")

    output_fields.append('"module_activation": [ { ' + ", ".join(module_fields) + " } ]")
    output_fields.append(textwrap.dedent("""
    "global_notes": [
      "Ambiguity / noisy fields / uncertainty notes"
    ]""").strip())

    output_schema_text = "{\n  " + ",\n  ".join(output_fields) + "\n}"

    soft_limits = []
    if inputs.max_activated_modules is not None:
        soft_limits.append(f"- Soft limit: activate at most {inputs.max_activated_modules} modules unless evidence strongly supports more.")
    if inputs.max_activated_groups_per_module is not None:
        soft_limits.append(
            f"- Soft limit: activate at most {inputs.max_activated_groups_per_module} groups per module unless evidence strongly supports more."
        )
    soft_limits_text = "\n".join(soft_limits) if soft_limits else "- No explicit activation count limit."

    context_payload = {
        "repo_name": inputs.repo_name,
        "review_scope": inputs.review_scope,
        "heuristic_hints": inputs.heuristic_hints,
        "allowed_language_keys": TOOL_REGISTRY_LANG_KEYS,
    }

        # NEW: Signal glossary and evidence heuristics (compact but explicit)
    signal_glossary = textwrap.dedent("""
    REPO SIGNAL GLOSSARY (infer from SiMAL schema + hints; do NOT require explicit booleans):

    repo_kind (choose one):
    - "library": reusable package/SDK; signs: package metadata (pyproject/setup.cfg, package.json w/ exports, go.mod with library layout, Cargo.toml lib crate),
                public API surface, semantic versioning/changelog.
    - "web_frontend": browser UI; signs: src/components, React/Vue/Angular, bundlers (vite/webpack/next), .tsx/.jsx, public/ assets.
    - "mobile_app": Android/iOS; signs: android/, ios/, *.swift, *.kt, gradle, Xcode project, mobile SDK patterns.
    - "ml_project": training/eval pipelines; signs: notebooks, training scripts, configs, datasets, model artifacts, wandb/mlflow/dvc.
    - "infra": deployment-only or infra-as-code; signs: terraform/, helm/, k8s/, charts/, mostly YAML/Terraform with minimal app source.
    - "backend": typical server/service; signs: src/, cmd/, app/, server/, *.go/.py/.java with API patterns, Dockerfile, database access, async patterns.
    - "unknown": if mixed or unclear.

    uses_api:
    - true if schema suggests server or API endpoints: routes/controllers/handlers, OpenAPI/Swagger, GraphQL schema, gRPC proto, "server", "router".
    - false if only internal libraries/scripts with no API indicators.

    uses_db:
    - true if schema suggests DB drivers/ORM/migrations/SQL: "migrations", "sequelize", "gorm", "sqlalchemy", "jdbc", "*.sql", connection pools.
    - false if only in-memory or no persistence indicators.

    uses_queue:
    - true if schema suggests async jobs/queues/events: kafka/rabbitmq/sqs/pubsub/celery/bullmq, consumers/producers, "worker", "job".
    - false if no messaging indicators.

    uses_filesystem:
    - true if repo manages file storage/uploads/import/export/pipelines: "uploads", "storage", "file", "s3", "blob", "filesystem".
    - false if no storage indicators.

    uses_frontend_framework:
    - true if React/Vue/Angular/Svelte or frontend build chain is present (package.json + src UI patterns).
    - false otherwise.

    uses_mobile_sdk:
    - true if Android/iOS frameworks present, or mobile build configs.
    - false otherwise.

    uses_ml_framework:
    - true if ML libraries/pipelines indicated: pytorch/tensorflow/jax/sklearn, training scripts, wandb/mlflow/dvc, checkpoints.
    - false otherwise.

    has_container:
    - true if Dockerfile/docker-compose/container build artifacts present or referenced.
    - false otherwise.

    has_k8s:
    - true if k8s manifests, Helm charts, or Kubernetes deployment configs exist.
    - false otherwise.

    has_iac:
    - true if Terraform/Pulumi/CloudFormation/Ansible/etc. infra-as-code appears.
    - false otherwise.

    uses_concurrency / uses_async:
    - true if schema indicates concurrency patterns: goroutines/channels, async/await, threads/executors, background workers.
    - If unsure, set false with low confidence note.

    IMPORTANT:
    - These are semantic signals; use best-effort inference.
    - If evidence is weak or conflicting, pick the most plausible value, lower confidence, and explain in global_notes.
    """).strip()

    if not inputs.ask_for_repo_signals:
        signal_glossary = ""

    SIMAL_CHEAT_SHEET = textwrap.dedent("""
SiMAL QUICK REFERENCE (System Modeling and Annotation Language)

Purpose:
- SiMAL is a compact DSL that describes a software repository/system structure, configs, APIs, and internal components.
- You should treat it as a structured summary of a repository, not as executable code.

Core structure:
- Exactly one top-level block: `system { ... }`
- Inside `system`, there may be multiple `service <name> { ... }` blocks.
  - A library-only repo may still be represented as a single `service`.

Attributes:
- Key-value lines look like: `key: value`
- Lists look like: `key: [item1, item2, ...]`
- Nested blocks may appear as: `key: { ... }` or nested named configs: `runtime: { dev: { ... } prod: { ... } }`
- Important attributes for activation:
  - `langs:` or similar language/stack fields (may be noisy; do not trust blindly)
  - `runtime:` / `ci:` / `dependencies:` (signals for infra, build, deploy)

Annotations:
- An annotation starts with `@` and applies to the next block or list item.
- The most important one is `@PATH(<path>)` which indicates a file/folder path in the repo.
  - Use @PATH to infer languages (file extensions) and detect Docker/K8s/IaC artifacts.
- Other annotations like `@CALLS(...)` may exist; they indicate dependencies between services.

APIs and components:
- `api: [...]` may include maps like `{ type: http|grpc|graphql, endpoints: [...] }`
  - Presence of `type: http` or `type: grpc` or OpenAPI/GraphQL hints, etc. => uses_api=true.
- `components: [...]` is a list of internal components; each component is often a block:
  - Example: `database UserRepo { ... }`, `cache SessionCache { ... }`, `struct UserService { ... }`
  - Component "kind" words (database/cache/queue/topic/job/table) are strong signals for uses_db/uses_queue/etc.
- Components/structs may include `methods: [...]` and `fields: [...]` but these are not needed for activation unless they contain clear keywords (router/controller/handler).

What to rely on most:
1) `@PATH(...)` file extensions and well-known config paths (Dockerfile, k8s/, helm/, terraform/)
2) `api` types (http/graphql/grpc/event) and endpoint keywords (route/controller/handler)
3) component kinds (database/queue/cache/job/topic)
4) only then free-text `stack:` or `langs:` lists (can be noisy)

Do NOT assume SiMAL syntax is perfect:
- The schema may contain noisy free-text stack fields, mixed casing, or informal descriptions.
- Prefer evidence you can point to (paths, api types, component kinds).
""").strip()

    prompt = f"""
You are performing TOP-DOWN NODE ACTIVATION (GATING) for a predefined repository-quality rubric DAG.

IMPORTANT METHODOLOGICAL RULES:
1) DO NOT generate new questions, nodes, modules, or groups.
2) DO NOT modify weights.
3) Your task is ONLY to activate/deactivate predefined modules and groups.
4) Treat `applies_if` as a SEMANTIC APPLICABILITY HINT, not as a strict executable condition.
   - Example: `uses_api == true` means "enable if the schema strongly suggests an API backend".
   - You should infer applicability from schema text, component names, file paths, config artifacts, and heuristic hints.
5) If uncertain, use lower confidence and mention ambiguity in `global_notes`.

LANGUAGE NORMALIZATION RULES:
- Extract implementation languages and normalize them to EXACT tool-registry keys only.
- Allowed keys: {TOOL_REGISTRY_LANG_KEYS}
- Do NOT output frameworks/tools/formats as languages (e.g., React, Next.js, Docker, Kubernetes, YAML, JSON, SQL).
- If schema contains noisy stack text, infer languages from stronger signals (paths/extensions/toolchain files/components).
- If uncertain between javascript and typescript:
  - prefer "typescript" if TS evidence exists (.ts/.tsx/tsconfig),
  - else "javascript".

SIMAL UNDERSTANDING:
{SIMAL_CHEAT_SHEET}

{signal_glossary}

ACTIVATION POLICY:
- Universal/core modules usually apply to most repositories.
- Domain-specific modules should be activated only if supported by evidence/signals.
- `applies_if` hints in the rubric are guidance for semantic routing by you (LLM), not machine predicates.

EVIDENCE_HINTS FORMAT (IMPORTANT)
For each ACTIVATED group, you MUST output `evidence_hints` as a list of OBJECTS (not strings).
These hints will be used to retrieve repository evidence for the scoring stage.

Allowed hint object schema:
- type: one of ["path", "glob", "component", "keyword"]
- value: string (max 160 chars)
- priority: integer 1..10 (10 = strongest, most specific anchor)
- note: optional string (max 120 chars)

Rules:
1) Prefer "path" and "component" hints. Use "keyword" only when you cannot point to a concrete artifact.
2) Only include hints you can justify from the provided SiMAL schema or heuristic hints. Do NOT hallucinate paths/files/classes.
3) Use repo-relative paths (no absolute paths). Use forward slashes (e.g., "config/database.php").
4) For directories use trailing slash (e.g., "database/migrations/").
5) Use "glob" only when you expect many similar files (e.g., "database/migrations/*.php") and cannot list them individually.
6) "component" values must be exact names as they appear in SiMAL (case-sensitive if possible).
7) Max 10 hints per group. If unsure, output fewer hints with lower priority.
8) If you cannot provide any justified hints, output an empty list [] (do NOT fabricate).

Examples (GOOD):
- {{"type":"path","value":"config/database.php","priority":10}}
- {{"type":"path","value":"database/migrations/","priority":9}}
- {{"type":"glob","value":"k8s/**/*.yaml","priority":8}}
- {{"type":"component","value":"UserRepo","priority":7,"note":"database component"}}
- {{"type":"keyword","value":"rate limit","priority":3}}

Examples (BAD):
- "path:config/database.php"                       (strings are forbidden)
- {{"type":"path","value":"/home/user/repo/..."}} (absolute paths forbidden)
- {{"type":"class","value":"DbConfig"}}            (unknown type; use component or keyword)
- {{"type":"path","value":"some/file/that/does/not/exist","priority":10}} (hallucination)

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- Include all modules from the provided activation rubric subset.
- If a module is disabled, still include it with enabled=false and empty activated_groups.
- Do not invent IDs.
- Reasons must be concise and grounded in provided evidence.
- Confidence values must be in [0.0, 1.0].

SOFT LIMITS:
{soft_limits_text}

OUTPUT JSON SCHEMA (example shape, exact IDs depend on rubric):
{output_schema_text}

CONTEXT PAYLOAD:
{_json(context_payload)}

SiMAL SCHEMA (raw):
```text
{inputs.simal_schema_text}
```
PREDEFINED RUBRIC ACTIVATION SUBSET (source of truth for module/group IDs):
```
{inputs.rubric_activation_yaml}
```
FINAL REMINDER:
- applies_if is a semantic hint interpreted by you from the schema.
- Do not treat missing explicit booleans as absence of applicability.
- Prefer evidence-grounded activation with confidence scores.

Return JSON only.
""".strip()

    return prompt
