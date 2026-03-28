from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


_LANG_EXT = {
    "python": {".py"},
    "go": {".go"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "java": {".java"},
    "kotlin": {".kt", ".kts"},
    "rust": {".rs"},
    "csharp": {".cs"},
    "php": {".php"},
    "ruby": {".rb"},
    "swift": {".swift"},
    "dart": {".dart"},
    "scala": {".scala"},
    "c": {".c", ".h"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"},
}


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


def _iter_files(repo_root: Path, max_files: int = 50_000) -> Iterable[Path]:
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
        if p.is_dir():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        count += 1
        yield p


def detect_languages(repo_root: Path, max_files: int = 8000) -> List[str]:
    """Naive fallback language detection by file extensions."""
    counts = {k: 0 for k in _LANG_EXT.keys()}
    for p in _iter_files(repo_root, max_files=max_files):
        ext = p.suffix.lower()
        for lang, exts in _LANG_EXT.items():
            if ext in exts:
                counts[lang] += 1
    langs = [k for k, v in counts.items() if v > 0]
    return sorted(langs)


def infer_repo_signals(repo_root: Path) -> Dict[str, Any]:
    """Naive fallback repo signals based on filenames/layout."""
    top = {p.name.lower() for p in repo_root.iterdir() if p.exists()}
    any_file = lambda *names: any((repo_root / n).exists() for n in names)

    has_container = any_file("Dockerfile", "docker-compose.yml", "docker-compose.yaml") or any(
        p.name.lower().startswith("dockerfile") for p in repo_root.glob("Dockerfile*")
    )
    has_k8s = (repo_root / "k8s").exists() or (repo_root / "helm").exists() or (repo_root / "charts").exists()
    has_iac = (repo_root / "terraform").exists() or any(repo_root.rglob("*.tf"))

    uses_frontend_framework = (repo_root / "package.json").exists() and any(
        (repo_root / f).exists() for f in ["vite.config.ts", "next.config.js", "next.config.ts"]
    )

    uses_api = any(repo_root.rglob("openapi*.yml")) or any(repo_root.rglob("openapi*.yaml")) or any(
        repo_root.rglob("*.graphql")
    )

    uses_db = (repo_root / "migrations").exists() or any(repo_root.rglob("*.sql"))
    uses_queue = any(
        k in _read_text(p)
        for p in [repo_root / "docker-compose.yml", repo_root / "docker-compose.yaml", repo_root / "compose.yml"]
        if p.exists()
        for k in ["kafka", "rabbitmq", "sqs", "celery", "redis"]
    )
    uses_filesystem = any(k in " ".join(top) for k in ["data", "storage", "uploads"])

    repo_kind = "unknown"
    if uses_frontend_framework:
        repo_kind = "web_frontend"
    if has_iac and not (repo_root / "src").exists() and not (repo_root / "services").exists():
        repo_kind = "infra"

    return {
        "uses_api": bool(uses_api),
        "uses_db": bool(uses_db),
        "uses_queue": bool(uses_queue),
        "uses_filesystem": bool(uses_filesystem),
        "uses_frontend_framework": bool(uses_frontend_framework),
        "uses_mobile_sdk": False,
        "uses_ml_framework": any((repo_root / x).exists() for x in ["mlruns", "wandb", "dvc.yaml"]),
        "has_container": bool(has_container),
        "has_k8s": bool(has_k8s),
        "has_iac": bool(has_iac),
        "uses_concurrency": False,
        "uses_async": False,
        "repo_kind": repo_kind,
    }
