from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Deterministic candidates (policy knobs)
NODE_MAJORS = [22, 20, 18, 16, 14, 12, 10]
PHP_MINORS = ["8.3", "8.2", "8.1", "8.0", "7.4", "7.3", "7.2", "7.1", "7.0", "5.6"]
DEFAULT_PYTHON_MINOR = "3.10"

def read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def scan_project_files(repo: Path) -> Dict[str, List[Path]]:
    """Recursively scans for relevant config files, ignoring dependency directories,
    and sorts them by depth (shallowest/root-level first)."""
    TARGETS = {
        ".python-version", ".tool-versions", "runtime.txt", "pyproject.toml", "setup.cfg", "requirements.txt", "setup.py", "Pipfile", "tox.ini",
        ".nvmrc", ".node-version", "package.json", "yarn.lock", "package-lock.json", "pnpm-lock.yaml", "tsconfig.json",
        "go.mod", "composer.json"
    }
    IGNORE_DIRS = {".git", "node_modules", "vendor", ".venv", "venv", "dist", "build", ".terraform"}

    found = {t: [] for t in TARGETS}

    for root, dirs, files in os.walk(repo):
        # Modify dirs in-place to prevent os.walk from entering ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        root_path = Path(root)
        for f in files:
            if f in TARGETS:
                found[f].append(root_path / f)

    # Sort each list by depth (number of path parts) to prioritize root files
    for k in found:
        found[k].sort(key=lambda p: len(p.parts))

    return found

def get_tool_versions(files_dict: Dict[str, List[Path]]) -> Dict[str, str]:
    """Parses the shallowest .tool-versions file found."""
    out: Dict[str, str] = {}
    for p in files_dict.get(".tool-versions", []):
        txt = read_text(p)
        if not txt:
            continue
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in out:
                out[parts[0].strip()] = parts[1].strip().lstrip("v")
        # Return after successfully parsing the highest level .tool-versions
        if out: 
            return out
    return out

def pick_python_minor(files_dict: Dict[str, List[Path]]) -> Tuple[Optional[str], str, Optional[str]]:
    def _extract_mm(s: str) -> Optional[str]:
        m = re.search(r"(\d+\.\d+)", s)
        return m.group(1) if m else None

    for p in files_dict.get(".python-version", []):
        txt = read_text(p)
        if txt and (mm := _extract_mm(txt.strip())):
            return mm, f"{p.name}", txt.strip()

    tv = get_tool_versions(files_dict)
    if "python" in tv and (mm := _extract_mm(tv["python"])):
        return mm, ".tool-versions:python", tv["python"]

    for p in files_dict.get("runtime.txt", []):
        txt = read_text(p)
        if txt and txt.strip().startswith("python-") and (mm := _extract_mm(txt.strip())):
            return mm, f"{p.name}", txt.strip()

    for p in files_dict.get("pyproject.toml", []):
        txt = read_text(p)
        if txt:
            m = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', txt)
            if m and (mm := _extract_mm(m.group(1))):
                return mm, f"{p.name}:requires-python", m.group(1)

    for p in files_dict.get("setup.cfg", []):
        txt = read_text(p)
        if txt:
            m = re.search(r'python_requires\s*=\s*(.*)', txt)
            if m and (mm := _extract_mm(m.group(1).strip())):
                return mm, f"{p.name}:python_requires", m.group(1).strip()

    # Hints fallback
    python_hints = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile", "tox.ini"]
    found_hints = [h for h in python_hints if files_dict.get(h)]
    if found_hints:
        return DEFAULT_PYTHON_MINOR, f"python_hints:{','.join(found_hints)} (default)", None

    return None, "no_python_version_found", None

def pick_node_major(files_dict: Dict[str, List[Path]]) -> Tuple[Optional[int], str, Optional[str]]:
    DEFAULT_NODE_MAJOR = 20

    def _major_from(s: str) -> Optional[int]:
        m = re.search(r"(?<!\d)(\d{1,3})(?!\d)", s.strip())
        return int(m.group(1)) if m else None

    for fname in (".nvmrc", ".node-version"):
        for p in files_dict.get(fname, []):
            txt = read_text(p)
            if txt and (maj := _major_from(txt.strip().lstrip("v"))):
                return maj, f"{p.name}", txt.strip()

    tv = get_tool_versions(files_dict)
    for key in ("nodejs", "node"):
        if key in tv and (maj := _major_from(tv[key].lstrip("v"))):
            return maj, f".tool-versions:{key}", tv[key]

    for p in files_dict.get("package.json", []):
        txt = read_text(p)
        if txt:
            try:
                data = json.loads(txt)
                eng = (data.get("engines") or {}).get("node")
                if isinstance(eng, str) and eng.strip():
                    eng = eng.strip()
                    m_ge = re.search(r">=\s*(\d+)", eng)
                    if m_ge:
                        return int(m_ge.group(1)), f"{p.name}:engines.node (>=)", eng
                    majors = sorted({int(x) for x in re.findall(r"(\d+)(?:\.\d+){0,2}", eng)})
                    if majors:
                        return majors[0], f"{p.name}:engines.node (major mentioned)", eng
                    return DEFAULT_NODE_MAJOR, f"{p.name}:engines.node (unparsed -> default)", eng
            except Exception:
                pass # Try the next package.json if this one fails to parse

    node_hints = ["yarn.lock", "package-lock.json", "pnpm-lock.yaml", "tsconfig.json", "package.json"]
    found_hints = [h for h in node_hints if files_dict.get(h)]
    if found_hints:
        return DEFAULT_NODE_MAJOR, f"node_hints:{','.join(found_hints)} (default)", None

    return None, "no_node_version_found", None

def pick_go_version(files_dict: Dict[str, List[Path]]) -> Tuple[Optional[str], str, Optional[str]]:
    for p in files_dict.get("go.mod", []):
        txt = read_text(p)
        if not txt: continue

        m = re.search(r"(?m)^\s*toolchain\s+go(\d+\.\d+(?:\.\d+)?)\s*$", txt)
        if m: return m.group(1), f"{p.name}:toolchain", m.group(1)

        m = re.search(r"(?m)^\s*go\s+(\d+\.\d+(?:\.\d+)?)\s*$", txt)
        if m:
            v = m.group(1)
            return ".".join(v.split(".")[:2]), f"{p.name}:go", v

    return None, "no_go.mod", None

def pick_php_minor(files_dict: Dict[str, List[Path]]) -> Tuple[Optional[str], str, Optional[str]]:
    for p in files_dict.get("composer.json", []):
        txt = read_text(p)
        if not txt: continue
        try:
            data = json.loads(txt)
            
            platform_php = (((data.get("config") or {}).get("platform") or {}).get("php"))
            if isinstance(platform_php, str) and platform_php.strip():
                v = platform_php.strip().lstrip("v")
                return ".".join(v.split(".")[:2]), f"{p.name}:config.platform.php", platform_php.strip()

            req_php = ((data.get("require") or {}).get("php"))
            if isinstance(req_php, str) and req_php.strip():
                mentioned_mm = re.findall(r"(\d+\.\d+)", req_php)
                if mentioned_mm:
                    mentioned_mm_sorted = sorted(set(mentioned_mm), key=lambda s: (int(s.split(".")[0]), int(s.split(".")[1])))
                    for mm in mentioned_mm_sorted:
                        if mm in PHP_MINORS:
                            return mm, f"{p.name}:require.php (heuristic)", req_php.strip()
                
                majors = sorted({int(x) for x in re.findall(r"(\d+)\.", req_php)})
                for maj in majors:
                    for mm in PHP_MINORS:
                        if int(mm.split(".")[0]) == maj:
                            return mm, f"{p.name}:require.php (major fallback)", req_php.strip()
        except Exception:
            pass

    tv = get_tool_versions(files_dict)
    if "php" in tv:
        return ".".join(tv["php"].split(".")[:2]), ".tool-versions:php", tv["php"]

    return None, "no_php_version_found", None

def find_docker_and_k8s(repo: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    docker_images = []
    k8s_files = []
    IGNORE_DIRS = {".git", "node_modules", "vendor", ".venv", "venv", "dist", "build", ".terraform"}

    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            p = Path(root) / f
            # Docker parsing
            if f == "Dockerfile" or f.startswith("Dockerfile."):
                txt = read_text(p)
                if txt:
                    for line in txt.splitlines():
                        m = re.match(r"^FROM\s+([^\s]+)", line.strip(), re.IGNORECASE)
                        if m: docker_images.append({"dockerfile": str(p), "from": m.group(1)})
            
            # K8s parsing
            if f.endswith((".yml", ".yaml")):
                txt = read_text(p)
                if txt and ("apiVersion:" in txt) and ("kind:" in txt):
                    k8s_files.append(str(p))

    # De-dup docker images
    uniq_docker = []
    seen = set()
    for x in docker_images:
        key = (x["dockerfile"], x["from"])
        if key not in seen:
            uniq_docker.append(x)
            seen.add(key)

    return uniq_docker, sorted(k8s_files)

def main() -> None:
    repo = Path(os.environ.get("REPO_DIR", ".")).resolve()
    
    # 1. Scan and index all relevant files deeply
    files_dict = scan_project_files(repo)

    # 2. Extract versions using the indexed files
    node_major, node_ev, node_raw = pick_node_major(files_dict)
    go_ver, go_ev, go_raw = pick_go_version(files_dict)
    php_mm, php_ev, php_raw = pick_php_minor(files_dict)
    python_mm, python_ev, python_raw = pick_python_minor(files_dict)

    # 3. Handle Docker and K8s in a single pass
    docker_images, k8s_files = find_docker_and_k8s(repo)

    out = {
        "repo_dir": str(repo),
        "js_ts": {"node_major": node_major, "evidence": node_ev, "raw": node_raw},
        "go": {"version": go_ver, "evidence": go_ev, "raw": go_raw},
        "php": {"minor": php_mm, "evidence": php_ev, "raw": php_raw},
        "python": {"minor": python_mm, "evidence": python_ev, "raw": python_raw},
        "docker": {"base_images": docker_images},
        "kubernetes": {"manifest_files": k8s_files},
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
