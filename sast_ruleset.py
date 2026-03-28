from __future__ import annotations

from typing import Any, Dict

SAST_FOCUS_MAP: Dict[str, Dict[str, Any]] = {
    "reliability-errors-timeouts": {
        "description": "Error handling, timeout usage, retries, exception handling patterns.",
        "semgrep": {
            # Start broad, later replace with tighter configs
            "configs": ["auto"],
            "post_filter": {
                "rule_id_contains_any": [
                    "timeout", "exception", "error", "retry", "resource", "handle"
                ],
                "message_contains_any": [
                    "timeout", "exception", "error", "retry", "resource leak"
                ],
                "path_ext_any": [".py", ".go", ".ts", ".tsx", ".js", ".java", ".kt", ".rs"],
            },
        },
        "fallback_code_search_queries": [
            "timeout", "retry", "backoff", "except", "catch", "raise", "throw", "finally", "defer"
        ],
        "expected_strength": "medium",
    },

    "injection-and-unsafe": {
        "description": "Injection risks and unsafe execution/deserialization patterns.",
        "semgrep": {
            "configs": ["auto"],  # can later be specific security configs
            "post_filter": {
                "rule_id_contains_any": [
                    "sql", "command", "injection", "xss", "deserialization", "ssrf", "path-traversal", "eval"
                ],
                "message_contains_any": [
                    "injection", "xss", "sql", "command", "deserial", "eval", "traversal"
                ],
                "severity_any": ["ERROR", "WARNING"],
                "path_ext_any": [".py", ".go", ".ts", ".tsx", ".js", ".java", ".php", ".rb"],
            },
        },
        "fallback_code_search_queries": [
            "eval(", "exec(", "os.system", "subprocess", "pickle", "yaml.load", "SELECT ", "INSERT ", "../"
        ],
        "expected_strength": "high",
    },

    "frontend-xss-dom": {
        "description": "Frontend DOM/XSS-unsafe patterns.",
        "semgrep": {
            "configs": ["auto"],
            "post_filter": {
                "rule_id_contains_any": ["xss", "dom", "react", "html"],
                "message_contains_any": ["xss", "innerHTML", "unsafe html", "dom injection"],
                "path_ext_any": [".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"],
            },
        },
        "fallback_code_search_queries": [
            "innerHTML", "dangerouslySetInnerHTML", "document.write", "v-html"
        ],
        "expected_strength": "high",
    },

    "concurrency-timeouts": {
        "description": "Concurrency misuse, missing cancellation/timeouts, async hazards.",
        "semgrep": {
            "configs": ["auto"],  # semgrep coverage here is limited/mixed
            "post_filter": {
                "rule_id_contains_any": ["concurr", "deadlock", "race", "timeout", "async", "thread"],
                "message_contains_any": ["deadlock", "race", "timeout", "thread", "async"],
                "path_ext_any": [".go", ".py", ".java", ".kt", ".rs", ".cs", ".ts"],
            },
        },
        "fallback_code_search_queries": [
            "timeout", "context", "cancel", "await", "async", "goroutine", "mutex", "lock", "deadlock"
        ],
        "expected_strength": "low_to_medium",
    },

    "performance-smells": {
        "description": "Potential performance anti-patterns (heuristic).",
        "semgrep": {
            "configs": ["auto"],  # broad scan + filtering only
            "post_filter": {
                "rule_id_contains_any": ["performance", "inefficient", "loop", "query"],
                "message_contains_any": ["performance", "inefficient", "loop", "query", "allocation"],
                "path_ext_any": [".py", ".go", ".ts", ".tsx", ".js", ".java", ".kt", ".rs", ".php"],
            },
        },
        "fallback_code_search_queries": [
            "for ", "while ", "select", "query", "join", "cache", "memo", "serialize", "deserialize"
        ],
        "expected_strength": "low",
    },
}
