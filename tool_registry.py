# SAGES Tool Registry v0.1
# Resolves evidence_profile.dynamic_tool.evidence_class -> concrete CLI commands
#
# Design notes:
# - "any" entries are generic cross-language tools / fallbacks.
# - "python", "go", "typescript", etc. are language-specific overrides/additions.
# - Each value is a LIST because the runner may try multiple tools in priority order.
# - "requires" is optional metadata for preflight checks (file presence, network, etc.).
# - Commands are written to produce machine-readable output whenever possible.
# - Use a shell wrapper that captures:
#     exit_code, stdout_path, stderr_path, duration_sec, timed_out, tool_version
#
# IMPORTANT: some tools exit non-zero when findings are present (not only on failures).
# Your runner should classify exit codes per tool.

TOOL_REGISTRY = {
    # ------------------------------------------------------------
    # Cross-cutting security / SAST / secrets
    # ------------------------------------------------------------
    "secrets_scanner": {
        "any": [
            {
                "tool": "gitleaks",
                "description": "Scans the repository for hardcoded secrets (tokens/keys/passwords) using signature rules and heuristics.",
                "cmd": "gitleaks detect --source . --report-format json --report-path gitleaks.json",
                "outputs": ["gitleaks.json"],
                "parsing": {"format": "gitleaks_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True},
                "priority": 100,
            }
        ]
    },

    "sast_scanner": {
        "any": [
            {
                "tool": "semgrep",
                "description": "Static analysis scanner that runs security rules to detect insecure patterns across the codebase.",
                "cmd": "semgrep scan --config auto --json --output semgrep.json",
                "outputs": ["semgrep.json"],
                "parsing": {"format": "semgrep_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True, "network_maybe": True},
                "priority": 100,
            }
        ],

        # Optional language-specialized semgrep presets (same tool, different rule focus)
        "python": [
            {
                "tool": "semgrep",
                "description": "Semgrep preset for Python-focused security rules (static analysis for insecure patterns).",
                "cmd": "semgrep scan --config p/python --json --output semgrep_python.json",
                "outputs": ["semgrep_python.json"],
                "parsing": {"format": "semgrep_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True, "network_maybe": True},
                "priority": 90,
            }
        ],
        "go": [
            {
                "tool": "semgrep",
                "description": "Semgrep preset for Go-focused security rules (static analysis for insecure patterns).",
                "cmd": "semgrep scan --config p/golang --json --output semgrep_go.json",
                "outputs": ["semgrep_go.json"],
                "parsing": {"format": "semgrep_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True, "network_maybe": True},
                "priority": 90,
            }
        ],
        "typescript": [
            {
                "tool": "semgrep",
                "description": "Semgrep preset for TypeScript-focused security rules (static analysis for insecure patterns).",
                "cmd": "semgrep scan --config p/typescript --json --output semgrep_ts.json",
                "outputs": ["semgrep_ts.json"],
                "parsing": {"format": "semgrep_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True, "network_maybe": True},
                "priority": 90,
            }
        ],
        "javascript": [
            {
                "tool": "semgrep",
                "description": "Semgrep preset for JavaScript-focused security rules (static analysis for insecure patterns).",
                "cmd": "semgrep scan --config p/javascript --json --output semgrep_js.json",
                "outputs": ["semgrep_js.json"],
                "parsing": {"format": "semgrep_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True, "network_maybe": True},
                "priority": 90,
            }
        ],
    },

    # ------------------------------------------------------------
    # Dependency / supply-chain scanning
    # ------------------------------------------------------------
    "dependency_scanner": {
        # Generic fallback (OSV scanner is a good baseline for many ecosystems)
        "any": [
            {
                "tool": "osv-scanner",
                "description": "Dependency vulnerability scan using OSV (reports known CVEs for detected package manifests/lockfiles).",
                "cmd": "osv-scanner scan --format json . > osv.json",
                "outputs": ["osv.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"repo_root": True, "network_maybe": True},
                "priority": 70,
            }
        ],

        "python": [
            {
                "tool": "pip-audit",
                "description": "Audits Python dependencies for known vulnerabilities using advisory databases (pip-audit JSON report).",
                "cmd": "pip-audit -f json -o pip_audit.json",
                "outputs": ["pip_audit.json"],
                "parsing": {"format": "pip_audit_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"python_env": True, "network_maybe": True},
                "priority": 100,
            },
            {
                "tool": "osv-scanner",
                "description": "Dependency vulnerability scan using OSV (reports known CVEs for detected manifests/lockfiles).",
                "cmd": "osv-scanner scan --format json . > osv.json",
                "outputs": ["osv.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": True,
                "priority": 70,
            },
        ],

        "go": [
            {
                "tool": "govulncheck",
                "description": "Scans Go modules for known vulnerabilities and call-path reachability (govulncheck JSON stream).",
                "cmd": "govulncheck -json ./... > govuln.json",
                "outputs": ["govuln.json"],
                "parsing": {"format": "govulncheck_json_stream"},
                "treat_nonzero_as_findings": True,
                "requires": {"go_toolchain": True, "network_maybe": True},
                "priority": 100,
            },
            {
                "tool": "osv-scanner",
                "description": "Dependency vulnerability scan using OSV (reports known CVEs for detected manifests/lockfiles).",
                "cmd": "osv-scanner scan --format json . > osv.json",
                "outputs": ["osv.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": True,
                "priority": 70,
            },
        ],

        "typescript": [
            {
                "tool": "npm-audit",
                "description": "Audits Node.js dependencies for known vulnerabilities using npm audit (JSON report).",
                "cmd": "npm audit --json > npm_audit.json",
                "outputs": ["npm_audit.json"],
                "parsing": {"format": "npm_audit_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"node_toolchain": True, "lockfile_preferred": True, "network_maybe": True},
                "priority": 100,
            },
            {
                "tool": "osv-scanner",
                "description": "Dependency vulnerability scan using OSV (reports known CVEs for detected manifests/lockfiles).",
                "cmd": "osv-scanner scan --format json . > osv.json",
                "outputs": ["osv.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": True,
                "priority": 70,
            },
        ],

        "javascript": [
            {
                "tool": "npm-audit",
                "description": "Audits Node.js dependencies for known vulnerabilities using npm audit (JSON report).",
                "cmd": "npm audit --json > npm_audit.json",
                "outputs": ["npm_audit.json"],
                "parsing": {"format": "npm_audit_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"node_toolchain": True, "lockfile_preferred": True, "network_maybe": True},
                "priority": 100,
            },
            {
                "tool": "osv-scanner",
                "description": "Dependency vulnerability scan using OSV (reports known CVEs for detected manifests/lockfiles).",
                "cmd": "osv-scanner scan --format json . > osv.json",
                "outputs": ["osv.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": True,
                "priority": 70,
            },
        ],

    },

    # ------------------------------------------------------------
    # Linters / format / style
    # ------------------------------------------------------------
    # NOTE: your evidence profile currently uses evidence_class: standard_linter
    # so this registry uses "standard_linter" as the key (not language_linter)
    "standard_linter": {
        "python": [
            {
                "tool": "ruff",
                "description": "Python linter (ruff) running fast static checks; JSON output contains rule IDs and locations.",
                "cmd": "ruff check . --output-format json > ruff.json",
                "outputs": ["ruff.json"],
                "parsing": {"format": "ruff_json"},
                "treat_nonzero_as_findings": True,
                "priority": 100,
            },
            {
                "tool": "flake8",
                "description": "Python linter (flake8) for style/quality checks; text output lists issues and locations.",
                "cmd": "flake8 --output-file flake8.txt .",
                "outputs": ["flake8.txt"],
                "parsing": {"format": "flake8_text"},
                "treat_nonzero_as_findings": True,
                "priority": 80,
            },
        ],

        "go": [
            {
                "tool": "golangci-lint",
                "description": "Go linter aggregator (golangci-lint) running multiple linters; JSON output contains issues and locations.",
                "cmd": "golangci-lint run --out-format json ./... > golangci.json",
                "outputs": ["golangci.json"],
                "parsing": {"format": "golangci_json"},
                "treat_nonzero_as_findings": True,
                "priority": 100,
            }
        ],

        "typescript": [
            {
                "tool": "eslint",
                "description": "JavaScript/TypeScript linter (ESLint); JSON output lists rule violations and locations.",
                "cmd": "npx --yes eslint . --format json --output-file eslint.json",
                "outputs": ["eslint.json"],
                "parsing": {"format": "eslint_json"},
                "treat_nonzero_as_findings": True,
                "priority": 100,
            }
        ],

        "javascript": [
            {
                "tool": "eslint",
                "description": "JavaScript linter (ESLint); JSON output lists rule violations and locations.",
                "cmd": "npx --yes eslint . --format json --output-file eslint.json",
                "outputs": ["eslint.json"],
                "parsing": {"format": "eslint_json"},
                "treat_nonzero_as_findings": True,
                "priority": 100,
            }
        ],

        # fallback if no language-specific linter is mapped
        "any": [
            {
                "tool": "semgrep",
                "description": "Fallback linter-like scan using Semgrep auto config; reports findings as JSON.",
                "cmd": "semgrep scan --config auto --json --output semgrep_lintish.json",
                "outputs": ["semgrep_lintish.json"],
                "parsing": {"format": "semgrep_json"},
                "treat_nonzero_as_findings": True,
                "priority": 30,
            }
        ],
    },

    # ------------------------------------------------------------
    # Complexity / duplication analyzers
    # ------------------------------------------------------------
    "complexity_analyzer": {
        "python": [
            {
                "tool": "lizard",
                "description": "Computes code complexity metrics (cyclomatic complexity, LOC) across the repository; XML output is summarized.",
                "cmd": "lizard -X -o lizard.xml .",
                "outputs": ["lizard.xml"],
                "parsing": {"format": "lizard_cppncss_xml"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            }
        ],
        "go": [
            {
                "tool": "gocyclo",
                "description": "Computes cyclomatic complexity for Go functions; text output lists functions and complexity scores.",
                "cmd": "gocyclo -over 0 ./... > gocyclo.txt",
                "outputs": ["gocyclo.txt"],
                "parsing": {"format": "gocyclo_text"},
                "treat_nonzero_as_findings": False,
                "priority": 80,
            },
            {
                "tool": "lizard",
                "description": "Computes code complexity metrics (cyclomatic complexity, LOC) across the repository; XML output is summarized.",
                "cmd": "lizard -X -o lizard.xml .",
                "outputs": ["lizard.xml"],
                "parsing": {"format": "lizard_cppncss_xml"},
                "priority": 60,
            },
        ],
        "typescript": [
            {
                "tool": "lizard",
                "description": "Computes code complexity metrics across the repository; XML output is summarized.",
                "cmd": "lizard -X -o lizard.xml .",
                "outputs": ["lizard.xml"],
                "parsing": {"format": "lizard_cppncss_xml"},
                "priority": 90,
            }
        ],
        "javascript": [
            {
                "tool": "lizard",
                "description": "Computes code complexity metrics across the repository; XML output is summarized.",
                "cmd": "lizard -X -o lizard.xml .",
                "outputs": ["lizard.xml"],
                "parsing": {"format": "lizard_cppncss_xml"},
                "priority": 90,
            }
        ],
        "any": [
            {
                "tool": "lizard",
                "description": "Computes code complexity metrics across the repository; XML output is summarized.",
                "cmd": "lizard -X -o lizard.xml .",
                "outputs": ["lizard.xml"],
                "parsing": {"format": "lizard_cppncss_xml"},
                "priority": 50,
            }
        ],
    },

    "duplication_analyzer": {
        "python": [
            {
                "tool": "jscpd",
                "description": "Detects duplicated code blocks across the repository (JSCPD); JSON report summarizes clones.",
                "cmd": "npx --yes jscpd --reporters json --output .jscpd-report .",
                "outputs": [".jscpd-report/jscpd-report.json"],
                "parsing": {"format": "jscpd_json"},
                "treat_nonzero_as_findings": False,
                "priority": 90,
            }
        ],
        "go": [
            {
                "tool": "dupl",
                "description": "Detects duplicated code in Go packages (dupl); plumbing text output lists duplicate blocks.",
                "cmd": "dupl -plumbing ./... > dupl.txt",
                "outputs": ["dupl.txt"],
                "parsing": {"format": "dupl_plumbing_text"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            },
            {
                "tool": "jscpd",
                "description": "Detects duplicated code blocks across the repository (JSCPD); JSON report summarizes clones.",
                "cmd": "npx --yes jscpd --reporters json --output .jscpd-report .",
                "outputs": [".jscpd-report/jscpd-report.json"],
                "parsing": {"format": "jscpd_json"},
                "priority": 70,
            }
        ],
        "typescript": [
            {
                "tool": "jscpd",
                "description": "Detects duplicated code blocks across the repository (JSCPD); JSON report summarizes clones.",
                "cmd": "npx --yes jscpd --reporters json --output .jscpd-report .",
                "outputs": [".jscpd-report/jscpd-report.json"],
                "parsing": {"format": "jscpd_json"},
                "priority": 100,
            }
        ],
        "javascript": [
            {
                "tool": "jscpd",
                "description": "Detects duplicated code blocks across the repository (JSCPD); JSON report summarizes clones.",
                "cmd": "npx --yes jscpd --reporters json --output .jscpd-report .",
                "outputs": [".jscpd-report/jscpd-report.json"],
                "parsing": {"format": "jscpd_json"},
                "priority": 100,
            }
        ],
        "any": [
            {
                "tool": "jscpd",
                "description": "Detects duplicated code blocks across the repository (JSCPD); JSON report summarizes clones.",
                "cmd": "npx --yes jscpd --reporters json --output .jscpd-report .",
                "outputs": [".jscpd-report/jscpd-report.json"],
                "parsing": {"format": "jscpd_json"},
                "priority": 50,
            }
        ],
    },

    # ------------------------------------------------------------
    # Dependency inventory (for "dependency_minimization_hints")
    # ------------------------------------------------------------
    "dependency_inventory": {
        "python": [
            {
                "tool": "pip-inspect",
                "description": "Inventories installed Python packages in the active environment (pip inspect JSON).",
                "cmd": "python -m pip inspect > pip_inspect.json",
                "outputs": ["pip_inspect.json"],
                "parsing": {"format": "pip_inspect_json"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            }
        ],
        "go": [
            {
                "tool": "go-list-modules",
                "description": "Inventories Go module dependencies (go list -m -json all) as a JSON stream.",
                "cmd": "go list -m -json all > go_modules.json",
                "outputs": ["go_modules.json"],
                "parsing": {"format": "go_list_modules_json_stream"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            }
        ],
        "typescript": [
            {
                "tool": "npm-ls",
                "description": "Inventories Node.js dependency tree (npm ls --all) as JSON.",
                "cmd": "npm ls --all --json > npm_ls.json",
                "outputs": ["npm_ls.json"],
                "parsing": {"format": "npm_ls_json"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            }
        ],
        "javascript": [
            {
                "tool": "npm-ls",
                "description": "Inventories Node.js dependency tree (npm ls --all) as JSON.",
                "cmd": "npm ls --all --json > npm_ls.json",
                "outputs": ["npm_ls.json"],
                "parsing": {"format": "npm_ls_json"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            }
        ],
        "any": [
            {
                "tool": "osv-scanner",
                "description": "Fallback inventory-like scan using OSV scanner output as a proxy dependency list.",
                "cmd": "osv-scanner scan --format json . > osv_inventory_like.json",
                "outputs": ["osv_inventory_like.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": False,
                "priority": 20,
            }
        ],
    },

    # ------------------------------------------------------------
    # Test runner / test report harvesting
    # ------------------------------------------------------------
    "test_runner": {
        "python": [
            {
                "tool": "pytest",
                "description": "Runs Python tests with pytest and emits a JUnit XML report for pass/fail and failures.",
                "cmd": "pytest -q --junitxml=pytest.junit.xml",
                "outputs": ["pytest.junit.xml"],
                "parsing": {"format": "junit_xml"},
                "treat_nonzero_as_findings": False,  # non-zero likely means test failures
                "treat_nonzero_as_execution_result": True,
                "priority": 100,
            }
        ],
        "go": [
            {
                "tool": "gotestsum",
                "description": "Runs Go tests and emits a JUnit XML report (gotestsum).",
                "cmd": "gotestsum --junitfile go.junit.xml --format testname",
                "outputs": ["go.junit.xml"],
                "parsing": {"format": "junit_xml"},
                "treat_nonzero_as_execution_result": True,
                "priority": 100,
            },
            {
                "tool": "go-test-json",
                "description": "Runs Go tests and emits JSON event stream output (go test -json).",
                "cmd": "go test ./... -json > go_test.json",
                "outputs": ["go_test.json"],
                "parsing": {"format": "go_test_json_stream"},
                "treat_nonzero_as_execution_result": True,
                "priority": 80,
            }
        ],
        "typescript": [
            {
                "tool": "jest",
                "description": "Runs JavaScript/TypeScript tests with Jest and writes a JSON report.",
                "cmd": "npx --yes jest --json --outputFile=jest.json",
                "outputs": ["jest.json"],
                "parsing": {"format": "jest_json"},
                "treat_nonzero_as_execution_result": True,
                "priority": 90,
            }
        ],
        "javascript": [
            {
                "tool": "jest",
                "description": "Runs JavaScript tests with Jest and writes a JSON report.",
                "cmd": "npx --yes jest --json --outputFile=jest.json",
                "outputs": ["jest.json"],
                "parsing": {"format": "jest_json"},
                "treat_nonzero_as_execution_result": True,
                "priority": 90,
            }
        ],
        "any": [
            {
                "tool": "noop-test-runner",
                "description": "No-op placeholder when no test runner is mapped for the detected language.",
                "cmd": "echo '{\"status\":\"no_runner_mapped\"}' > no_test_runner.json",
                "outputs": ["no_test_runner.json"],
                "parsing": {"format": "generic_json"},
                "treat_nonzero_as_execution_result": False,
                "priority": 1,
            }
        ],
    },

    # ------------------------------------------------------------
    # Container / K8s / IaC
    # ------------------------------------------------------------
    "container_linter": {
        "any": [
            {
                "tool": "hadolint",
                "description": "Lints Dockerfiles for common best-practice and security issues (Hadolint JSON output).",
                "cmd": "hadolint Dockerfile -f json > hadolint.json",
                "outputs": ["hadolint.json"],
                "parsing": {"format": "hadolint_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"dockerfile_present": True},
                "priority": 100,
            }
        ]
    },

    "k8s_linter": {
        "any": [
            {
                "tool": "kube-linter",
                "description": "Scans Kubernetes YAML manifests for common configuration and security issues (kube-linter JSON output).",
                "cmd": "kube-linter lint . --format json > kube_linter.json",
                "outputs": ["kube_linter.json"],
                "parsing": {"format": "kube_linter_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"k8s_manifests_present": True},
                "priority": 90,
            },
            {
                "tool": "kube-score",
                "description": "Scores Kubernetes YAML manifests against best practices (kube-score JSON output).",
                "cmd": "find . -type f \\( -name '*.yml' -o -name '*.yaml' \\) -print0 | xargs -0 -r kube-score score --output-format json > kube_score.json",
                "outputs": ["kube_score.json"],
                "parsing": {"format": "kube_score_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"k8s_manifests_present": True},
                "priority": 70,
            }
        ]
    },

}

# TOOL_REGISTRY extension patch (PHP + additional ecosystems)

TOOL_REGISTRY_PATCH = {
    # ------------------------------------------------------------
    # Dependency / supply-chain scanning additions
    # ------------------------------------------------------------
    "dependency_scanner": {
        "php": [
            {
                "tool": "composer-audit",
                "description": "Audits PHP dependencies for known vulnerabilities using Composer audit (JSON report).",
                "cmd": "composer audit --format=json > composer_audit.json",
                "outputs": ["composer_audit.json"],
                "parsing": {"format": "composer_audit_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"composer_present": True, "network_maybe": True},
                "priority": 100,
            },
            {
                "tool": "osv-scanner",
                "description": "Dependency vulnerability scan using OSV (reports known CVEs for detected manifests/lockfiles).",
                "cmd": "osv-scanner scan --format json . > osv.json",
                "outputs": ["osv.json"],
                "parsing": {"format": "osv_scanner_json"},
                "treat_nonzero_as_findings": True,
                "priority": 70,
            },
        ],
    },

    # ------------------------------------------------------------
    # Standard lint / static style additions
    # ------------------------------------------------------------
    "standard_linter": {
        "php": [
            {
                "tool": "phpstan",
                "description": "PHP static analyzer (PHPStan); JSON output reports type and code issues.",
                "cmd": "phpstan analyse . --no-progress --error-format=json > phpstan.json",
                "outputs": ["phpstan.json"],
                "parsing": {"format": "phpstan_json"},
                "treat_nonzero_as_findings": True,
                "requires": {"composer_or_phpstan_present": True},
                "priority": 100,
            },
            {
                "tool": "phpcs",
                "description": "PHP code style checker (PHP_CodeSniffer); JSON report lists standard violations.",
                "cmd": "phpcs --report=json . > phpcs.json",
                "outputs": ["phpcs.json"],
                "parsing": {"format": "phpcs_json"},
                "treat_nonzero_as_findings": True,
                "priority": 90,
            },
            {
                "tool": "psalm",
                "description": "PHP static analyzer (Psalm); JSON output reports type and code issues.",
                "cmd": "psalm --output-format=json > psalm.json",
                "outputs": ["psalm.json"],
                "parsing": {"format": "psalm_json"},
                "treat_nonzero_as_findings": True,
                "priority": 85,
            },
        ],
    },

    # ------------------------------------------------------------
    # Complexity analyzer additions
    # ------------------------------------------------------------
    "complexity_analyzer": {
        "php": [
            {
                "tool": "lizard",
                "description": "Computes code complexity metrics across the repository (lizard XML output).",
                "cmd": "lizard -X -o lizard.xml .",
                "outputs": ["lizard.xml"],
                "parsing": {"format": "lizard_cppncss_xml"},
                "priority": 100,
            }
        ],
    },

    # ------------------------------------------------------------
    # Duplication analyzer additions
    # ------------------------------------------------------------
    "duplication_analyzer": {
        "php": [
            {
                "tool": "phpcpd",
                "description": "Detects duplicated PHP code (phpcpd); PMD XML output reports clones.",
                "cmd": "phpcpd --log-pmd phpcpd.xml .",
                "outputs": ["phpcpd.xml"],
                "parsing": {"format": "phpcpd_pmd_xml"},
                "treat_nonzero_as_findings": True,
                "priority": 100,
            },
            {
                "tool": "jscpd",
                "description": "Detects duplicated code blocks across the repository (JSCPD); JSON report summarizes clones.",
                "cmd": "npx --yes jscpd --reporters json --output .jscpd-report .",
                "outputs": [".jscpd-report/jscpd-report.json"],
                "parsing": {"format": "jscpd_json"},
                "priority": 70,
            },
        ],
    },

    # ------------------------------------------------------------
    # Dependency inventory additions
    # ------------------------------------------------------------
    "dependency_inventory": {
        "php": [
            {
                "tool": "composer-show",
                "description": "Inventories PHP dependencies from composer.lock (composer show --locked JSON).",
                "cmd": "if composer help show >/dev/null 2>&1; then composer show --locked --format=json > composer_show.json; else echo 'composer show unsupported (needs newer Composer)' >&2; exit 127; fi",
                "outputs": ["composer_show.json"],
                "parsing": {"format": "composer_show_json"},
                "treat_nonzero_as_findings": False,
                "priority": 100,
            }
        ],
    },

    # ------------------------------------------------------------
    # Test runner additions
    # ------------------------------------------------------------
    "test_runner": {
        "php": [
            {
                "tool": "phpunit",
                "description": "Runs PHP unit tests with PHPUnit and emits a JUnit XML report.",
                "cmd": "if [ -x ./vendor/bin/phpunit ]; then ./vendor/bin/phpunit --log-junit phpunit.junit.xml; else phpunit --log-junit phpunit.junit.xml; fi",
                "outputs": ["phpunit.junit.xml"],
                "parsing": {"format": "junit_xml"},
                "treat_nonzero_as_execution_result": True,
                "priority": 100,
            },
            {
                "tool": "pest",
                "description": "Runs PHP tests with Pest and emits a JUnit XML report.",
                "cmd": "if [ -x ./vendor/bin/pest ]; then ./vendor/bin/pest --log-junit pest.junit.xml; else pest --log-junit pest.junit.xml; fi",
                "outputs": ["pest.junit.xml"],
                "parsing": {"format": "junit_xml"},
                "treat_nonzero_as_execution_result": True,
                "priority": 90,
            },
        ],
    },
}


def get_tool_registry() -> dict:
    """Return the effective tool registry (base + patch).

    The runner should use this instead of referencing TOOL_REGISTRY directly.
    """

    def _merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return _merge(TOOL_REGISTRY, TOOL_REGISTRY_PATCH)
