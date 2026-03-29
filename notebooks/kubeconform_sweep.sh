#!/usr/bin/env bash
set -euo pipefail

TARGETS_JSON="${1:-targets.json}"

# candidate list (tune to your needs)
VERSIONS=(
  1.20.0 1.21.0 1.22.0 1.23.0 1.24.0 1.25.0 1.26.0 1.27.0 1.28.0 1.29.0 1.30.0
)

mapfile -t FILES < <(jq -r '.kubernetes.manifest_files[]?' "$TARGETS_JSON")
if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo '{"selected": null, "mode": "no_manifests", "note": "No Kubernetes manifests detected"}'
  exit 0
fi

run_one() {
  local ver="$1"
  local mode="$2" # strict|ignore-missing-schemas
  local extra=()
  if [[ "$mode" == "ignore-missing-schemas" ]]; then
    extra+=("-ignore-missing-schemas")
  fi

  # -kubernetes-version sets which schemas to validate against :contentReference[oaicite:5]{index=5}
  kubeconform -strict -summary -output json -kubernetes-version "$ver" "${extra[@]}" "${FILES[@]}" 2>/dev/null
}

only_missing_schema_errors() {
  # returns 0 if all ERROR resources are missing-schema type
  jq -e '
    (.resources // [])
    | map(select(.status=="ERROR"))
    | if length==0 then false
      else all(.msg? | tostring | test("schema"; "i"))
    end
  ' >/dev/null
}

for ver in "${VERSIONS[@]}"; do
  if out="$(run_one "$ver" "strict")"; then
    inv="$(jq -r '.summary.invalid // 999' <<<"$out")"
    err="$(jq -r '.summary.errors  // 999' <<<"$out")"
    if [[ "$inv" == "0" && "$err" == "0" ]]; then
      jq -n --arg v "$ver" --arg m "strict" --argjson s "$(jq '.summary' <<<"$out")" \
        '{selected:$v, mode:$m, summary:$s}'
      exit 0
    fi
  else
    # kubeconform exits non-zero on invalid/error :contentReference[oaicite:6]{index=6}
    out="$(run_one "$ver" "strict" || true)"
    if only_missing_schema_errors <<<"$out"; then
      out2="$(run_one "$ver" "ignore-missing-schemas" || true)"
      inv="$(jq -r '.summary.invalid // 999' <<<"$out2")"
      err="$(jq -r '.summary.errors  // 999' <<<"$out2")"
      if [[ "$inv" == "0" && "$err" == "0" ]]; then
        jq -n --arg v "$ver" --arg m "ignore-missing-schemas" --argjson s "$(jq '.summary' <<<"$out2")" \
          '{selected:$v, mode:$m, summary:$s, note:"strict mode failed due to missing schemas; fell back to ignore-missing-schemas"}'
        exit 0
      fi
    fi
  fi
done

echo '{"selected": null, "mode": "no_version_passed", "note": "No tested Kubernetes version validated cleanly"}'
exit 1
