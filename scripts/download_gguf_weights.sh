#!/usr/bin/env bash
# Download the published native GGUF bundle into Kimodo's standard paths.
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1

ORG="${GGUF_ORG:-LocalAI-io}"
TEXT_REPO_DEFAULT="$ORG/Llama-3-Kimodo-GGML"

usage() {
    printf '%s\n' "usage: $0 --output DIR [--model MODEL]... [--motion-repo HF_REPO] [--text-repo HF_REPO] [--text-quantization LEVEL] [--revision REVISION] [--motion-only]" >&2
    printf '%s\n' "models: soma-rp-v1.1, soma-seed-v1.1, g1-rp-v1, g1-seed-v1" >&2
    printf '%s\n' "text levels: bf16, q8_0, q6_k, q5_k, q4_k, q4_k_m (default: q8_0)" >&2
    exit 2
}

output='' motion_repo_override='' text_repo="$TEXT_REPO_DEFAULT" text_quantization='q8_0' revision='main' motion_only=0 models=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) [ "$#" -ge 2 ] || usage; output=$2; shift 2 ;;
        --model) [ "$#" -ge 2 ] || usage; models+=("$2"); shift 2 ;;
        --motion-repo) [ "$#" -ge 2 ] || usage; motion_repo_override=$2; shift 2 ;;
        --text-repo) [ "$#" -ge 2 ] || usage; text_repo=$2; shift 2 ;;
        --text-quantization) [ "$#" -ge 2 ] || usage; text_quantization=$2; shift 2 ;;
        --revision) [ "$#" -ge 2 ] || usage; revision=$2; shift 2 ;;
        --motion-only) motion_only=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$output" ] || usage
case "$text_quantization" in bf16|q8_0|q6_k|q5_k|q4_k|q4_k_m) ;; *) usage ;; esac
command -v hf >/dev/null || { echo "hf not found; enter the Nix shell first" >&2; exit 1; }
[ "${#models[@]}" -gt 0 ] || models=(soma-rp-v1.1)
[ -z "$motion_repo_override" ] || [ "${#models[@]}" -eq 1 ] || { echo "--motion-repo requires exactly one --model" >&2; exit 2; }

mkdir -p "$output"

download_and_verify() { # repo relative-path...
    local repo=$1; shift
    local manifest_dir="$output/.kimodo-manifests/${repo//\//__}"
    mkdir -p "$manifest_dir"
    echo "Downloading $repo at $revision into $output"
    local args=(download "$repo" --revision "$revision" --local-dir "$output")
    local pattern
    for pattern in "$@"; do args+=(--include "$pattern"); done
    hf "${args[@]}" >/dev/null
    hf download "$repo" --revision "$revision" --local-dir "$manifest_dir" --include MANIFEST.json >/dev/null
    python - "$manifest_dir/MANIFEST.json" "$output" "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output = Path(sys.argv[2])
root = output.resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("format") != "kimodo-gguf-manifest-v1":
    raise SystemExit("unsupported or malformed GGUF manifest")
requested = set(sys.argv[3:])
matched = set()
for entry in manifest.get("files", []):
    relative = Path(entry.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".gguf":
        raise SystemExit(f"unsafe manifest path: {relative}")
    if relative.as_posix() not in requested:
        continue
    matched.add(relative.as_posix())
    path = root / relative
    if not path.is_file() or path.stat().st_size != entry.get("bytes"):
        raise SystemExit(f"missing or wrong-sized file: {relative}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != entry.get("sha256"):
        raise SystemExit(f"checksum mismatch: {relative}")
missing = requested - matched
if missing:
    raise SystemExit(f"selected paths absent from manifest: {', '.join(sorted(missing))}")
print("verified native Kimodo GGUF bundle")
PY
}

select_text_paths() { # repo quantization
    local repo=$1 quantization=$2
    local manifest_dir="$output/.kimodo-manifests/${repo//\//__}"
    mkdir -p "$manifest_dir"
    hf download "$repo" --revision "$revision" --local-dir "$manifest_dir" --include MANIFEST.json >/dev/null
    python - "$manifest_dir/MANIFEST.json" "$quantization" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
variant = manifest.get("variants", {}).get(sys.argv[2])
if not isinstance(variant, dict):
    raise SystemExit(f"text quantization {sys.argv[2]!r} is not present in the manifest")
for key in ("weights", "tokenizer"):
    value = variant.get(key)
    path = Path(value) if isinstance(value, str) else None
    if path is None or path.is_absolute() or ".." in path.parts or path.suffix != ".gguf":
        raise SystemExit(f"unsafe or malformed text variant path: {value!r}")
    print(value)
PY
}

for model in "${models[@]}"; do
    case "$model" in
        soma-rp-v1.1) motion_repo="$ORG/Kimodo-SOMA-RP-v1.1-GGML"; motion_file='models/kimodo-soma-rp-v1.1-f32.gguf' ;;
        soma-seed-v1.1) motion_repo="$ORG/Kimodo-SOMA-SEED-v1.1-GGML"; motion_file='models/kimodo-soma-seed-v1.1-f32.gguf' ;;
        g1-rp-v1) motion_repo="$ORG/Kimodo-G1-RP-v1-GGML"; motion_file='models/kimodo-g1-rp-v1-f32.gguf' ;;
        g1-seed-v1) motion_repo="$ORG/Kimodo-G1-SEED-v1-GGML"; motion_file='models/kimodo-g1-seed-v1-f32.gguf' ;;
        *) echo "unknown motion model: $model" >&2; usage ;;
    esac
    [ -z "$motion_repo_override" ] || motion_repo=$motion_repo_override
    download_and_verify "$motion_repo" "$motion_file"
done
if [ "$motion_only" -eq 0 ]; then
    text_selection=$(select_text_paths "$text_repo" "$text_quantization")
    mapfile -t text_paths <<<"$text_selection"
    download_and_verify "$text_repo" "${text_paths[@]}"
fi
