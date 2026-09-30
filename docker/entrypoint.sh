#!/bin/sh
# Fetch missing GGUF weights into /data, then start the kimodo.cpp demo
# web server on 0.0.0.0:${KIMODO_PORT:-8094}.
set -eu

DATA="${KIMODO_DATA:-/data}"
PORT="${KIMODO_PORT:-8094}"
# ${VAR-default} (not ${VAR:-default}): an explicitly empty KIMODO_MODELS
# means "download nothing"; only an unset variable takes the default.
MODELS="${KIMODO_MODELS-soma-rp-v1.1}"
# Text encoder variant fetched on first start: bf16, q8_0 (default),
# q6_k, q5_k, q4_k or q4_k_m.  Smaller variants trade some prompt
# fidelity for download size and memory.
TEXT_QUANTIZATION="${KIMODO_TEXT_QUANTIZATION-q8_0}"

mkdir -p "$DATA/models" "$DATA/generated"

# An explicitly empty KIMODO_MODELS skips downloading entirely (weights
# mounted by hand, or smoke-testing the server image).
if [ -n "$MODELS" ]; then
    # Map model ids to their GGUF paths; an unknown id is a hard error so
    # typos in KIMODO_MODELS fail fast instead of being silently skipped.
    motion_paths=""
    for model in $MODELS; do
        case "$model" in
            soma-rp-v1.1)   path=models/kimodo-soma-rp-v1.1-f32.gguf ;;
            soma-seed-v1.1) path=models/kimodo-soma-seed-v1.1-f32.gguf ;;
            g1-rp-v1)       path=models/kimodo-g1-rp-v1-f32.gguf ;;
            g1-seed-v1)     path=models/kimodo-g1-seed-v1-f32.gguf ;;
            *) printf 'entrypoint: unknown model "%s" in KIMODO_MODELS\n' "$model" >&2; exit 1 ;;
        esac
        motion_paths="$motion_paths $path"
    done

    # Text encoder presence: any packed monolithic bundle or the legacy
    # F32 component directory counts.
    text_present=0
    for packed in "$DATA"/Llama-3-Kimodo-*.gguf; do
        [ -f "$packed" ] && text_present=1
    done
    [ -f "$DATA/generated/llm2vec-text-bundle/layer-31.gguf" ] && text_present=1

    # The downloader verifies SHA-256 against the published manifests, but
    # re-hashes every file it walks; only run it when something is missing.
    missing=0
    for path in $motion_paths; do
        [ -f "$DATA/$path" ] || missing=1
    done
    [ "$text_present" -eq 1 ] || missing=1

    if [ "$missing" -eq 1 ]; then
        set -- --output "$DATA"
        for model in $MODELS; do
            set -- "$@" --model "$model"
        done
        if [ "$text_present" -eq 1 ]; then
            set -- --motion-only "$@"
        else
            set -- "$@" --text-quantization "$TEXT_QUANTIZATION"
        fi
        echo "entrypoint: fetching missing weights into $DATA (first start only; several GB)"
        python3 /app/scripts/download_gguf_weights.py "$@"
    fi
fi

# Run from the data directory so the demo's packed-over-legacy text
# bundle defaults resolve against /data: a downloaded Llama-3-Kimodo-*.gguf
# is preferred, generated/llm2vec-text-bundle is the fallback.
cd "$DATA"
exec /app/bin/kimodo-demo \
    -addr "0.0.0.0:$PORT" \
    -generator /app/bin/kmd-generate \
    -output "$DATA/demo-output" \
    -motion-model models/kimodo-smplx-rp-v1-f32.gguf \
    -soma-rp-model models/kimodo-soma-rp-v1.1-f32.gguf \
    -soma-seed-model models/kimodo-soma-seed-v1.1-f32.gguf \
    -g1-rp-model models/kimodo-g1-rp-v1-f32.gguf \
    -g1-seed-model models/kimodo-g1-seed-v1-f32.gguf
