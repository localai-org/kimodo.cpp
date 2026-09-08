#!/bin/sh
# Fetch missing GGUF weights into /data, then start the kimodo.cpp demo
# web server on 0.0.0.0:${KIMODO_PORT:-8094}.
set -eu

DATA="${KIMODO_DATA:-/data}"
PORT="${KIMODO_PORT:-8094}"
# ${VAR-default} (not ${VAR:-default}): an explicitly empty KIMODO_MODELS
# means "download nothing"; only an unset variable takes the default.
MODELS="${KIMODO_MODELS-soma-rp-v1.1}"

mkdir -p "$DATA/models" "$DATA/generated"

TEXT_BUNDLE="$DATA/generated/llm2vec-text-bundle"
TEXT_SENTINEL="$TEXT_BUNDLE/layer-31.gguf"

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

    # The downloader verifies SHA-256 against the published manifests, but
    # re-hashes every file it walks; only run it when something is missing.
    missing=0
    for path in $motion_paths; do
        [ -f "$DATA/$path" ] || missing=1
    done
    [ -f "$TEXT_SENTINEL" ] || missing=1

    if [ "$missing" -eq 1 ]; then
        set -- --output "$DATA"
        for model in $MODELS; do
            set -- "$@" --model "$model"
        done
        # The text bundle is shared by all motion models and weighs the
        # most; skip it when it is already complete.
        if [ -f "$TEXT_SENTINEL" ]; then
            set -- --motion-only "$@"
        fi
        echo "entrypoint: fetching missing weights into $DATA (first start only; several GB)"
        python3 /app/scripts/download_gguf_weights.py "$@"
    fi
fi

# Absent GGUFs simply show up as unavailable in the web UI (the demo
# checks each path with os.Stat), so pass all five catalogue entries.
exec /app/bin/kimodo-demo \
    -addr "0.0.0.0:$PORT" \
    -generator /app/bin/kmd-generate \
    -text-bundle "$TEXT_BUNDLE" \
    -output "$DATA/demo-output" \
    -motion-model "$DATA/models/kimodo-smplx-rp-v1-f32.gguf" \
    -soma-rp-model "$DATA/models/kimodo-soma-rp-v1.1-f32.gguf" \
    -soma-seed-model "$DATA/models/kimodo-soma-seed-v1.1-f32.gguf" \
    -g1-rp-model "$DATA/models/kimodo-g1-rp-v1-f32.gguf" \
    -g1-seed-model "$DATA/models/kimodo-g1-seed-v1-f32.gguf"
