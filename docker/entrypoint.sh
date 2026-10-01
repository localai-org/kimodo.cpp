#!/bin/sh
# Fetch missing GGUF weights into /data, then start the kimodo.cpp demo
# web server on 0.0.0.0:${KIMODO_PORT:-8094}.
set -eu

DATA="${KIMODO_DATA:-/data}"
PORT="${KIMODO_PORT:-8094}"
# App root carrying scripts/ and bin/; /app in the image, overridden by the
# entrypoint test to point at stub downloader/server scripts.
APP="${KIMODO_APP_DIR:-/app}"
# ${VAR-default} (not ${VAR:-default}): an explicitly empty KIMODO_MODELS
# means "download nothing"; only an unset variable takes the default.
MODELS="${KIMODO_MODELS-soma-rp-v1.1}"
# Text encoder variant fetched on first start: bf16, q8_0 (default),
# q6_k, q5_k, q4_k or q4_k_m.  Smaller variants trade some prompt
# fidelity for download size and memory.
TEXT_QUANTIZATION="${KIMODO_TEXT_QUANTIZATION-q8_0}"
# ${VAR+set} records whether the variable was provided explicitly; an
# unset variable keeps the "reuse whatever bundle already exists" default.
TEXT_QUANT_EXPLICIT=0
[ "${KIMODO_TEXT_QUANTIZATION+set}" ] && TEXT_QUANT_EXPLICIT=1

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

    # Map the text variant to its packed bundle file name (mirrors the
    # downloader manifest); an unknown id is a hard error so a typo in
    # KIMODO_TEXT_QUANTIZATION fails fast instead of starting half-ready.
    case "$TEXT_QUANTIZATION" in
        bf16)   PACKED_NAME="Llama-3-Kimodo-BF16.gguf" ;;
        q8_0)   PACKED_NAME="Llama-3-Kimodo-Q8_0.gguf" ;;
        q6_k)   PACKED_NAME="Llama-3-Kimodo-Q6_K.gguf" ;;
        q5_k)   PACKED_NAME="Llama-3-Kimodo-Q5_K.gguf" ;;
        q4_k)   PACKED_NAME="Llama-3-Kimodo-Q4_K.gguf" ;;
        q4_k_m) PACKED_NAME="Llama-3-Kimodo-Q4_K_M.gguf" ;;
        *) printf 'entrypoint: unknown KIMODO_TEXT_QUANTIZATION "%s"\n' "$TEXT_QUANTIZATION" >&2; exit 1 ;;
    esac

    # Text encoder presence: a packed monolithic bundle only counts when
    # its sibling tokenizer.gguf is there too (the demo marks a packed
    # bundle unavailable without it, so an interrupted first start that
    # stopped between the two must re-enter the downloader), or the
    # legacy F32 component directory counts.
    text_present=0
    for packed in "$DATA"/Llama-3-Kimodo-*.gguf; do
        [ -f "$packed" ] || continue
        if [ -f "${packed%/*}/tokenizer.gguf" ]; then
            text_present=1
        else
            printf 'entrypoint: ignoring %s (sibling tokenizer.gguf missing; resuming text download)\n' "$packed" >&2
        fi
    done
    [ -f "$DATA/generated/llm2vec-text-bundle/layer-31.gguf" ] && text_present=1

    # An explicit KIMODO_TEXT_QUANTIZATION opts in to fetching that packed
    # bundle even when a legacy directory or another packed variant already
    # lets the server start ("set once to add a packed bundle").
    text_selected_present=0
    if [ -f "$DATA/$PACKED_NAME" ] && [ -f "$DATA/tokenizer.gguf" ]; then
        text_selected_present=1
    fi

    # The downloader verifies SHA-256 against the published manifests, but
    # re-hashes every file it walks; only run it when something is missing.
    missing=0
    for path in $motion_paths; do
        [ -f "$DATA/$path" ] || missing=1
    done
    [ "$text_present" -eq 1 ] || missing=1
    [ "$TEXT_QUANT_EXPLICIT" -eq 1 ] && [ "$text_selected_present" -eq 0 ] && missing=1

    # Motion-only when some text bundle is complete and the selected packed
    # bundle needs no fetch: without an explicit KIMODO_TEXT_QUANTIZATION
    # whatever bundle exists is reused as-is, so e.g. adding a motion model
    # on a legacy-only volume stays a motion-only download.
    motion_only=0
    if [ "$text_present" -eq 1 ]; then
        if [ "$text_selected_present" -eq 1 ] || [ "$TEXT_QUANT_EXPLICIT" -eq 0 ]; then
            motion_only=1
        fi
    fi

    if [ "$missing" -eq 1 ]; then
        set -- --output "$DATA"
        for model in $MODELS; do
            set -- "$@" --model "$model"
        done
        if [ "$motion_only" -eq 1 ]; then
            set -- --motion-only "$@"
        else
            set -- "$@" --text-quantization "$TEXT_QUANTIZATION"
        fi
        echo "entrypoint: fetching missing weights into $DATA (first start only; several GB)"
        python3 "$APP/scripts/download_gguf_weights.py" "$@"
    fi
fi

# Run from the data directory so the demo's packed-over-legacy text
# bundle defaults resolve against /data: a downloaded Llama-3-Kimodo-*.gguf
# is preferred, generated/llm2vec-text-bundle is the fallback.
cd "$DATA"
exec "$APP/bin/kimodo-demo" \
    -addr "0.0.0.0:$PORT" \
    -generator "$APP/bin/kmd-generate" \
    -output "$DATA/demo-output" \
    -motion-model models/kimodo-smplx-rp-v1-f32.gguf \
    -soma-rp-model models/kimodo-soma-rp-v1.1-f32.gguf \
    -soma-seed-model models/kimodo-soma-seed-v1.1-f32.gguf \
    -g1-rp-model models/kimodo-g1-rp-v1-f32.gguf \
    -g1-seed-model models/kimodo-g1-seed-v1-f32.gguf
