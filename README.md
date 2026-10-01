# kimodo.cpp

GGML/C++ implementation of NVIDIA's Kimodo text-to-motion model.

## Status

The five released Kimodo motion checkpoints accept either a UTF-8 prompt or a
precomputed LLM2Vec embedding and generate local rotations plus root
translations on CPU or Vulkan:

- SMPL-X RP v1: 22 joints
- SOMA RP/SEED v1.1: the predicted compact 30-joint control skeleton
- G1 RP/SEED v1: 34 Unitree G1 joints

NVIDIA's Python API expands SOMA's predicted 30 joints to a relaxed-hand
77-joint presentation skeleton. The native API currently returns the 30 joints
the model actually predicts. The text encoder uses eight-layer Vulkan chunks by
default; set `KIMODO_TEXT_LAYER_CHUNK=1..32` to tune VRAM use.

Included: checked GGUF loading, safetensors conversion, DDIM sampling, C/C++
APIs, conditioned multi-prompt transitions, CPU/Vulkan parity tests,
skeleton-only GLB export, selective LLM2Vec quantisation, divergence reporting,
and local generation/comparison viewers. General constraint input, 77-joint
SOMA expansion, skinned-mesh GLB export, and motion-denoiser quantisation are
not implemented yet.

## Build and test on Linux

Install a C++23 compiler, CMake 3.25+, Ninja, Python 3 with the Hugging Face
CLI (`pip install huggingface_hub`), and the Vulkan loader/headers for Vulkan
support. GGML is a pinned Git submodule:

```sh
git submodule update --init --recursive
scripts/download_gguf_weights.sh --output "$PWD" --model soma-rp-v1.1
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The standard test suite requires the local motion GGUF, text bundle, and
fixtures. It never downloads weights by itself. `release`, `asan-ubsan`, and
`fuzz` presets are also available.

Nix is optional and provides these dependencies reproducibly:

```sh
nix develop path:. --command cmake --preset debug
nix develop path:. --command cmake --build --preset debug
```

For sanitizer work:

```sh
nix develop path:. --command cmake --preset asan-ubsan
nix develop path:. --command cmake --build --preset asan-ubsan
nix develop path:. --command env \
  LD_LIBRARY_PATH="$PWD/build/asan-ubsan/ggml/src:$PWD/build/asan-ubsan/ggml/src/ggml-vulkan:$LD_LIBRARY_PATH" \
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --preset asan-ubsan --output-on-failure
```

Leak detection is disabled because Vulkan loader/driver allocations are global
to the process. The GGUF parser fuzzer requires Clang.

## API

`include/kimodo/kimodo_capi.h` is the C API. Model loading checks the motion
GGUF and text model before inference. The text model can be a monolithic GGUF
beside `tokenizer.gguf` or the legacy component directory. Use
`kimodo_generate_embedding` for
4096 F32 values or `kimodo_generate` for text. Both return the selected model's
root translations and local XYZW rotations; query the joint count from the
result rather than assuming a fixed skeleton.

## Demo

After building the debug preset and downloading the native GGUF bundle:

```sh
go run ./demo -addr 0.0.0.0:8094
```

Open `http://localhost:8094`. The left sidebar contains the prompt and a
persistent history plus motion-model and text-encoder-quantization selectors;
choosing a previous animation restores its prompt and encoder choice for a new
generation. Every successful animation also writes a standalone
`animation.glb` beside its raw streams, for example
`demo-output/<animation-id>/animation.glb`. It contains the selected animated
node hierarchy (no mesh), ready to copy into a Three.js project. It is also
available from `/api/animations/<animation-id>/animation.glb` while the demo
is running.

The demo keeps all 32 layers of its default Q8 text encoder in VRAM for maximum
throughput, while executing them as bounded eight-layer GGML graphs. Its 10 GiB
residency ceiling makes the larger BF16 reference stream in bounded groups on a
16 GiB GPU. Library and command-line callers retain the lower-memory eight-layer
default; set `KIMODO_TEXT_LAYER_CHUNK=32` for residency and optionally set
`KIMODO_TEXT_RESIDENT_LIMIT_MIB` to enforce a VRAM-safe bundle-size ceiling.
The demo reuses one native worker while the selected motion model and text
quantization remain unchanged, preserving both weight sets across requests.
Profiling controls, measurements, and the next optimization targets are in
[`docs/PROFILING.md`](docs/PROFILING.md).

Quantisation comparisons produced by the workflow in
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) can be opened at
`http://localhost:8094/compare`. Supply their parent directory with the demo's
`--comparisons` option. The comparison view overlays every variant on a shared
timeline and provides world-space and root-position-aligned modes.

### Docker

To run the demo without a local build toolchain:

```sh
docker compose up --build
```

The demo is then served at `http://localhost:8094`, and the weights (a motion
model plus a monolithic text encoder) are downloaded into the `kimodo-data`
volume on first start. Inference uses the CPU backend by default; see
[docs/docker_portainer.md](docs/docker_portainer.md) for GPU passthrough,
Portainer deployment and configuration.

## Weights

Ready-to-run native GGML weights are published under the Hugging Face
`LocalAI-io` organisation (not GitHub's `localai-org`). The reusable
[Llama-3-Kimodo-GGML](https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML)
text encoder is separate from the four redistributable motion repositories,
each of which preserves a one-to-one relationship to its NVIDIA upstream:

- [Kimodo-SOMA-RP-v1.1-GGML](https://huggingface.co/LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML)
- [Kimodo-SOMA-SEED-v1.1-GGML](https://huggingface.co/LocalAI-io/Kimodo-SOMA-SEED-v1.1-GGML)
- [Kimodo-G1-RP-v1-GGML](https://huggingface.co/LocalAI-io/Kimodo-G1-RP-v1-GGML)
- [Kimodo-G1-SEED-v1-GGML](https://huggingface.co/LocalAI-io/Kimodo-G1-SEED-v1-GGML)

Download one or repeat `--model` to install several:

```sh
scripts/download_gguf_weights.sh --output "$PWD" \
  --model soma-rp-v1.1 --model g1-rp-v1
```

Q8_0 is the default text encoder. Pass `--text-quantization` with `bf16`,
`q8_0`, `q6_k`, `q5_k`, `q4_k`, or `q4_k_m` to install another level. Each
level is one weight GGUF and all levels share `tokenizer.gguf`; the runtime can
still selectively stream layer tensors from the monolith. The installer
verifies each published manifest and SHA-256 hashes. Use
`--motion-only` when supplying a precomputed 4096-float LLM2Vec embedding.
SMPL-X RP is deliberately absent from the published-weight installer: its
internal-R&D licence prohibits distributing derivative models, so it must be
converted locally after the user obtains the upstream checkpoint under its
gated terms.

The text bundle includes converted Meta Llama 3 material and retains its
separate terms. Review every selected model card before downloading or
redistributing.

## License

The C++ port and its original tooling are licensed under Apache-2.0; see
[LICENSE](LICENSE). GGML and the model weights retain their respective
licences.

| Motion checkpoint | Upstream terms | Commercial use |
| --- | --- | --- |
| Kimodo-SMPLX-RP-v1 | [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/) | No; internal, non-production R&D only; derivative model redistribution is prohibited |
| SOMA RP/SEED v1.1 | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Permitted by the model licence |
| G1 RP/SEED v1 | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Permitted by the model licence |

The SMPL-X warning is about NVIDIA's trained Kimodo checkpoint, not the mere
fact that its output uses an SMPL-X-shaped 22-joint hierarchy. Converting that
checkpoint to GGUF is a new runtime representation of the same weights and does
not replace its licence. Skeleton names, parent links, and the Apache-2.0 port
source do not by themselves make the SOMA or G1 checkpoints non-commercial.
The SMPL-X Hugging Face metadata, model card, and access terms identify the
internal-R&D licence; treat those restrictive terms as controlling even though
an apparently inconsistent `LICENSE` file has also appeared in that upstream
repository.

### Regenerating the bundle

This is only needed to reproduce a conversion. The SMPL-X checkpoint and Llama
base model are gated. After accepting their Hugging Face licences and
authenticating, download the exact revisions and hash manifests with:

```sh
nix develop path:. --command hf auth login
scripts/download_weights.sh --output "$PWD/models" --with-text \
  --model smplx-rp-v1 --model soma-rp-v1.1 --model soma-seed-v1.1 \
  --model g1-rp-v1 --model g1-seed-v1
```

Convert the local LLM2Vec model to the native component bundle with:

```sh
nix develop path:. --command scripts/convert_llm2vec_bundle.sh \
  "$PWD/models/llama3-8b-instruct-base" "$PWD/generated/llm2vec-text-bundle"
```

Quantize and pack the release variants as described in
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md). `kmd-pack-text` streams tensor
payloads between files, so creating the BF16 monolith does not require enough
RAM to hold the complete encoder.

Validate a prospective release without network access, then explicitly upload
it from an account allowed to publish to `LocalAI-io`:

```sh
nix develop path:. --command python scripts/publish_gguf.py --component motion \
  --motion-model soma-rp-v1.1
nix develop path:. --command python scripts/publish_gguf.py --component motion \
  --motion-model soma-rp-v1.1 --upload --confirm-upstream-licences
nix develop path:. --command python scripts/publish_gguf.py --component text \
  --upload --confirm-upstream-licences
```
