# Running the kimodo.cpp demo in Docker (Compose & Portainer)

This guide runs the kimodo.cpp text-to-motion web demo as a container:

- **`kimodo-demo`** (Go HTTP server) serves the web UI and animation
  gallery on port **8094** and shells out to **`kmd-generate`** for each
  generation job (one job at a time).
- Inference uses the **CPU backend by default** and works on any x86-64
  host; an NVIDIA GPU can be passed through for Vulkan inference (see the
  GPU section — two lines to change).
- Model weights are downloaded into the data volume on **first start**
  and verified against the published SHA-256 manifests: a motion model
  (~1.1 GB for SOMA RP v1.1) plus a monolithic text encoder (7.6 GB for
  the default Q8_0 variant; smaller Q4/Q5/Q6 and the 14 GB BF16 reference
  are selectable).

Requirements: Docker (Compose v2), ~12 GB free disk (image plus weights
plus room for animations).

## Quick start (Docker Compose)

From a checkout of this repository:

```sh
docker compose up --build
```

First start downloads the weights (several GB — watch the logs), then the
demo is served at `http://localhost:8094`. Subsequent starts skip the
download; the volume keeps both weights and the animation gallery.

## Deploying with Portainer

1. On the Docker host, from a checkout of this repository:

   ```sh
   docker build -t kimodo:latest .
   ```

2. Portainer → **Stacks** → **Add stack**, name it `kimodo`, choose
   **Web editor**, paste `docker-compose.yml` **without the `build:` key**
   (Portainer web-editor stacks have no build context), then
   **Deploy the stack**.

   Alternatively deploy as a **Repository** stack pointing at this
   repository — Portainer then honours `build:` and builds the image on
   first deploy. Note that updating the repository does not rebuild the
   image automatically (portainer/portainer#6288); rebuild on the host and
   re-deploy for code updates.

### First start

The container shows as *starting/unhealthy* until the download completes
and the HTTP server binds; the healthcheck allows 30 minutes for this.
An interrupted download resumes on restart. On shared or rate-limited
connections, set `HF_TOKEN` in the environment to raise the Hugging Face
rate limits.

## Configuration

All knobs are environment variables on the `kimodo` service (edit them in
the Portainer stack editor or `docker-compose.yml` and redeploy — the
volume keeps weights and gallery):

| Variable | Default | Meaning |
| --- | --- | --- |
| `KIMODO_MODELS` | `soma-rp-v1.1` | Space-separated motion models to fetch: `soma-rp-v1.1 soma-seed-v1.1 g1-rp-v1 g1-seed-v1`. Adding one later triggers a download of just that GGUF on next start. Explicitly empty disables downloading (mount weights yourself). |
| `KIMODO_BACKEND` | `cpu` (compose) | The compose default works everywhere. With the variable removed, the binary auto-selects Vulkan when a usable device is present. |
| `KIMODO_TEXT_QUANTIZATION` | `q8_0` | Text encoder fetched on first start: `bf16`, `q8_0`, `q6_k`, `q5_k`, `q4_k`, `q4_k_m`. Smaller variants trade prompt fidelity for download size and memory. A pre-existing bundle (packed or legacy F32) is reused regardless. |
| `KIMODO_TEXT_LAYER_CHUNK` | `32` | Text-encoder layers per GGML graph. Lower values trade throughput for memory headroom. |
| `KIMODO_TEXT_RESIDENT_LIMIT_MIB` | `10240` | VRAM/RAM ceiling for resident text-encoder weights; layers stream beyond it. Lower this on small GPUs (see GPU section). |
| `KIMODO_THREADS` | all cores | CPU threads for CPU inference. Set when capping the container's CPUs, since the auto value does not follow cgroup limits. |
| `KIMODO_PORT` | `8094` | Port inside the container (change the `ports:` mapping instead for host-side changes). |
| `HF_TOKEN` | unset | Optional Hugging Face token for the first-start download. |

Volume layout (`kimodo-data` → `/data`):

```
/data/models/…f32.gguf                  # motion GGUFs (~1.1 GB each)
/data/Llama-3-Kimodo-<variant>.gguf     # packed text encoder (Q8_0: 7.6 GB)
/data/tokenizer.gguf
/data/generated/llm2vec-text-bundle/    # legacy F32 component directory (reused if present)
/data/demo-output/<id>/                 # animations: prompt, .f32 buffers, animation.glb
```

To place the data on a specific disk, replace the named volume with a
bind mount:

```yaml
    volumes:
      - /srv/kimodo-data:/data
```

## Using the animations

- The web UI's **Download animated GLB** gives a skeleton-animated GLB
  (node hierarchy + rotation/translation tracks, ready for Three.js and
  most DCC tools).
- For a **skinned mesh** (Unreal Engine 5 / Blender retargeting, SOMA
  only), run the exporter inside the container against a finished
  animation directory:

  ```sh
  docker exec kimodo python3 /app/scripts/export_glb.py \
      --motion-dir /data/demo-output/<id> --output /data/demo-output/<id>/skinned.glb
  ```

  BVH export (metre-to-centimetre scaling for Unreal) works the same way
  with `/app/scripts/export_bvh.py`.

## GPU passthrough (optional)

The image contains the GGML Vulkan backend; once a usable device is
visible in the container, inference picks it up. Three changes:

1. Pass the GPU through via CDI (requires the NVIDIA container toolkit on
   the host; generate the spec once with
   `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`):

   ```yaml
    devices:
      - nvidia.com/gpu=all
    environment:
      NVIDIA_DRIVER_CAPABILITIES: "graphics,compute,utility"
   ```

2. Remove the `KIMODO_BACKEND: "cpu"` line.
3. On a 4 GB GPU, bound the text-encoder residency: the default ceiling
   (10 GiB) assumes larger cards. Set
   `KIMODO_TEXT_RESIDENT_LIMIT_MIB: "512"`–`"1024"` so encoder layers
   stream instead of filling VRAM alongside the motion model (~2.8 GB).
   A smaller `KIMODO_TEXT_QUANTIZATION` (e.g. `q4_k_m`) shrinks the
   per-layer footprint further.

Verify with `docker exec kimodo vulkaninfo --summary` (should list the
GPU) and by checking `docker logs` during a generation.

The image ships the GLVND libraries (`libEGL.so.1`/`libGLX.so.0`) that
the NVIDIA ICD dlopens during Vulkan init even without a display — a
minimal image without them makes the ICD fail inside the container with
`Could not get 'vkCreateInstance'` while the host works fine.

Measured reference point (24-core CPU, GTX 1050 Ti 4 GB, legacy F32 text
bundle): a 60-frame, 10-step clip took ~45 s on CPU and ~36 s on GPU.
Pascal-class GPUs see a modest speedup only (no fp16/coopmat paths; the
project enforces F32 Vulkan for reference parity) — the GPU's main win is
keeping weights out of system RAM.

## Security notes

The demo serves plain HTTP with **no authentication** and runs one
generation job at a time. Keep it on a trusted network; do not expose
port 8094 to the internet directly — put it behind a reverse proxy with
TLS and your own access control if remote access is needed.

## Platform notes

The image is built and tested on **linux/amd64**. The build itself is
architecture-agnostic (Debian trixie base, GGML CPU backend); arm64
should build with `KIMODO_ENABLE_VULKAN=OFF` but is untested. There is no
published registry image yet — build from source as above; adding a
multi-arch CI build to a registry (ghcr.io) is a natural follow-up for
maintainers.

## Updating

```sh
git pull
docker compose up --build        # or: docker build -t kimodo:latest . + Portainer re-deploy
```

The container is recreated; weights and gallery survive in the volume. To
reset the gallery, remove the data volume.

## Troubleshooting

- **Container stuck in first start / unhealthy** — watch the logs; an
  interrupted download resumes on restart.
- **Generation fails with `vk::createInstance: ErrorIncompatibleDriver`**
  — the binary is Vulkan-enabled but the container has no usable Vulkan
  device, and GGML's Vulkan enumeration can abort instead of falling
  back. Set `KIMODO_BACKEND: "cpu"` (the shipped compose default).
- **GPU not detected** — `docker exec kimodo vulkaninfo --summary` should
  list the GPU. Check the device block, the host toolkit, and that
  `NVIDIA_DRIVER_CAPABILITIES` includes `graphics`.
- **Out of memory (host RAM or 4 GB-class GPUs)** — lower
  `KIMODO_TEXT_RESIDENT_LIMIT_MIB` (e.g. `512`) and/or pick a smaller
  `KIMODO_TEXT_QUANTIZATION`; the encoder and the denoiser are never
  resident at the same time, so this bounds the peak.
- **Slow generation** — diffusion runs 100 steps by default; fewer steps
  via the UI trade quality for speed.
