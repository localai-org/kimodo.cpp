# Running the kimodo.cpp demo in Docker on Portainer

This guide deploys the kimodo.cpp text-to-motion web demo as a Portainer
stack:

- **`kimodo-demo`** (Go HTTP server) serves the web UI and animation
  gallery on port **8094** and shells out to **`kmd-generate`** for each
  generation job (one job at a time).
- Inference picks the **GGML Vulkan backend automatically when the
  container has a GPU**, and falls back to the **CPU backend** otherwise —
  no configuration needed either way.
- Model weights (SOMA RP v1.1: ~1.1 GB motion model + ~15 GB shared
  LLM2Vec text encoder, F32) are downloaded into the data volume on
  **first start** and verified against the published SHA-256 manifests.

The image is built on the Docker host from this repository; Portainer then
deploys `docker-compose.yml` as a stack. Total disk needed: ~20 GB
(image plus weights, plus room for animations).

## 1. Build the image on the Docker host

On the host (or any machine that controls its Docker daemon), from a
checkout of this repository:

```sh
docker build -t kimodo:latest .
```

The build compiles `kmd-generate`/`kmd-inspect` (CMake/Ninja, C++23) and
the Go demo, and checks out GGML itself at the pinned submodule commit —
you do **not** need `git submodule update --init` before building.

## 2. Deploy the stack in Portainer

1. Open Portainer (`https://192.168.1.231:9443`) → **Stacks** →
   **Add stack**.
2. Name it `kimodo`, choose **Web editor**, and paste the contents of
   `docker-compose.yml` from this repository.
3. Click **Deploy the stack**.

Portainer reuses the `kimodo:latest` image built in step 1. (The
`build:` key in the compose file only takes effect when deploying with
`docker compose up` from the repository itself.)

### GPU passthrough

The image contains the GGML Vulkan backend and auto-selects it whenever a
usable Vulkan device is visible in the container; otherwise it runs on
the CPU backend. To pass the host GPU through, add this block to the
`kimodo` service (requires the NVIDIA container toolkit and a working
host driver):

```yaml
    devices:
      - nvidia.com/gpu=all
    environment:
      NVIDIA_DRIVER_CAPABILITIES: "graphics,compute,utility"
```

The CDI form above (`devices:`) is preferred on modern Docker/toolkit
setups; generate the CDI spec on the host with
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.

The image ships the GLVND libraries (`libEGL.so.1`/`libGLX.so.0`) that
the NVIDIA ICD dlopens during Vulkan init even without a display — a
minimal image without them makes the ICD fail with
`Could not get 'vkCreateInstance'` inside the container while everything
works on the host.

**VRAM sizing:** the text encoder is loaded onto the GPU in
`KIMODO_TEXT_LAYER_CHUNK`-sized chunks (~440 MB per layer at F32), and
the motion model needs a bit under 3 GB. On a 4 GB GPU use
`KIMODO_TEXT_LAYER_CHUNK: "4"`; larger cards can keep the default `8`.

Verify GPU visibility with:

```sh
docker exec kimodo vulkaninfo --summary   # should list the GPU
```

> **Note for this homelab (192.168.1.231):** the GPU now works. The saga,
> for the record: the original 24.04 + `580-server` stack had a driver
> whose GLX/Vulkan component refused to initialize (fixed only by a full
> OS upgrade to the 7.0 kernel plus a DKMS-built 580.173.02 driver);
> the OS upgrade in turn removed `nvidia-container-toolkit`, which was
> reinstalled, and the CDI spec regenerated. Measured on the GTX 1050 Ti:
> Vulkan 1.4.312, a 60-frame/10-step clip in 36 s (vs 45 s on the 24 CPU
> cores) at ~2.8 GB VRAM with chunk=4.


### First start

The container logs (`Stacks → kimodo → Logs` or `docker logs -f kimodo`)
show the weight download progress. The HTTP server binds **after** the
download completes, so the container may show as *starting/unhealthy* for
the duration (the healthcheck allows 30 minutes). Subsequent starts skip
the download entirely.

Open `http://192.168.1.231:8094`, pick a model, enter a prompt
(60–150 frames per segment), and generate. Results appear in the gallery
with a **Download animated GLB** button.

## 3. Configuration

All knobs are environment variables on the `kimodo` service (edit them in
the Portainer stack editor and redeploy — the volume keeps your weights
and gallery):

| Variable | Default | Meaning |
| --- | --- | --- |
| `KIMODO_MODELS` | `soma-rp-v1.1` | Space-separated motion models to fetch: `soma-rp-v1.1 soma-seed-v1.1 g1-rp-v1 g1-seed-v1`. Adding one later triggers a download of just that GGUF on next start. |
| `KIMODO_BACKEND` | unset (auto) | Vulkan when a device is present, else CPU. Set `cpu` to force CPU even with a GPU. |
| `KIMODO_THREADS` | all cores | CPU threads for CPU inference. Set when capping the container's CPUs, since the auto value does not follow cgroup limits. |
| `KIMODO_TEXT_LAYER_CHUNK` | `8` | Text-encoder layers resident at once (1–32). Lower halves peak memory (VRAM or RAM) at some encoding-speed cost. |
| `KIMODO_PORT` | `8094` | Port inside the container (change the `ports:` mapping instead for host-side changes). |

Volume layout (`kimodo-data` → `/data`):

```
/data/models/…f32.gguf                  # motion GGUFs (~1.1 GB each)
/data/generated/llm2vec-text-bundle/    # shared LLM2Vec text encoder (~15 GB)
/data/demo-output/<id>/                 # animations: prompt, .f32 buffers, animation.glb
```

To avoid re-downloading the ~16 GB of weights when redeploying (or to
place them on a specific disk), replace the named volume with a bind
mount:

```yaml
    volumes:
      - /home/ai/kimodo-data:/data
```

## 4. Using the animations

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

## 5. Updating

```sh
# on the host, from the repository checkout
git pull && docker build -t kimodo:latest .
```

Then Portainer → stack → **Update the stack** (the container is recreated;
weights and gallery survive in the volume). To also reset the gallery,
remove the `kimodo-data` volume afterwards via Portainer's Volumes view.

## 6. Troubleshooting

- **Container stuck in first start / unhealthy** — watch the logs; an
  interrupted download resumes on restart.
- **Generation fails with `vk::createInstance: ErrorIncompatibleDriver`**
  — the binary is Vulkan-enabled but the container has no usable Vulkan
  device, and GGML's Vulkan enumeration can abort instead of falling
  back. Set `KIMODO_BACKEND: "cpu"` in the stack environment (this is
  the shipped default) and remove it only when GPU passthrough works.
- **Vulkan not detected** — `docker exec kimodo vulkaninfo --summary`
  should list the GPU. If not, check the device block, the host toolkit,
  and that `NVIDIA_DRIVER_CAPABILITIES` includes `graphics`.  The image
  ships an NVIDIA Vulkan ICD manifest for the injected driver.
- **Out of memory (host RAM or 4 GB VRAM class GPUs)** — lower
  `KIMODO_TEXT_LAYER_CHUNK` (e.g. `4`); the encoder and the denoiser are
  never resident at the same time, so this bounds the peak.
- **Slow generation** — diffusion runs 100 steps by default (fewer steps
  = faster, at quality cost). On a GTX 1050 Ti the Vulkan backend mainly
  wins by keeping weights out of system RAM rather than raw speed; Pascal
  has no cooperative matrices, and the project disables those code paths
  anyway to preserve its F32 reference parity.
