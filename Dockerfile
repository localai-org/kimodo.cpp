# kimodo.cpp container: text-to-motion web demo (Go server + kmd-generate).
#
# Build from the repository root:
#   docker build -t kimodo:latest .
#
# Inference runs on the GGML Vulkan backend when the container has a GPU
# (NVIDIA container toolkit + device request) and transparently falls back
# to the CPU backend otherwise.  Weights are fetched into /data on first
# start; see docker/entrypoint.sh and docs/docker_portainer.md.

# --- builder -----------------------------------------------------------------
FROM golang:1.26-trixie AS builder

# Pinned GGML submodule commit (git ls-tree HEAD ggml in this repository).
ARG GGML_REF=8c63e70982c95ceb862e3a1073a2c1beef75d60a
ARG GGML_REPO=https://github.com/ggml-org/ggml.git

# cmake/ninja/git drive the C++ build; libvulkan-dev, glslc and
# spirv-headers are needed to compile the GGML Vulkan backend shaders
# (spirv-headers ships the SPIRV-HeadersConfig.cmake that GGML's CMake
# looks for).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake ninja-build git libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*

# GGML is a pinned submodule; check it out explicitly so the build context
# does not depend on the submodule being initialised (plain git clones and
# Portainer builds do not include it).
RUN git clone "$GGML_REPO" /src/ggml \
    && git -C /src/ggml checkout "$GGML_REF"

WORKDIR /src/kimodo
COPY . .
RUN mkdir -p /app/bin /app/lib \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DKIMODO_BUILD_TESTS=OFF \
        -DKIMODO_ENABLE_FUZZERS=OFF \
        -DKIMODO_ENABLE_VULKAN=ON \
        -DKIMODO_GGML_SOURCE_DIR=/src/ggml \
    && cmake --build build --target kmd-generate kmd-inspect --parallel "$(nproc)" \
    && go build -o /app/bin/kimodo-demo ./demo \
    && cp -L build/kmd-generate build/kmd-inspect /app/bin/ \
    && find build/ggml/src -maxdepth 2 -name 'libggml*.so*' -exec cp -L '{}' /app/lib/ \;

# --- runtime -------------------------------------------------------------------
FROM debian:trixie-slim

# python3 + huggingface_hub fetch and verify the GGUF weights on first
# start; libvulkan1 and vulkan-tools let the GGML Vulkan backend enumerate
# the GPU (vulkaninfo is handy when debugging).  libx11/libxext satisfy
# the injected libGLX_nvidia.so (the NVIDIA Vulkan ICD library), libgomp1
# is the OpenMP runtime the GGML backends link against, and the libglvnd
# pieces provide libEGL.so.1/libGLX.so.0, which the NVIDIA ICD dlopens
# during Vulkan init even though no display is used.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-pip libvulkan1 vulkan-tools \
        libx11-6 libxext6 libgomp1 libegl1 libglx0 libglvnd0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir huggingface_hub

# The NVIDIA container runtime injects the host driver libraries but not
# always the Vulkan ICD manifest that lets the loader find them; ship it
# ourselves.  The manifest is inert when no NVIDIA driver is mounted.
RUN mkdir -p /usr/share/vulkan/icd.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json

COPY --from=builder /app/ /app/
COPY scripts/download_gguf_weights.py scripts/export_glb.py scripts/export_bvh.py /app/scripts/
COPY docker/entrypoint.sh /app/bin/entrypoint.sh
RUN chmod 0755 /app/bin/entrypoint.sh

ENV LD_LIBRARY_PATH=/app/lib \
    HF_HOME=/data/.cache/huggingface \
    KIMODO_MODELS=soma-rp-v1.1

VOLUME /data
EXPOSE 8094

# The first start downloads ~16 GB of weights (SOMA RP v1.1: ~1.1 GB
# motion + ~15 GB shared text encoder) before the HTTP server binds, so
# allow a generous start period; later starts are immediate because the
# volume keeps both weights and the animation gallery.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python3 -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('KIMODO_PORT', '8094') + '/api/models', timeout=3)"

ENTRYPOINT ["/app/bin/entrypoint.sh"]
