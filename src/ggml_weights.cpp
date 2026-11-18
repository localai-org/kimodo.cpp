#include "ggml_weights.hpp"
#include "gguf.hpp"
#include "motion_graph_cache.hpp"
#include "profile.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>
#if defined(KIMODO_HAVE_GGML_VULKAN)
#include <ggml-vulkan.h>
#endif
#if defined(KIMODO_HAVE_GGML_METAL)
#include <ggml-metal.h>
#endif

namespace kimodo::detail {
namespace {

void configure_vulkan_f32_parity() noexcept {
#if defined(__unix__)
    // Kimodo's reference model is F32.  Current Vulkan cooperative-matrix
    // paths convert F32 inputs to FP16 on this GPU, which breaks parity.
    // Keep callers free to supply their own stricter environment, but make
    // the correct reference-first path the default.
    setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
    setenv("GGML_VK_DISABLE_COOPMAT2", "1", 0);
    setenv("GGML_VK_DISABLE_F16", "1", 0);
#endif
}

// GGML's CPU backend defaults to four threads.  That is a sensible library
// default, but makes a full diffusion sample use only a small fraction of a
// typical workstation.  Honour an explicit cap for predictable deployment
// and otherwise use the machine's advertised concurrency.
int cpu_thread_count() noexcept {
    constexpr unsigned fallback = 4;
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = fallback;

    if (const char *value = std::getenv("KIMODO_THREADS")) {
        char *end = nullptr;
        errno = 0;
        const long requested = std::strtol(value, &end, 10);
        if (errno == 0 && end != value && *end == '\0' && requested > 0 &&
            requested <= std::numeric_limits<int>::max()) {
            threads = static_cast<unsigned>(requested);
        }
    }
    return static_cast<int>(std::min<unsigned>(threads, std::numeric_limits<int>::max()));
}

} // namespace

std::expected<std::unique_ptr<ggml_motion_weights>, std::string> ggml_motion_weights::load(std::string_view path) {
    const auto load_started = std::chrono::steady_clock::now();
    auto checked = read_gguf_header(path);
    if (!checked) return std::unexpected(checked.error());
    if (auto valid = validate_motion_gguf(*checked); !valid) return std::unexpected(valid.error());
    auto result = std::unique_ptr<ggml_motion_weights>(new ggml_motion_weights);
    result->skeleton_ = checked->strings.at("kimodo.skeleton");
    result->motion_dim_ = static_cast<size_t>(checked->uints.at("kimodo.motion_dim"));
    result->body_dim_ = static_cast<size_t>(checked->uints.at("kimodo.body_dim"));
    gguf_init_params params{true, &result->context_};
    result->gguf_ = gguf_init_from_file(std::string(path).c_str(), params);
    if (!result->gguf_ || !result->context_) return std::unexpected("GGML could not load checked motion GGUF");
    // Vulkan is the normal inference path.  Keep the CPU backend as a
    // portability fallback, including for CI systems without a Vulkan ICD.
#if defined(KIMODO_HAVE_GGML_VULKAN)
    // Retain a deterministic CPU escape hatch for parity triage.  It is not
    // a performance mode; a captured fixture can establish whether a drift
    // belongs to the GGML graph or specifically to Vulkan.
    const bool force_cpu = [] {
        const char *value = std::getenv("KIMODO_BACKEND");
        return value && std::string_view(value) == "cpu";
    }();
    if (!force_cpu) {
        configure_vulkan_f32_parity();
        if (ggml_backend_vk_get_device_count() > 0) result->backend_ = ggml_backend_vk_init(0);
    }
#endif
#if defined(KIMODO_HAVE_GGML_METAL)
    // Apple GPUs.  Opt-in (KIMODO_BACKEND=metal) so the CPU path stays the
    // reference; a failed initialization falls through to the CPU backend.
    if (!result->backend_) {
        const char *value = std::getenv("KIMODO_BACKEND");
        if (value && std::string_view(value) == "metal") result->backend_ = ggml_backend_metal_init();
    }
#endif
    if (!result->backend_) {
        result->backend_ = ggml_backend_cpu_init();
        if (!result->backend_) return std::unexpected("GGML CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(result->backend_, cpu_thread_count());
    }
    result->buffer_ = ggml_backend_alloc_ctx_tensors(result->context_, result->backend_);
    if (!result->buffer_) return std::unexpected("GGML motion weight allocation failed");
    result->allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(result->backend_));
    if (!result->allocator_) return std::unexpected("GGML motion compute allocator creation failed");
    const double allocation_ms = profile_elapsed_ms(load_started);
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) return std::unexpected("cannot reopen motion GGUF");
    const size_t data_start = gguf_get_data_offset(result->gguf_);
    std::vector<char> scratch(8U*1024U*1024U);
    for (int64_t i=0;i<gguf_get_n_tensors(result->gguf_);++i) {
        auto *tensor = ggml_get_tensor(result->context_, gguf_get_tensor_name(result->gguf_, i));
        if (!tensor || tensor->type != GGML_TYPE_F32) return std::unexpected("motion GGUF contains an invalid non-F32 tensor");
        const size_t bytes=ggml_nbytes(tensor), offset=gguf_get_tensor_offset(result->gguf_, i);
        input.seekg(static_cast<std::streamoff>(data_start+offset));
        for(size_t done=0;done<bytes;) {
            const size_t chunk=std::min(scratch.size(), bytes-done);
            input.read(scratch.data(), static_cast<std::streamsize>(chunk));
            if (!input) return std::unexpected("short tensor data in motion GGUF");
            ggml_backend_tensor_set(tensor, scratch.data(), done, chunk); done+=chunk;
        }
    }
    if (profile_enabled()) {
        std::fprintf(stderr, "profile motion.weights allocation_ms=%.3f upload_ms=%.3f total_ms=%.3f\n",
                     allocation_ms, profile_elapsed_ms(load_started) - allocation_ms,
                     profile_elapsed_ms(load_started));
    }
    return result;
}
ggml_motion_weights::~ggml_motion_weights() {
    graph_cache_.reset();
    if (allocator_) ggml_gallocr_free(allocator_);
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (gguf_) gguf_free(gguf_);
    if (context_) ggml_free(context_);
    if (backend_) ggml_backend_free(backend_);
}
motion_graph_cache *ggml_motion_weights::graph_cache() const noexcept {
    return graph_cache_.get();
}
void ggml_motion_weights::graph_cache(std::unique_ptr<motion_graph_cache> cache) const noexcept {
    graph_cache_ = std::move(cache);
}
ggml_tensor *ggml_motion_weights::tensor(std::string_view name) const {
    return context_ ? ggml_get_tensor(context_, std::string(name).c_str()) : nullptr;
}
std::expected<std::vector<float>, std::string> ggml_motion_weights::f32_values(std::string_view name) const {
    const bool cache_on_host = name.starts_with("stats.");
    if (cache_on_host) {
        const auto found = host_f32_cache_.find(std::string(name));
        if (found != host_f32_cache_.end()) return found->second;
    }
    auto *value = tensor(name);
    if (!value || value->type != GGML_TYPE_F32) return std::unexpected("missing F32 GGML tensor: " + std::string(name));
    std::vector<float> result(static_cast<size_t>(ggml_nelements(value)));
    ggml_backend_tensor_get(value, result.data(), 0, result.size()*sizeof(float));
    if (cache_on_host) host_f32_cache_.emplace(std::string(name), result);
    return result;
}
} // namespace kimodo::detail
