// The operation ordering and attention tensor views follow llama.cpp
// src/llama-graph.cpp at 78ec4c378031811671d1c76a067acbee4f4c56ce.
// This is an independent implementation; no llama.cpp source is copied.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
struct weights {
    ggml_context *context = nullptr;
    gguf_context *file = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~weights() { if (buffer) ggml_backend_buffer_free(buffer); if (file) gguf_free(file); if (context) ggml_free(context); if (backend) ggml_backend_free(backend); }
    ggml_tensor *get(std::string_view name) const { return ggml_get_tensor(context, std::string(name).c_str()); }
};

std::vector<float> read_f32(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float))) throw std::runtime_error("invalid F32 fixture " + path.string());
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0); input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) throw std::runtime_error("short F32 fixture " + path.string());
    return values;
}

void write_f32(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write " + path.string());
    output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("short write " + path.string());
}

std::string layer_fixture_name(int layer) {
    char name[32];
    std::snprintf(name, sizeof(name), "layer_%02d_output.f32", layer);
    return name;
}

std::unique_ptr<weights> load(const char *path, bool use_vulkan) {
    auto value = std::make_unique<weights>();
    gguf_init_params params{true, &value->context};
    value->file = gguf_init_from_file(path, params);
#if defined(KIMODO_HAVE_GGML_VULKAN)
    if (use_vulkan && ggml_backend_vk_get_device_count() > 0) value->backend = ggml_backend_vk_init(0);
#else
    (void)use_vulkan;
#endif
    if (!value->backend) { value->backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(value->backend, 24); }
    value->buffer = ggml_backend_alloc_ctx_tensors(value->context, value->backend);
    if (!value->buffer) throw std::runtime_error("cannot allocate layer weights");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot reopen layer GGUF");
    const auto start = gguf_get_data_offset(value->file);
    std::vector<char> scratch(8 * 1024 * 1024);
    for (int64_t index = 0; index < gguf_get_n_tensors(value->file); ++index) {
        auto *tensor = ggml_get_tensor(value->context, gguf_get_tensor_name(value->file, index));
        if (!tensor || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_BF16)) throw std::runtime_error("layer GGUF must contain F32 or BF16 tensors");
        const size_t bytes = ggml_nbytes(tensor), offset = gguf_get_tensor_offset(value->file, index);
        input.seekg(static_cast<std::streamoff>(start + offset));
        for (size_t done = 0; done < bytes;) {
            const size_t chunk = std::min(scratch.size(), bytes - done);
            input.read(scratch.data(), static_cast<std::streamsize>(chunk));
            if (!input) throw std::runtime_error("short GGUF tensor payload");
            ggml_backend_tensor_set(tensor, scratch.data(), done, chunk); done += chunk;
        }
    }
    return value;
}

ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight) {
    if (!weight || weight->type != GGML_TYPE_BF16) throw std::runtime_error("expected BF16 RMSNorm weight");
    auto *normalized = ggml_rms_norm(ctx, x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32), 1e-5f);
    if (x->type == GGML_TYPE_BF16) {
        // Transformers LlamaRMSNorm normalizes in F32, then casts to its
        // input dtype before multiplying its BF16 scale.  GGML Vulkan has no
        // BF16 elementwise MUL shader, so perform the multiply in F32 and
        // explicitly round its result back to BF16; this is the same tensor
        // precision boundary as the upstream operation.
        normalized = ggml_cast(ctx, normalized, GGML_TYPE_BF16);
        auto *scale = ggml_repeat(ctx, weight, normalized);
        auto *product = ggml_mul(ctx, ggml_cast(ctx, normalized, GGML_TYPE_F32), ggml_cast(ctx, scale, GGML_TYPE_F32));
        return ggml_cast(ctx, product, GGML_TYPE_BF16);
    }
    return ggml_mul(ctx, normalized, ggml_repeat(ctx, ggml_cast(ctx, weight, GGML_TYPE_F32), normalized));
}

float bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> read_tensor(ggml_tensor *tensor) {
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
    if (tensor->type == GGML_TYPE_BF16) {
        std::vector<uint16_t> raw(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(uint16_t));
        std::vector<float> values(raw.size());
        std::transform(raw.begin(), raw.end(), values.begin(), bf16_to_f32);
        return values;
    }
    throw std::runtime_error("unexpected snapshot tensor type");
}

// Grouped-query KV expansion: repeat each KV head consecutively.  The tensor
// ordering is independently expressed, following the MHA layout conventions
// documented in llama.cpp's build_attn_mha (commit noted in this file header).
ggml_tensor *repeat_kv(ggml_context *ctx, ggml_tensor *value, int64_t head_dim, int64_t kv_heads, int64_t groups, int64_t seq) {
    auto *grouped = ggml_reshape_4d(ctx, value, head_dim, kv_heads, 1, seq);
    auto *shape = ggml_new_tensor_4d(ctx, value->type, head_dim, kv_heads, groups, seq);
    auto *repeated = ggml_repeat(ctx, grouped, shape);
    repeated = ggml_cont(ctx, ggml_permute(ctx, repeated, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, repeated, head_dim, kv_heads * groups, seq);
}

struct layer_result {
    std::vector<float> output;
    std::vector<std::pair<std::string, std::vector<float>>> snapshots;
};

layer_result run(const weights &w, const std::vector<float> &input, bool use_vulkan) {
    constexpr int64_t dim = 4096, seq = 16, heads = 32, kv_heads = 8, head_dim = 128, ff = 14336;
    if (input.size() != static_cast<size_t>(dim * seq)) throw std::runtime_error("unexpected layer input size");
    ggml_init_params params{96ULL * 1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("cannot allocate GGML graph context");
    auto cleanup = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ctx, ggml_free);
    auto *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
    auto *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_set_input(x); ggml_set_input(pos);
    auto base_linear = [&](const char *name, ggml_tensor *value) {
        const std::string stem(name);
        auto *base = w.get(stem + "_base.weight");
        auto *a = w.get(stem + "_lora_a.weight");
        auto *b = w.get(stem + "_lora_b.weight");
        if (!base || !a || !b || base->type != GGML_TYPE_BF16 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            throw std::runtime_error("invalid linear tensors for " + stem);
        }
        auto *base_input = value->type == GGML_TYPE_BF16 ? value : ggml_cast(ctx, value, GGML_TYPE_BF16);
        return ggml_mul_mat(ctx, base, base_input);
    };
    auto linear = [&](const char *name, ggml_tensor *value) {
        const std::string stem(name);
        auto *a = w.get(stem + "_lora_a.weight");
        auto *b = w.get(stem + "_lora_b.weight");
        auto *base_result = base_linear(name, value);
        if (!a || !b) throw std::runtime_error("missing LoRA tensors for " + stem);
        auto *lora_input = value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32);
        auto *lora_result = ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a, lora_input));
        return ggml_add(ctx, ggml_cast(ctx, base_result, GGML_TYPE_F32), ggml_scale(ctx, lora_result, 2.f));
    };
    auto *residual = ggml_cast(ctx, x, GGML_TYPE_BF16);
    auto *attn_norm = norm(ctx, residual, w.get("attn_norm.weight"));
    auto *h = attn_norm;
    auto *q_linear = linear("attn_q_proj", h);
    auto *k_linear = linear("attn_k_proj", h);
    auto *v_linear = linear("attn_v_proj", h);
    auto *q = ggml_reshape_3d(ctx, q_linear, head_dim, heads, seq);
    auto *k = ggml_reshape_3d(ctx, k_linear, head_dim, kv_heads, seq);
    auto *v = ggml_reshape_3d(ctx, v_linear, head_dim, kv_heads, seq);
    // Hugging Face Llama's rotate_half uses the NeoX half-split layout.
    q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 500000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 500000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
    k = repeat_kv(ctx, k, head_dim, kv_heads, heads / kv_heads, seq);
    v = repeat_kv(ctx, v, head_dim, kv_heads, heads / kv_heads, seq);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    auto *scores = ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.f / std::sqrt(static_cast<float>(head_dim)));
    auto *probabilities = ggml_soft_max(ctx, scores);
    v = ggml_cont(ctx, ggml_transpose(ctx, v));
    auto *attended = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, v, probabilities), 0, 2, 1, 3));
    auto *o_linear = linear("attn_o_proj", ggml_reshape_2d(ctx, attended, dim, seq));
    h = ggml_add(ctx, ggml_cast(ctx, residual, GGML_TYPE_F32), o_linear);
    residual = h;
    auto *ffn_norm = norm(ctx, h, w.get("ffn_norm.weight"));
    h = ffn_norm;
    auto *gate_linear = linear("ffn_gate_proj", h);
    auto *gate = ggml_silu(ctx, gate_linear);
    auto *up = linear("ffn_up_proj", h);
    auto *down = linear("ffn_down_proj", ggml_mul(ctx, gate, up));
    h = ggml_add(ctx, residual, down);
    const std::pair<const char *, ggml_tensor *> snapshot_tensors[] = {
        {"debug_input_norm", attn_norm}, {"debug_q", q_linear}, {"debug_k", k_linear},
        {"debug_v", v_linear}, {"debug_o", o_linear}, {"debug_post_norm", ffn_norm},
        {"debug_gate", gate_linear}, {"debug_up", up}, {"debug_down", down},
    };
    // Keep diagnostics live through execution.  Without this, the backend
    // allocator may reuse an intermediate's buffer after its final consumer.
    for (const auto &[_, tensor] : snapshot_tensors) ggml_set_output(tensor);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, h);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, w.backend);
    if (!buffer) throw std::runtime_error("cannot allocate layer graph");
    auto release = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(buffer, ggml_backend_buffer_free);
    std::vector<int32_t> positions(seq); for (int32_t index = 0; index < seq; ++index) positions[static_cast<size_t>(index)] = index;
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(w.backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("GGML layer graph failed");
    layer_result result;
    result.output = read_tensor(h);
    for (const auto &[name, tensor] : snapshot_tensors) {
        result.snapshots.emplace_back(name, read_tensor(tensor));
    }
    return result;
}
} // namespace

int main(int argc, char **argv) try {
    if (argc != 5 && argc != 7) { std::fprintf(stderr, "usage: %s LAYER.gguf FIXTURE_DIR LAYER_INDEX cpu|vulkan [INPUT.f32 OUTPUT.f32]\n", argv[0]); return 2; }
    const int layer = std::atoi(argv[3]);
    if (layer < 0 || layer >= 32) throw std::runtime_error("layer index must be in [0, 31]");
    const bool use_vulkan = std::string_view(argv[4]) == "vulkan";
    if (!use_vulkan && std::string_view(argv[4]) != "cpu") throw std::runtime_error("backend must be cpu or vulkan");
    const auto fixture = std::filesystem::path(argv[2]);
    const auto input = read_f32(argc == 7 ? std::filesystem::path(argv[5]) : fixture / (layer == 0 ? "token_embeddings.f32" : layer_fixture_name(layer - 1)));
    const auto expected = read_f32(fixture / layer_fixture_name(layer));
    const auto loaded = load(argv[1], use_vulkan);
    const auto actual = run(*loaded, input, use_vulkan);
    if (argc == 7) write_f32(argv[6], actual.output);
    float error = 0.f; for (size_t index = 0; index < actual.output.size(); ++index) error = std::max(error, std::abs(actual.output[index] - expected[index]));
    for (const auto &[name, values] : actual.snapshots) {
        if (!std::getenv("KIMODO_LLM_DEBUG")) continue;
        const auto reference = read_f32(fixture / (name + ".f32"));
        if (reference.size() != values.size()) throw std::runtime_error("unexpected snapshot size for " + name);
        float snapshot_error = 0.f;
        double squared_error = 0.0, squared_reference = 0.0;
        for (size_t index = 0; index < values.size(); ++index) {
            const float difference = values[index] - reference[index];
            snapshot_error = std::max(snapshot_error, std::abs(difference));
            squared_error += static_cast<double>(difference) * difference;
            squared_reference += static_cast<double>(reference[index]) * reference[index];
        }
        std::printf("%s max_abs=%g rel_l2=%g\n", name.c_str(), snapshot_error, std::sqrt(squared_error / squared_reference));
    }
    std::printf("layer%d %s max_abs=%g\n", layer, argv[4], error);
    return error < 2e-2f ? 0 : 1;
} catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
