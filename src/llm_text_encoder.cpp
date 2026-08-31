// Attention layout is independently implemented with llama.cpp
// src/llama-graph.cpp at 78ec4c378031811671d1c76a067acbee4f4c56ce as a
// reference. No llama.cpp source is copied.
#include "llm_text_encoder.hpp"
#include "llm_tokenizer.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <gguf.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kimodo::detail {
namespace {
constexpr int64_t hidden = 4096, heads = 32, kv_heads = 8, head_dim = 128;

bool use_vulkan() {
    const char *choice = std::getenv("KIMODO_BACKEND");
    return !choice || std::string_view(choice) != "cpu";
}

int layer_chunk_size() {
    constexpr int fallback = 8;
    const char *value = std::getenv("KIMODO_TEXT_LAYER_CHUNK");
    if (!value) return fallback;
    int parsed = 0;
    const auto [end, error] = std::from_chars(value, value + std::strlen(value), parsed);
    if (error != std::errc{} || *end != '\0' || parsed < 1 || parsed > 32)
        throw std::runtime_error("KIMODO_TEXT_LAYER_CHUNK must be in 1..32");
    return parsed;
}

struct component {
    ggml_context *ctx = nullptr;
    gguf_context *file = nullptr;
    ggml_backend_buffer_t weights = nullptr;
    ~component() {
        if (weights) ggml_backend_buffer_free(weights);
        if (file) gguf_free(file);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor *tensor(const char *name) const { return ggml_get_tensor(ctx, name); }
};

std::unique_ptr<component> open_component(const std::filesystem::path &path, ggml_backend_t backend) {
    auto result = std::make_unique<component>();
    gguf_init_params params{true, &result->ctx};
    result->file = gguf_init_from_file(path.string().c_str(), params);
    if (!result->file || !result->ctx)
        throw std::runtime_error("cannot load text component " + path.string());
    result->weights = ggml_backend_alloc_ctx_tensors(result->ctx, backend);
    if (!result->weights) throw std::runtime_error("cannot allocate text component " + path.string());

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot reopen text component " + path.string());
    const auto data_offset = gguf_get_data_offset(result->file);
    std::vector<char> scratch(8U * 1024U * 1024U);
    for (int64_t i = 0; i < gguf_get_n_tensors(result->file); ++i) {
        auto *tensor = ggml_get_tensor(result->ctx, gguf_get_tensor_name(result->file, i));
        if (!tensor || (tensor->type != GGML_TYPE_BF16 && tensor->type != GGML_TYPE_F32))
            throw std::runtime_error("invalid text tensor");
        const size_t bytes = ggml_nbytes(tensor);
        const size_t offset = gguf_get_tensor_offset(result->file, i);
        in.seekg(static_cast<std::streamoff>(data_offset + offset));
        for (size_t done = 0; done < bytes;) {
            const size_t n = std::min(scratch.size(), bytes - done);
            in.read(scratch.data(), static_cast<std::streamsize>(n));
            if (!in) throw std::runtime_error("truncated text tensor");
            ggml_backend_tensor_set(tensor, scratch.data(), done, n);
            done += n;
        }
    }
    return result;
}

std::vector<float> read_output(ggml_tensor *tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
    if (tensor->type == GGML_TYPE_BF16) {
        std::vector<uint16_t> raw(result.size());
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(uint16_t));
        for (size_t i = 0; i < result.size(); ++i) {
            const uint32_t bits = uint32_t(raw[i]) << 16;
            std::memcpy(&result[i], &bits, sizeof(float));
        }
        return result;
    }
    throw std::runtime_error("unsupported text output type");
}

ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *weight) {
    auto *normalized = ggml_rms_norm(ctx, x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32), 1e-5F);
    if (x->type == GGML_TYPE_BF16) {
        normalized = ggml_cast(ctx, normalized, GGML_TYPE_BF16);
        auto *repeated = ggml_repeat(ctx, weight, normalized);
        return ggml_cast(ctx, ggml_mul(ctx, ggml_cast(ctx, normalized, GGML_TYPE_F32),
            ggml_cast(ctx, repeated, GGML_TYPE_F32)), GGML_TYPE_BF16);
    }
    return ggml_mul(ctx, normalized, ggml_repeat(ctx, ggml_cast(ctx, weight, GGML_TYPE_F32), normalized));
}

ggml_tensor *repeat_kv(ggml_context *ctx, ggml_tensor *x, int64_t seq) {
    auto *value = ggml_reshape_4d(ctx, x, head_dim, kv_heads, 1, seq);
    auto *shape = ggml_new_tensor_4d(ctx, x->type, head_dim, kv_heads, heads / kv_heads, seq);
    value = ggml_repeat(ctx, value, shape);
    value = ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, value, head_dim, heads, seq);
}

ggml_tensor *layer_graph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *positions,
                         const component &model, int64_t seq) {
    auto base = [&](const char *name, ggml_tensor *value) {
        const std::string prefix(name);
        auto *weight = model.tensor((prefix + "_base.weight").c_str());
        if (!weight || weight->type != GGML_TYPE_BF16) throw std::runtime_error("missing base projection");
        // Vulkan's BF16 matrix-vector kernel rejects BF16 right operands. A
        // F32 cast preserves the BF16 values while taking its supported path.
        return ggml_mul_mat(ctx, weight, value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32));
    };
    auto linear = [&](const char *name, ggml_tensor *value) {
        const std::string prefix(name);
        auto *a = model.tensor((prefix + "_lora_a.weight").c_str());
        auto *b = model.tensor((prefix + "_lora_b.weight").c_str());
        if (!a || !b) throw std::runtime_error("missing LoRA projection");
        auto *lora = ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a,
            value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32)));
        return ggml_add(ctx, ggml_cast(ctx, base(name, value), GGML_TYPE_F32), ggml_scale(ctx, lora, 2.F));
    };
    auto *attn_norm = model.tensor("attn_norm.weight");
    auto *ffn_norm = model.tensor("ffn_norm.weight");
    if (!attn_norm || !ffn_norm) throw std::runtime_error("missing layer norm");
    auto *residual = ggml_cast(ctx, x, GGML_TYPE_BF16);
    auto *q = linear("attn_q_proj", norm(ctx, residual, attn_norm));
    auto *k = linear("attn_k_proj", norm(ctx, residual, attn_norm));
    auto *v = linear("attn_v_proj", norm(ctx, residual, attn_norm));
    q = ggml_reshape_3d(ctx, q, head_dim, heads, seq);
    k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, seq);
    v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, seq);
    q = ggml_rope_ext(ctx, q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 500000.F, 1, 0, 1, 0, 0);
    k = ggml_rope_ext(ctx, k, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 500000.F, 1, 0, 1, 0, 0);
    k = repeat_kv(ctx, k, seq);
    v = repeat_kv(ctx, v, seq);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    auto *probability = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.F / std::sqrt(float(head_dim))));
    v = ggml_cont(ctx, ggml_transpose(ctx, v));
    auto *attention = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, v, probability), 0, 2, 1, 3));
    auto *output = ggml_add(ctx, ggml_cast(ctx, residual, GGML_TYPE_F32),
        linear("attn_o_proj", ggml_reshape_2d(ctx, attention, hidden, seq)));
    auto *hidden_norm = norm(ctx, output, ffn_norm);
    auto *gate = ggml_silu(ctx, linear("ffn_gate_proj", hidden_norm));
    output = ggml_add(ctx, output, linear("ffn_down_proj", ggml_mul(ctx, gate, linear("ffn_up_proj", hidden_norm))));
    return output;
}

std::vector<float> run_layer_chunk(const std::vector<std::unique_ptr<component>> &layers,
                                   const std::vector<float> &input, ggml_backend_t backend) {
    const int64_t seq = static_cast<int64_t>(input.size() / hidden);
    auto *ctx = ggml_init({128ULL * 1024ULL * 1024ULL, nullptr, true});
    if (!ctx) throw std::runtime_error("layer chunk graph allocation failed");
    auto cleanup = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ctx, ggml_free);
    auto *state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
    auto *input_state = state;
    auto *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_set_input(state);
    ggml_set_input(positions);
    for (const auto &layer : layers) state = layer_graph(ctx, state, positions, *layer, seq);
    auto *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, state);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("layer chunk backend allocation failed");
    auto release = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(buffer, ggml_backend_buffer_free);
    std::vector<int32_t> position_values(static_cast<size_t>(seq));
    for (int32_t i = 0; i < seq; ++i) position_values[static_cast<size_t>(i)] = i;
    ggml_backend_tensor_set(input_state, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("layer chunk graph failed");
    return read_output(state);
}
} // namespace

struct llm_text_encoder::impl {
    std::filesystem::path directory;
    std::unique_ptr<llm_tokenizer> tokenizer;
    ggml_backend_t backend = nullptr;
    ~impl() { if (backend) ggml_backend_free(backend); }
};

llm_text_encoder::~llm_text_encoder() = default;

std::expected<std::unique_ptr<llm_text_encoder>, std::string> llm_text_encoder::load(std::string_view directory) try {
    const auto path = std::filesystem::path(directory);
    if (!std::filesystem::is_directory(path)) return std::unexpected("text model must be a component directory");
    for (const auto &name : {"tokenizer.gguf", "embedding.gguf", "final-norm.gguf"})
        if (!std::filesystem::is_regular_file(path / name)) return std::unexpected("text bundle missing " + std::string(name));
    for (int i = 0; i < 32; ++i) { char name[32]; std::snprintf(name, sizeof(name), "layer-%02d.gguf", i); if (!std::filesystem::is_regular_file(path / name)) return std::unexpected("text bundle missing " + std::string(name)); }
    auto result = std::unique_ptr<llm_text_encoder>(new llm_text_encoder);
    result->impl_ = std::make_unique<impl>();
#if defined(KIMODO_HAVE_GGML_VULKAN)
    if (use_vulkan() && ggml_backend_vk_get_device_count()) result->impl_->backend = ggml_backend_vk_init(0);
#endif
    if (!result->impl_->backend) {
        result->impl_->backend = ggml_backend_cpu_init();
        if (!result->impl_->backend) return std::unexpected("cannot initialize text backend");
        ggml_backend_cpu_set_n_threads(result->impl_->backend, static_cast<int>(std::max(1U, std::thread::hardware_concurrency())));
    }
    auto tokenizer = llm_tokenizer::load((path / "tokenizer.gguf").string());
    if (!tokenizer) return std::unexpected(tokenizer.error());
    result->impl_->directory = path;
    result->impl_->tokenizer = std::move(*tokenizer);
    return result;
} catch (const std::exception &error) { return std::unexpected(error.what()); }

std::expected<std::array<float, 4096>, std::string> llm_text_encoder::encode(std::string_view prompt) const try {
    auto ids = impl_->tokenizer->encode(prompt);
    if (!ids) return std::unexpected(ids.error());
    if (ids->size() < 2 || ids->size() > 512) return std::unexpected("prompt token count must be in 1..511 excluding BOS");
    std::vector<float> state;
    {
        auto embedding = open_component(impl_->directory / "embedding.gguf", impl_->backend);
        auto *weight = embedding->tensor("token_embedding.weight");
        if (!weight) throw std::runtime_error("text bundle missing token_embedding.weight");
        auto *ctx = ggml_init({2ULL * 1024ULL * 1024ULL, nullptr, true});
        if (!ctx) throw std::runtime_error("embedding graph allocation failed");
        auto cleanup = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ctx, ggml_free);
        auto *indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids->size());
        ggml_set_input(indices);
        auto *rows = ggml_get_rows(ctx, weight, indices);
        auto *graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, rows);
        auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);
        if (!buffer) throw std::runtime_error("embedding graph backend allocation failed");
        auto release = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(buffer, ggml_backend_buffer_free);
        std::vector<int32_t> values(ids->begin(), ids->end());
        ggml_backend_tensor_set(indices, values.data(), 0, values.size() * sizeof(int32_t));
        if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("embedding graph failed");
        state = read_output(rows);
    }
    const int chunk = layer_chunk_size();
    for (int first = 0; first < 32; first += chunk) {
        std::vector<std::unique_ptr<component>> layers;
        for (int i = first; i < std::min(first + chunk, 32); ++i) {
            char name[32]; std::snprintf(name, sizeof(name), "layer-%02d.gguf", i);
            layers.push_back(open_component(impl_->directory / name, impl_->backend));
        }
        state = run_layer_chunk(layers, state, impl_->backend);
    }
    auto final = open_component(impl_->directory / "final-norm.gguf", impl_->backend);
    auto *weight = final->tensor("final_norm.weight");
    if (!weight) throw std::runtime_error("text bundle missing final_norm.weight");
    auto *ctx = ggml_init({8ULL * 1024ULL * 1024ULL, nullptr, true});
    if (!ctx) throw std::runtime_error("final norm graph allocation failed");
    auto cleanup = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ctx, ggml_free);
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ids->size()); ggml_set_input(input);
    auto *normalized = ggml_rms_norm(ctx, input, 1e-5F);
    auto *output = ggml_mul(ctx, normalized, ggml_repeat(ctx, ggml_cast(ctx, weight, GGML_TYPE_F32), normalized));
    auto *graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
    auto *buffer = ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);
    if (!buffer) throw std::runtime_error("final norm graph backend allocation failed");
    auto release = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(buffer, ggml_backend_buffer_free);
    ggml_backend_tensor_set(input, state.data(), 0, state.size() * sizeof(float));
    if (ggml_backend_graph_compute(impl_->backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("final norm graph failed");
    state = read_output(output);
    std::array<float, 4096> pooled{};
    for (size_t token = 1; token < ids->size(); ++token)
        for (size_t dim = 0; dim < pooled.size(); ++dim) pooled[dim] += state[token * pooled.size() + dim];
    for (float &value : pooled) value /= float(ids->size() - 1);
    return pooled;
} catch (const std::exception &error) { return std::unexpected(error.what()); }
} // namespace kimodo::detail
