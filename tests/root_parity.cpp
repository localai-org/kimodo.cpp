#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <gguf.h>
#include "motion_rep.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int D = 1024, LLM = 4096, H = 8, HD = 128, TEXT = 50, PREFIX = 52, T = 8, B = 3, S = PREFIX + T;
thread_local std::vector<std::pair<ggml_tensor *, std::vector<float>>> graph_inputs;

std::vector<float> read_f32(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() < 0 || static_cast<std::size_t>(in.tellg()) % sizeof(float)) throw std::runtime_error("invalid F32 fixture: " + path);
    std::vector<float> result(static_cast<std::size_t>(in.tellg()) / sizeof(float));
    in.seekg(0); in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!in) throw std::runtime_error("short F32 fixture: " + path); return result;
}
struct weights {
    ggml_context *ctx = nullptr; gguf_context *file = nullptr; ggml_backend_t backend = nullptr; ggml_backend_buffer_t buffer = nullptr; std::unordered_map<std::string, ggml_tensor *> tensors;
    explicit weights(const char *path) {
        gguf_init_params p{true, &ctx}; file = gguf_init_from_file(path, p);
        if (!file || !ctx) throw std::runtime_error("GGML could not load GGUF");
        for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
            const char *n = gguf_get_tensor_name(file, i); tensors.emplace(n, ggml_get_tensor(ctx, n));
        }
#if defined(_WIN32)
        _putenv_s("GGML_VK_DISABLE_COOPMAT", "1");
        _putenv_s("GGML_VK_DISABLE_COOPMAT2", "1");
        _putenv_s("GGML_VK_DISABLE_F16", "1");
#else
        setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
        setenv("GGML_VK_DISABLE_COOPMAT2", "1", 0);
        setenv("GGML_VK_DISABLE_F16", "1", 0);
#endif
#if defined(KIMODO_HAVE_GGML_VULKAN)
        if (ggml_backend_vk_get_device_count() > 0) backend = ggml_backend_vk_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("backend init failed");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); if (!buffer) throw std::runtime_error("weight buffer allocation failed");
        std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("cannot re-open GGUF");
        std::vector<char> scratch(8*1024*1024);
        const size_t data_start = gguf_get_data_offset(file);
        for (int64_t i=0;i<gguf_get_n_tensors(file);++i) {
            auto *tensor=ggml_get_tensor(ctx,gguf_get_tensor_name(file,i)); const size_t bytes=ggml_nbytes(tensor), offset=gguf_get_tensor_offset(file,i);
            input.seekg(static_cast<std::streamoff>(data_start+offset));
            for(size_t done=0;done<bytes;) { const size_t n=std::min(scratch.size(),bytes-done); input.read(scratch.data(),static_cast<std::streamsize>(n)); if(!input) throw std::runtime_error("short GGUF tensor read"); ggml_backend_tensor_set(tensor,scratch.data(),done,n); done+=n; }
        }
    }
    ~weights() { if (buffer) ggml_backend_buffer_free(buffer); if (file) gguf_free(file); if (ctx) ggml_free(ctx); if (backend) ggml_backend_free(backend); }
    ggml_tensor *get(const std::string &name) const { auto it = tensors.find(name); if (it == tensors.end()) throw std::runtime_error("missing tensor " + name); return it->second; }
};
std::vector<float> tensor_values(const weights &w, const std::string &name) {
    auto *tensor = w.get(name);
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
    return values;
}
std::vector<float> root_local_reference(const weights &w, const std::vector<float> &root, const std::vector<float> &mask) {
    const auto global_mean = tensor_values(w, "stats.global_root.mean");
    const auto global_std = tensor_values(w, "stats.global_root.std");
    const auto local_mean = tensor_values(w, "stats.local_root.mean");
    const auto local_std = tensor_values(w, "stats.local_root.std");
    if (global_mean.size()!=5 || global_std.size()!=5 || local_mean.size()!=4 || local_std.size()!=4) throw std::runtime_error("unexpected root statistics size");
    auto result = kimodo::detail::global_root_to_local_root(root, mask, B, T, global_mean, global_std, local_mean, local_std);
    if (!result) throw std::runtime_error(result.error());
    return std::move(*result);
}
ggml_tensor *input3(ggml_context *ctx, const float *data, int a, int b, int c) {
    auto *r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c); graph_inputs.emplace_back(r, std::vector<float>(data, data + static_cast<std::size_t>(a)*b*c)); return r;
}
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *bias) {
    if (w->ne[0] != x->ne[0]) throw std::runtime_error("linear dimension mismatch: weight=" + std::to_string(w->ne[0]) + " input=" + std::to_string(x->ne[0]));
    auto *y = ggml_mul_mat(ctx, w, x); ggml_mul_mat_set_prec(y, GGML_PREC_F32); return ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
}
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale, ggml_tensor *bias) {
    auto *n = ggml_norm(ctx, x, 1.e-5f); n = ggml_mul(ctx, n, ggml_repeat(ctx, scale, n)); return ggml_add(ctx, n, ggml_repeat(ctx, bias, n));
}
using capture = std::pair<ggml_tensor *, std::vector<float> *>;
std::vector<float> execute(ggml_context *ctx, ggml_tensor *out, std::size_t values, ggml_backend_t backend, const std::vector<capture> &captures = {}) {
    ggml_cgraph *graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, out);
    // Expand captured intermediates as graph outputs too.  Otherwise gallocr
    // is free to reuse their storage after the final output has consumed them.
    for (const auto &[tensor, _] : captures) ggml_build_forward_expand(graph, tensor);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) throw std::runtime_error("graph allocation failed");
    for (const auto &[tensor, data] : graph_inputs) ggml_backend_tensor_set(tensor, data.data(), 0, data.size()*sizeof(float));
    graph_inputs.clear();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("GGML graph failed");
    for (const auto &[tensor, values_out] : captures) { values_out->resize(static_cast<size_t>(ggml_nelements(tensor))); ggml_backend_tensor_get(tensor, values_out->data(), 0, values_out->size()*sizeof(float)); }
    std::vector<float> result(values); ggml_backend_tensor_get(out, result.data(), 0, values*sizeof(float));
    ggml_gallocr_free(alloc); return result;
}
ggml_tensor *transformer_layer(ggml_context *ctx, ggml_tensor *x, const weights &w, const std::string &p) {
    auto *qkv = linear(ctx, x, w.get(p+"self_attn.in_proj_weight"), w.get(p+"self_attn.in_proj_bias"));
    // Keep the reference layout explicit.  qkv is [3D, S, B], with the
    // PyTorch head dimension contiguous inside D.  The generic operations
    // below avoid relying on flash-attention's different head/sequence
    // conventions while we establish F32 parity.
    auto head = [&](size_t block, int h, int b) {
        return ggml_view_2d(ctx, qkv, HD, S, qkv->nb[1],
                            block*static_cast<size_t>(D)*sizeof(float) + static_cast<size_t>(b)*qkv->nb[2] + static_cast<size_t>(h)*HD*sizeof(float));
    };
    std::vector<ggml_tensor *> batches;
    for (int b = 0; b < B; ++b) {
        std::vector<ggml_tensor *> heads;
        for (int h = 0; h < H; ++h) {
            auto *q = ggml_cont(ctx, head(0, h, b));
            auto *k = ggml_cont(ctx, head(1, h, b));
            auto *v = ggml_cont(ctx, head(2, h, b));
            // scores are [key, query]; softmax's first dimension is exactly
            // the key axis required by PyTorch's attention implementation.
            auto *scores = ggml_mul_mat(ctx, k, q);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores = ggml_scale(ctx, scores, 1.f/std::sqrt(static_cast<float>(HD)));
            auto *prob = ggml_soft_max(ctx, scores);
            auto *value_product = ggml_mul_mat(ctx, prob, ggml_cont(ctx, ggml_transpose(ctx, v)));
            ggml_mul_mat_set_prec(value_product, GGML_PREC_F32);
            heads.push_back(ggml_transpose(ctx, value_product));
        }
        auto *joined = heads.front();
        for (int h = 1; h < H; ++h) joined = ggml_concat(ctx, joined, heads[h], 0);
        batches.push_back(ggml_reshape_3d(ctx, joined, D, S, 1));
    }
    auto *a = batches.front();
    for (int b = 1; b < B; ++b) a = ggml_concat(ctx, a, batches[b], 2);
    a = linear(ctx, a, w.get(p+"self_attn.out_proj.weight"), w.get(p+"self_attn.out_proj.bias"));
    x = layer_norm(ctx, ggml_add(ctx, x, a), w.get(p+"norm1.weight"), w.get(p+"norm1.bias"));
    auto *ff = linear(ctx, x, w.get(p+"linear1.weight"), w.get(p+"linear1.bias"));
    ff = ggml_gelu_erf(ctx, ff);
    ff = linear(ctx, ff, w.get(p+"linear2.weight"), w.get(p+"linear2.bias"));
    return layer_norm(ctx, ggml_add(ctx, x, ff), w.get(p+"norm2.weight"), w.get(p+"norm2.bias"));
}
}

int main(int argc, char **argv) try {
    if (argc != 4 || (std::string_view(argv[3]) != "root" && std::string_view(argv[3]) != "body")) {
        std::fprintf(stderr, "usage: kimodo-root-parity MODEL.gguf FIXTURE_DIR {root|body}\n"); return 2;
    }
    const std::string stage(argv[3]), prefix = stage + "_model.", f = std::string(argv[2]) + "/";
    const int input_dim = stage == "root" ? 546 : 545;
    const int output_dim = stage == "root" ? 5 : 268;
    weights w(argv[1]);
    const auto motion = read_f32(f+stage+"_input_0.f32"), mask = read_f32(f+stage+"_input_1.f32"), text = read_f32(f+stage+"_input_2.f32"), time = read_f32(f+stage+"_input_4.f32"), heading = read_f32(f+stage+"_input_5.f32"), expected = read_f32(f+stage+"_output.f32");
    const auto layer0_expected = stage == "root" ? read_f32(f+"root_layer0_output.f32") : std::vector<float>{};
    const auto layer0_input = stage == "root" ? read_f32(f+"root_layer0_input.f32") : std::vector<float>{};
    std::vector<float> padded_text(static_cast<size_t>(LLM)*TEXT*B), time_pe(static_cast<size_t>(D)*B), angle(static_cast<size_t>(2)*B), pos(static_cast<size_t>(D)*S);
    for (int b=0;b<B;++b) { std::memcpy(padded_text.data()+static_cast<size_t>(b)*TEXT*LLM, text.data()+static_cast<size_t>(b)*LLM, LLM*sizeof(float));
        const int t = static_cast<int>(time[b]); for (int d=0;d<D;d+=2) { const float z=t*std::pow(10000.f,-static_cast<float>(d)/D); time_pe[static_cast<size_t>(b)*D+d]=std::sin(z); time_pe[static_cast<size_t>(b)*D+d+1]=std::cos(z); }
        angle[2*b]=std::cos(heading[b]); angle[2*b+1]=std::sin(heading[b]); }
    for (int s=0;s<S;++s) for (int d=0;d<D;d+=2) { const float z=s*std::pow(10000.f,-static_cast<float>(d)/D); pos[static_cast<size_t>(s)*D+d]=std::sin(z); pos[static_cast<size_t>(s)*D+d+1]=std::cos(z); }
    std::vector<float> state;
    { ggml_init_params ip{128ULL*1024*1024,nullptr,true}; ggml_context *ctx=ggml_init(ip); if(!ctx) throw std::runtime_error("graph allocation failed");
      auto *motion_input=input3(ctx,motion.data(),input_dim,T,B); auto *m=linear(ctx,motion_input,w.get(prefix+"input_linear.weight"),w.get(prefix+"input_linear.bias"));
      auto *te=linear(ctx,input3(ctx,padded_text.data(),LLM,TEXT,B),w.get(prefix+"embed_text.weight"),w.get(prefix+"embed_text.bias"));
      auto *ti=linear(ctx,input3(ctx,time_pe.data(),D,1,B),w.get(prefix+"embed_timestep.time_embed.0.weight"),w.get(prefix+"embed_timestep.time_embed.0.bias")); ti=linear(ctx,ggml_silu(ctx,ti),w.get(prefix+"embed_timestep.time_embed.2.weight"),w.get(prefix+"embed_timestep.time_embed.2.bias"));
      auto *heading_input=input3(ctx,angle.data(),2,1,B); auto *he=linear(ctx,heading_input,w.get(prefix+"linear_first_heading_angle.weight"),w.get(prefix+"linear_first_heading_angle.bias"));
      auto *x=ggml_concat(ctx,ggml_concat(ctx,ggml_concat(ctx,te,ti,1),he,1),m,1); auto *position=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,D,S); graph_inputs.emplace_back(position,pos); x=ggml_add(ctx,x,ggml_repeat(ctx,position,x));
      state=execute(ctx,x,static_cast<size_t>(D)*S*B,w.backend); if(stage == "root") { float input_max=0.f; for(size_t j=0;j<state.size();++j)input_max=std::max(input_max,std::abs(state[j]-layer0_input[j])); std::printf("root layer0 input max_abs=%g\n",input_max); } ggml_free(ctx); }
    for(int i=0;i<16;++i) { ggml_init_params ip{128ULL*1024*1024,nullptr,true}; ggml_context *ctx=ggml_init(ip); auto *x=input3(ctx,state.data(),D,S,B); x=transformer_layer(ctx,x,w,prefix+"seqTransEncoder.layers."+std::to_string(i)+"."); state=execute(ctx,x,static_cast<size_t>(D)*S*B,w.backend); ggml_free(ctx);
      if(stage == "root") { const auto layer_expected=read_f32(f+"root_layer"+std::to_string(i)+"_output.f32"); float m=0.f; for(size_t j=0;j<state.size();++j)m=std::max(m,std::abs(state[j]-layer_expected[j])); std::printf("root layer%d output max_abs=%g\n",i,m); } }
    std::vector<float> output;
    { ggml_init_params ip{32ULL*1024*1024,nullptr,true}; ggml_context *ctx=ggml_init(ip); auto *all=input3(ctx,state.data(),D,S,B); auto *motion_out=ggml_view_3d(ctx,all,D,T,B,all->nb[1],all->nb[2],static_cast<size_t>(PREFIX)*D*sizeof(float)); motion_out=ggml_cont(ctx,motion_out); auto *y=linear(ctx,motion_out,w.get(prefix+"output_linear.weight"),w.get(prefix+"output_linear.bias")); output=execute(ctx,y,static_cast<size_t>(output_dim)*T*B,w.backend); ggml_free(ctx); }
    float max_abs=0.f, sq=0.f, ref=0.f; for(size_t i=0;i<output.size();++i){ float d=output[i]-expected[i]; max_abs=std::max(max_abs,std::abs(d)); sq+=d*d; ref+=expected[i]*expected[i]; }
    const float rel=std::sqrt(sq/std::max(ref,1.e-20f)); std::printf("%s parity: max_abs=%g rel_l2=%g\n", stage.c_str(), max_abs,rel);
    if (stage == "root") {
        const auto expected_local = read_f32(f+"root_local.f32"), local = root_local_reference(w, output, mask);
        float local_max=0.f, local_sq=0.f, local_ref=0.f;
        for(size_t i=0;i<local.size();++i) { const float d=local[i]-expected_local[i]; local_max=std::max(local_max,std::abs(d)); local_sq+=d*d; local_ref+=expected_local[i]*expected_local[i]; }
        const float local_rel=std::sqrt(local_sq/std::max(local_ref,1.e-20f));
        std::printf("root local conversion: max_abs=%g rel_l2=%g\n",local_max,local_rel);
        if (local_max >= 2.e-3f || local_rel >= 2.e-4f) return 1;
        const auto expected_body_input = read_f32(f+"body_input_0.f32");
        float body_input_max=0.f;
        for(int b=0;b<B;++b) for(int t=0;t<T;++t) {
            const size_t root_base=(static_cast<size_t>(b)*T+t)*546, body_base=(static_cast<size_t>(b)*T+t)*545;
            for(int d=0;d<4;++d) body_input_max=std::max(body_input_max,std::abs(local[(static_cast<size_t>(b)*T+t)*4+d]-expected_body_input[body_base+d]));
            for(int d=0;d<541;++d) body_input_max=std::max(body_input_max,std::abs(motion[root_base+5+d]-expected_body_input[body_base+4+d]));
        }
        std::printf("root-to-body input max_abs=%g\n",body_input_max);
        if (body_input_max >= 2.e-3f) return 1;
    }
    return (max_abs < 2.e-3f && rel < 2.e-4f) ? 0 : 1;
} catch(const std::exception &e) { std::fprintf(stderr,"root parity error: %s\n",e.what()); return 1; }
