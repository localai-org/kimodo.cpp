// GGML graph structure follows the operation conventions in llama.cpp
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
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
struct loaded {
    ggml_context *ctx = nullptr; gguf_context *file = nullptr; ggml_backend_t backend = nullptr; ggml_backend_buffer_t buffer = nullptr;
    ~loaded() { if (buffer) ggml_backend_buffer_free(buffer); if (file) gguf_free(file); if (ctx) ggml_free(ctx); if (backend) ggml_backend_free(backend); }
};
template<class T> std::vector<T> read(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate); if (!in) throw std::runtime_error("cannot read " + path.string());
    const auto bytes=in.tellg(); if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T))) throw std::runtime_error("invalid fixture");
    std::vector<T> out(static_cast<size_t>(bytes)/sizeof(T)); in.seekg(0); in.read(reinterpret_cast<char *>(out.data()),bytes); if(!in) throw std::runtime_error("short fixture"); return out;
}
std::unique_ptr<loaded> load(const char *path, bool vulkan) {
    auto out=std::make_unique<loaded>(); gguf_init_params p{true,&out->ctx}; out->file=gguf_init_from_file(path,p);
    if(!out->file||!out->ctx) throw std::runtime_error("cannot load GGUF");
#if defined(KIMODO_HAVE_GGML_VULKAN)
    if(vulkan && ggml_backend_vk_get_device_count()) out->backend=ggml_backend_vk_init(0);
#else
    (void)vulkan;
#endif
    if(!out->backend) { out->backend=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(out->backend,24); }
    out->buffer=ggml_backend_alloc_ctx_tensors(out->ctx,out->backend); if(!out->buffer) throw std::runtime_error("cannot allocate weights");
    std::ifstream in(path,std::ios::binary); const auto start=gguf_get_data_offset(out->file); auto *weight=ggml_get_tensor(out->ctx,"final_norm.weight");
    if(!weight||weight->type!=GGML_TYPE_BF16) throw std::runtime_error("missing BF16 final norm");
    std::vector<char> data(ggml_nbytes(weight)); in.seekg(static_cast<std::streamoff>(start+gguf_get_tensor_offset(out->file,0))); in.read(data.data(),static_cast<std::streamsize>(data.size()));
    if(!in) throw std::runtime_error("short GGUF"); ggml_backend_tensor_set(weight,data.data(),0,data.size()); return out;
}
void report(std::string_view name,const std::vector<float>& actual,const std::vector<float>& expected) {
    if(actual.size()!=expected.size()) throw std::runtime_error("size mismatch"); float maximum=0.f; double err=0,ref=0;
    for(size_t i=0;i<actual.size();++i){const float d=actual[i]-expected[i];maximum=std::max(maximum,std::abs(d));err+=double(d)*d;ref+=double(expected[i])*expected[i];}
    std::printf("%.*s max_abs=%g rel_l2=%g\n",int(name.size()),name.data(),maximum,std::sqrt(err/ref));
}
}
int main(int argc,char **argv) try {
    if(argc!=5){std::fprintf(stderr,"usage: %s FINAL_NORM.gguf HIDDEN.f32 FIXTURE_DIR cpu|vulkan\n",argv[0]);return 2;}
    const bool vulkan=std::string_view(argv[4])=="vulkan"; if(!vulkan&&std::string_view(argv[4])!="cpu") throw std::runtime_error("backend must be cpu or vulkan");
    const auto input=read<float>(argv[2]); if(input.size()!=16*4096) throw std::runtime_error("expected [1,16,4096] hidden state");
    auto model=load(argv[1],vulkan); auto *weight=ggml_get_tensor(model->ctx,"final_norm.weight");
    auto *ctx=ggml_init({8ULL*1024*1024,nullptr,true}); if(!ctx) throw std::runtime_error("cannot allocate graph"); auto guard=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ctx,ggml_free);
    auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,16);ggml_set_input(x);auto *normal=ggml_rms_norm(ctx,x,1e-5f);auto *out=ggml_mul(ctx,normal,ggml_repeat(ctx,ggml_cast(ctx,weight,GGML_TYPE_F32),normal));
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,model->backend);if(!buffer)throw std::runtime_error("cannot allocate graph");auto release=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(buffer,ggml_backend_buffer_free);
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));if(ggml_backend_graph_compute(model->backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML graph failed");std::vector<float> actual(input.size());ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
    const auto fixture=std::filesystem::path(argv[3]);report("final_hidden",actual,read<float>(fixture/"final_hidden_state.f32"));
    const auto mask=read<int64_t>(fixture/"embed_mask.i64");if(mask.size()!=16)throw std::runtime_error("expected 16 embed-mask values");std::vector<float> pooled(4096);int count=0;for(int t=0;t<16;++t)if(mask[t]){++count;for(int d=0;d<4096;++d)pooled[d]+=actual[size_t(t)*4096+d];}for(float &v:pooled)v/=count;report("pooled_embedding",pooled,read<float>(fixture/"pooled_embedding.f32"));return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
