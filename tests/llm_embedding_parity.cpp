#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
template<class T> std::vector<T> read(const std::filesystem::path &path) { std::ifstream in(path,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("cannot read");const auto n=in.tellg();if(n<0||n%static_cast<std::streamoff>(sizeof(T)))throw std::runtime_error("bad fixture");std::vector<T>v(static_cast<size_t>(n)/sizeof(T));in.seekg(0);in.read(reinterpret_cast<char*>(v.data()),n);return v; }
float bf16(uint16_t x) { uint32_t b=uint32_t(x)<<16;float f;std::memcpy(&f,&b,4);return f; }
struct loaded { ggml_context*c=nullptr;gguf_context*f=nullptr;ggml_backend_t b=nullptr;ggml_backend_buffer_t w=nullptr;~loaded(){if(w)ggml_backend_buffer_free(w);if(f)gguf_free(f);if(c)ggml_free(c);if(b)ggml_backend_free(b);} };
std::unique_ptr<loaded> load(const char *path,bool vk){auto r=std::make_unique<loaded>();gguf_init_params p{true,&r->c};r->f=gguf_init_from_file(path,p);if(!r->f||!r->c)throw std::runtime_error("load GGUF");
#if defined(KIMODO_HAVE_GGML_VULKAN)
if(vk&&ggml_backend_vk_get_device_count())r->b=ggml_backend_vk_init(0);
#else
(void)vk;
#endif
if(!r->b){r->b=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(r->b,24);}r->w=ggml_backend_alloc_ctx_tensors(r->c,r->b);auto*t=ggml_get_tensor(r->c,"token_embedding.weight");if(!r->w||!t||t->type!=GGML_TYPE_BF16)throw std::runtime_error("missing BF16 embedding");std::ifstream in(path,std::ios::binary);const auto base=gguf_get_data_offset(r->f),off=gguf_get_tensor_offset(r->f,0);std::vector<char>v(ggml_nbytes(t));in.seekg(static_cast<std::streamoff>(base+off));in.read(v.data(),static_cast<std::streamsize>(v.size()));if(!in)throw std::runtime_error("short GGUF");ggml_backend_tensor_set(t,v.data(),0,v.size());return r;}
}
int main(int argc,char**argv)try{if(argc!=4){std::fprintf(stderr,"usage: %s EMBEDDING.gguf FIXTURE cpu|vulkan\n",argv[0]);return 2;}const bool vk=std::string_view(argv[3])=="vulkan";if(!vk&&std::string_view(argv[3])!="cpu")throw std::runtime_error("backend");auto model=load(argv[1],vk);const auto fixture=std::filesystem::path(argv[2]);const auto ids64=read<int64_t>(fixture/"input_ids.i64");std::vector<int32_t>ids(ids64.begin(),ids64.end());auto*ctx=ggml_init({2ULL*1024*1024,nullptr,true});auto cleanup=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ctx,ggml_free);auto*indices=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,ids.size());ggml_set_input(indices);auto*output=ggml_get_rows(ctx,ggml_get_tensor(model->c,"token_embedding.weight"),indices);auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,model->b);auto release=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(buffer,ggml_backend_buffer_free);if(!buffer)throw std::runtime_error("allocation");ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*sizeof(int32_t));if(ggml_backend_graph_compute(model->b,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");const auto expected=read<float>(fixture/"token_embeddings.f32");std::vector<float>actual(expected.size());if(output->type==GGML_TYPE_F32)ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));else if(output->type==GGML_TYPE_BF16){std::vector<uint16_t>raw(actual.size());ggml_backend_tensor_get(output,raw.data(),0,raw.size()*sizeof(uint16_t));for(size_t i=0;i<raw.size();++i)actual[i]=bf16(raw[i]);}else throw std::runtime_error("unexpected embedding output type");float max=0;double e=0,r=0;for(size_t i=0;i<actual.size();++i){float d=actual[i]-expected[i];max=std::max(max,std::abs(d));e+=double(d)*d;r+=double(expected[i])*expected[i];}std::printf("embedding %s max_abs=%g rel_l2=%g type=%d\n",argv[3],max,std::sqrt(e/r),output->type);return max==0?0:1;}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
