#include "denoiser.hpp"
#include "ggml_weights.hpp"
#include "motion_rep.hpp"
#include "diffusion.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace kimodo::detail {
namespace {
constexpr int width=1024, heads=8, head_width=128, text_tokens=50, prefix_tokens=52;
thread_local std::vector<std::pair<ggml_tensor *, std::vector<float>>> inputs;
ggml_tensor *input(ggml_context *ctx, std::span<const float> values, int a, int b, int c) {
    auto *r=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,a,b,c); inputs.emplace_back(r, std::vector<float>(values.begin(),values.end())); return r;
}
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *bias) {
    auto *y=ggml_mul_mat(ctx,w,x);
    // F32 parity takes precedence over Tensor Core throughput.  In
    // particular, do not let a Vulkan backend lower the accumulation
    // precision for the reference model.
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return ggml_add(ctx,y,ggml_repeat(ctx,bias,y));
}
ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale, ggml_tensor *bias) {
    auto *n=ggml_norm(ctx,x,1.e-5f); return ggml_add(ctx,ggml_mul(ctx,n,ggml_repeat(ctx,scale,n)),ggml_repeat(ctx,bias,n));
}
std::expected<std::vector<float>,std::string> execute(ggml_context *ctx, ggml_tensor *out, size_t values, ggml_backend_t backend) {
    auto *graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!alloc || !ggml_gallocr_reserve(alloc,graph) || !ggml_gallocr_alloc_graph(alloc,graph)) return std::unexpected("GGML graph allocation failed");
    for(const auto &[t,data]:inputs) ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));
    inputs.clear();
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) { ggml_gallocr_free(alloc); return std::unexpected("GGML graph execution failed"); }
    std::vector<float> r(values); ggml_backend_tensor_get(out,r.data(),0,r.size()*sizeof(float)); ggml_gallocr_free(alloc); return r;
}
ggml_tensor *weight(const ggml_motion_weights&w,std::string_view n) { auto*t=w.tensor(n); if(!t) throw std::runtime_error("missing GGML tensor: "+std::string(n)); return t; }
ggml_tensor *layer(ggml_context *ctx,ggml_tensor*x,const ggml_motion_weights&w,std::string_view p,int seq,int batch) {
    const std::string s(p); auto*qkv=linear(ctx,x,weight(w,s+"self_attn.in_proj_weight"),weight(w,s+"self_attn.in_proj_bias"));
    // Use explicit [head, batch] branches for the F32 reference graph.  The
    // packed 4-D variant is faster, but differs slightly across Vulkan
    // backends; this layout exactly matches the PyTorch tensor boundaries.
    auto head = [&](int block, int h, int b) {
        return ggml_view_2d(ctx, qkv, head_width, seq, qkv->nb[1],
            static_cast<size_t>(block*width)*sizeof(float) +
            static_cast<size_t>(b)*qkv->nb[2] +
            static_cast<size_t>(h*head_width)*sizeof(float));
    };
    std::vector<ggml_tensor *> batches;
    batches.reserve(static_cast<size_t>(batch));
    for (int b=0; b<batch; ++b) {
        std::vector<ggml_tensor *> joined_heads;
        joined_heads.reserve(heads);
        for (int h=0; h<heads; ++h) {
            auto *q=ggml_cont(ctx,head(0,h,b)), *k=ggml_cont(ctx,head(1,h,b)), *v=ggml_cont(ctx,head(2,h,b));
            auto *scores=ggml_mul_mat(ctx,k,q);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            auto *prob=ggml_soft_max(ctx,ggml_scale(ctx,scores,1.f/std::sqrt(float(head_width))));
            auto *value_product=ggml_mul_mat(ctx,prob,ggml_cont(ctx,ggml_transpose(ctx,v)));
            ggml_mul_mat_set_prec(value_product, GGML_PREC_F32);
            joined_heads.push_back(ggml_transpose(ctx,value_product));
        }
        auto *joined=joined_heads.front();
        for (int h=1; h<heads; ++h) joined=ggml_concat(ctx,joined,joined_heads[static_cast<size_t>(h)],0);
        batches.push_back(ggml_reshape_3d(ctx,joined,width,seq,1));
    }
    auto *a=batches.front();
    for (int b=1; b<batch; ++b) a=ggml_concat(ctx,a,batches[static_cast<size_t>(b)],2);
    a=linear(ctx,a,weight(w,s+"self_attn.out_proj.weight"),weight(w,s+"self_attn.out_proj.bias")); x=norm(ctx,ggml_add(ctx,x,a),weight(w,s+"norm1.weight"),weight(w,s+"norm1.bias")); auto*ff=linear(ctx,x,weight(w,s+"linear1.weight"),weight(w,s+"linear1.bias")); ff=ggml_gelu_erf(ctx,ff); ff=linear(ctx,ff,weight(w,s+"linear2.weight"),weight(w,s+"linear2.bias")); return norm(ctx,ggml_add(ctx,x,ff),weight(w,s+"norm2.weight"),weight(w,s+"norm2.bias"));
}
}
std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned(
    const ggml_motion_weights &, std::span<const float>, std::span<const float>,
    std::span<const float>, std::span<const float>, float, float, float, float, std::size_t);
std::expected<std::vector<float>, std::string> run_motion_transformer(const ggml_motion_weights&w,std::string_view prefix,std::span<const float> motion,size_t motion_dim,std::span<const float> embedding,std::span<const float> timesteps,std::span<const float> headings,size_t batch,size_t frames) try {
    if(!batch||!frames||motion.size()!=batch*frames*motion_dim||embedding.size()!=batch*4096||timesteps.size()!=batch||headings.size()!=batch) return std::unexpected("invalid Transformer input dimensions");
    const int seq=prefix_tokens+static_cast<int>(frames); std::vector<float> text(batch*text_tokens*4096),time(batch*width),angle(batch*2),position(size_t(seq)*width);
    for(size_t b=0;b<batch;++b) { std::memcpy(text.data()+b*text_tokens*4096,embedding.data()+b*4096,4096*sizeof(float)); for(int d=0;d<width;d+=2){float z=timesteps[b]*std::pow(10000.f,-float(d)/width);time[b*width+d]=std::sin(z);time[b*width+d+1]=std::cos(z);} angle[2*b]=std::cos(headings[b]);angle[2*b+1]=std::sin(headings[b]); }
    for(int s=0;s<seq;++s)for(int d=0;d<width;d+=2){float z=float(s)*std::pow(10000.f,-float(d)/width);position[size_t(s)*width+d]=std::sin(z);position[size_t(s)*width+d+1]=std::cos(z);}
    const std::string p(prefix); std::vector<float> state;
    { auto*ctx=ggml_init({128ULL*1024*1024,nullptr,true}); if(!ctx)return std::unexpected("GGML context allocation failed"); auto*m=linear(ctx,input(ctx,motion,int(motion_dim),int(frames),int(batch)),weight(w,p+"input_linear.weight"),weight(w,p+"input_linear.bias")); auto*te=linear(ctx,input(ctx,text,4096,text_tokens,int(batch)),weight(w,p+"embed_text.weight"),weight(w,p+"embed_text.bias")); auto*ti=linear(ctx,input(ctx,time,width,1,int(batch)),weight(w,p+"embed_timestep.time_embed.0.weight"),weight(w,p+"embed_timestep.time_embed.0.bias"));ti=linear(ctx,ggml_silu(ctx,ti),weight(w,p+"embed_timestep.time_embed.2.weight"),weight(w,p+"embed_timestep.time_embed.2.bias"));auto*he=linear(ctx,input(ctx,angle,2,1,int(batch)),weight(w,p+"linear_first_heading_angle.weight"),weight(w,p+"linear_first_heading_angle.bias"));auto*x=ggml_concat(ctx,ggml_concat(ctx,ggml_concat(ctx,te,ti,1),he,1),m,1);auto*pos=input(ctx,position,width,seq,1);x=ggml_add(ctx,x,ggml_repeat(ctx,pos,x));auto r=execute(ctx,x,size_t(width)*seq*batch,w.backend());ggml_free(ctx);if(!r)return std::unexpected(r.error());state=std::move(*r); }
    for(int i=0;i<16;++i){auto*ctx=ggml_init({128ULL*1024*1024,nullptr,true});if(!ctx)return std::unexpected("GGML context allocation failed");auto*x=layer(ctx,input(ctx,state,width,seq,int(batch)),w,p+"seqTransEncoder.layers."+std::to_string(i)+".",seq,int(batch));auto r=execute(ctx,x,size_t(width)*seq*batch,w.backend());ggml_free(ctx);if(!r)return std::unexpected(r.error());state=std::move(*r);}
    auto*ctx=ggml_init({32ULL*1024*1024,nullptr,true});if(!ctx)return std::unexpected("GGML context allocation failed");auto*all=input(ctx,state,width,seq,int(batch));auto*part=ggml_view_3d(ctx,all,width,frames,batch,all->nb[1],all->nb[2],size_t(prefix_tokens)*width*sizeof(float));part=ggml_cont(ctx,part);auto*y=linear(ctx,part,weight(w,p+"output_linear.weight"),weight(w,p+"output_linear.bias"));const size_t outdim=size_t(weight(w,p+"output_linear.bias")->ne[0]);auto r=execute(ctx,y,outdim*frames*batch,w.backend());ggml_free(ctx);return r;
} catch(const std::exception&e){inputs.clear();return std::unexpected(e.what());}

std::expected<std::vector<float>, std::string> run_two_stage_denoiser(
    const ggml_motion_weights &weights, std::span<const float> x,
    std::span<const float> embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::span<const float> mask,
    std::size_t batch, std::size_t frames) {
    const size_t dim=weights.motion_dim(), root_input_dim=2*dim, body_input_dim=2*dim-1;
    if (!batch || !frames || !dim || x.size()!=batch*frames*root_input_dim || mask.size()!=batch*frames)
        return std::unexpected("invalid two-stage denoiser input dimensions");
    auto root=run_motion_transformer(weights,"root_model.",x,root_input_dim,embedding,timesteps,headings,batch,frames);
    if(!root)return std::unexpected(root.error());
    auto gm=weights.f32_values("stats.global_root.mean"), gs=weights.f32_values("stats.global_root.std"), lm=weights.f32_values("stats.local_root.mean"), ls=weights.f32_values("stats.local_root.std");
    if(!gm)return std::unexpected(gm.error());
    if(!gs)return std::unexpected(gs.error());
    if(!lm)return std::unexpected(lm.error());
    if(!ls)return std::unexpected(ls.error());
    auto local=global_root_to_local_root(*root,mask,batch,frames,*gm,*gs,*lm,*ls);
    if(!local)return std::unexpected(local.error());
    std::vector<float> body_input(batch*frames*body_input_dim);
    for(std::size_t b=0;b<batch;++b) for(std::size_t t=0;t<frames;++t) {
        const auto src=(b*frames+t)*root_input_dim, dst=(b*frames+t)*body_input_dim;
        std::memcpy(body_input.data()+dst,local->data()+(b*frames+t)*4,4*sizeof(float));
        std::memcpy(body_input.data()+dst+4,x.data()+src+5,(root_input_dim-5)*sizeof(float));
    }
    auto body=run_motion_transformer(weights,"body_model.",body_input,body_input_dim,embedding,timesteps,headings,batch,frames);
    if(!body)return std::unexpected(body.error());
    std::vector<float> output(batch*frames*dim);
    for(std::size_t b=0;b<batch;++b)for(std::size_t t=0;t<frames;++t){const auto r=(b*frames+t)*5, q=(b*frames+t)*(dim-5), o=(b*frames+t)*dim;std::memcpy(output.data()+o,root->data()+r,5*sizeof(float));std::memcpy(output.data()+o+5,body->data()+q,(dim-5)*sizeof(float));}
    return output;
}

std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, float timestep, float text_weight,
    float constraint_weight, std::size_t frames) {
    const std::vector<float> empty(frames*weights.motion_dim(), 0.f);
    return run_separated_cfg_denoiser_conditioned(weights, motion, embedding, empty, empty,
                                                  timestep, 0.f, text_weight, constraint_weight, frames);
}

std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float timestep, float heading,
    float text_weight, float constraint_weight, std::size_t frames) {
    const size_t dim=weights.motion_dim();
    if (!dim || motion.size()!=frames*dim || embedding.size()!=4096 || !std::isfinite(timestep) || !std::isfinite(text_weight) || !std::isfinite(constraint_weight))
        return std::unexpected("invalid separated CFG denoiser input");
    if (observed.size()!=frames*dim || observed_mask.size()!=frames*dim || !std::isfinite(heading))
        return std::unexpected("invalid separated CFG condition dimensions");
    constexpr size_t cfg_batch=3; std::vector<float> extended(cfg_batch*frames*2*dim), text(cfg_batch*4096), times(cfg_batch,timestep), headings(cfg_batch,heading), mask(cfg_batch*frames,1.f);
    for(size_t b=0;b<cfg_batch;++b) for(size_t t=0;t<frames;++t) {
        auto *dst=extended.data()+(b*frames+t)*2*dim;
        std::memcpy(dst,motion.data()+t*dim,dim*sizeof(float));
        // Upstream separated CFG is [text, constraint, unconditional]. Only
        // the constraint branch receives observed motion and its feature mask.
        if (b==1) for (size_t d=0;d<dim;++d) dst[d]=motion[t*dim+d]*(1.f-observed_mask[t*dim+d])+observed[t*dim+d]*observed_mask[t*dim+d];
        if (b==1) std::memcpy(dst+dim,observed_mask.data()+t*dim,dim*sizeof(float));
    }
    // Only branch zero has text. Branch one is constraint-only; branch two is
    // unconditional. This is the upstream separated-CFG batch order.
    std::memcpy(text.data(),embedding.data(),4096*sizeof(float));
    auto all=run_two_stage_denoiser(weights,extended,text,times,headings,mask,cfg_batch,frames);
    if(!all)return std::unexpected(all.error());
    std::vector<float> result(frames*dim);
    for(size_t i=0;i<result.size();++i) result[i]=(*all)[2*result.size()+i]+text_weight*((*all)[i]-(*all)[2*result.size()+i])+constraint_weight*((*all)[result.size()+i]-(*all)[2*result.size()+i]);
    return result;
}

std::expected<std::vector<float>, std::string> sample_motion_from_noise(
    const ggml_motion_weights &weights, std::span<const float> initial,
    std::span<const float> embedding, std::size_t frames, unsigned steps,
    float text_weight, float constraint_weight) {
    if(initial.size()!=frames*weights.motion_dim()) return std::unexpected("invalid initial motion noise dimensions");
    auto schedule=make_cosine_schedule(1000,steps); if(!schedule)return std::unexpected(schedule.error());
    std::vector<float> state(initial.begin(),initial.end()), next(state.size());
    for(unsigned i=steps;i-->0;) {
        auto clean=run_separated_cfg_denoiser(weights,state,embedding,float(schedule->use_timesteps[i]),text_weight,constraint_weight,frames);
        if(!clean)return std::unexpected(clean.error());
        auto stepped=ddim_step(*schedule,i,state.data(),clean->data(),next.data(),state.size());
        if(!stepped)return std::unexpected(stepped.error());
        state.swap(next);
    }
    return state;
}

std::expected<std::vector<float>, std::string> sample_motion_from_noise_conditioned(
    const ggml_motion_weights &weights, std::span<const float> initial,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float heading, std::size_t frames,
    unsigned steps, float text_weight, float constraint_weight) {
    if(initial.size()!=frames*weights.motion_dim() || observed.size()!=initial.size() || observed_mask.size()!=initial.size())
        return std::unexpected("invalid conditioned motion noise dimensions");
    auto schedule=make_cosine_schedule(1000,steps); if(!schedule)return std::unexpected(schedule.error());
    std::vector<float> state(initial.begin(),initial.end()), next(state.size());
    for(unsigned i=steps;i-->0;) {
        auto clean=run_separated_cfg_denoiser_conditioned(weights,state,embedding,observed,observed_mask,
                                                           float(schedule->use_timesteps[i]),heading,text_weight,constraint_weight,frames);
        if(!clean)return std::unexpected(clean.error());
        auto stepped=ddim_step(*schedule,i,state.data(),clean->data(),next.data(),state.size());
        if(!stepped)return std::unexpected(stepped.error());
        state.swap(next);
    }
    return state;
}
}
