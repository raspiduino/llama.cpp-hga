#include "ggml-backend.h"  
#include "ggml-cuda/common.cuh"  
#include "ggml-cuda/hga.cuh"  
#include "ggml.h"  
#include <cuda_runtime.h>  
#include <cuda_fp16.h>  
#include <cuda_bf16.h>  
#include <vector>  
#include <cstring>  
#include <cmath>  
  
// ==============================================================================  
// HELPER: NVCC-Bulletproof Type Casting  
// ==============================================================================  
template <typename T> struct HGA_Cast {  
    static __device__ inline float to_float(T v);  
    static __device__ inline T from_float(float v);  
};  
template <> struct HGA_Cast<half> {  
    static __device__ inline float to_float(half v) { return __half2float(v); }  
    static __device__ inline half from_float(float v) { return __float2half(v); }  
};  
template <> struct HGA_Cast<nv_bfloat16> {  
    static __device__ inline float to_float(nv_bfloat16 v) { return __bfloat162float(v); }  
    static __device__ inline nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }  
};  

// Mapped State Block Definition
struct HGA_Decode_State {
    int32_t real_unclosed_tokens;
    int32_t real_total_history;
    int32_t store_offset_k;
    int32_t store_offset_v;
    int32_t carry_src_bytes_k;
    int32_t carry_dst_bytes_k;
    int32_t carry_bytes_k;
    int32_t carry_src_bytes_v;
    int32_t carry_dst_bytes_v;
    int32_t carry_bytes_v;
};

extern "C" void* hga_get_carry_ptr(int il, int is_v);
  
// ==============================================================================  
// 1. HGA SUMMARY: Mixed-Strategy + On-The-Fly RoPE  
// ==============================================================================  
template <typename T>  
__global__ void hga_summary_kernel_mixed(  
    const T* __restrict__ K, T* __restrict__ out,  
    int chunk_size, int head_dim, int start_pos, float theta_base) {  
    int head_idx = blockIdx.x;  
    int d = threadIdx.x;  
    int HALF_DIM = head_dim / 2;  
    if (d >= HALF_DIM) return;  
  
    int HIGH_FREQ_HALF = HALF_DIM / 2;  
    bool is_high_freq = (d < HIGH_FREQ_HALF);  
    float acc1 = 0.0f, acc2 = 0.0f;  
    int stride_chunk = head_dim * gridDim.x;  
    int stride_head = head_dim;  
    float freq = 1.0f / powf(theta_base, 2.0f * (float)d / (float)head_dim);  
  
    for (int i = 0; i < chunk_size; ++i) {  
        int p = start_pos + i;  
        int idx1 = i * stride_chunk + head_idx * stride_head + d;  
        int idx2 = i * stride_chunk + head_idx * stride_head + d + HALF_DIM;  
        float k1 = HGA_Cast<T>::to_float(K[idx1]);  
        float k2 = HGA_Cast<T>::to_float(K[idx2]);  
        if (is_high_freq) {  
            acc1 += k1; acc2 += k2;  
        } else {  
            float c = cosf(p * freq); float s = sinf(p * freq);  
            acc1 += k1 * c + k2 * s;  
            acc2 += k2 * c - k1 * s;  
        }  
    }  
  
    float inv_chunk = 1.0f / (float)chunk_size;  
    float avg1 = acc1 * inv_chunk;  
    float avg2 = acc2 * inv_chunk;  
    int out_idx1 = head_idx * head_dim + d;  
    int out_idx2 = head_idx * head_dim + d + HALF_DIM;  
  
    if (is_high_freq) {  
        out[out_idx1] = HGA_Cast<T>::from_float(avg1);  
        out[out_idx2] = HGA_Cast<T>::from_float(avg2);  
    } else {  
        int mid_pos = start_pos + chunk_size / 2;  
        float cm = cosf(mid_pos * freq); float sm = sinf(mid_pos * freq);  
        out[out_idx1] = HGA_Cast<T>::from_float(avg1 * cm - avg2 * sm);  
        out[out_idx2] = HGA_Cast<T>::from_float(avg2 * cm + avg1 * sm);  
    }  
}  
  
void ggml_cuda_op_hga_summary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    const ggml_tensor * src_k = dst->src[0];  
    int32_t start_pos; memcpy(&start_pos, dst->op_params, sizeof(int32_t));  
    float theta_base; memcpy(&theta_base, (char*)dst->op_params + sizeof(int32_t), sizeof(float));  
    int chunk_size = src_k->ne[2], n_head_kv = src_k->ne[1], head_dim = src_k->ne[0];  
    dim3 grid(n_head_kv), block(head_dim / 2);  
    if (src_k->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16)  
        hga_summary_kernel_mixed<half><<<grid, block, 0, ctx.stream()>>>((const half*)src_k->data, (half*)dst->data, chunk_size, head_dim, start_pos, theta_base);  
    else if (src_k->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16)  
        hga_summary_kernel_mixed<nv_bfloat16><<<grid, block, 0, ctx.stream()>>>((const nv_bfloat16*)src_k->data, (nv_bfloat16*)dst->data, chunk_size, head_dim, start_pos, theta_base);  
}  
  
// ==============================================================================  
// 2. HGA ROUTE: Parallel GQA Dot-Product  
// ==============================================================================  
__global__ void hga_route_kernel(  
    const float* __restrict__ Q, const nv_bfloat16* __restrict__ S, float* __restrict__ scores,  
    int head_dim, int n_head_q, int n_head_kv, int n_tokens, int valid_chunks)  
{  
    int c = blockIdx.x;  
    extern __shared__ float s_Q_means[];  
    for (int idx = threadIdx.x; idx < n_head_kv * head_dim; idx += blockDim.x) {  
        int h_kv = idx / head_dim;  
        int d = idx % head_dim;  
        float q_sum = 0.0f;  
        int G = n_head_q / n_head_kv;  
        for (int g = 0; g < G; ++g) {  
            int h_q = h_kv * G + g;  
            for (int t = 0; t < n_tokens; ++t) {  
                q_sum += Q[d + h_q * head_dim + t * head_dim * n_head_q];  
            }  
        }  
        s_Q_means[h_kv * head_dim + d] = q_sum / (float)(G * n_tokens);  
    }  
    __syncthreads();  
  
    if (c < valid_chunks) {  
        float total_score = 0.0f;  
        for (int h_kv = 0; h_kv < n_head_kv; ++h_kv) {  
            for (int d = 0; d < head_dim; ++d) {  
                total_score += s_Q_means[h_kv * head_dim + d] * __bfloat162float(S[d + h_kv * head_dim + c * head_dim * n_head_kv]);  
            }  
        }  
        float final_score = total_score / (float)n_head_kv;
        // BULLETPROOF: Prevent NaNs from propagating to the sorting kernel
        scores[c] = isnan(final_score) ? 0.0f : final_score;  
    }
}  
  
void ggml_cuda_op_hga_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    const ggml_tensor * src_q = dst->src[0]; const ggml_tensor * src_s = dst->src[1];  
    int32_t params[4]; memcpy(params, dst->op_params, sizeof(params));  
    int valid_chunks = params[0], n_head_q = params[1], n_head_kv = params[2], n_tokens = params[3];  
    int head_dim = src_q->ne[0];  
    int block_size = (head_dim <= 128) ? 128 : 256;  
    size_t shared_mem = n_head_kv * head_dim * sizeof(float);  
    hga_route_kernel<<<valid_chunks, block_size, shared_mem, ctx.stream()>>>(  
        (const float*)src_q->data, (const nv_bfloat16*)src_s->data, (float*)dst->data,  
        head_dim, n_head_q, n_head_kv, n_tokens, valid_chunks);  
}  
  
// ==============================================================================  
// 3. HGA GATHER: Chunk-Level Tiered Byte-Copy (PCIe -> VRAM)  
// ==============================================================================  
extern "C" void* hga_get_pcie_ptr(int il, int is_v);  
  
__global__ void hga_gather_tiered_kernel(  
    const int32_t* __restrict__ routed_idxs, const char* __restrict__ src, char* __restrict__ dst,  
    int sink_end, int k_to_route, int local_start, int valid_chunks,  
    int chunk_size, size_t chunk_bytes, const HGA_Decode_State* __restrict__ state, int static_max_seq)  
{  
    int out_c = blockIdx.x;  
    int unclosed_tokens = state->real_unclosed_tokens;
    int total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);  
    size_t token_bytes = chunk_bytes / chunk_size;  
      
    if (out_c == total_out_chunks) {  
        if (unclosed_tokens == 0) return;  
        size_t unclosed_bytes = unclosed_tokens * token_bytes;  
        int src_tok_start = valid_chunks * chunk_size;  
        const char* src_ptr = src + src_tok_start * token_bytes;  
        char* dst_ptr = dst + total_out_chunks * chunk_bytes;  
          
        size_t i = threadIdx.x * 16;  
        for (; i + 16 <= unclosed_bytes; i += blockDim.x * 16) {  
            uint4 val = *reinterpret_cast<const uint4*>(src_ptr + i);  
            *reinterpret_cast<uint4*>(dst_ptr + i) = val;  
        }  
        size_t base = (unclosed_bytes / 16) * 16;  
        if (threadIdx.x == 0) for (size_t j = base; j < unclosed_bytes; ++j) dst_ptr[j] = src_ptr[j];  
        return;  
    }  
  
    int src_c = routed_idxs[out_c];
    
    // BULLETPROOF BOUNDS CHECK:
    // If the index array contains uninitialized VRAM garbage or -1, clamp it to 0.
    // This mathematically guarantees the pointer can never go negative or out of bounds.
    if (src_c < 0 || src_c >= valid_chunks) {
        src_c = 0; 
    }
      
    const char* src_ptr = src + src_c * chunk_bytes;  
    char* dst_ptr = dst + out_c * chunk_bytes;  
      
    size_t i = threadIdx.x * 16;  
    for (; i + 16 <= chunk_bytes; i += blockDim.x * 16) {  
        uint4 val = *reinterpret_cast<const uint4*>(src_ptr + i);  
        *reinterpret_cast<uint4*>(dst_ptr + i) = val;  
    }  
    size_t base = (chunk_bytes / 16) * 16;  
    if (threadIdx.x == 0) for (size_t j = base; j < chunk_bytes; ++j) dst_ptr[j] = src_ptr[j];  
}  
  
void ggml_cuda_op_hga_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    const ggml_tensor * routed_idxs = dst->src[0];  
    int32_t params[10]; memcpy(params, dst->op_params, sizeof(params));  
    int sink_end = params[0], k_to_route = params[1], local_start = params[2], valid_chunks = params[3];  
    int chunk_size = params[4]; size_t chunk_bytes = (size_t)params[5]; int il = params[6]; int is_v = params[7];  
    HGA_Decode_State* state; memcpy(&state, &params[8], sizeof(HGA_Decode_State*));
    int static_max_seq = dst->ne[1];
      
    const char* src = (const char*)hga_get_pcie_ptr(il, is_v);  
    int total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);  
    int grid_size = total_out_chunks + 1; 
      
    hga_gather_tiered_kernel<<<grid_size, 256, 0, ctx.stream()>>>(  
        (const int32_t*)routed_idxs->data, src, (char*)dst->data,  
        sink_end, k_to_route, local_start, valid_chunks, chunk_size, chunk_bytes, state, static_max_seq);  
}

// ==============================================================================  
// 4. HGA STITCH: Pure F16/BF16 Byte-Copy  
// ==============================================================================  
__global__ void hga_stitch_kernel(  
    const char* __restrict__ hist, const char* __restrict__ cur, char* __restrict__ out,  
    const HGA_Decode_State* __restrict__ state, int cur_tokens, size_t token_bytes, int static_max_seq) {  
    int tok_idx = blockIdx.x;  
    if (tok_idx >= static_max_seq) return;
    int real_hist = state->real_total_history;
    
    const char* src_ptr = nullptr; int src_tok_idx = 0;  
    if (tok_idx < real_hist) { 
        src_ptr = hist; src_tok_idx = tok_idx; 
    }  
    else if (tok_idx < real_hist + cur_tokens) { 
        src_ptr = cur; src_tok_idx = tok_idx - real_hist; 
    }  
    else {
        // PADDING TRAP FIX:
        // FlashAttention dequantizes the ENTIRE static_max_seq buffer.
        // If we leave the padding as uninitialized VRAM garbage, the Q4_0 dequantizer
        // will read random bytes as NaN/INF scale factors, destroying the attention logits.
        // Writing 0 bytes creates a valid Q4_0 block that dequantizes to exactly 0.0f.
        for (size_t i = threadIdx.x; i < token_bytes; i += blockDim.x) {  
            out[tok_idx * token_bytes + i] = 0;  
        }
        return;
    }
    
    for (size_t i = threadIdx.x; i < token_bytes; i += blockDim.x) {  
        out[tok_idx * token_bytes + i] = src_ptr[src_tok_idx * token_bytes + i];  
    }  
}  

void ggml_cuda_op_hga_stitch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    int32_t params[4]; memcpy(params, dst->op_params, sizeof(params));  
    int cur_tokens = params[0]; size_t token_bytes = (size_t)params[1];  
    HGA_Decode_State* state; memcpy(&state, &params[2], sizeof(HGA_Decode_State*));
    int static_max_seq = dst->ne[1]; // 2D tensor: ne[1] is static_max_seq
    
    hga_stitch_kernel<<<static_max_seq, 256, 0, ctx.stream()>>>(  
        (const char*)dst->src[0]->data, (const char*)dst->src[1]->data, (char*)dst->data,  
        state, cur_tokens, token_bytes, static_max_seq);  
}

// ==============================================================================  
// 5. HGA MASK: Causal Triangle Stamping  
// ==============================================================================  
__global__ void hga_mask_kernel(half* mask, int n_tokens, int static_max_seq, const HGA_Decode_State* __restrict__ state) {  
    int q = blockIdx.y; int k = blockIdx.x * blockDim.x + threadIdx.x;  
    if (k >= static_max_seq) return;
    int real_history = state->real_total_history;
    half zero = __float2half(0.0f); half neg_inf = __float2half(-10000.0f);  
    int total_valid_k = real_history + n_tokens;
    
    if (k < total_valid_k) {
        if (k < real_history) mask[q * static_max_seq + k] = zero;  
        else {  
            int k_local = k - real_history;  
            mask[q * static_max_seq + k] = (k_local <= q) ? zero : neg_inf;  
        }  
    } else {
        mask[q * static_max_seq + k] = neg_inf;
    }
}  

void ggml_cuda_op_hga_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    int32_t params[4]; memcpy(params, dst->op_params, sizeof(params));  
    int n_tokens = params[0], static_max_seq = params[1];  
    HGA_Decode_State* state; memcpy(&state, &params[2], sizeof(HGA_Decode_State*));
    
    dim3 block(256); dim3 grid((static_max_seq + block.x - 1) / block.x, n_tokens);  
    hga_mask_kernel<<<grid, block, 0, ctx.stream()>>>((half*)dst->data, n_tokens, static_max_seq, state);  
}

// ==============================================================================  
// 6. HGA BUILD IDXS: CUB-Free Top-K and Tiered Assembly  
// ==============================================================================  
__global__ void hga_build_idxs_kernel(    
    const float* __restrict__ scores, int32_t* __restrict__ out_idxs,    
    int valid_chunks, int sink_end, int k_to_route, int local_start)    
{    
    extern __shared__ char shared_mem[];    
    float* s_scores = (float*)shared_mem;    
    int* s_idxs = (int*)&s_scores[valid_chunks];    
    
    int tid = threadIdx.x;    
    for (int i = tid; i < valid_chunks; i += blockDim.x) {    
        s_scores[i] = scores[i];    
        s_idxs[i] = i;    
    }    
    __syncthreads();    
    
    if (tid == 0) {    
        int out_idx = 0;    
        
        // 1. Sinks
        for (int i = 0; i < sink_end; i++) out_idxs[out_idx++] = i;    
  
        // 2. Top-K    
        int selected[64];    
        for (int i = 0; i < k_to_route; i++) {    
            float max_val = -1e30f; int max_idx = -1;    
            for (int j = 0; j < valid_chunks; j++) {    
                bool is_selected = false;    
                for (int s = 0; s < i; s++) if (selected[s] == j) { is_selected = true; break; }    
                if (is_selected) continue;    
                if (s_scores[j] > max_val) { max_val = s_scores[j]; max_idx = j; }    
            }    
              
            if (max_idx == -1) {  
                max_idx = sink_end + i;  
                if (max_idx >= local_start) max_idx = sink_end;  
            }  
              
            selected[i] = max_idx; out_idxs[out_idx++] = max_idx;    
        }    
        
        // 3. Local (THIS WAS MISSING AND CAUSING THE 40TB OOB READ!)
        for (int i = local_start; i < valid_chunks; i++) {
            out_idxs[out_idx++] = i;
        }
    }
}
  
void ggml_cuda_op_hga_build_idxs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    const ggml_tensor * scores = dst->src[0];  
    int32_t params[4]; memcpy(params, dst->op_params, sizeof(params));  
    int valid_chunks = params[0], sink_end = params[1], k_to_route = params[2], local_start = params[3];  
    size_t shared_mem = valid_chunks * (sizeof(float) + sizeof(int));  
    hga_build_idxs_kernel<<<1, 256, shared_mem, ctx.stream()>>>(  
        (const float*)scores->data, (int32_t*)dst->data, valid_chunks, sink_end, k_to_route, local_start);  
}  
  
// ==============================================================================  
// 7. HGA STORE: Async DMA + Fused Carry Buffer Update
// ==============================================================================  
__global__ void hga_store_and_carry_kernel(
    const char* __restrict__ src, char* __restrict__ dst_host, char* __restrict__ dst_carry_k, char* __restrict__ dst_carry_v,
    const HGA_Decode_State* __restrict__ state, int bytes_to_copy, int is_v) 
{
    int host_offset = is_v ? state->store_offset_v : state->store_offset_k;
    char* dst_h = dst_host + host_offset;
    
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (idx + 16 <= bytes_to_copy) {
        uint4 val = *reinterpret_cast<const uint4*>(src + idx);
        *reinterpret_cast<uint4*>(dst_h + idx) = val;
    }
    
    int c_bytes = is_v ? state->carry_bytes_v : state->carry_bytes_k;
    if (c_bytes > 0) {
        int c_src_off = is_v ? state->carry_src_bytes_v : state->carry_src_bytes_k;
        int c_dst_off = is_v ? state->carry_dst_bytes_v : state->carry_dst_bytes_k;
        char* dst_c = is_v ? dst_carry_v : dst_carry_k;
        if (idx + 16 <= c_bytes) {
            uint4 val = *reinterpret_cast<const uint4*>(src + c_src_off + idx);
            *reinterpret_cast<uint4*>(dst_c + c_dst_off + idx) = val;
        }
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        size_t base_h = (bytes_to_copy / 16) * 16;
        for (size_t j = base_h; j < bytes_to_copy; ++j) dst_h[j] = src[j];
        if (c_bytes > 0) {
            size_t base_c = (c_bytes / 16) * 16;
            int c_src_off = is_v ? state->carry_src_bytes_v : state->carry_src_bytes_k;
            int c_dst_off = is_v ? state->carry_dst_bytes_v : state->carry_dst_bytes_k;
            char* dst_c = is_v ? dst_carry_v : dst_carry_k;
            for (size_t j = base_c; j < c_bytes; ++j) dst_c[c_dst_off + j] = src[c_src_off + j];
        }
    }
}

void ggml_cuda_op_hga_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {  
    const ggml_tensor * src = dst->src[0];  
    int32_t params[6]; memcpy(params, dst->op_params, sizeof(params));  
    int il = params[0], is_v = params[1], bytes_to_copy = params[2];  
    HGA_Decode_State* state; memcpy(&state, &params[3], sizeof(HGA_Decode_State*));
      
    char* dst_host_ptr = (char*)hga_get_pcie_ptr(il, is_v);  
    char* dst_carry_k = (char*)hga_get_carry_ptr(il, 0);
    char* dst_carry_v = (char*)hga_get_carry_ptr(il, 1);
      
    int max_bytes = bytes_to_copy;
    int c_bytes = is_v ? state->carry_bytes_v : state->carry_bytes_k;
    if (c_bytes > max_bytes) max_bytes = c_bytes;
    
    int grid_size = ((((max_bytes) + 15) / 16) + 255) / 256;
    hga_store_and_carry_kernel<<<grid_size, 256, 0, ctx.stream()>>>(  
        (const char*)src->data, dst_host_ptr, dst_carry_k, dst_carry_v, state, bytes_to_copy, is_v);  
}
  
// ==============================================================================  
// ZERO-COPY PINNED MEMORY BRIDGE  
// ==============================================================================  
extern "C" {  
    void hga_alloc_pinned_mapped(size_t bytes, void** host_ptr, void** dev_ptr);  
    void hga_free_pinned_mapped(void* host_ptr);  
  
    void hga_alloc_pinned_mapped(size_t bytes, void** host_ptr, void** dev_ptr) {  
        cudaError_t err = cudaHostAlloc(host_ptr, bytes, cudaHostAllocMapped);  
        if (err == cudaSuccess) {  
            cudaHostGetDevicePointer(dev_ptr, *host_ptr, 0);  
        } else {  
            *host_ptr = nullptr;  
            *dev_ptr = nullptr;  
        }  
    }  
      
    void hga_free_pinned_mapped(void* host_ptr) {  
        if (host_ptr) cudaFreeHost(host_ptr);  
    }  
}  