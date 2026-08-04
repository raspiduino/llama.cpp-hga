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

// ==============================================================================
// 1. HGA SUMMARY: Mixed-Strategy + On-The-Fly RoPE (Zero VRAM Tables)
// ==============================================================================
template <typename T>  
__global__ void hga_summary_kernel_mixed(  
    const T* __restrict__ K, T* __restrict__ out,             
    int chunk_size, int head_dim, int start_pos, float theta_base) {  
      
    int head_idx = blockIdx.x;  
    int d = threadIdx.x;  
    int HALF_DIM = head_dim / 2;  
    if (d >= HALF_DIM) return;  
  
    // The first quarter of the head dimensions are high-frequency
    int HIGH_FREQ_HALF = HALF_DIM / 2;  
    bool is_high_freq = (d < HIGH_FREQ_HALF);  
    
    float acc1 = 0.0f, acc2 = 0.0f;  
    int stride_chunk = head_dim * gridDim.x;  
    int stride_head = head_dim;  
  
    // FIX #1 & #2: Correct exponent (2*d) and a single shared frequency for both halves.
    // This frequency is only used for the low-freq inverse/re-rotation.
    float freq = 1.0f / powf(theta_base, 2.0f * (float)d / (float)head_dim);  
  
    for (int i = 0; i < chunk_size; ++i) {  
        int p = start_pos + i;  
        int idx1 = i * stride_chunk + head_idx * stride_head + d;  
        int idx2 = i * stride_chunk + head_idx * stride_head + d + HALF_DIM;  
        float k1 = HGA_Cast<T>::to_float(K[idx1]);  
        float k2 = HGA_Cast<T>::to_float(K[idx2]);  
          
        if (is_high_freq) {  
            // HIGH-FREQ: Average the already-rotated keys directly (per HGA paper)
            acc1 += k1; 
            acc2 += k2;  
        } else {  
            // LOW-FREQ: Inverse RoPE to un-rotate to the origin before averaging
            float c = cosf(p * freq);
            float s = sinf(p * freq);
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
        // HIGH-FREQ: Already averaged in rotated space, just write out
        out[out_idx1] = HGA_Cast<T>::from_float(avg1);  
        out[out_idx2] = HGA_Cast<T>::from_float(avg2);  
    } else {
        // LOW-FREQ: Re-rotate the averaged raw keys to the midpoint of the chunk
        int mid_pos = start_pos + chunk_size / 2;  
        float cm = cosf(mid_pos * freq);
        float sm = sinf(mid_pos * freq);
        
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
    
    if (src_k->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        hga_summary_kernel_mixed<half><<<grid, block, 0, ctx.stream()>>>(
            (const half*)src_k->data, (half*)dst->data, chunk_size, head_dim, start_pos, theta_base);
    } else if (src_k->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        hga_summary_kernel_mixed<nv_bfloat16><<<grid, block, 0, ctx.stream()>>>(
            (const nv_bfloat16*)src_k->data, (nv_bfloat16*)dst->data, chunk_size, head_dim, start_pos, theta_base);
    }
}

// ==============================================================================
// 2. HGA ROUTE: Mathematically pure GQA + Token + Head routing
// ==============================================================================
__global__ void hga_route_kernel(
    const float* __restrict__ Q, const nv_bfloat16* __restrict__ S, float* __restrict__ scores,
    int head_dim, int n_head_q, int n_head_kv, int n_tokens, int valid_chunks) {
    
    int c = blockIdx.x; if (c >= valid_chunks) return;
    int d = threadIdx.x; 
    int G = n_head_q / n_head_kv;
    float total_score = 0.0f;
    
    if (d < head_dim) {
        for (int h_kv = 0; h_kv < n_head_kv; ++h_kv) {
            float q_sum = 0.0f;
            for (int g = 0; g < G; ++g) {
                int h_q = h_kv * G + g;
                for (int t = 0; t < n_tokens; ++t) {
                    q_sum += Q[d + h_q * head_dim + t * head_dim * n_head_q];
                }
            }
            float q_mean = q_sum / (float)(G * n_tokens);
            float s_val = __bfloat162float(S[d + h_kv * head_dim + c * head_dim * n_head_kv]);
            total_score += q_mean * s_val;
        }
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        total_score += __shfl_down_sync(0xffffffff, total_score, offset);
    }
    __shared__ float warp_sums[8]; 
    int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp_id] = total_score;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float final_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; ++i) final_sum += warp_sums[i];
        scores[c] = final_sum / (float)n_head_kv;
    }
}

void ggml_cuda_op_hga_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q = dst->src[0]; const ggml_tensor * src_s = dst->src[1];
    int32_t params[4]; memcpy(params, dst->op_params, sizeof(params));
    int valid_chunks = params[0], n_head_q = params[1], n_head_kv = params[2], n_tokens = params[3];
    int head_dim = src_q->ne[0];
    int block_size = (head_dim <= 128) ? 128 : 256;
    hga_route_kernel<<<valid_chunks, block_size, 0, ctx.stream()>>>(
        (const float*)src_q->data, (const nv_bfloat16*)src_s->data, (float*)dst->data, 
        head_dim, n_head_q, n_head_kv, n_tokens, valid_chunks);
}

// ==============================================================================
// 3. HGA STITCH: Quantization-aware VRAM Concatenation
// ==============================================================================
__global__ void hga_stitch_kernel(
    const char* __restrict__ sink, const char* __restrict__ rout, 
    const char* __restrict__ local, const char* __restrict__ cur, char* __restrict__ out,
    int sink_tokens, int rout_tokens, int local_tokens, int cur_tokens, size_t token_bytes) {
    
    int tok_idx = blockIdx.x;
    int total_hist = sink_tokens + rout_tokens + local_tokens;
    const char* src_ptr = nullptr; int src_tok_idx = 0;
    
    if (tok_idx < sink_tokens) { src_ptr = sink; src_tok_idx = tok_idx; } 
    else if (tok_idx < sink_tokens + rout_tokens) { src_ptr = rout; src_tok_idx = tok_idx - sink_tokens; } 
    else if (tok_idx < total_hist) { src_ptr = local; src_tok_idx = tok_idx - sink_tokens - rout_tokens; } 
    else { src_ptr = cur; src_tok_idx = tok_idx - total_hist; }
    
    for (size_t i = threadIdx.x; i < token_bytes; i += blockDim.x) {
        out[tok_idx * token_bytes + i] = src_ptr[src_tok_idx * token_bytes + i];
    }
}

void ggml_cuda_op_hga_stitch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int32_t params[5]; memcpy(params, dst->op_params, sizeof(params));
    int sink_tokens = params[0], rout_tokens = params[1], local_tokens = params[2], cur_tokens = params[3];
    size_t token_bytes = (size_t)params[4];
    int total_tokens = sink_tokens + rout_tokens + local_tokens + cur_tokens;
    hga_stitch_kernel<<<total_tokens, 256, 0, ctx.stream()>>>(
        (const char*)dst->src[0]->data, (const char*)dst->src[1]->data,
        (const char*)dst->src[2]->data, (const char*)dst->src[3]->data, (char*)dst->data,
        sink_tokens, rout_tokens, local_tokens, cur_tokens, token_bytes);
}

// ==============================================================================
// 4. HGA MASK: Stamps the causal triangle for Sparse Prefill
// ==============================================================================
__global__ void hga_mask_kernel(half* mask, int history_tokens, int n_tokens, int stride) {
    int q = blockIdx.y; 
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
    half zero = __float2half(0.0f);
    half neg_inf = __float2half(-10000.0f); 
    
    if (k < history_tokens + n_tokens) {
        if (k < history_tokens) {
            mask[q * stride + k] = zero; // Sparse history is fully visible
        } else {
            int k_local = k - history_tokens;
            mask[q * stride + k] = (k_local <= q) ? zero : neg_inf; // Causal for current ubatch
        }
    }
}

void ggml_cuda_op_hga_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int32_t params[2];
    memcpy(params, dst->op_params, sizeof(params));
    int history_tokens = params[0];
    int n_tokens = params[1];
    int stride = history_tokens + n_tokens;
    
    dim3 block(256);
    dim3 grid((history_tokens + n_tokens + block.x - 1) / block.x, n_tokens);
    
    hga_mask_kernel<<<grid, block, 0, ctx.stream()>>>(
        (half*)dst->data, history_tokens, n_tokens, stride);
}