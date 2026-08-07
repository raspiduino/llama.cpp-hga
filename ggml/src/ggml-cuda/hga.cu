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
                for (int t = 0; t < n_tokens; ++t) q_sum += Q[d + h_q * head_dim + t * head_dim * n_head_q];
            }
            float q_mean = q_sum / (float)(G * n_tokens);
            float s_val = __bfloat162float(S[d + h_kv * head_dim + c * head_dim * n_head_kv]);
            total_score += q_mean * s_val;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) total_score += __shfl_down_sync(0xffffffff, total_score, offset);
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
// 3. HGA GATHER: Chunk-Level Tiered Byte-Copy (PCIe -> VRAM)
// ==============================================================================
extern "C" void* hga_get_pcie_ptr(int il, int is_v);

__global__ void hga_gather_tiered_kernel(
    const int32_t* __restrict__ routed_idxs, const char* __restrict__ src, char* __restrict__ dst,
    int sink_end, int k_to_route, int local_start, int valid_chunks, 
    int chunk_size, size_t chunk_bytes, int unclosed_tokens) 
{
    int out_c = blockIdx.x;
    int total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);
    size_t token_bytes = chunk_bytes / chunk_size;
    
    // Handle unclosed tokens in the final block
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
        if (threadIdx.x == 0) {
            for (size_t j = base; j < unclosed_bytes; ++j) dst_ptr[j] = src_ptr[j];
        }
        return;
    }

    // Normal closed chunk block
    int src_c = -1;
    if (out_c < sink_end) src_c = out_c;
    else if (out_c < sink_end + k_to_route) src_c = routed_idxs[out_c - sink_end];
    else src_c = local_start + (out_c - (sink_end + k_to_route));
    
    const char* src_ptr = src + src_c * chunk_bytes;
    char* dst_ptr = dst + out_c * chunk_bytes;
    
    size_t i = threadIdx.x * 16;
    for (; i + 16 <= chunk_bytes; i += blockDim.x * 16) {
        uint4 val = *reinterpret_cast<const uint4*>(src_ptr + i);
        *reinterpret_cast<uint4*>(dst_ptr + i) = val;
    }
    size_t base = (chunk_bytes / 16) * 16;
    if (threadIdx.x == 0) {
        for (size_t j = base; j < chunk_bytes; ++j) dst_ptr[j] = src_ptr[j];
    }
}

void ggml_cuda_op_hga_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * routed_idxs = dst->src[0];
    
    int32_t params[9];
    memcpy(params, dst->op_params, sizeof(params));
    int sink_end = params[0], k_to_route = params[1], local_start = params[2], valid_chunks = params[3];
    int chunk_size = params[4];
    int unclosed_tokens = params[5];
    size_t chunk_bytes = (size_t)params[6];
    int il = params[7];
    int is_v = params[8];
    
    const char* src = (const char*)hga_get_pcie_ptr(il, is_v);
    
    int total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);
    // Add 1 extra block if there are unclosed tokens to handle the tail
    int grid_size = total_out_chunks + (unclosed_tokens > 0 ? 1 : 0);
    
    hga_gather_tiered_kernel<<<grid_size, 256, 0, ctx.stream()>>>(
        (const int32_t*)routed_idxs->data, src, (char*)dst->data,
        sink_end, k_to_route, local_start, valid_chunks, chunk_size, chunk_bytes, unclosed_tokens
    );
}

// ==============================================================================
// 4. HGA STITCH: Pure F16/BF16 Byte-Copy
// ==============================================================================
__global__ void hga_stitch_kernel(
    const char* __restrict__ hist, const char* __restrict__ unclosed, const char* __restrict__ cur, char* __restrict__ out,
    int hist_tokens, int unclosed_tokens, int cur_tokens, size_t token_bytes) {
    int tok_idx = blockIdx.x;
    const char* src_ptr = nullptr; int src_tok_idx = 0;
    if (tok_idx < hist_tokens) { src_ptr = hist; src_tok_idx = tok_idx; }
    else if (tok_idx < hist_tokens + unclosed_tokens) { src_ptr = unclosed; src_tok_idx = tok_idx - hist_tokens; }
    else { src_ptr = cur; src_tok_idx = tok_idx - hist_tokens - unclosed_tokens; }
    for (size_t i = threadIdx.x; i < token_bytes; i += blockDim.x) {
        out[tok_idx * token_bytes + i] = src_ptr[src_tok_idx * token_bytes + i];
    }
}

void ggml_cuda_op_hga_stitch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int32_t params[4]; memcpy(params, dst->op_params, sizeof(params));
    int hist_tokens = params[0], unclosed_tokens = params[1], cur_tokens = params[2];
    size_t token_bytes = (size_t)params[3];
    int total_tokens = hist_tokens + unclosed_tokens + cur_tokens;
    hga_stitch_kernel<<<total_tokens, 256, 0, ctx.stream()>>>(
        (const char*)dst->src[0]->data, (const char*)dst->src[1]->data, (const char*)dst->src[2]->data, (char*)dst->data,
        hist_tokens, unclosed_tokens, cur_tokens, token_bytes);
}

// ==============================================================================
// 5. HGA MASK: Causal Triangle Stamping
// ==============================================================================
__global__ void hga_mask_kernel(half* mask, int history_tokens, int n_tokens, int stride) {
    int q = blockIdx.y; int k = blockIdx.x * blockDim.x + threadIdx.x;
    half zero = __float2half(0.0f); half neg_inf = __float2half(-10000.0f);
    if (k < history_tokens + n_tokens) {
        if (k < history_tokens) mask[q * stride + k] = zero;
        else {
            int k_local = k - history_tokens;
            mask[q * stride + k] = (k_local <= q) ? zero : neg_inf;
        }
    }
}

void ggml_cuda_op_hga_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int32_t params[2]; memcpy(params, dst->op_params, sizeof(params));
    int history_tokens = params[0], n_tokens = params[1];
    int stride = history_tokens + n_tokens;
    dim3 block(256); dim3 grid((stride + block.x - 1) / block.x, n_tokens);
    hga_mask_kernel<<<grid, block, 0, ctx.stream()>>>((half*)dst->data, history_tokens, n_tokens, stride);
}

// ==============================================================================
// 6. HGA BUILD IDXS: CUB-Free Top-K and Tiered Assembly (Guarantees Graph Capture)
// ==============================================================================
__global__ void hga_build_idxs_kernel(
    const float* __restrict__ scores, int32_t* __restrict__ out_idxs,
    int valid_chunks, int sink_end, int k_to_route, int local_start) 
{
    // Single thread execution: N=400, K=16 takes < 5 microseconds.
    // Zero dynamic memory, zero CUB dependency.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int out_idx = 0;
        
        // 1. Sinks
        for (int i = 0; i < sink_end; i++) {
            out_idxs[out_idx++] = i;
        }
        
        // 2. Top-K (Selection Sort)
        int selected[64]; // Assuming k_to_route <= 64
        for (int i = 0; i < k_to_route; i++) {
            float max_val = -1e30f;
            int max_idx = -1;
            for (int j = 0; j < valid_chunks; j++) {
                bool is_selected = false;
                for (int s = 0; s < i; s++) {
                    if (selected[s] == j) { is_selected = true; break; }
                }
                if (is_selected) continue;
                
                if (scores[j] > max_val) {
                    max_val = scores[j];
                    max_idx = j;
                }
            }
            selected[i] = max_idx;
            out_idxs[out_idx++] = max_idx;
        }
        
        // 3. Local
        for (int i = local_start; i < valid_chunks; i++) {
            out_idxs[out_idx++] = i;
        }
    }
}

void ggml_cuda_op_hga_build_idxs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * scores = dst->src[0];
    
    int32_t params[4];
    memcpy(params, dst->op_params, sizeof(params));
    int valid_chunks = params[0];
    int sink_end = params[1];
    int k_to_route = params[2];
    int local_start = params[3];
    
    hga_build_idxs_kernel<<<1, 1, 0, ctx.stream()>>>(
        (const float*)scores->data, (int32_t*)dst->data,
        valid_chunks, sink_end, k_to_route, local_start
    );
}

// ==============================================================================
// 7. HGA STORE: Async DMA (VRAM Staging -> Pinned Host Memory)
// ==============================================================================
void ggml_cuda_op_hga_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    
    int32_t params[4];
    memcpy(params, dst->op_params, sizeof(params));
    int il = params[0];
    int is_v = params[1];
    int offset_bytes = params[2];
    int bytes_to_copy = params[3];
    
    char* dst_ptr = (char*)hga_get_pcie_ptr(il, is_v) + offset_bytes;
    
    // Pure asynchronous DMA from VRAM staging buffer to Pinned Host Memory.
    // Zero host blocking. Stream ordering guarantees correctness.
    cudaMemcpyAsync(dst_ptr, src->data, bytes_to_copy, cudaMemcpyDeviceToHost, ctx.stream());
}

// ==============================================================================
// ZERO-COPY PINNED MEMORY BRIDGE (Exposed to core llama.cpp)
// ==============================================================================
extern "C" {
    // Prototypes to silence -Wmissing-declarations
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