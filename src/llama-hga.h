#pragma once
#include "llama-graph.h"
#include "ggml.h"
#include <cstdint>
#include <cstdlib>

struct HGAConfig {
    uint32_t chunk_size = 64, num_sink_chunks = 2, num_local_chunks = 8, num_routed_chunks = 16;
    uint32_t get_working_chunks() const { return num_sink_chunks + num_local_chunks + num_routed_chunks; }
    static HGAConfig load_from_env() {
        HGAConfig cfg;
        if (const char* val = getenv("HGA_CHUNK_SIZE")) cfg.chunk_size = atoi(val);
        if (const char* val = getenv("HGA_SINK_CHUNKS")) cfg.num_sink_chunks = atoi(val);
        if (const char* val = getenv("HGA_LOCAL_CHUNKS")) cfg.num_local_chunks = atoi(val);
        if (const char* val = getenv("HGA_ROUTED_CHUNKS")) cfg.num_routed_chunks = atoi(val);
        return cfg;
    }
};
extern HGAConfig g_hga_config;

struct llama_hga_layer {
    ggml_tensor * cpu_hist_k, * cpu_hist_v, * gpu_scratch_k, * gpu_scratch_v, * gpu_summaries, * gpu_carry_k;    
    
    ggml_context * ctx_cpu;
    ggml_context * ctx_gpu;
    ggml_backend_buffer_t buf_cpu;
    ggml_backend_buffer_t buf_gpu;

    void * pinned_k_host;
    void * pinned_v_host;
    void * cpu_hist_k_dev;
    void * cpu_hist_v_dev;

    uint32_t tokens_processed = 0, n_chunks_closed = 0, carry_count = 0; 
};

void hga_init_layers(int n_layers);
void set_hga_layer(int32_t il, const llama_hga_layer & hga);
llama_hga_layer & get_hga_layer(int32_t il);
void hga_truncate_layers(int32_t keep_tokens);

#ifdef __cplusplus
extern "C" {
#endif

// Bridge functions for backend-specific memory allocation
void hga_alloc_pinned_mapped(size_t bytes, void** host_ptr, void** dev_ptr);
void hga_free_pinned_mapped(void* host_ptr);

#ifdef __cplusplus
}
#endif

ggml_tensor * ggml_hga_summary(ggml_context * ctx, ggml_tensor * k_acc, int32_t start_pos, float theta_base, ggml_tensor * out_slot);
ggml_tensor * ggml_hga_route(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * summaries, int32_t valid_chunks);
ggml_tensor * ggml_hga_stitch(ggml_context * ctx, ggml_tensor * hist, ggml_tensor * unclosed, ggml_tensor * cur, int32_t hist_tokens, int32_t unclosed_tokens);
ggml_tensor * ggml_hga_gather(ggml_context * ctx, ggml_tensor * routed_idxs, ggml_tensor * src, int32_t sink_end, int32_t k_to_route, int32_t local_start, int32_t valid_chunks, int32_t chunk_size);
ggml_tensor * llm_build_hga_attn(const llm_graph_context & llm, llm_graph_input_attn_kv * inp, ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur, ggml_tensor * kq_b, ggml_tensor * sinks, float kq_scale, int il);