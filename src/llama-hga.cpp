#include "llama-hga.h"  
#include "ggml.h"  
#include <vector>  
#include <cstring>  
  
HGAConfig g_hga_config = HGAConfig::load_from_env();  
static std::vector<llama_hga_layer> g_hga_layers;  
  
void hga_init_layers(int n_layers) { g_hga_layers.resize(n_layers); }  
void set_hga_layer(int32_t il, const llama_hga_layer & hga) { g_hga_layers[il] = hga; }  
llama_hga_layer & get_hga_layer(int32_t il) { return g_hga_layers[il]; }  
  
void hga_truncate_layers(int32_t keep_tokens) {  
    uint32_t u_keep = (keep_tokens < 0) ? 0 : (uint32_t)keep_tokens;  
    uint32_t chunk_size = g_hga_config.chunk_size;  
    for (auto & hga : g_hga_layers) {  
        if (hga.tokens_processed > u_keep) {  
            hga.tokens_processed = u_keep;  
            hga.n_chunks_closed = u_keep / chunk_size;  
        }  
    }  
}  
  
#ifndef GGML_USE_CUDA  
extern "C" {  
void hga_alloc_pinned_mapped(size_t bytes, void** host_ptr, void** dev_ptr) { *host_ptr = malloc(bytes); *dev_ptr = *host_ptr; }  
void hga_free_pinned_mapped(void* host_ptr) { free(host_ptr); }  
} // extern "C"  
#endif  

extern "C" void* hga_get_carry_ptr(int il, int is_v) {  
    auto& hga = get_hga_layer(il);  
    return is_v ? hga.gpu_carry_v->data : hga.gpu_carry_k->data;  
}
  
ggml_tensor * ggml_hga_summary(ggml_context * ctx, ggml_tensor * k_acc, int32_t start_pos, float theta_base, ggml_tensor * out_slot) {  
    ggml_tensor * op = ggml_new_tensor(ctx, out_slot->type, GGML_MAX_DIMS, out_slot->ne);  
    op->op = GGML_OP_HGA_SUMMARY; op->src[0] = k_acc;  
    memcpy(op->op_params, &start_pos, sizeof(int32_t));  
    memcpy((char*)op->op_params + sizeof(int32_t), &theta_base, sizeof(float));  
    op->data = out_slot->data; return op;  
}  
  
ggml_tensor * ggml_hga_route(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * summaries, int32_t valid_chunks) {  
    ggml_tensor * op = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, valid_chunks);  
    op->op = GGML_OP_HGA_ROUTE; op->src[0] = q_cur; op->src[1] = summaries;  
    int32_t params[4] = {valid_chunks, (int32_t)q_cur->ne[1], (int32_t)summaries->ne[1], (int32_t)q_cur->ne[2]};  
    memcpy(op->op_params, params, sizeof(params)); return op;  
}  
  
ggml_tensor * ggml_hga_stitch(ggml_context * ctx, ggml_tensor * hist, ggml_tensor * cur, int32_t n_tokens, int32_t static_max_seq, int32_t token_bytes, HGA_Decode_State* dev_state) {  
    // Output is strictly 2D: [n_embd_gqa, static_max_seq]
    ggml_tensor * out = ggml_new_tensor_2d(ctx, hist->type, hist->ne[0], static_max_seq);  
    out->op = GGML_OP_HGA_STITCH;  
    out->src[0] = hist; 
    out->src[1] = cur; // Keep-alive anchor for the staging buffer
    int32_t params[4] = {n_tokens, token_bytes, 0, 0};  
    memcpy(&params[2], &dev_state, sizeof(HGA_Decode_State*));
    memcpy(out->op_params, params, sizeof(params));  
    return out;  
}
  
extern "C" void* hga_get_pcie_ptr(int il, int is_v) {  
    auto& hga = get_hga_layer(il);  
    return is_v ? hga.cpu_hist_v_dev : hga.cpu_hist_k_dev;  
}  
  
ggml_tensor * ggml_hga_gather(ggml_context * ctx, ggml_tensor * routed_idxs, ggml_tensor * dummy_src,  
                              int32_t il, int32_t is_v,  
                              int32_t sink_end, int32_t k_to_route, int32_t local_start, int32_t valid_chunks,  
                              int32_t chunk_size, int32_t static_max_seq, HGA_Decode_State* dev_state) {  
    auto& hga = get_hga_layer(il);  
    ggml_tensor * hist = is_v ? hga.cpu_hist_v : hga.cpu_hist_k;  
    int64_t ne0 = hist->ne[0];  
    ggml_tensor * out = ggml_new_tensor_2d(ctx, hist->type, ne0, static_max_seq);  
    out->op = GGML_OP_HGA_GATHER;  
    out->src[0] = routed_idxs; out->src[1] = dummy_src;  
    int32_t chunk_bytes = (int32_t)(ggml_row_size(hist->type, ne0) * chunk_size);  
    int32_t params[10] = {sink_end, k_to_route, local_start, valid_chunks, chunk_size, chunk_bytes, il, is_v, 0, 0};  
    memcpy(&params[8], &dev_state, sizeof(HGA_Decode_State*));
    memcpy(out->op_params, params, sizeof(params));  
    return out;  
}  
  
ggml_tensor * ggml_hga_build_idxs(ggml_context * ctx, ggml_tensor * scores,  
                                  int32_t valid_chunks, int32_t sink_end, int32_t k_to_route, int32_t local_start) {  
    int32_t total_idxs = sink_end + k_to_route + (valid_chunks - local_start);  
    ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_idxs);  
    out->op = GGML_OP_HGA_BUILD_IDXS; out->src[0] = scores;  
    int32_t params[4] = {valid_chunks, sink_end, k_to_route, local_start};  
    memcpy(out->op_params, params, sizeof(params)); return out;  
}  
  
ggml_tensor * ggml_hga_store(ggml_context * ctx, ggml_tensor * src, ggml_tensor * dummy1, ggml_tensor * dummy2, int32_t il, int32_t is_v, int32_t bytes_to_copy, HGA_Decode_State* dev_state) {  
    ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);  
    out->op = GGML_OP_HGA_STORE;  
    out->src[0] = src; out->src[1] = dummy1; out->src[2] = dummy2;  
    int32_t params[6] = {il, is_v, bytes_to_copy, 0, 0, 0};  
    memcpy(&params[3], &dev_state, sizeof(HGA_Decode_State*));
    memcpy(out->op_params, params, sizeof(params));  
    return out;  
}

ggml_tensor * ggml_hga_mask(ggml_context * ctx, ggml_tensor * dummy_dep, int32_t n_tokens, int32_t static_max_seq, HGA_Decode_State* dev_state) {
    ggml_tensor * out = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, static_max_seq, n_tokens, 1, 1);
    out->op = GGML_OP_HGA_MASK;
    
    // CLEAN ANCHOR: Satisfies the graph allocator so it assigns a compute buffer,
    // but doesn't trigger shape-change invalidations.
    out->src[0] = dummy_dep; 
    
    int32_t params[4] = {n_tokens, static_max_seq, 0, 0};
    memcpy(&params[2], &dev_state, sizeof(HGA_Decode_State*));
    memcpy(out->op_params, params, sizeof(params));
    return out;
}
  
ggml_tensor * llm_build_hga_attn(      
        const llm_graph_context & llm,      
        llm_graph_input_attn_kv * inp,      
        ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,      
        ggml_tensor * staging_k, ggml_tensor * staging_v,
        ggml_tensor * kq_b, ggml_tensor * sinks,      
        float kq_scale, int il) {
    
    GGML_UNUSED(inp); GGML_UNUSED(kq_b);    
    auto & hga = get_hga_layer(il);    
    ggml_context * ctx0 = llm.ctx0; ggml_cgraph * gf = llm.gf;    
    
    uint32_t valid_chunks = hga.n_chunks_closed;    
    int n_tokens = k_cur->ne[2];    
    int64_t chunk_size = g_hga_config.chunk_size;    
    
    uint32_t sink_end = std::min(g_hga_config.num_sink_chunks, valid_chunks);    
    uint32_t local_start = std::max(sink_end, valid_chunks - g_hga_config.num_local_chunks);    
    uint32_t mid_chunks = local_start - sink_end;    
    uint32_t k_to_route = std::min(g_hga_config.num_routed_chunks, mid_chunks);    
    
    ggml_tensor * final_idxs = nullptr;    
    if (k_to_route > 0) {    
        ggml_tensor * scores = ggml_hga_route(ctx0, q_cur, hga.gpu_summaries, valid_chunks);    
        final_idxs = ggml_hga_build_idxs(ctx0, scores, valid_chunks, sink_end, k_to_route, local_start);    
    } else {    
        ggml_tensor * dummy_scores = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);    
        final_idxs = ggml_hga_build_idxs(ctx0, dummy_scores, valid_chunks, sink_end, 0, local_start);    
    }    
    
    ggml_tensor * k_gathered = nullptr;    
    ggml_tensor * v_gathered = nullptr;    
    int total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);    
    int closed_history_tokens = total_out_chunks * chunk_size;    
    int unclosed_tokens = (n_tokens == 1 && hga.carry_count > 0) ? hga.carry_count : 0;    
    int total_history_tokens = closed_history_tokens + unclosed_tokens;  
  
    // Update state block BEFORE executing graph nodes  
    hga.cpu_state->real_unclosed_tokens = unclosed_tokens;  
    hga.cpu_state->real_total_history = total_history_tokens;  
    
    // DYNAMIC SIZING WITH FA BLOCK ALIGNMENT:
    int64_t max_out_chunks = g_hga_config.num_sink_chunks + g_hga_config.num_routed_chunks + g_hga_config.num_local_chunks;
    int64_t max_history_capacity = (max_out_chunks + 1) * chunk_size;
    
    // Pad n_tokens to a multiple of 64 to prevent FlashAttention block-reads from triggering illegal memory access
    int64_t padded_n_tokens = (n_tokens + 63) & ~63;
    int64_t static_max_seq = max_history_capacity + padded_n_tokens;
    
    // CRITICAL: Ensure static_max_seq itself is a multiple of 64 for FA internal block alignment
    static_max_seq = (static_max_seq + 63) & ~63;
    
    if (total_history_tokens > 0) {    
        // FIX: Removed the inner `ggml_tensor *` declarations that were shadowing the outer variables.
        k_gathered = ggml_hga_gather(ctx0, final_idxs, k_cur, il, 0, sink_end, k_to_route, local_start, valid_chunks, chunk_size, static_max_seq, hga.dev_state);    
        v_gathered = ggml_hga_gather(ctx0, final_idxs, v_cur, il, 1, sink_end, k_to_route, local_start, valid_chunks, chunk_size, static_max_seq, hga.dev_state);    
    
        ggml_build_forward_expand(gf, k_gathered);    
        ggml_build_forward_expand(gf, v_gathered);    
    }    
    
    // Calculate exact byte size of one quantized token using the transient tensor's type
    int32_t token_bytes_k = ggml_row_size(staging_k->type, k_cur->ne[0] * k_cur->ne[1]);    
    int32_t token_bytes_v = ggml_row_size(staging_v->type, v_cur->ne[0] * v_cur->ne[1]);    

    // If history is empty, use the transient staging buffer as a dummy 'hist' source.
    // The stitch kernel reads state->real_total_history (which is 0) and will safely ignore it.
    ggml_tensor * k_hist_src = k_gathered ? k_gathered : staging_k;    
    ggml_tensor * v_hist_src = v_gathered ? v_gathered : staging_v;    

    // Stitch the gathered history and the transient staging buffer
    ggml_tensor * k_full = ggml_hga_stitch(ctx0, k_hist_src, staging_k, n_tokens, static_max_seq, token_bytes_k, hga.dev_state);      
    ggml_tensor * v_full = ggml_hga_stitch(ctx0, v_hist_src, staging_v, n_tokens, static_max_seq, token_bytes_v, hga.dev_state);   
    
    ggml_build_forward_expand(gf, k_full);    
    ggml_build_forward_expand(gf, v_full);    
  
    // Reshape to 3D [head_dim, n_head_kv, static_max_seq] for FlashAttention  
    ggml_tensor * k_full_3d = ggml_reshape_3d(ctx0, k_full, k_cur->ne[0], k_cur->ne[1], static_max_seq);  
    ggml_tensor * v_full_3d = ggml_reshape_3d(ctx0, v_full, v_cur->ne[0], v_cur->ne[1], static_max_seq);  
  
    ggml_tensor * q_perm = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);    
    ggml_tensor * k_perm = ggml_permute(ctx0, k_full_3d, 0, 2, 1, 3);    
    ggml_tensor * v_perm = ggml_permute(ctx0, v_full_3d, 0, 2, 1, 3);    
    
    // CLEAN ANCHOR: We pass k_full_3d as a dummy dependency.
    // It is already in the graph, and its shape (static_max_seq) is strictly constant during decode.
    // This satisfies the graph allocator WITHOUT triggering the per-token shape change of kq_mask.
    ggml_tensor * hga_mask = ggml_hga_mask(ctx0, k_full_3d, n_tokens, static_max_seq, hga.dev_state);
    ggml_build_forward_expand(gf, hga_mask);
    ggml_tensor * fa_mask = (n_tokens > 1) ? hga_mask : nullptr;
    
    // PASS Q4_0 DIRECTLY TO FLASH ATTENTION!  
    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q_perm, k_perm, v_perm, fa_mask, kq_scale,    
                                  llm.hparams.f_max_alibi_bias,    
                                  llm.hparams.attn_soft_cap ? llm.hparams.f_attn_logit_softcapping : 0.0f);    
    
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);    
    if (sinks) ggml_flash_attn_ext_add_sinks(cur, sinks);    
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);    
    return cur;    
}    