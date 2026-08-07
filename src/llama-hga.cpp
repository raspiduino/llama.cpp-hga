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

void hga_alloc_pinned_mapped(size_t bytes, void** host_ptr, void** dev_ptr) {
    *host_ptr = malloc(bytes);
    *dev_ptr = *host_ptr;
}

void hga_free_pinned_mapped(void* host_ptr) {
    free(host_ptr);
}

} // extern "C"
#endif

ggml_tensor * ggml_hga_summary(ggml_context * ctx, ggml_tensor * k_acc, int32_t start_pos, float theta_base, ggml_tensor * out_slot) {
    ggml_tensor * op = ggml_new_tensor(ctx, out_slot->type, GGML_MAX_DIMS, out_slot->ne);
    op->op = GGML_OP_HGA_SUMMARY; op->src[0] = k_acc;
    memcpy(op->op_params, &start_pos, sizeof(int32_t));
    memcpy((char*)op->op_params + sizeof(int32_t), &theta_base, sizeof(float));
    op->data = out_slot->data; return op;
}

ggml_tensor * ggml_hga_route(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * summaries, int32_t valid_chunks) {
    ggml_tensor * op = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, valid_chunks);
    op->op = GGML_OP_HGA_ROUTE; 
    op->src[0] = q_cur; 
    op->src[1] = summaries;
    int32_t params[4] = {valid_chunks, (int32_t)q_cur->ne[1], (int32_t)summaries->ne[1], (int32_t)q_cur->ne[2]};
    memcpy(op->op_params, params, sizeof(params)); 
    return op;
}

ggml_tensor * ggml_hga_stitch(ggml_context * ctx, ggml_tensor * hist, ggml_tensor * unclosed, ggml_tensor * cur,
                              int32_t hist_tokens, int32_t unclosed_tokens) {
    int32_t cur_tokens = cur->ne[2];
    int32_t total_tokens = hist_tokens + unclosed_tokens + cur_tokens;
    ggml_tensor * out = ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], total_tokens);
    out->op = GGML_OP_HGA_STITCH;
    out->src[0] = hist;
    out->src[1] = unclosed;
    out->src[2] = cur;
    int32_t params[4] = {hist_tokens, unclosed_tokens, cur_tokens, (int32_t)cur->nb[2]};
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
                              int32_t chunk_size, int32_t unclosed_tokens) {
    auto& hga = get_hga_layer(il);
    ggml_tensor * hist = is_v ? hga.cpu_hist_v : hga.cpu_hist_k;
    
    int64_t ne0 = hist->ne[0];
    int32_t total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);
    int64_t total_closed_tokens = total_out_chunks * chunk_size;
    int64_t total_tokens = total_closed_tokens + unclosed_tokens;
    
    ggml_tensor * out = ggml_new_tensor_2d(ctx, hist->type, ne0, total_tokens);
    out->op = GGML_OP_HGA_GATHER;
    out->src[0] = routed_idxs;
    out->src[1] = dummy_src; 
    
    size_t chunk_bytes = ggml_row_size(hist->type, ne0) * chunk_size;
    int32_t params[9] = {sink_end, k_to_route, local_start, valid_chunks, chunk_size, unclosed_tokens, (int32_t)chunk_bytes, il, is_v};
    memcpy(out->op_params, params, sizeof(params));
    return out;
}

ggml_tensor * ggml_hga_build_idxs(ggml_context * ctx, ggml_tensor * scores, 
                                  int32_t valid_chunks, int32_t sink_end, int32_t k_to_route, int32_t local_start) {
    int32_t total_idxs = sink_end + k_to_route + (valid_chunks - local_start);
    
    ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_idxs);
    out->op = GGML_OP_HGA_BUILD_IDXS;
    out->src[0] = scores;
    
    int32_t params[4] = {valid_chunks, sink_end, k_to_route, local_start};
    memcpy(out->op_params, params, sizeof(params));
    return out;
}

ggml_tensor * ggml_hga_store(ggml_context * ctx, ggml_tensor * src, ggml_tensor * dummy1, ggml_tensor * dummy2, int32_t il, int32_t is_v, int32_t offset_bytes, int32_t bytes_to_copy) {
    // Dummy output tensor to satisfy the graph allocator
    ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    out->op = GGML_OP_HGA_STORE;
    out->src[0] = src;
    out->src[1] = dummy1; // Keep-alive anchor for k_idxs
    out->src[2] = dummy2; // Keep-alive anchor for v_idxs
    
    int32_t params[4] = {il, is_v, offset_bytes, bytes_to_copy};
    memcpy(out->op_params, params, sizeof(params));
    return out;
}

ggml_tensor * llm_build_hga_attn(
        const llm_graph_context & llm,
        llm_graph_input_attn_kv * inp,
        ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,
        ggml_tensor * kq_b, ggml_tensor * sinks,
        float kq_scale, int il) {

    GGML_UNUSED(inp);
    GGML_UNUSED(kq_b);

    auto & hga = get_hga_layer(il);
    ggml_context * ctx0 = llm.ctx0;
    ggml_cgraph * gf = llm.gf;

    uint32_t valid_chunks = hga.n_chunks_closed;
    int n_tokens = k_cur->ne[2];

    int64_t chunk_size = g_hga_config.chunk_size;

    // =====================================================================
    // THE TIERED GATHER (CUB-Free, Graph-Capture Safe)
    // =====================================================================
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

    // =====================================================================
    // GATHER CLOSED CHUNKS + UNCLOSED "LIMBO" TOKENS 
    // =====================================================================
    ggml_tensor * k_gathered = nullptr;
    ggml_tensor * v_gathered = nullptr;
    
    int total_out_chunks = sink_end + k_to_route + (valid_chunks - local_start);
    int closed_history_tokens = total_out_chunks * chunk_size;
    
    int unclosed_tokens = 0;
    if (n_tokens == 1 && hga.carry_count > 0) {
        unclosed_tokens = hga.carry_count;
    }

    int total_history_tokens = closed_history_tokens + unclosed_tokens;

    if (total_history_tokens > 0) {
        // Pass k_cur/v_cur as dummy sources. Pass 'il' and 'is_v' (0 or 1).
        // The CUDA kernel fetches the real PCIe pointer via hga_get_pcie_ptr(il, is_v).
        // It handles BOTH scattered closed chunks AND contiguous unclosed tokens in one launch.
        ggml_tensor * k_gath_raw = ggml_hga_gather(ctx0, final_idxs, k_cur, il, 0, sink_end, k_to_route, local_start, valid_chunks, chunk_size, unclosed_tokens);
        ggml_tensor * v_gath_raw = ggml_hga_gather(ctx0, final_idxs, v_cur, il, 1, sink_end, k_to_route, local_start, valid_chunks, chunk_size, unclosed_tokens);

        k_gathered = (k_gath_raw->type == k_cur->type) ? k_gath_raw : ggml_cast(ctx0, k_gath_raw, k_cur->type);
        v_gathered = (v_gath_raw->type == v_cur->type) ? v_gath_raw : ggml_cast(ctx0, v_gath_raw, v_cur->type);

        ggml_build_forward_expand(gf, k_gathered);
        ggml_build_forward_expand(gf, v_gathered);
    }

    // =====================================================================
    // FUSED STITCH (History + Current Ubarch)
    // =====================================================================
    ggml_tensor * dummy = k_cur;

    // k_gathered now contains BOTH closed chunks and unclosed tokens.
    // We pass it as 'hist', and 0 for 'unclosed_tokens' to the stitch kernel.
    ggml_tensor * k_full = ggml_hga_stitch(ctx0, k_gathered ? k_gathered : dummy, dummy, k_cur, total_history_tokens, 0);
    ggml_tensor * v_full = ggml_hga_stitch(ctx0, v_gathered ? v_gathered : dummy, dummy, v_cur, total_history_tokens, 0);

    ggml_build_forward_expand(gf, k_full);
    ggml_build_forward_expand(gf, v_full);

    // =====================================================================
    // THE PERMUTE TRICK
    // =====================================================================
    ggml_tensor * q_perm = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx0, k_full, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx0, v_full, 0, 2, 1, 3);

    // =====================================================================
    // THE MASK TRAP FIX & KEEP-ALIVE ANCHOR
    // =====================================================================
    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * hga_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, total_history_tokens + n_tokens, n_tokens, 1, 1);
    hga_mask->op = GGML_OP_HGA_MASK;
    hga_mask->src[0] = kq_mask;

    int32_t mask_params[2] = {total_history_tokens, n_tokens};
    memcpy(hga_mask->op_params, mask_params, sizeof(mask_params));
    ggml_build_forward_expand(gf, hga_mask);

    ggml_tensor * fa_mask = (n_tokens > 1) ? hga_mask : nullptr;

    // =====================================================================
    // EXECUTE FLASH ATTENTION
    // =====================================================================
    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q_perm, k_perm, v_perm, fa_mask, kq_scale,
                                  llm.hparams.f_max_alibi_bias,
                                  llm.hparams.attn_soft_cap ? llm.hparams.f_attn_logit_softcapping : 0.0f);

    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    if (sinks) ggml_flash_attn_ext_add_sinks(cur, sinks);

    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);

    return cur;
}