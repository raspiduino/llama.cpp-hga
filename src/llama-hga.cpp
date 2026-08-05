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

ggml_tensor * ggml_hga_stitch(ggml_context * ctx, ggml_tensor * sink, ggml_tensor * rout, ggml_tensor * local, ggml_tensor * cur, int32_t sink_tokens, int32_t rout_tokens, int32_t local_tokens) {
    int32_t cur_tokens = cur->ne[2];
    int32_t total_tokens = sink_tokens + rout_tokens + local_tokens + cur_tokens;
    ggml_tensor * out = ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], total_tokens);
    out->op = GGML_OP_HGA_STITCH; out->src[0] = sink; out->src[1] = rout; out->src[2] = local; out->src[3] = cur;
    int32_t params[5] = {sink_tokens, rout_tokens, local_tokens, cur_tokens, (int32_t)cur->nb[2]};
    memcpy(out->op_params, params, sizeof(params)); return out;
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

    // 1. Mathematically Pure Routing (Runs on GPU)
    ggml_tensor * scores = ggml_hga_route(ctx0, q_cur, hga.gpu_summaries, valid_chunks);

    int64_t chunk_size = g_hga_config.chunk_size;
    int64_t embd_k = k_cur->ne[0] * k_cur->ne[1];
    int64_t embd_v = v_cur->ne[0] * v_cur->ne[1];
    
    // Reshape [embd, tokens] -> [embd * chunk_size, max_chunks]
    // Because chunks are contiguous in memory, this is a perfectly valid zero-cost view.
    ggml_tensor * hist_k_chunked = ggml_reshape_2d(ctx0, hga.cpu_hist_k, embd_k * chunk_size, hga.cpu_hist_k->ne[1] / chunk_size);
    ggml_tensor * hist_v_chunked = ggml_reshape_2d(ctx0, hga.cpu_hist_v, embd_v * chunk_size, hga.cpu_hist_v->ne[1] / chunk_size);

    // =====================================================================
    // THE TIERED GATHER: Sinks + Routed + Local (Mirroring vLLM)
    // =====================================================================
    uint32_t sink_end = std::min(g_hga_config.num_sink_chunks, valid_chunks);
    uint32_t local_start = std::max(sink_end, valid_chunks - g_hga_config.num_local_chunks);
    uint32_t mid_chunks = local_start - sink_end;
    uint32_t k_to_route = std::min(g_hga_config.num_routed_chunks, mid_chunks);

    ggml_tensor * final_idxs = nullptr;

    // A. Sinks (Always attend to the beginning of the prompt)
    if (sink_end > 0) {
        final_idxs = ggml_arange(ctx0, 0.0f, (float)sink_end, 1.0f);
        final_idxs = ggml_cast(ctx0, final_idxs, GGML_TYPE_I32);
    }

    // B. Routed (The sparse middle history)
    if (k_to_route > 0) {
        // Note: top_k might pick a sink/local index. FlashAttn handles duplicate keys gracefully.
        ggml_tensor * routed_idxs = ggml_top_k(ctx0, scores, k_to_route);
        if (final_idxs) {
            final_idxs = ggml_concat(ctx0, final_idxs, routed_idxs, 0);
        } else {
            final_idxs = routed_idxs;
        }
    }

    // C. Local (The sliding window immediately before the current ubatch)
    if (local_start < valid_chunks) {
        ggml_tensor * local_idxs = ggml_arange(ctx0, (float)local_start, (float)valid_chunks, 1.0f);
        local_idxs = ggml_cast(ctx0, local_idxs, GGML_TYPE_I32);
        if (final_idxs) {
            final_idxs = ggml_concat(ctx0, final_idxs, local_idxs, 0);
        } else {
            final_idxs = local_idxs;
        }
    }

    // =====================================================================
    // GATHER & STITCH
    // =====================================================================
    ggml_tensor * k_full = nullptr;
    ggml_tensor * v_full = nullptr;
    int history_tokens = 0;

    if (final_idxs) {
        ggml_tensor * k_gathered_chunked = ggml_get_rows(ctx0, hist_k_chunked, final_idxs);
        ggml_tensor * v_gathered_chunked = ggml_get_rows(ctx0, hist_v_chunked, final_idxs);

        int64_t gathered_chunks = final_idxs->ne[0];
        history_tokens = gathered_chunks * chunk_size;

        ggml_tensor * k_gathered = ggml_reshape_2d(ctx0, k_gathered_chunked, embd_k, history_tokens);
        ggml_tensor * v_gathered = ggml_reshape_2d(ctx0, v_gathered_chunked, embd_v, history_tokens);

        ggml_build_forward_expand(gf, k_gathered);
        ggml_build_forward_expand(gf, v_gathered);

        ggml_tensor * dummy = ggml_new_tensor_3d(ctx0, k_cur->type, k_cur->ne[0], k_cur->ne[1], 1);
        k_full = ggml_hga_stitch(ctx0, dummy, k_gathered, dummy, k_cur, 0, history_tokens, 0);
        v_full = ggml_hga_stitch(ctx0, dummy, v_gathered, dummy, v_cur, 0, history_tokens, 0);
        
        ggml_build_forward_expand(gf, k_full);
        ggml_build_forward_expand(gf, v_full);
    } else {
        // Edge case: No history yet (first ubatch)
        k_full = k_cur;
        v_full = v_cur;
    }

    // 5. THE PERMUTE TRICK: [D, H, N, B] -> [D, N, H, B]
    ggml_tensor * q_perm = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx0, k_full, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx0, v_full, 0, 2, 1, 3);

    // 6. Build the HGA Causal Mask
    int n_tokens = k_cur->ne[2];
    ggml_tensor * kq_mask = inp->get_kq_mask();
    
    ggml_tensor * hga_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, history_tokens + n_tokens, n_tokens, 1, 1);
    hga_mask->op = GGML_OP_HGA_MASK;
    hga_mask->src[0] = kq_mask; // Keep-alive anchor
    
    int32_t mask_params[2] = {history_tokens, n_tokens};
    memcpy(hga_mask->op_params, mask_params, sizeof(mask_params));
    ggml_build_forward_expand(gf, hga_mask);

    // 7. Execute Flash Attention
    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q_perm, k_perm, v_perm, hga_mask, kq_scale,
                                  llm.hparams.f_max_alibi_bias,
                                  llm.hparams.attn_soft_cap ? llm.hparams.f_attn_logit_softcapping : 0.0f);
    
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    if (sinks) ggml_flash_attn_ext_add_sinks(cur, sinks);
    
    // 8. FLATTEN TO 2D (The llama.cpp Attention Contract)
    // ggml_flash_attn_ext natively returns [D, H, N, B] (e.g., [256, 24, 512, 1]).
    // We flatten D and H into n_embd (6144) to match the 2D gate_sigmoid.
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    
    return cur;
}