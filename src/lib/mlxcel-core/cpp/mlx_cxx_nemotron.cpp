// Copyright 2025 mlx-lm-rs authors
// NemotronH full-forward decode path for the mlx_cxx bridge. Split out of
// mlx_cxx_bridge.cpp; calls the SSM/MoE fused-kernel FFI functions
// (fused_mamba2_forward, fused_moe_forward) across translation units.

#include "mlx_cxx_internal.h"

namespace mlx_cxx {

// ============ NemotronH Full Forward Decode ============

namespace {
struct NemotronModel {
    // Global weights
    const MlxArray *embed_w, *embed_s, *embed_b;
    const MlxArray *final_norm_w;
    const MlxArray *lm_head_w, *lm_head_s, *lm_head_b;
    std::vector<const MlxArray*> norm_weights;
    std::vector<int32_t> block_types;

    struct Mamba {
        const MlxArray *in_w, *in_s, *in_b;
        const MlxArray *conv_w, *conv_b;
        const MlxArray *a_log, *d, *dt_bias, *norm_w;
        const MlxArray *out_w, *out_s, *out_b;
    };
    struct MoE {
        const MlxArray *gate_w, *corr_bias;
        const MlxArray *fc1_w, *fc1_s, *fc1_b;
        const MlxArray *fc2_w, *fc2_s, *fc2_b;
        const MlxArray *su_w, *su_s, *su_b;
        const MlxArray *sd_w, *sd_s, *sd_b;
    };
    struct Attn {
        const MlxArray *q_w, *q_s, *q_b;
        const MlxArray *k_w, *k_s, *k_b;
        const MlxArray *v_w, *v_s, *v_b;
        const MlxArray *o_w, *o_s, *o_b;
    };
    std::vector<Mamba> mamba_layers;
    std::vector<MoE> moe_layers;
    std::vector<Attn> attn_layers;

    // Config
    float norm_eps; int gs, bits;
    int m_inter, m_cdim, m_ck, m_heads, m_hdim, m_groups, m_state;
    float m_ts_min, m_ts_max, m_neps;
    int moe_tk; float moe_sc; bool moe_norm;
    int a_heads, a_kvh, a_hdim; float a_rope, a_scale;
};

static std::unordered_map<uint64_t, std::unique_ptr<NemotronModel>> g_models;
static uint64_t g_next_handle = 1;
} // anonymous namespace

uint64_t nemotron_register_model(
    const MlxArray& embed_w, const MlxArray& embed_s, const MlxArray& embed_b,
    const MlxArray& final_norm_w,
    const MlxArray& lm_head_w, const MlxArray& lm_head_s, const MlxArray* lm_head_b,
    rust::Slice<const MlxArray* const> norm_weights,
    rust::Slice<const int32_t> block_types,
    rust::Slice<const MlxArray* const> mw,
    rust::Slice<const MlxArray* const> ew,
    rust::Slice<const MlxArray* const> aw,
    float norm_eps, int32_t gs, int32_t bits,
    int32_t m_inter, int32_t m_cdim, int32_t m_ck,
    int32_t m_heads, int32_t m_hdim, int32_t m_groups, int32_t m_state,
    float m_ts_min, float m_ts_max, float m_neps,
    int32_t moe_tk, float moe_sc, bool moe_norm,
    int32_t a_heads, int32_t a_kvh, int32_t a_hdim,
    float a_rope, float a_scale
) {
    auto model = std::make_unique<NemotronModel>();
    model->embed_w = &embed_w; model->embed_s = &embed_s; model->embed_b = &embed_b;
    model->final_norm_w = &final_norm_w;
    model->lm_head_w = &lm_head_w; model->lm_head_s = &lm_head_s;
    model->lm_head_b = lm_head_b;
    for (auto* p : norm_weights) model->norm_weights.push_back(p);
    for (auto bt : block_types) model->block_types.push_back(bt);

    // Parse Mamba weights: 12 per layer
    // Order: in_w, in_s, in_b, conv_w, conv_b, a_log, d, dt_bias, norm_w, out_w, out_s, out_b
    for (size_t i = 0; i + 11 < mw.size(); i += 12) {
        model->mamba_layers.push_back({
            mw[i], mw[i+1], mw[i+2],  // in(w,s,b)
            mw[i+3], mw[i+4],          // conv(w,b)
            mw[i+5], mw[i+6], mw[i+7], mw[i+8],  // a_log, d, dt_bias, norm_w
            mw[i+9], mw[i+10], mw[i+11] // out(w,s,b)
        });
    }
    // Parse MoE weights: 14 per layer
    for (size_t i = 0; i + 13 < ew.size(); i += 14) {
        model->moe_layers.push_back({
            ew[i], ew[i+1],
            ew[i+2], ew[i+3], ew[i+4],
            ew[i+5], ew[i+6], ew[i+7],
            ew[i+8], ew[i+9], ew[i+10],
            ew[i+11], ew[i+12], ew[i+13]
        });
    }
    // Parse Attention weights: 12 per layer
    for (size_t i = 0; i + 11 < aw.size(); i += 12) {
        model->attn_layers.push_back({
            aw[i], aw[i+1], aw[i+2],
            aw[i+3], aw[i+4], aw[i+5],
            aw[i+6], aw[i+7], aw[i+8],
            aw[i+9], aw[i+10], aw[i+11]
        });
    }

    model->norm_eps = norm_eps; model->gs = gs; model->bits = bits;
    model->m_inter = m_inter; model->m_cdim = m_cdim; model->m_ck = m_ck;
    model->m_heads = m_heads; model->m_hdim = m_hdim; model->m_groups = m_groups;
    model->m_state = m_state; model->m_ts_min = m_ts_min; model->m_ts_max = m_ts_max;
    model->m_neps = m_neps;
    model->moe_tk = moe_tk; model->moe_sc = moe_sc; model->moe_norm = moe_norm;
    model->a_heads = a_heads; model->a_kvh = a_kvh; model->a_hdim = a_hdim;
    model->a_rope = a_rope; model->a_scale = a_scale;

    uint64_t handle = g_next_handle++;
    g_models[handle] = std::move(model);
    return handle;
}

void nemotron_free_model(uint64_t handle) {
    g_models.erase(handle);
}

void nemotron_decode_step(
    uint64_t handle,
    const MlxArray& input_ids,
    rust::Slice<const MlxArray* const> mamba_conv_in,
    rust::Slice<const MlxArray* const> mamba_ssm_in,
    rust::Slice<const MlxArray* const> attn_kv_keys,
    rust::Slice<const MlxArray* const> attn_kv_values,
    rust::Slice<const int32_t> attn_kv_offsets,
    std::unique_ptr<MlxArray>& logits,
    rust::Slice<std::unique_ptr<MlxArray>> mamba_conv_out,
    rust::Slice<std::unique_ptr<MlxArray>> mamba_ssm_out
) {
    using namespace mlx::core;
    auto& m = *g_models.at(handle);
    int num_layers = (int)m.block_types.size();

    // Embedding (quantized)
    auto flat_ids = reshape(input_ids.inner, {-1});
    auto w_idx = take(m.embed_w->inner, flat_ids, 0);
    auto s_idx = take(m.embed_s->inner, flat_ids, 0);
    auto b_idx = take(m.embed_b->inner, flat_ids, 0);
    auto h = dequantize(w_idx, s_idx, b_idx, m.gs, m.bits, "affine");
    auto id_shape = input_ids.inner.shape();
    h = reshape(h, {id_shape[0], id_shape[1], (int)h.shape().back()});

    int mamba_idx = 0, moe_idx = 0, attn_idx = 0;

    for (int i = 0; i < num_layers; ++i) {
        // RMSNorm
        auto normed = fast::rms_norm(h, m.norm_weights[i]->inner, m.norm_eps);

        int bt = m.block_types[i];
        array out = normed; // placeholder

        if (bt == 0) { // Mamba
            auto& ml = m.mamba_layers[mamba_idx];
            MlxArray h_w{normed};
            MlxArray in_w{ml.in_w->inner}, in_s{ml.in_s->inner};
            MlxArray cw{ml.conv_w->inner}, al{ml.a_log->inner}, dd{ml.d->inner}, dtb{ml.dt_bias->inner};
            MlxArray nw{ml.norm_w->inner}, ow{ml.out_w->inner}, os{ml.out_s->inner};
            MlxArray cs{mamba_conv_in[mamba_idx]->inner}, ss{mamba_ssm_in[mamba_idx]->inner};

            std::unique_ptr<MlxArray> m_out, m_cs, m_ss;
            fused_mamba2_forward(
                h_w, in_w, in_s, ml.in_b, cw, ml.conv_b,
                al, dd, dtb, nw, ow, os, ml.out_b,
                cs, ss,
                m.m_inter, m.m_cdim, m.m_ck,
                m.m_heads, m.m_hdim, m.m_groups, m.m_state,
                m.m_ts_min, m.m_ts_max, m.m_neps,
                m.gs, m.bits,
                m_out, m_cs, m_ss
            );
            out = m_out->inner;
            mamba_conv_out[mamba_idx] = std::move(m_cs);
            mamba_ssm_out[mamba_idx] = std::move(m_ss);
            mamba_idx++;
        } else if (bt == 3) { // MoE
            auto& el = m.moe_layers[moe_idx];
            // Flatten to [batch*seq, hidden] for MoE (expects 2D input)
            auto normed_flat = reshape(normed, {-1, (int)normed.shape().back()});
            auto moe_result = fused_moe_forward(
                MlxArray{normed_flat},
                *el.gate_w, *el.corr_bias,
                *el.fc1_w, *el.fc1_s, *el.fc1_b,
                *el.fc2_w, *el.fc2_s, *el.fc2_b,
                el.su_w, el.su_s, el.su_b,
                el.sd_w, el.sd_s, el.sd_b,
                m.moe_tk, m.moe_sc, m.moe_norm,
                m.gs, m.bits
            );
            // Reshape back to [batch, seq, hidden]
            out = reshape(moe_result->inner, h.shape());
            moe_idx++;
        } else if (bt == 1) { // Attention
            auto& al = m.attn_layers[attn_idx];
            int batch = (int)h.shape()[0];
            int seq = (int)h.shape()[1];
            int hidden = (int)h.shape()[2];
            auto x_flat = reshape(normed, {batch * seq, hidden});

            // QKV projections
            std::optional<array> qb = al.q_b ? std::optional(al.q_b->inner) : std::nullopt;
            std::optional<array> kb = al.k_b ? std::optional(al.k_b->inner) : std::nullopt;
            std::optional<array> vb = al.v_b ? std::optional(al.v_b->inner) : std::nullopt;
            std::optional<array> ob = al.o_b ? std::optional(al.o_b->inner) : std::nullopt;
            auto q = quantized_matmul(x_flat, al.q_w->inner, al.q_s->inner, qb, true, m.gs, m.bits);
            auto k = quantized_matmul(x_flat, al.k_w->inner, al.k_s->inner, kb, true, m.gs, m.bits);
            auto v = quantized_matmul(x_flat, al.v_w->inner, al.v_s->inner, vb, true, m.gs, m.bits);

            q = reshape(q, {batch, seq, m.a_heads, m.a_hdim});
            k = reshape(k, {batch, seq, m.a_kvh, m.a_hdim});
            v = reshape(v, {batch, seq, m.a_kvh, m.a_hdim});
            q = transpose(q, {0, 2, 1, 3});
            k = transpose(k, {0, 2, 1, 3});
            v = transpose(v, {0, 2, 1, 3});

            int offset = attn_kv_offsets[attn_idx];
            q = fast::rope(q, m.a_hdim, false, m.a_rope, 1.0f, offset);
            k = fast::rope(k, m.a_hdim, false, m.a_rope, 1.0f, offset);

            // Simple KV cache: concatenate with existing
            if (attn_kv_keys[attn_idx]) {
                k = concatenate(std::vector<array>{attn_kv_keys[attn_idx]->inner, k}, 2);
                v = concatenate(std::vector<array>{attn_kv_values[attn_idx]->inner, v}, 2);
            }

            // GQA repeat
            int n_rep = m.a_heads / m.a_kvh;
            if (n_rep > 1) {
                auto ks = k.shape();
                k = reshape(k, {ks[0], ks[1], 1, ks[2], ks[3]});
                k = broadcast_to(k, {ks[0], ks[1], n_rep, ks[2], ks[3]});
                k = reshape(k, {ks[0], ks[1] * n_rep, ks[2], ks[3]});
                auto vs = v.shape();
                v = reshape(v, {vs[0], vs[1], 1, vs[2], vs[3]});
                v = broadcast_to(v, {vs[0], vs[1], n_rep, vs[2], vs[3]});
                v = reshape(v, {vs[0], vs[1] * n_rep, vs[2], vs[3]});
            }

            auto attn = fast::scaled_dot_product_attention(q, k, v, m.a_scale);
            attn = transpose(attn, {0, 2, 1, 3});
            attn = reshape(attn, {batch, seq, m.a_heads * m.a_hdim});
            auto o_flat = reshape(attn, {batch * seq, m.a_heads * m.a_hdim});
            out = reshape(quantized_matmul(o_flat, al.o_w->inner, al.o_s->inner, ob, true, m.gs, m.bits),
                         {batch, seq, hidden});
            attn_idx++;
        }

        // Residual
        h = add(h, out);
    }

    // Final norm + lm_head
    h = fast::rms_norm(h, m.final_norm_w->inner, m.norm_eps);
    auto h_flat = reshape(h, {(int)h.shape()[0], (int)h.shape().back()});
    std::optional<array> lm_b = m.lm_head_b ? std::optional(m.lm_head_b->inner) : std::nullopt;
    auto lm_out = quantized_matmul(h_flat, m.lm_head_w->inner, m.lm_head_s->inner, lm_b, true, m.gs, m.bits);
    lm_out = reshape(lm_out, {(int)h.shape()[0], 1, (int)lm_out.shape().back()});

    logits = std::make_unique<MlxArray>(std::move(lm_out));
}
}  // namespace mlx_cxx
