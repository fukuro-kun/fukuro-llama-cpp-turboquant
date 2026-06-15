#include "models.h"
#include "gemma4-common.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// MONOLITHIC PORT NOTE: Ported from upstream's llama_model_diffusion_gemma class
// to our fork's monolithic llama_model structure.

// Region-aware additive mask for the unified [prompt | canvas] forward.
class llm_graph_input_attn_diffusion : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion(const llama_hparams & hparams, const llama_cparams & cparams,
                                   int64_t n_prompt) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt) {}
    ~llm_graph_input_attn_diffusion() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;
        const int64_t P        = n_prompt;

        const auto fill = [&](auto * data, bool swa) {
            using T = std::remove_reference_t<decltype(*data)>;
            std::fill(data, data + n_tokens * n_tokens, (T)(-INFINITY));
            for (int64_t q = 0; q < n_tokens; ++q) {
                const bool q_is_canvas = q >= P;
                const uint64_t row = q * n_tokens;
                const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;
                for (int64_t k = 0; k < n_tokens; ++k) {
                    const bool k_is_canvas = k >= P;
                    bool allow;
                    if (q_is_canvas) {
                        if (swa) {
                            allow = k_is_canvas || (k >= canvas_prompt_lo);
                        } else {
                            allow = true;
                        }
                    } else {
                        allow = (!k_is_canvas) && (k <= q);
                    }
                    if (allow && swa && !q_is_canvas &&
                        llama_hparams::is_masked_swa(hparams.n_swa, hparams.swa_type, k, q)) {
                        allow = false;
                    }
                    if (allow) {
                        data[row + k] = (T)(0.0f);
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill((ggml_fp16_t *) self_kq_mask->data, false);
        } else {
            fill((float *) self_kq_mask->data, false);
        }

        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            if (self_kq_mask_swa->type == GGML_TYPE_F16) {
                fill((ggml_fp16_t *) self_kq_mask_swa->data, true);
            } else {
                fill((float *) self_kq_mask_swa->data, true);
            }
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }
    int64_t n_prompt;
};

// Self-conditioning input.
class llm_graph_input_sc : public llm_graph_input_i {
public:
    llm_graph_input_sc(const float * src, int64_t n_vocab, int64_t C) :
        src(src), n_vocab(n_vocab), C(C) {}
    ~llm_graph_input_sc() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        if (sc_logits && src) {
            GGML_ASSERT(ggml_nelements(sc_logits) == n_vocab * C);
            ggml_backend_tensor_set(sc_logits, src, 0, (size_t) n_vocab * C * sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    ggml_tensor * sc_logits = nullptr;
    const float * src;
    int64_t       n_vocab;
    int64_t       C;
};

// Decode-phase mask.
class llm_graph_input_attn_diffusion_decode : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion_decode(const llama_hparams & hparams, const llama_cparams & cparams,
                                          int64_t n_prompt, int64_t n_canvas) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt), n_canvas(n_canvas) {}
    ~llm_graph_input_attn_diffusion_decode() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        const int64_t P    = n_prompt;
        const int64_t C    = n_canvas;
        const int64_t n_kv = P + C;
        const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;

        const auto fill = [&](auto * data, bool swa) {
            using T = std::remove_reference_t<decltype(*data)>;
            std::fill(data, data + n_kv * C, (T)(-INFINITY));
            for (int64_t q = 0; q < C; ++q) {
                const uint64_t row = q * n_kv;
                for (int64_t k = 0; k < n_kv; ++k) {
                    bool allow;
                    if (k < P) {
                        allow = swa ? (k >= canvas_prompt_lo) : true;
                    } else {
                        allow = true;
                    }
                    if (allow) {
                        data[row + k] = (T)(0.0f);
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill((ggml_fp16_t *) self_kq_mask->data, false);
        } else {
            fill((float *) self_kq_mask->data, false);
        }
        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            if (self_kq_mask_swa->type == GGML_TYPE_F16) {
                fill((ggml_fp16_t *) self_kq_mask_swa->data, true);
            } else {
                fill((float *) self_kq_mask_swa->data, true);
            }
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }
    int64_t n_prompt;
    int64_t n_canvas;
};

// Build the SC soft-embedding weight once.
static void dg_ensure_sc_embT(const llama_model & m) {
    if (m.sc_embT != nullptr) {
        return;
    }
    ggml_tensor * src = m.tok_embd;
    GGML_ASSERT(src != nullptr);
    const int64_t n_embd  = src->ne[0];
    const int64_t n_vocab = src->ne[1];

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    const_cast<llama_model &>(m).sc_embT_ctx = ggml_init(ip);
    GGML_ASSERT(m.sc_embT_ctx != nullptr);
    const_cast<llama_model &>(m).sc_embT = ggml_new_tensor_2d(m.sc_embT_ctx, GGML_TYPE_F16, n_vocab, n_embd);
    ggml_set_name(m.sc_embT, "sc_embT");

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
    const_cast<llama_model &>(m).sc_embT_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.sc_embT_ctx, buft);
    GGML_ASSERT(m.sc_embT_buf != nullptr);
    ggml_backend_buffer_set_usage(m.sc_embT_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const ggml_type st       = src->type;
    const size_t    row_size = ggml_row_size(st, n_embd);
    std::vector<char> host_src((size_t) row_size * n_vocab);
    ggml_backend_tensor_get(src, host_src.data(), 0, host_src.size());

    std::vector<ggml_fp16_t> dstT((size_t) n_vocab * n_embd);
    const ggml_type_traits * tr = ggml_get_type_traits(st);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));
    auto worker = [&](int64_t v0, int64_t v1) {
        std::vector<float> tmp(n_embd);
        for (int64_t v = v0; v < v1; ++v) {
            const char * row = host_src.data() + (size_t) v * row_size;
            if (st == GGML_TYPE_F32) {
                std::memcpy(tmp.data(), row, (size_t) n_embd * sizeof(float));
            } else {
                tr->to_float(row, tmp.data(), n_embd);
            }
            for (int64_t e = 0; e < n_embd; ++e) {
                dstT[(size_t) e * n_vocab + v] = ggml_fp32_to_fp16(tmp[e]);
            }
        }
    };
    std::vector<std::thread> pool;
    const int64_t chunk = (n_vocab + nth - 1) / nth;
    for (unsigned t = 0; t < nth; ++t) {
        const int64_t v0 = (int64_t) t * chunk;
        const int64_t v1 = std::min(v0 + chunk, n_vocab);
        if (v0 < v1) {
            pool.emplace_back(worker, v0, v1);
        }
    }
    for (auto & th : pool) {
        th.join();
    }

    ggml_backend_tensor_set(m.sc_embT, dstT.data(), 0, dstT.size() * sizeof(ggml_fp16_t));
}

// Lazily (re)allocate the device prev-step canvas-logits buffer.
static void dg_ensure_sc_dev(const llama_model & m, int64_t C) {
    const int64_t n_vocab = m.tok_embd->ne[1];
    if (m.sc_dev != nullptr && m.sc_dev_C >= C) {
        return;
    }
    if (m.sc_dev_buf) { ggml_backend_buffer_free(const_cast<ggml_backend_buffer_t>(m.sc_dev_buf)); const_cast<llama_model &>(m).sc_dev_buf = nullptr; }
    if (m.sc_dev_ctx) { ggml_free(const_cast<ggml_context *>(m.sc_dev_ctx)); const_cast<llama_model &>(m).sc_dev_ctx = nullptr; }

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    const_cast<llama_model &>(m).sc_dev_ctx = ggml_init(ip);
    GGML_ASSERT(m.sc_dev_ctx != nullptr);
    const_cast<llama_model &>(m).sc_dev = ggml_new_tensor_2d(m.sc_dev_ctx, GGML_TYPE_F32, n_vocab, C);
    ggml_set_name(m.sc_dev, "sc_dev");

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
    const_cast<llama_model &>(m).sc_dev_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.sc_dev_ctx, buft);
    GGML_ASSERT(m.sc_dev_buf != nullptr);
    ggml_backend_buffer_clear(m.sc_dev_buf, 0);
    const_cast<llama_model &>(m).sc_dev_C = C;
}

// Lazily (re)allocate the device-resident F32 prompt-KV store.
static void dg_ensure_pkv_store(const llama_model & m, int64_t P) {
    if (m.pkv_buf != nullptr && m.pkv_cap >= P) {
        return;
    }
    if (m.pkv_buf) { ggml_backend_buffer_free(const_cast<ggml_backend_buffer_t>(m.pkv_buf)); const_cast<llama_model &>(m).pkv_buf = nullptr; }
    if (m.pkv_ctx) { ggml_free(const_cast<ggml_context *>(m.pkv_ctx)); const_cast<llama_model &>(m).pkv_ctx = nullptr; }
    const_cast<llama_model &>(m).pkv_k.clear();
    const_cast<llama_model &>(m).pkv_v.clear();

    const int     n_layer = (int) m.hparams.n_layer;
    const int64_t cap     = P;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (2 * n_layer + 4),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const_cast<llama_model &>(m).pkv_ctx = ggml_init(ip);
    GGML_ASSERT(m.pkv_ctx != nullptr);
    const_cast<llama_model &>(m).pkv_k.resize(n_layer);
    const_cast<llama_model &>(m).pkv_v.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const int64_t hd  = m.hparams.n_embd_head_k(il);
        const int64_t nkv = m.hparams.n_head_kv(il);
        const_cast<llama_model &>(m).pkv_k[il] = ggml_new_tensor_3d(m.pkv_ctx, GGML_TYPE_F32, hd, nkv, cap);
        const_cast<llama_model &>(m).pkv_v[il] = ggml_new_tensor_3d(m.pkv_ctx, GGML_TYPE_F32, hd, nkv, cap);
        ggml_format_name(m.pkv_k[il], "pkv_k_l%d", il);
        ggml_format_name(m.pkv_v[il], "pkv_v_l%d", il);
    }

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
    const_cast<llama_model &>(m).pkv_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.pkv_ctx, buft);
    GGML_ASSERT(m.pkv_buf != nullptr);
    const_cast<llama_model &>(m).pkv_cap = cap;
}

// Graph builder
llm_build_diffusion_gemma::llm_build_diffusion_gemma(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {

    ggml_tensor * cur;
    ggml_tensor * inpL;

    const auto & hparams = model.hparams;
    const auto   phase   = model.pkv_phase;
    const bool is_prefill = (phase == llama_model::PKV_PREFILL);
    const bool is_decode  = (phase == llama_model::PKV_DECODE);

    const uint32_t canvas_length = model.canvas_length;
    int64_t P, C;
    if (is_decode) {
        P = model.pkv_P;
        C = n_tokens;
    } else if (is_prefill) {
        P = n_tokens;
        C = 0;
    } else {
        P = (canvas_length > 0 && n_tokens > canvas_length) ? (n_tokens - canvas_length) : 0;
        C = n_tokens - P;
    }

    if (is_prefill || is_decode) {
        const int64_t need = is_prefill ? n_tokens : P;
        GGML_ASSERT(!model.pkv_k.empty() && !model.pkv_v.empty() && model.pkv_cap >= need &&
                    "DiffusionGemma prompt-KV store not allocated/sized for this phase");
    }

    auto dg_canvas_embed = [&](ggml_tensor * canvas) -> ggml_tensor * {
        if (model.sc_enabled) {
            const int64_t Cc      = canvas->ne[1];
            const int64_t n_vocab = model.tok_embd->ne[1];
            canvas = ggml_cont(ctx0, canvas);

            ggml_tensor * sc_logits;
            if (model.sc_device_resident) {
                dg_ensure_sc_dev(model, Cc);
                sc_logits = model.sc_dev;
            } else {
                auto inp_sc = std::make_unique<llm_graph_input_sc>(model.sc_logits_ptr, n_vocab, Cc);
                sc_logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, Cc);
                ggml_set_name(sc_logits, "sc_logits");
                ggml_set_input(sc_logits);
                inp_sc->sc_logits = sc_logits;
                res->add_input(std::move(inp_sc));
            }

            ggml_tensor * sc_emb;
            {
                dg_ensure_sc_embT(model);
                ggml_tensor * sc_probs = ggml_soft_max(ctx0, ggml_scale(ctx0, sc_logits, model.sc_temp_inv));
                sc_emb = ggml_mul_mat(ctx0, model.sc_embT, sc_probs);
            }

            ggml_tensor * sc_gated = ggml_mul_mat(ctx0, model.sc_gate, ggml_silu(ctx0, ggml_mul_mat(ctx0, model.sc_pre_norm, canvas)));
            ggml_tensor * sc_upped  = ggml_mul_mat(ctx0, model.sc_up, canvas);
            ggml_tensor * sc_proj   = ggml_mul_mat(ctx0, model.sc_down, ggml_mul(ctx0, sc_gated, sc_upped));
            canvas = ggml_add(ctx0, canvas, ggml_scale(ctx0, sc_proj, model.sc_use));
            canvas = ggml_add(ctx0, canvas, ggml_scale(ctx0, sc_emb, model.sc_use));
        }
        return ggml_rms_norm(ctx0, canvas, hparams.f_norm_rms_eps);
    };

    // Build input embeddings; res->t_inp_tokens is set by build_inp_embd
    ggml_tensor * inp_embd = build_inp_embd(model.tok_embd);

    if (C > 0 && n_tokens > C) {
        // prompt = first P tokens, canvas = last C tokens
        ggml_tensor * prompt = ggml_view_2d(ctx0, inp_embd, n_embd, P, inp_embd->nb[1], 0);
        prompt = ggml_scale(ctx0, ggml_cont(ctx0, prompt), sqrtf((float) hparams.n_embd));
        ggml_tensor * canvas = ggml_view_2d(ctx0, inp_embd, n_embd, C, inp_embd->nb[1], P * inp_embd->nb[1]);
        canvas = dg_canvas_embed(ggml_cont(ctx0, canvas));
        inpL = ggml_concat(ctx0, prompt, canvas, 1);
    } else {
        inpL = inp_embd;
    }
    cb(inpL, "inp_region", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    llm_graph_input_attn_no_cache * inp_attn = nullptr;
    if (is_decode) {
        const int64_t n_kv = P + C;
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion_decode>(hparams, cparams, P, C);
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv, C, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            uptr->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_kv, C, 1, 1);
            ggml_set_input(uptr->self_kq_mask_swa);
            uptr->self_kq_mask_swa_cnv = uptr->self_kq_mask_swa;
        }
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    } else {
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion>(hparams, cparams, P);
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            uptr->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
            ggml_set_input(uptr->self_kq_mask_swa);
            uptr->self_kq_mask_swa_cnv = uptr->self_kq_mask_swa;
        }
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));
        const int64_t n_head_kv = hparams.n_head_kv(il);

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = gemma4_build_q(*this, model, cur, inp_pos, il);
        llm_graph_qkv kv   = gemma4_build_kv(*this, model, cur, inp_pos, il);
        ggml_tensor * Kcur = kv.k;
        ggml_tensor * Vcur = kv.v;

        if (is_prefill) {
            ggml_tensor * sk = ggml_view_3d(ctx0, model.pkv_k[il], n_embd_head, n_head_kv, n_tokens,
                                            model.pkv_k[il]->nb[1], model.pkv_k[il]->nb[2], 0);
            ggml_tensor * sv = ggml_view_3d(ctx0, model.pkv_v[il], n_embd_head, n_head_kv, n_tokens,
                                            model.pkv_v[il]->nb[1], model.pkv_v[il]->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, sk));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, sv));
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr,
                             Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                             hparams.f_attention_scale, il);
        } else if (is_decode) {
            ggml_tensor * pk = ggml_view_3d(ctx0, model.pkv_k[il], n_embd_head, n_head_kv, P,
                                            model.pkv_k[il]->nb[1], model.pkv_k[il]->nb[2], 0);
            ggml_tensor * pv = ggml_view_3d(ctx0, model.pkv_v[il], n_embd_head, n_head_kv, P,
                                            model.pkv_v[il]->nb[1], model.pkv_v[il]->nb[2], 0);
            ggml_tensor * Kfull = ggml_concat(ctx0, pk, Kcur, 2);
            ggml_tensor * Vfull = ggml_concat(ctx0, pv, Vcur, 2);
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr,
                             Qcur, Kfull, Vfull, nullptr, nullptr, nullptr,
                             hparams.f_attention_scale, il);
        } else {
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr,
                             Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                             hparams.f_attention_scale, il);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = gemma4_build_ffn_moe(*this, model, attn_out, il);

        if (is_prefill) {
            cur = ggml_mul(ctx0, cur, model.layers[il].enc_out_scale);
        } else if (is_decode) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        } else if (P > 0 && P < n_tokens) {
            ggml_tensor * prompt = ggml_view_2d(ctx0, cur, n_embd, P, cur->nb[1], 0);
            ggml_tensor * canvas = ggml_view_2d(ctx0, cur, n_embd, C, cur->nb[1], P * cur->nb[1]);
            prompt = ggml_mul(ctx0, ggml_cont(ctx0, prompt), model.layers[il].enc_out_scale);
            canvas = ggml_mul(ctx0, ggml_cont(ctx0, canvas), model.layers[il].out_scale);
            cur = ggml_concat(ctx0, prompt, canvas, 1);
        } else {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        }
        cb(cur, "out_scaled", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (model.sc_enabled && model.sc_device_resident && C > 0) {
        dg_ensure_sc_dev(model, C);
        const int64_t n_out   = cur->ne[1];
        const int64_t n_vocab = cur->ne[0];
        GGML_ASSERT(n_out >= C && "device SC expects the canvas rows to be present in the output logits");
        ggml_tensor * canvas_logits = ggml_view_2d(ctx0, cur, n_vocab, C,
                                                   cur->nb[1], (n_out - C) * cur->nb[1]);
        ggml_tensor * sc_cpy = ggml_cpy(ctx0, canvas_logits, model.sc_dev);
        ggml_build_forward_expand(gf, sc_cpy);
    }
}

extern "C" {

// Public API: set per-request self-conditioning (no-op on other models).
void llama_diffusion_set_sc(struct llama_model * model, const float * sc_logits,
                            float use_sc, float temp_inv, bool enabled) {
    if (model->arch != LLM_ARCH_DIFFUSION_GEMMA) {
        return;
    }
    model->sc_logits_ptr = sc_logits;
    model->sc_use        = use_sc;
    model->sc_temp_inv   = temp_inv;
    model->sc_enabled    = enabled;
}

// Public API: opt into device-resident self-conditioning.
void llama_diffusion_set_device_sc(struct llama_model * model, bool enabled) {
    if (model->arch != LLM_ARCH_DIFFUSION_GEMMA) {
        return;
    }
    model->sc_device_resident = enabled;
}

// Stage-1 device sampling entry.
typedef bool (*dg_cuda_sample_fn)(struct ggml_tensor *, const float *, int *, float *, int *, int, float);

bool llama_diffusion_device_sample(const struct llama_model * model, const float * u, int * argmax,
                                   float * entropy, int * sampled, int n_tokens, float inv_temp) {
    if (model->arch != LLM_ARCH_DIFFUSION_GEMMA || model->sc_dev == nullptr || !u || !argmax || !entropy || !sampled || n_tokens <= 0) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        return false;
    }
    static dg_cuda_sample_fn fn =
        (dg_cuda_sample_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_diffusion_sample");
    if (!fn) {
        return false;
    }
    return fn(model->sc_dev, u, argmax, entropy, sampled, n_tokens, inv_temp);
}

// Public API: select the prompt-KV-caching phase for the next llama_decode.
void llama_diffusion_set_phase(struct llama_model * model, int phase, int32_t P) {
    if (model->arch != LLM_ARCH_DIFFUSION_GEMMA) {
        return;
    }
    switch (phase) {
        case 0: model->pkv_phase = llama_model::PKV_UNIFIED; break;
        case 1: model->pkv_phase = llama_model::PKV_PREFILL; break;
        case 2: model->pkv_phase = llama_model::PKV_DECODE;  break;
        default: model->pkv_phase = llama_model::PKV_UNIFIED; break;
    }
    model->pkv_P = P;
    if (model->pkv_phase == llama_model::PKV_PREFILL || model->pkv_phase == llama_model::PKV_DECODE) {
        dg_ensure_pkv_store(*model, P > 0 ? P : (int32_t)model->hparams.n_ctx_train);
    }
}

} // extern "C"
