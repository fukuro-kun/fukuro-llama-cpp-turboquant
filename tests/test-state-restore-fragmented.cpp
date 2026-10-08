// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <cstring>

static bool test_multistream_restore(llama_model * model, const common_params & params) {
    std::vector<ggml_type> types_k = { GGML_TYPE_F16 };
    if (params.cache_type_k != GGML_TYPE_F16) {
        types_k.push_back(params.cache_type_k);
    }
    for (const auto type_k : types_k) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_ctx = 768;
        cparams.n_batch = 32;
        cparams.n_ubatch = 32;
        cparams.n_seq_max = 3;
        cparams.kv_unified = false;
        cparams.type_k = type_k;
        cparams.type_v = GGML_TYPE_F16;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "%s: failed to create multistream context (%s K)\n", __func__, ggml_type_name(type_k));
            return false;
        }
        llama_batch batch = llama_batch_init(32, 0, 1);
        bool ok = true;
        for (int s = 0; s < 3 && ok; ++s) {
            common_batch_clear(batch);
            for (int pos = 0; pos < 8 + s; ++pos) {
                common_batch_add(batch, 1 + s, pos, { s }, pos == 7 + s);
            }
            ok = llama_decode(ctx, batch) == 0;
        }
        if (ok) {
            std::vector<std::vector<uint8_t>> seq_states(3);
            for (int s = 0; s < 3 && ok; ++s) {
                seq_states[s].resize(llama_state_seq_get_size(ctx, s));
                ok = !seq_states[s].empty() && llama_memory_seq_pos_max(llama_get_memory(ctx), s) == 7 + s &&
                     llama_state_seq_get_data(ctx, seq_states[s].data(), seq_states[s].size(), s) == seq_states[s].size();
            }
            std::vector<uint8_t> state(llama_state_get_size(ctx));
            ok = ok && llama_state_get_data(ctx, state.data(), state.size()) == state.size();
            llama_memory_clear(llama_get_memory(ctx), true);
            ok = ok && llama_state_set_data(ctx, state.data(), state.size()) == state.size();
            for (int s = 0; s < 3 && ok; ++s) {
                std::vector<uint8_t> restored(llama_state_seq_get_size(ctx, s));
                ok = llama_state_seq_get_data(ctx, restored.data(), restored.size(), s) == restored.size() &&
                     restored == seq_states[s];
                if (!ok) {
                    fprintf(stderr, "%s: stream %d (%s) differs after full restore\n", __func__, s, ggml_type_name(type_k));
                }
            }
        }
        llama_batch_free(batch);
        llama_free(ctx);
        if (!ok) {
            return false;
        }
        fprintf(stderr, "%s: full restore preserves all three streams (%s K)\n", __func__, ggml_type_name(type_k));
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // init

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    if (!test_multistream_restore(model, params)) {
        return 1;
    }

    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    llama_batch batch = llama_batch_init(params.n_parallel*tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Save state of seq 1
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);

    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)tokens.size(), {1}, true);

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return 0;
}
