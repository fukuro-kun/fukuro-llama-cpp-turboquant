/**
 * expert-overlap: Measure cross-layer expert selection overlap for MoE models.
 *
 * Captures ffn_moe_topk-{il} tensors via eval callback and computes:
 *   overlap(N, N+1) = |experts_N ∩ experts_N+1| / |experts_N|
 *
 * This validates the Fate (arXiv:2502.12224) assumption that adjacent layers
 * select similar experts (>83% cosine similarity in gate inputs → high expert overlap).
 *
 * Usage:
 *   llama-expert-overlap -m model.gguf -p "prompt" -n 128 -ngl 99 [-c 4096]
 */

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

struct OverlapCollector {
    std::map<int, std::vector<std::set<int32_t>>> layer_experts; // layer → per-token expert sets
    std::mutex mtx;
    int64_t n_expert_used = 0;

    static std::string clean_name(const char * raw) {
        const char * p = strchr(raw, '#');
        if (p) {
            ++p;
            const char * q = strchr(p, '#');
            return q ? std::string(p, q - p) : std::string(p);
        }
        return raw;
    }

    bool wants(struct ggml_tensor * t) {
        if (!t->name[0]) return false;
        const std::string n = clean_name(t->name);
        return n.compare(0, 13, "ffn_moe_topk-") == 0;
    }

    bool on_tensor(struct ggml_tensor * t) {
        const std::string n = clean_name(t->name);
        int il = -1;
        if (sscanf(n.c_str(), "ffn_moe_topk-%d", &il) != 1) return false;

        const int64_t n_eu  = t->ne[0];  // n_expert_used
        const int64_t n_tok = t->ne[1];  // n_tokens
        n_expert_used = n_eu;

        std::vector<std::set<int32_t>> token_experts(n_tok);
        const int32_t * data = (const int32_t *) t->data;
        for (int64_t t_idx = 0; t_idx < n_tok; ++t_idx) {
            for (int64_t k = 0; k < n_eu; ++k) {
                int32_t eid = data[k + t_idx * n_eu];
                if (eid >= 0) {
                    token_experts[t_idx].insert(eid);
                }
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        auto & existing = layer_experts[il];
        for (auto & te : token_experts) {
            existing.push_back(std::move(te));
        }
        return true;
    }

    void report() {
        printf("\n=== Cross-Layer Expert Overlap Report ===\n\n");
        printf("n_expert_used: %lld\n\n", (long long) n_expert_used);

        // Per-layer expert set sizes
        printf("Layer | Tokens | Avg Experts/Token\n");
        printf("------|--------|------------------\n");
        for (auto & [il, tokens] : layer_experts) {
            double avg = 0;
            for (auto & s : tokens) avg += s.size();
            avg /= tokens.empty() ? 1 : tokens.size();
            printf("%5d | %6zu | %16.1f\n", il, tokens.size(), avg);
        }

        // Adjacent layer overlap
        printf("\nAdjacent Layer Expert Overlap (Fate assumption: >80%%)\n");
        printf("Layer Pair | Avg Overlap %% | Min %% | Max %% | Exact Match %%\n");
        printf("-----------|--------------|-------|-------|---------------\n");

        std::vector<int> layer_ids;
        for (auto & [il, _] : layer_experts) layer_ids.push_back(il);
        std::sort(layer_ids.begin(), layer_ids.end());

        double total_avg = 0;
        int pair_count = 0;

        for (size_t i = 0; i + 1 < layer_ids.size(); ++i) {
            int la = layer_ids[i];
            int lb = layer_ids[i + 1];
            if (lb != la + 1) continue;

            auto & tokens_a = layer_experts[la];
            auto & tokens_b = layer_experts[lb];
            size_t n = std::min(tokens_a.size(), tokens_b.size());
            if (n == 0) continue;

            double sum_overlap = 0;
            double min_overlap = 100;
            double max_overlap = 0;
            int exact_match = 0;

            for (size_t t = 0; t < n; ++t) {
                std::set<int32_t> intersection;
                std::set_intersection(
                    tokens_a[t].begin(), tokens_a[t].end(),
                    tokens_b[t].begin(), tokens_b[t].end(),
                    std::inserter(intersection, intersection.begin()));

                double overlap = tokens_a[t].empty() ? 0 :
                    100.0 * intersection.size() / tokens_a[t].size();
                sum_overlap += overlap;
                min_overlap = std::min(min_overlap, overlap);
                max_overlap = std::max(max_overlap, overlap);
                if (intersection.size() == tokens_a[t].size()) exact_match++;
            }

            double avg_overlap = sum_overlap / n;
            double exact_pct = 100.0 * exact_match / n;
            printf("%4d-%-4d | %12.1f | %5.1f | %5.1f | %13.1f\n",
                   la, lb, avg_overlap, min_overlap, max_overlap, exact_pct);

            total_avg += avg_overlap;
            pair_count++;
        }

        if (pair_count > 0) {
            printf("\nOverall average overlap: %.1f%%\n", total_avg / pair_count);
            printf("Fate threshold: >80%% → %s\n",
                   (total_avg / pair_count) > 80 ? "PASS ✅" : "FAIL ❌");
        }

        // Multi-step overlap
        printf("\nMulti-Step Expert Overlap (Layer N vs N+k)\n");
        printf("Step k | Avg Overlap %%\n");
        printf("-------|--------------\n");
        for (int step = 1; step <= 3; ++step) {
            double sum = 0;
            int cnt = 0;
            for (size_t i = 0; i + step < layer_ids.size(); ++i) {
                int la = layer_ids[i];
                int lb = layer_ids[i + step];
                if (lb != la + step) continue;

                auto & tokens_a = layer_experts[la];
                auto & tokens_b = layer_experts[lb];
                size_t n = std::min(tokens_a.size(), tokens_b.size());
                if (n == 0) continue;

                double s = 0;
                for (size_t t = 0; t < n; ++t) {
                    std::set<int32_t> inter;
                    std::set_intersection(
                        tokens_a[t].begin(), tokens_a[t].end(),
                        tokens_b[t].begin(), tokens_b[t].end(),
                        std::inserter(inter, inter.begin()));
                    s += tokens_a[t].empty() ? 0 :
                        100.0 * inter.size() / tokens_a[t].size();
                }
                sum += s / n;
                cnt++;
            }
            if (cnt > 0) {
                printf("  %d    | %12.1f\n", step, sum / cnt);
            }
        }
        printf("\n");
    }
};

static OverlapCollector g_collector;

static bool expert_eval_callback(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return g_collector.wants(t);
    }
    return g_collector.on_tensor(t);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog. "
                         "In a world of artificial intelligence, mixture of experts models "
                         "have become increasingly important for scaling language models. "
                         "The key challenge is efficiently routing tokens to the right experts.";
    int n_gpu_layers = 99;
    int n_threads = 8;
    int n_predict = 128;
    int ctx_size = 4096;
    enum ggml_type kv_type_k = GGML_TYPE_F16;
    enum ggml_type kv_type_v = GGML_TYPE_F16;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i]); exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model")        model_path = next();
        else if (a == "-p" || a == "--prompt")  prompt = next();
        else if (a == "-n" || a == "--n-predict") n_predict = std::stoi(next());
        else if (a == "-ngl")                   n_gpu_layers = std::stoi(next());
        else if (a == "-t" || a == "--threads") n_threads = std::stoi(next());
        else if (a == "-c" || a == "--ctx-size") ctx_size = std::stoi(next());
        else if (a == "--type-k")               kv_type_k = (enum ggml_type) atoi(next().c_str());
        else if (a == "--type-v")               kv_type_v = (enum ggml_type) atoi(next().c_str());
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s -m model.gguf [options]\n"
                   "  -m, --model PATH      Model file\n"
                   "  -p, --prompt TEXT     Prompt text (default: built-in)\n"
                   "  -n, --n-predict N     Number of tokens to generate (default: 128)\n"
                   "  -ngl N                GPU layers (default: 99)\n"
                   "  -t, --threads N       CPU threads (default: 8)\n"
                   "  -c, --ctx-size N      Context size (default: 4096)\n"
                   "  --type-k TYPE         KV cache K type (default: f16)\n"
                   "  --type-v TYPE         KV cache V type (default: f16)\n\n", argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m model.gguf [options]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { LOG_ERR("failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = ctx_size;
    cparams.n_batch           = ctx_size;
    cparams.n_ubatch          = std::min(ctx_size, 512);
    cparams.n_threads         = n_threads;
    cparams.type_k            = kv_type_k;
    cparams.type_v            = kv_type_v;
    cparams.cb_eval           = expert_eval_callback;
    cparams.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { LOG_ERR("failed to create context\n"); return 1; }

    fprintf(stderr, "DEBUG: context created, tokenizing...\n");

    // Tokenize prompt
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    if ((int) tokens.size() > 256) tokens.resize(256);

    fprintf(stderr, "DEBUG: tokenized %zu tokens, decoding...\n", tokens.size());

    printf("Processing %zu prompt tokens...\n", tokens.size());

    // Process prompt in chunks to avoid memory issues
    int chunk_size = 128;
    for (int offset = 0; offset < (int) tokens.size(); offset += chunk_size) {
        int n = std::min(chunk_size, (int) tokens.size() - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, n);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("failed to decode prompt chunk at offset %d\n", offset);
            return 1;
        }
    }

    // Generate tokens one by one (greedy: pick argmax from logits)
    printf("Generating %d tokens (greedy)...\n", n_predict);
    for (int i = 0; i < n_predict; ++i) {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            LOG_ERR("failed to get logits for token %d\n", i);
            break;
        }
        llama_vocab * vocab = llama_get_model(ctx) ? (llama_vocab *) llama_model_get_vocab(llama_get_model(ctx)) : nullptr;
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        llama_token new_token = 0;
        float max_logit = -1e30f;
        for (int32_t t = 0; t < n_vocab; ++t) {
            if (logits[t] > max_logit) {
                max_logit = logits[t];
                new_token = t;
            }
        }

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("failed to decode token %d\n", i);
            break;
        }
    }

    // Report
    g_collector.report();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
