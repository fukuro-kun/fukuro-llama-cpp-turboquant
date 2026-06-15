#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// Minimal DiffusionGemma CLI
// Usage: llama-diffusion-cli -m <model.gguf> -p "prompt" [-ngl N]

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m <model.gguf> -p <prompt> [-ngl N] [-n <max_tokens>]\n", prog);
    fprintf(stderr, "  -m    Model path (required)\n");
    fprintf(stderr, "  -p    Prompt text (required)\n");
    fprintf(stderr, "  -ngl  Number of GPU layers (default: 0)\n");
    fprintf(stderr, "  -n    Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  --temp  Sampling temperature (default: 0.8)\n");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt = nullptr;
    int n_gpu_layers = 0;
    int n_predict = 256;
    float temp = 0.8f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path || !prompt) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "DiffusionGemma CLI\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Prompt: %s\n", prompt);
    fprintf(stderr, "GPU layers: %d\n", n_gpu_layers);
    fprintf(stderr, "Max tokens: %d\n", n_predict);
    fprintf(stderr, "Temperature: %.2f\n\n", temp);

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    fprintf(stderr, "Loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    fprintf(stderr, "Creating context...\n");
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize prompt
    std::vector<llama_token> prompt_tokens(4096);
    int n_prompt_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                         prompt_tokens.data(), prompt_tokens.size(), true, false);
    if (n_prompt_tokens < 0) {
        fprintf(stderr, "Error: Failed to tokenize prompt\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    prompt_tokens.resize(n_prompt_tokens);

    fprintf(stderr, "Prompt tokenized to %d tokens\n", n_prompt_tokens);

    // ============================================
    // Phase 1: Prefill (encode prompt)
    // ============================================
    fprintf(stderr, "\n[Phase 1: Prefill]\n");

    // Set phase to PREFILL
    llama_diffusion_set_phase(model, 1, n_prompt_tokens);

    llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
    for (int i = 0; i < n_prompt_tokens; i++) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    batch.n_tokens = n_prompt_tokens;
    batch.logits[n_prompt_tokens - 1] = 1;  // logits for last token

    auto t_start = std::chrono::high_resolution_clock::now();
    int ret = llama_decode(ctx, batch);
    auto t_end = std::chrono::high_resolution_clock::now();

    llama_batch_free(batch);

    if (ret != 0) {
        fprintf(stderr, "Error: Prefill decode failed with code %d\n", ret);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    float prefill_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    fprintf(stderr, "Prefill: %d tokens in %.2f ms (%.2f t/s)\n",
            n_prompt_tokens, prefill_ms, n_prompt_tokens / (prefill_ms / 1000.0f));

    // ============================================
    // Phase 2: Decode (generate tokens)
    // ============================================
    fprintf(stderr, "\n[Phase 2: Decode]\n");

    // Set phase to DECODE with prompt length
    llama_diffusion_set_phase(model, 2, n_prompt_tokens);

    std::string generated_text;
    std::vector<llama_token> all_tokens = prompt_tokens;

    for (int i = 0; i < n_predict && i < 256; i++) {  // Limit to 256 for safety
        int n_tokens = all_tokens.size();

        // Prepare batch with all tokens so far
        llama_batch gen_batch = llama_batch_init(n_tokens, 0, 1);
        for (int j = 0; j < n_tokens; j++) {
            gen_batch.token[j] = all_tokens[j];
            gen_batch.pos[j] = j;
            gen_batch.n_seq_id[j] = 1;
            gen_batch.seq_id[j][0] = 0;
            gen_batch.logits[j] = 0;
        }
        gen_batch.n_tokens = n_tokens;
        gen_batch.logits[n_tokens - 1] = 1;

        t_start = std::chrono::high_resolution_clock::now();
        ret = llama_decode(ctx, gen_batch);
        t_end = std::chrono::high_resolution_clock::now();

        llama_batch_free(gen_batch);

        if (ret != 0) {
            fprintf(stderr, "Error: Decode step %d failed with code %d\n", i, ret);
            break;
        }

        float step_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        // Greedy decode
        float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
        llama_token next_token = 0;
        float max_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = v;
            }
        }

        // Check for EOS
        if (next_token == llama_vocab_eos(vocab) ||
            next_token == llama_vocab_eot(vocab)) {
            fprintf(stderr, "EOS token reached at step %d\n", i);
            break;
        }

        // Decode token to text
        char buf[256];
        int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            buf[n] = '\0';
            generated_text += buf;
            printf("%s", buf);
            fflush(stdout);
        }

        fprintf(stderr, "  Step %d: token=%d, logit=%.2f, time=%.2f ms\n",
                i + 1, next_token, max_logit, step_ms);

        all_tokens.push_back(next_token);
    }

    printf("\n\n");
    fprintf(stderr, "\nGeneration complete.\n");
    fprintf(stderr, "Generated %zu characters.\n", generated_text.length());

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
