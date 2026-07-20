#include "llama.h"
#include "chat.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <random>

// ================================================================================================
// DiffusionGemma CLI — Entropy-Bound Decoder
// Portiert aus PR #24423 (Unsloth), angepasst an monolithische Architektur.
//
// Der Entropy-Bound Decoder ist DER Kern-Algorithmus von DiffusionGemma:
//   1. Canvas wird mit zufaelligen Tokens initialisiert (Rauschen)
//   2. N Schritte: [prompt|canvas] -> bidirektionaler Forward
//   3. Pro Position: Argmax, Entropie, Multinomial-Sample
//   4. Positionen werden nach Entropie sortiert
//   5. Niedrigst-entropische Positionen werden AKZEPTIERT
//   6. Rest wird mit frischem Rauschen RENOISED
//   7. Output ist der Argmax-Canvas (nicht der gesampelte!)
// ================================================================================================

// ================================================================================================
// Parameter-Struktur
// ================================================================================================
struct diffusion_eb_params {
    int32_t max_denoising_steps  = 48;
    float   t_min                = 0.4f;   // Temperatur am letzten Step
    float   t_max                = 0.8f;   // Temperatur am ersten Step
    float   entropy_bound        = 0.1f;   // Akzeptiere niedrigste Entropie innerhalb Bound
    int32_t stability_threshold  = 1;      // Steps Argmax-Canvas stabil fuer Stopp
    float   confidence_threshold = 0.005f; // Mittlere Entropie unter Schwelle
    int32_t seed                 = 0;
    int32_t max_length           = 0;      // n_input + canvas_length
    bool    kv_cache             = false;  // Prefix-KV-Cache (PREFILL/DECODE)
    bool    gpu_sampling         = false;  // Device-resident SC
    bool    gpu_sample_reduce    = false;  // Stage-1 on-device sampling
};

// ================================================================================================
// Hilfsfunktion: Canvas bei EOS/EOT trimmen
// ================================================================================================
static size_t trim_canvas(const llama_vocab * vocab,
                          const llama_token * canvas,
                          size_t C) {
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);
    for (size_t i = 0; i < C; i++) {
        if ((eos >= 0 && canvas[i] == eos) || (eot >= 0 && canvas[i] == eot)) {
            return i;
        }
    }
    return C;
}

// ================================================================================================
// Entropy-Bound Decoder
// Portiert aus PR #24423 examples/diffusion/diffusion.cpp
// ================================================================================================
static void diffusion_generate_entropy_bound(llama_context * ctx,
                                             const llama_token * input_tokens,
                                             llama_token * output_tokens,
                                             int32_t n_input,
                                             const diffusion_eb_params & params,
                                             int32_t & n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    llama_model * model   = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t C       = params.max_length - n_input;  // Canvas-Laenge
    const int32_t S       = std::max(1, params.max_denoising_steps);

    // Non-causal Attention fuer Diffusion
    llama_set_causal_attn(ctx, false);
    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    // Zufallsgenerator (deterministisch bei gegebenem Seed)
    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    // Arbeits-Canvas
    std::vector<llama_token> current_canvas(C);
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);  // random init (not mask)
    }

    // Puffer fuer Self-Conditioning (nur Host-Pfad)
    std::vector<float>       sc_buffer((size_t) C * n_vocab, 0.0f);
    std::vector<llama_token> argmax_canvas(C, 0);
    std::vector<llama_token> prev_argmax(C, -1);  // -1 = Step 0 ist instabil
    std::vector<float>       entropy(C);
    std::vector<llama_token> denoiser(C);
    std::vector<int32_t>     order(C);
    std::vector<float>       u(C);         // vorgezeichnete Zufallszahlen
    std::vector<llama_token> renoise(C);   // vorgezeichnete Renoising-Tokens

    // Thread-Pool fuer parallele Reduktionen
    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);

    // Cached path: PREFILL Prompt einmalig
    const int32_t logit_off = params.kv_cache ? 0 : n_input;
    if (params.kv_cache) {
        llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, n_input);
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "PREFILL decode failed\n");
            llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
            llama_batch_free(batch);
            return;
        }
    }

    float prev_temp_inv = 1.0f;
    int   held          = 0;
    bool  finished      = false;

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;  // 0-based
        // Lineare Temperatur-Schedule: t_max -> t_min
        const float t        = params.t_min + (params.t_max - params.t_min)
                               * ((float) cur_step / (float) S);
        const float temp_inv = 1.0f / t;

        // Batch aufbauen: PREFILL/DECODE oder UNIFIED
        if (params.kv_cache) {
            llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, n_input);
            batch.n_tokens = C;
            for (int32_t i = 0; i < C; i++) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_input + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        } else {
            llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
            batch.n_tokens = params.max_length;
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = (i < n_input) ? input_tokens[i] : current_canvas[i - n_input];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        }

        // Self-Conditioning: DEAKTIVIERT (SC-Tensoren nicht geladen in unserem Fork)
        // In der Referenz: softmax(prev logits / prev t), gated off on first step
        // Wir setzen enabled=false, da sc_gate/sc_up/sc_down nullptr sind.
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decode failed at step %d\n", step_idx);
            break;
        }

        // Logits holen
        const float * logits = llama_get_logits(ctx);

        // Zufallszahlen vorgezeichnet fuer Determinismus
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // Per-Position: argmax, entropy, multinomial sample
        // Parallele Reduktion ueber Thread-Pool
        auto worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; pos++) {
                const float * row = logits + (size_t) (logit_off + pos) * n_vocab;

                // Argmax
                float m = -INFINITY;
                int32_t amax = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float z = row[v] * temp_inv;
                    if (z > m) { m = z; amax = v; }
                }

                // Softmax + Entropie + Multinomial
                float Z = 0.0f;
                for (int32_t v = 0; v < n_vocab; v++) {
                    Z += expf(row[v] * temp_inv - m);
                }

                const float target = u[pos] * Z;
                float cum = 0.0f, H = 0.0f;
                int32_t sampled = n_vocab - 1;
                bool picked = false;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float e = expf(row[v] * temp_inv - m);
                    const float p = e / Z;
                    if (p > 0.0f) { H -= p * logf(p); }
                    cum += e;
                    if (!picked && cum >= target) { sampled = v; picked = true; }
                }

                entropy[pos]       = H;
                argmax_canvas[pos] = amax;
                denoiser[pos]      = sampled;

                // SC-Buffer fuer naechsten Step
                std::memcpy(sc_buffer.data() + (size_t) pos * n_vocab, row, n_vocab * sizeof(float));
            }
        };

        // Thread-Pool starten
        {
            std::vector<std::thread> pool;
            const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ti++) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, C);
                if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
            }
            for (auto & th : pool) { th.join(); }
        }

        // Acceptance: Sortiere nach Entropie, akzeptiere niedrigste innerhalb Bound
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });

        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= params.entropy_bound) {
                accepted[pos] = 1;
            }
        }

        // Renoising + Output
        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            current_canvas[pos]          = accepted[pos] ? denoiser[pos] : renoise[pos];
            output_tokens[n_input + pos] = argmax_canvas[pos];  // OUTPUT = argmax!
            entropy_sum += entropy[pos];
        }

        // Konvergenz-Check: stabil UND konfidenz
        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;
        if (held >= params.stability_threshold && confident) {
            finished = true;
        }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;
    }

    if (params.kv_cache) {
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
    }
    llama_batch_free(batch);
    n_generated = params.max_length;
}

// ================================================================================================
// CLI
// ================================================================================================
static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m <model.gguf> -p <prompt> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m          Model path (required)\n");
    fprintf(stderr, "  -p          Prompt text (required)\n");
    fprintf(stderr, "  -ngl        GPU layers (default: 0)\n");
    fprintf(stderr, "  --steps     Diffusion steps (default: 48)\n");
    fprintf(stderr, "  --t-min     Min temperature (default: 0.4)\n");
    fprintf(stderr, "  --t-max     Max temperature (default: 0.8)\n");
    fprintf(stderr, "  --eb        Entropy bound (default: 0.1)\n");
    fprintf(stderr, "  --seed      Random seed (default: 0)\n");
    fprintf(stderr, "  -h, --help  Show this help\n");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt_text = nullptr;
    int n_gpu_layers = 0;
    int n_steps = 48;
    float t_min = 0.4f;
    float t_max = 0.8f;
    float entropy_bound = 0.1f;
    int seed = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t-min") == 0 && i + 1 < argc) {
            t_min = atof(argv[++i]);
        } else if (strcmp(argv[i], "--t-max") == 0 && i + 1 < argc) {
            t_max = atof(argv[++i]);
        } else if (strcmp(argv[i], "--eb") == 0 && i + 1 < argc) {
            entropy_bound = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path || !prompt_text) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "  DiffusionGemma CLI — Entropy-Bound Decoder\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "Model:   %s\n", model_path);
    fprintf(stderr, "Prompt:  %s\n", prompt_text);
    fprintf(stderr, "GPU:     %d layers\n", n_gpu_layers);
    fprintf(stderr, "Steps:   %d\n", n_steps);
    fprintf(stderr, "T-range: %.2f → %.2f\n", t_max, t_min);
    fprintf(stderr, "EB:      %.3f\n", entropy_bound);
    fprintf(stderr, "Seed:    %d\n\n", seed);

    // ========================================================================
    // 1. Modell laden
    // ========================================================================
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    fprintf(stderr, "[1/4] Loading model...\n");
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Canvas-Laenge aus GGUF Metadaten
    char canvas_str[32] = {0};
    int32_t canvas_length = 256;
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
        canvas_length = (int32_t) strtol(canvas_str, nullptr, 10);
    }
    fprintf(stderr, "  Canvas length: %d\n", canvas_length);

    // ================================================================================================
    // Chat-Template: Prompt automatisch formatieren (falls im Modell vorhanden)
    // ================================================================================================
    std::string formatted_prompt = prompt_text;
    try {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
        if (tmpls) {
            common_chat_templates_inputs inputs;
            inputs.messages.push_back({"user", prompt_text, {}, {}, "", ""});
            inputs.add_generation_prompt = true;
            inputs.use_jinja = true;
            common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
            if (!params.prompt.empty()) {
                formatted_prompt = params.prompt;
                fprintf(stderr, "  Chat-Template angewendet.\n");
            } else {
                fprintf(stderr, "  Chat-Template leer, nutze Roh-Prompt.\n");
            }
        } else {
            fprintf(stderr, "  Kein Chat-Template gefunden, nutze Roh-Prompt.\n");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "  Chat-Template-Fehler (%s), nutze Roh-Prompt.\n", e.what());
    }

    // ========================================================================
    // 2. Kontext erstellen
    // ========================================================================
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    fprintf(stderr, "[2/4] Creating context...\n");
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // ========================================================================
    // 3. Prompt tokenisieren
    // ========================================================================
    fprintf(stderr, "[3/4] Tokenizing...\n");
    std::vector<llama_token> prompt_tokens(4096);
    int n_prompt = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(),
                                  prompt_tokens.data(), prompt_tokens.size(), true, false);
    if (n_prompt < 0) {
        fprintf(stderr, "Error: Failed to tokenize\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    prompt_tokens.resize(n_prompt);
    fprintf(stderr, "  Prompt: %d tokens\n", n_prompt);

    // ========================================================================
    // 4. Entropy-Bound Decoder ausfuehren
    // ========================================================================
    fprintf(stderr, "[4/4] Entropy-Bound Decoding...\n");
    fprintf(stderr, "--------------------------------------------------------------------------------\n");

    const int32_t max_length = n_prompt + canvas_length;
    std::vector<llama_token> output_tokens(max_length, 0);

    diffusion_eb_params eb;
    eb.max_denoising_steps  = n_steps;
    eb.t_min                = t_min;
    eb.t_max                = t_max;
    eb.entropy_bound        = entropy_bound;
    eb.seed                 = seed;
    eb.max_length           = max_length;
    eb.kv_cache             = false;  // UNIFIED mode (einfacher)

    auto t_start = std::chrono::high_resolution_clock::now();
    int32_t n_generated = 0;
    diffusion_generate_entropy_bound(ctx, prompt_tokens.data(), output_tokens.data(),
                                      n_prompt, eb, n_generated);
    auto t_end = std::chrono::high_resolution_clock::now();

    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    fprintf(stderr, "--------------------------------------------------------------------------------\n");
    fprintf(stderr, "Decoding: %.1f ms (%.2f ms/step avg)\n",
            total_ms, total_ms / n_steps);

    // ========================================================================
    // 5. Ergebnis dekodieren und ausgeben
    // ========================================================================
    // Canvas extrahieren (nach Prompt)
    const llama_token * canvas = output_tokens.data() + n_prompt;
    const size_t cut = trim_canvas(vocab, canvas, (size_t) canvas_length);

    // Debug: Roh-Canvas anzeigen (vor Trimming)
    fprintf(stderr, "\n=== Raw Canvas (first 20 tokens) ===\n");
    for (size_t j = 0; j < std::min((size_t)20, (size_t)canvas_length); j++) {
        char buf[256];
        int n = llama_token_to_piece(vocab, canvas[j], buf, sizeof(buf), 0, true);
        if (n > 0) {
            buf[n] = '\0';
            fprintf(stderr, "[%zu] id=%d '%s' ", j, canvas[j], buf);
        }
    }
    fprintf(stderr, "\n\n=== Trimmed Text (cut=%zu) ===\n", cut);
    std::string final_text;
    for (size_t j = 0; j < cut; j++) {
        char buf[256];
        int n = llama_token_to_piece(vocab, canvas[j], buf, sizeof(buf), 0, true);
        if (n > 0) {
            buf[n] = '\0';
            final_text += buf;
        }
    }
    printf("%s\n", final_text.c_str());

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
