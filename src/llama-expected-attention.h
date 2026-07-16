#pragma once

// Expected Attention KV Cache Compression — Phase 3 (Intelligent EA Scoring)
// Paper: arXiv:2510.00636
// ROADMAP-Item: #19
//
// Training-free KV-Cache-Compression durch Vorhersage der zukünftigen Wichtigkeit
// von KV-Paaren. Orthogonal zu TurboQuant (Pruning vs Quantisierung).
//
// Phase 1: Standalone CPU-Mathematik ohne KV-Cache-Integration.
// Phase 2: Integration in llama_kv_cache (Pruning-Logik).
// Phase 3: TurboQuant-Kombination (WHT-Forward) + Intelligent EA Scoring.

#include <cstdint>
#include <vector>

namespace llama_expected_attention {

// Parameters for Expected Attention compression
struct ea_params {
    float    compression_ratio   = 0.5f;   // fraction of KV pairs to prune (0.0 = no pruning)
    int      n_future_positions  = 512;    // RoPE prediction horizon
    int      n_sink              = 4;      // protected initial tokens (sink attention)
    int      n_local             = 128;    // protected recent tokens (local window)
    bool     use_covariance      = true;   // use covariance term (more accurate, O(d^2))
    bool     use_vnorm           = true;   // rescale scores by value L2 norm
    int      rolling_buffer_size = 128;    // number of recent queries for statistics
};

// Result of a compression decision for one attention head
struct ea_compress_result {
    std::vector<int32_t> keep_mask;  // [n_tokens] 1 = keep, 0 = prune
    std::vector<float>   scores;     // [n_tokens] expected attention scores
    int                  n_pruned;   // number of tokens pruned
    int                  n_kept;     // number of tokens kept
};

// Rolling buffer for query statistics (before RoPE)
// Stores the last `rolling_buffer_size` query hidden states per head.
struct ea_query_buffer {
    int  head_dim = 0;
    int  capacity = 0;  // == rolling_buffer_size
    int  count    = 0;  // number of queries stored so far (up to capacity)
    int  write_idx = 0; // circular buffer write position

    // [capacity * head_dim] flat storage, row-major
    std::vector<float> data;

    ea_query_buffer() = default;
    ea_query_buffer(int head_dim, int capacity);

    // Add a query (head_dim floats). Old queries are overwritten circularly.
    void add(const float * query);

    // Get pointer to query at logical index i (0 = oldest, count-1 = newest)
    const float * get(int i) const;

    // Reset buffer
    void clear();
};

// Compute mean (μ) of stored queries.
// Output: mu[head_dim]
void ea_compute_mean(const ea_query_buffer & buf, float * mu);

// Apply TurboQuant WHT-forward rotation in-place.
// Replicates turbo_cpu_fwht: signs1 → butterfly → normalize → signs2.
// group_size is always 128 (TurboQuant pads head_dim to multiples of 128).
// If scale_inv is non-null, applies per-channel scaling BEFORE WHT (InnerQ forward).
// x must have at least group_size elements. Operates on one group at a time.
void ea_wht_forward(float * x, int group_size, const float * scale_inv = nullptr);

// Compute covariance (Σ) of stored queries around the mean.
// Output: cov[head_dim * head_dim], row-major
void ea_compute_covariance(const ea_query_buffer & buf, const float * mu, float * cov);

// Compute RoPE rotation matrix for a given position.
// For Llama-style RoPE with theta_base and head_dim.
// Output: rot[head_dim * head_dim], row-major
// rope_mode: 0 = interleaved (GPT-J/NORM, pairs (2i, 2i+1)),
//            1 = NeoX (pairs (i, i+d/2))
void ea_compute_rope_matrix(float pos, float theta_base, int head_dim, float * rot, int rope_mode = 0);

// Transform statistics through RoPE: μ' = R @ μ, Σ' = R @ Σ @ R^T
void ea_transform_statistics(const float * mu, const float * cov, const float * rot,
                             int head_dim, float * mu_prime, float * cov_prime);

// Average RoPE rotation over n_future_positions starting from current_pos.
// This gives the expected rotation that future queries will undergo.
// rope_mode: 0 = interleaved (GPT-J/NORM), 1 = NeoX
void ea_average_rope(float current_pos, int n_future, float theta_base,
                     int head_dim, float * rot_avg, int rope_mode = 0);

// Compute expected attention scores for each KV pair.
//
// E(A)_i = exp(K_i @ μ'^T / sqrt(d) + 0.5 * K_i @ Σ' @ K_i^T / d)
//
// K:       [n_tokens, head_dim] key vectors (post-RoPE)
// V:       [n_tokens, head_dim] value vectors (optional, for vnorm)
// mu_prime:  [head_dim] transformed mean
// cov_prime: [head_dim * head_dim] transformed covariance (or nullptr if use_covariance=false)
// params:    compression parameters
//
// Output: scores[n_tokens]
void ea_compute_scores(const float * K, const float * V,
                       const float * mu_prime, const float * cov_prime,
                       int n_tokens, int head_dim,
                       const ea_params & params,
                       float * scores);

// Decide which tokens to keep and which to prune.
// Protects n_sink initial tokens and n_local recent tokens.
// Prunes bottom compression_ratio% by score.
//
// scores: [n_tokens] expected attention scores
// n_tokens: total number of tokens
// params:   compression parameters
//
// Returns: ea_compress_result with keep_mask and counts
ea_compress_result ea_compress(const float * scores, int n_tokens,
                                const ea_params & params);

// Full pipeline: given queries, keys, values, decide which to prune.
// This is the main entry point for Phase 1.
//
// queries:  [n_queries, head_dim] query hidden states (before RoPE, for statistics)
// keys:     [n_tokens, head_dim] key vectors (post-RoPE)
// values:   [n_tokens, head_dim] value vectors (optional, for vnorm)
// n_queries: number of queries in the rolling buffer (<= rolling_buffer_size)
// n_tokens:  total KV pairs to evaluate
// head_dim:  dimension per head
// current_pos: current position in the sequence (for RoPE prediction)
// theta_base:  RoPE theta base (e.g. 10000.0f for Llama, 1000000.0f for Gemma)
// params:    compression parameters
//
// Returns: ea_compress_result
ea_compress_result ea_compress_pipeline(
    const float * queries, int n_queries,
    const float * keys, const float * values,
    int n_tokens, int head_dim,
    float current_pos, float theta_base,
    const ea_params & params);

} // namespace llama_expected_attention
