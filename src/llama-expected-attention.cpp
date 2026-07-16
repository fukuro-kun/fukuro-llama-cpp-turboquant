// Expected Attention KV Cache Compression — Phase 3 (Intelligent EA Scoring)
// Paper: arXiv:2510.00636
//
// Implementation notes:
// - All math is CPU-only, single-threaded (Phase 3)
// - Uses simple loops (no BLAS) for clarity and portability
// - Covariance is O(d^2) per head — expensive for large head_dim
// - Mean-only mode (use_covariance=false) is O(d) and much faster
// - Covariance deferred to Phase 4 (Gemma 4 QK-Norm → <0.1% Gain, siehe Design-Doc)

#include "llama-expected-attention.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace llama_expected_attention {

// ============================================================================
// ea_query_buffer
// ============================================================================

ea_query_buffer::ea_query_buffer(int hd, int cap)
    : head_dim(hd), capacity(cap), count(0), write_idx(0),
      data((size_t)cap * hd, 0.0f) {}

void ea_query_buffer::add(const float * query) {
    if (capacity <= 0 || head_dim <= 0) return;
    std::memcpy(data.data() + (size_t)write_idx * head_dim, query, (size_t)head_dim * sizeof(float));
    write_idx = (write_idx + 1) % capacity;
    if (count < capacity) count++;
}

const float * ea_query_buffer::get(int i) const {
    if (i < 0 || i >= count) return nullptr;
    // Logical index 0 = oldest. In circular buffer, oldest is at write_idx if full,
    // or at index 0 if not yet full.
    int physical;
    if (count < capacity) {
        physical = i;  // not wrapped yet, oldest is at 0
    } else {
        physical = (write_idx + i) % capacity;
    }
    return data.data() + (size_t)physical * head_dim;
}

void ea_query_buffer::clear() {
    count = 0;
    write_idx = 0;
    std::fill(data.begin(), data.end(), 0.0f);
}

// ============================================================================
// Statistics
// ============================================================================

void ea_compute_mean(const ea_query_buffer & buf, float * mu) {
    if (buf.count == 0 || buf.head_dim <= 0) {
        std::memset(mu, 0, (size_t)buf.head_dim * sizeof(float));
        return;
    }

    std::memset(mu, 0, (size_t)buf.head_dim * sizeof(float));
    for (int i = 0; i < buf.count; i++) {
        const float * q = buf.get(i);
        for (int d = 0; d < buf.head_dim; d++) {
            mu[d] += q[d];
        }
    }
    const float inv_n = 1.0f / (float)buf.count;
    for (int d = 0; d < buf.head_dim; d++) {
        mu[d] *= inv_n;
    }
}

// ============================================================================
// TurboQuant WHT-Forward (replicates turbo_cpu_fwht from ggml-turbo-quant.c)
// ============================================================================

// WHT sign arrays — must match turbo_cpu_s1/s2 in ggml-turbo-quant.c
// and turbo_wht_s1/s2 in ggml-cpu/ops.cpp (seed=42)
static const float ea_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float ea_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

void ea_wht_forward(float * x, int group_size, const float * scale_inv) {
    const float * s1 = ea_wht_s1;
    const float * s2 = ea_wht_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // InnerQ forward: apply scale_inv BEFORE signs+WHT (for Q pre-rotation)
    if (scale_inv) {
        for (int i = 0; i < group_size; i++) x[i] *= scale_inv[i];
    }

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

void ea_compute_covariance(const ea_query_buffer & buf, const float * mu, float * cov) {
    const int d = buf.head_dim;
    if (buf.count <= 1 || d <= 0) {
        std::memset(cov, 0, (size_t)d * d * sizeof(float));
        return;
    }

    // cov[i][j] = E[(q_i - mu_i)(q_j - mu_j)] = E[q_i q_j] - mu_i * mu_j
    // We compute E[q_i q_j] first, then subtract outer product of mean.
    std::memset(cov, 0, (size_t)d * d * sizeof(float));

    for (int n = 0; n < buf.count; n++) {
        const float * q = buf.get(n);
        for (int i = 0; i < d; i++) {
            const float qi = q[i] - mu[i];
            cov[(size_t)i * d + i] += qi * qi;  // diagonal
            for (int j = i + 1; j < d; j++) {
                const float qj = q[j] - mu[j];
                const float val = qi * qj;
                cov[(size_t)i * d + j] += val;
                cov[(size_t)j * d + i] += val;  // symmetric
            }
        }
    }

    const float inv_n1 = 1.0f / (float)(buf.count - 1);  // sample covariance
    for (size_t k = 0; k < (size_t)d * d; k++) {
        cov[k] *= inv_n1;
    }
}

// ============================================================================
// RoPE Rotation Matrix
// ============================================================================

// Llama-style RoPE: pairs (i, i+d/2) are rotated by angle theta_i * pos
// where theta_i = theta_base^(-2i/d)
//
// Rotation matrix R is block-diagonal with 2x2 rotation blocks:
// R[2i][2i]     = cos(angle_i)
// R[2i][2i+1]   = -sin(angle_i)
// R[2i+1][2i]   = sin(angle_i)
// R[2i+1][2i+1] = cos(angle_i)
//
// Note: There are different RoPE conventions (GPT-J vs Llama).
// This uses the Llama convention where dimensions are paired as (0, d/2), (1, d/2+1), ...
// Actually, llama.cpp uses the "interleaved" convention: (0,1), (2,3), ...
// We use the interleaved convention here.

void ea_compute_rope_matrix(float pos, float theta_base, int head_dim, float * rot, int rope_mode) {
    const int d = head_dim;
    const int half = d / 2;

    // Initialize as identity
    std::memset(rot, 0, (size_t)d * d * sizeof(float));
    for (int i = 0; i < d; i++) {
        rot[(size_t)i * d + i] = 1.0f;
    }

    if (rope_mode == 1) {
        // NeoX convention: pairs (i, i+d/2)
        // R[i][i]       = cos(angle_i)
        // R[i][i+d/2]   = -sin(angle_i)
        // R[i+d/2][i]   = sin(angle_i)
        // R[i+d/2][i+d/2] = cos(angle_i)
        for (int i = 0; i < half; i++) {
            const float theta = std::pow(theta_base, (float)(-2 * i) / (float)d);
            const float angle = pos * theta;
            const float c = std::cos(angle);
            const float s = std::sin(angle);

            rot[(size_t)i * d + i]           = c;
            rot[(size_t)i * d + (i + half)]  = -s;
            rot[(size_t)(i + half) * d + i]  = s;
            rot[(size_t)(i + half) * d + (i + half)] = c;
        }
    } else {
        // Interleaved convention (GPT-J/NORM): pairs (2i, 2i+1)
        for (int i = 0; i < half; i++) {
            const float theta = std::pow(theta_base, (float)(-2 * i) / (float)d);
            const float angle = pos * theta;
            const float c = std::cos(angle);
            const float s = std::sin(angle);

            const int r0 = 2 * i;
            const int r1 = 2 * i + 1;

            rot[(size_t)r0 * d + r0] = c;
            rot[(size_t)r0 * d + r1] = -s;
            rot[(size_t)r1 * d + r0] = s;
            rot[(size_t)r1 * d + r1] = c;
        }
    }

    // Handle odd dimension (last dimension unpaired, no rotation)
    if (d % 2 == 1) {
        rot[(size_t)(d - 1) * d + (d - 1)] = 1.0f;
    }
}

void ea_average_rope(float current_pos, int n_future, float theta_base,
                     int head_dim, float * rot_avg, int rope_mode) {
    const int d = head_dim;
    if (n_future <= 0) {
        ea_compute_rope_matrix(current_pos, theta_base, d, rot_avg, rope_mode);
        return;
    }

    // Accumulate rotation matrices over n_future positions
    std::vector<float> rot_single((size_t)d * d);
    std::memset(rot_avg, 0, (size_t)d * d * sizeof(float));

    for (int t = 0; t < n_future; t++) {
        const float pos = current_pos + (float)t + 1.0f;
        ea_compute_rope_matrix(pos, theta_base, d, rot_single.data(), rope_mode);
        for (size_t k = 0; k < (size_t)d * d; k++) {
            rot_avg[k] += rot_single[k];
        }
    }

    const float inv_n = 1.0f / (float)n_future;
    for (size_t k = 0; k < (size_t)d * d; k++) {
        rot_avg[k] *= inv_n;
    }
}

// ============================================================================
// Transform Statistics
// ============================================================================

// μ' = R @ μ
void ea_transform_statistics(const float * mu, const float * cov, const float * rot,
                             int head_dim, float * mu_prime, float * cov_prime) {
    const int d = head_dim;

    // μ' = R @ μ
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += rot[(size_t)i * d + j] * mu[j];
        }
        mu_prime[i] = sum;
    }

    if (cov == nullptr || cov_prime == nullptr) return;

    // Σ' = R @ Σ @ R^T
    // Step 1: tmp = R @ Σ
    std::vector<float> tmp((size_t)d * d);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += rot[(size_t)i * d + k] * cov[(size_t)k * d + j];
            }
            tmp[(size_t)i * d + j] = sum;
        }
    }

    // Step 2: Σ' = tmp @ R^T
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += tmp[(size_t)i * d + k] * rot[(size_t)j * d + k];  // R^T[k][j] = R[j][k]
            }
            cov_prime[(size_t)i * d + j] = sum;
        }
    }
}

// ============================================================================
// Score Computation
// ============================================================================

void ea_compute_scores(const float * K, const float * V,
                       const float * mu_prime, const float * cov_prime,
                       int n_tokens, int head_dim,
                       const ea_params & params,
                       float * scores) {
    const int d = head_dim;
    const float inv_sqrt_d = 1.0f / std::sqrt((float)d);
    const float inv_d = 1.0f / (float)d;
    const float eps = 1e-6f;

    // Temporary buffer for covariance term: K_i @ Σ' @ K_i^T
    std::vector<float> tmp(d);

    for (int t = 0; t < n_tokens; t++) {
        const float * K_t = K + (size_t)t * d;

        // Mean term: K_i @ μ'^T / sqrt(d)
        float mean_term = 0.0f;
        for (int i = 0; i < d; i++) {
            mean_term += K_t[i] * mu_prime[i];
        }
        mean_term *= inv_sqrt_d;

        // Covariance term: 0.5 * K_i @ Σ' @ K_i^T / d
        float cov_term = 0.0f;
        if (params.use_covariance && cov_prime != nullptr) {
            // tmp = Σ' @ K_t
            for (int i = 0; i < d; i++) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += cov_prime[(size_t)i * d + j] * K_t[j];
                }
                tmp[i] = sum;
            }
            // K_t @ tmp = K_t @ Σ' @ K_t
            for (int i = 0; i < d; i++) {
                cov_term += K_t[i] * tmp[i];
            }
            cov_term *= 0.5f * inv_d;
        }

        // E(A) = exp(mean_term + cov_term)
        // Clamp to avoid overflow
        float logit = mean_term + cov_term;
        if (logit > 50.0f) logit = 50.0f;
        if (logit < -50.0f) logit = -50.0f;
        float ea = std::exp(logit);

        // Value norm rescaling
        if (params.use_vnorm && V != nullptr) {
            const float * V_t = V + (size_t)t * d;
            float vnorm = 0.0f;
            for (int i = 0; i < d; i++) {
                vnorm += V_t[i] * V_t[i];
            }
            vnorm = std::sqrt(vnorm);
            ea = (ea + eps) * vnorm;
        } else {
            ea += eps;
        }

        scores[t] = ea;
    }
}

// ============================================================================
// Compression Decision
// ============================================================================

ea_compress_result ea_compress(const float * scores, int n_tokens,
                                const ea_params & params) {
    ea_compress_result result;
    result.keep_mask.resize(n_tokens, 1);
    result.scores.assign(scores, scores + n_tokens);

    if (n_tokens <= params.n_sink + params.n_local) {
        // Not enough tokens to prune
        result.n_pruned = 0;
        result.n_kept = n_tokens;
        return result;
    }

    // Number of tokens eligible for pruning (exclude sink + local)
    const int n_eligible = n_tokens - params.n_sink - params.n_local;
    const int n_to_prune = (int)(params.compression_ratio * n_eligible);
    if (n_to_prune <= 0) {
        result.n_pruned = 0;
        result.n_kept = n_tokens;
        return result;
    }

    // Collect indices of eligible tokens (positions n_sink .. n_tokens - n_local - 1)
    struct score_idx {
        float score;
        int   idx;
    };
    std::vector<score_idx> eligible(n_eligible);
    for (int i = 0; i < n_eligible; i++) {
        eligible[i] = {scores[params.n_sink + i], params.n_sink + i};
    }

    // Partial sort: find the n_to_prune smallest scores
    std::nth_element(eligible.begin(),
                     eligible.begin() + n_to_prune,
                     eligible.end(),
                     [](const score_idx & a, const score_idx & b) {
                         return a.score < b.score;
                     });

    // Mark pruned tokens
    for (int i = 0; i < n_to_prune; i++) {
        result.keep_mask[eligible[i].idx] = 0;
    }

    result.n_pruned = n_to_prune;
    result.n_kept = n_tokens - n_to_prune;
    return result;
}

// ============================================================================
// Full Pipeline
// ============================================================================

ea_compress_result ea_compress_pipeline(
    const float * queries, int n_queries,
    const float * keys, const float * values,
    int n_tokens, int head_dim,
    float current_pos, float theta_base,
    const ea_params & params) {

    // Step 1: Fill rolling buffer with queries
    ea_query_buffer buf(head_dim, params.rolling_buffer_size);
    const int n_buf = std::min(n_queries, params.rolling_buffer_size);
    for (int i = 0; i < n_buf; i++) {
        buf.add(queries + (size_t)i * head_dim);
    }

    // Step 2: Compute mean and covariance
    std::vector<float> mu(head_dim);
    ea_compute_mean(buf, mu.data());

    std::vector<float> cov;
    if (params.use_covariance) {
        cov.resize((size_t)head_dim * head_dim);
        ea_compute_covariance(buf, mu.data(), cov.data());
    }

    // Step 3: Average RoPE rotation over future positions
    std::vector<float> rot_avg((size_t)head_dim * head_dim);
    ea_average_rope(current_pos, params.n_future_positions, theta_base,
                    head_dim, rot_avg.data());

    // Step 4: Transform statistics
    std::vector<float> mu_prime(head_dim);
    std::vector<float> cov_prime;
    if (params.use_covariance) {
        cov_prime.resize((size_t)head_dim * head_dim);
    }
    ea_transform_statistics(mu.data(), cov.empty() ? nullptr : cov.data(),
                           rot_avg.data(), head_dim,
                           mu_prime.data(),
                           cov_prime.empty() ? nullptr : cov_prime.data());

    // Step 5: Compute scores
    std::vector<float> scores(n_tokens);
    ea_compute_scores(keys, values,
                      mu_prime.data(),
                      cov_prime.empty() ? nullptr : cov_prime.data(),
                      n_tokens, head_dim, params, scores.data());

    // Step 6: Compression decision
    return ea_compress(scores.data(), n_tokens, params);
}

} // namespace llama_expected_attention
