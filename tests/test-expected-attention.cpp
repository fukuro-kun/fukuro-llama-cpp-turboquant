// Unit tests for Expected Attention KV Cache Compression — Phase 1
// Tests the CPU math: mean, covariance, RoPE matrix, score computation, compression

#include "llama-expected-attention.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace llama_expected_attention;

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond, ...) do { \
    g_tests_run++; \
    if (cond) { g_tests_passed++; } \
    else { g_tests_failed++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
} while(0)

#define ASSERT_NEAR(a, b, eps, ...) do { \
    g_tests_run++; \
    if (std::fabs((float)(a) - (float)(b)) < (eps)) { g_tests_passed++; } \
    else { g_tests_failed++; printf("FAIL: " __VA_ARGS__); printf(" (got %f, expected %f, eps %f)\n", (double)(a), (double)(b), (double)(eps)); } \
} while(0)

// ============================================================================
// Test 1: Rolling buffer circular behavior
// ============================================================================

static void test_query_buffer_circular() {
    ea_query_buffer buf(4, 3);  // head_dim=4, capacity=3

    float q0[4] = {1, 2, 3, 4};
    float q1[4] = {5, 6, 7, 8};
    float q2[4] = {9, 10, 11, 12};
    float q3[4] = {13, 14, 15, 16};

    buf.add(q0);
    buf.add(q1);
    buf.add(q2);
    ASSERT_TRUE(buf.count == 3, "buffer should have 3 entries");

    // Without wrapping: get(0) = oldest = q0
    const float * r0 = buf.get(0);
    ASSERT_NEAR(r0[0], 1.0f, 1e-6f, "oldest should be q0[0]");

    // Add one more → wraps, q0 is evicted, q1 becomes oldest
    buf.add(q3);
    ASSERT_TRUE(buf.count == 3, "buffer should still have 3 entries after wrap");

    const float * r1 = buf.get(0);
    ASSERT_NEAR(r1[0], 5.0f, 1e-6f, "after wrap, oldest should be q1[0]");

    const float * r2 = buf.get(2);
    ASSERT_NEAR(r2[0], 13.0f, 1e-6f, "newest should be q3[0]");
}

// ============================================================================
// Test 2: Mean computation
// ============================================================================

static void test_mean() {
    ea_query_buffer buf(2, 3);

    float q0[2] = {1.0f, 2.0f};
    float q1[2] = {3.0f, 4.0f};
    float q2[2] = {5.0f, 6.0f};

    buf.add(q0);
    buf.add(q1);
    buf.add(q2);

    float mu[2];
    ea_compute_mean(buf, mu);
    // mean = (1+3+5)/3 = 3, (2+4+6)/3 = 4
    ASSERT_NEAR(mu[0], 3.0f, 1e-5f, "mean[0]");
    ASSERT_NEAR(mu[1], 4.0f, 1e-5f, "mean[1]");
}

// ============================================================================
// Test 3: Covariance computation
// ============================================================================

static void test_covariance() {
    ea_query_buffer buf(2, 3);

    // Simple case: q0=(0,0), q1=(2,0), q2=(0,2)
    // mean = (2/3, 2/3)
    // cov[0][0] = var(x) = E[x^2] - mean_x^2 = (0+4+0)/3 - (2/3)^2 = 4/3 - 4/9 = 8/9
    // But sample covariance divides by (n-1) = 2
    // cov[0][0] = ((0-2/3)^2 + (2-2/3)^2 + (0-2/3)^2) / 2 = (4/9 + 16/9 + 4/9) / 2 = 24/9 / 2 = 4/3
    float q0[2] = {0.0f, 0.0f};
    float q1[2] = {2.0f, 0.0f};
    float q2[2] = {0.0f, 2.0f};

    buf.add(q0);
    buf.add(q1);
    buf.add(q2);

    float mu[2];
    ea_compute_mean(buf, mu);

    float cov[4];
    ea_compute_covariance(buf, mu, cov);

    // cov[0][0] = 4/3 ≈ 1.333
    ASSERT_NEAR(cov[0], 4.0f / 3.0f, 1e-4f, "cov[0][0]");
    // cov[1][1] = 4/3
    ASSERT_NEAR(cov[3], 4.0f / 3.0f, 1e-4f, "cov[1][1]");
    // cov[0][1] = cov[1][0] = -2/3 (x and y are negatively correlated:
    // when x is high, y tends to be low, and vice versa)
    ASSERT_NEAR(cov[1], -2.0f / 3.0f, 1e-4f, "cov[0][1]");
    ASSERT_NEAR(cov[2], -2.0f / 3.0f, 1e-4f, "cov[1][0]");
}

// ============================================================================
// Test 4: RoPE rotation matrix at position 0
// ============================================================================

static void test_rope_matrix_pos0() {
    // At position 0, all angles are 0, so R should be identity
    const int d = 4;
    float rot[d * d];
    ea_compute_rope_matrix(0.0f, 10000.0f, d, rot);

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            if (i == j) {
                ASSERT_NEAR(rot[(size_t)i * d + j], 1.0f, 1e-6f, "R[%d][%d] should be 1", i, j);
            } else {
                ASSERT_NEAR(rot[(size_t)i * d + j], 0.0f, 1e-6f, "R[%d][%d] should be 0", i, j);
            }
        }
    }
}

// ============================================================================
// Test 5: RoPE rotation matrix is orthogonal
// ============================================================================

static void test_rope_orthogonal() {
    const int d = 8;
    float rot[d * d];
    ea_compute_rope_matrix(42.0f, 10000.0f, d, rot);

    // Check R @ R^T = I
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += rot[(size_t)i * d + k] * rot[(size_t)j * d + k];
            }
            if (i == j) {
                ASSERT_NEAR(dot, 1.0f, 1e-4f, "R@R^T[%d][%d] should be 1", i, j);
            } else {
                ASSERT_NEAR(dot, 0.0f, 1e-4f, "R@R^T[%d][%d] should be 0", i, j);
            }
        }
    }
}

// ============================================================================
// Test 6: Score computation — uniform keys should give uniform scores
// ============================================================================

static void test_scores_uniform() {
    const int d = 4;
    const int n = 10;

    // All keys identical → all scores identical
    std::vector<float> K((size_t)n * d, 0.5f);
    std::vector<float> V((size_t)n * d, 1.0f);

    float mu_prime[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    // No covariance for simplicity
    ea_params params;
    params.use_covariance = false;
    params.use_vnorm = false;

    std::vector<float> scores(n);
    ea_compute_scores(K.data(), V.data(), mu_prime, nullptr,
                      n, d, params, scores.data());

    // All scores should be equal
    for (int i = 1; i < n; i++) {
        ASSERT_NEAR(scores[i], scores[0], 1e-5f, "uniform keys → uniform scores");
    }
}

// ============================================================================
// Test 7: Score computation — higher alignment with mu should give higher score
// ============================================================================

static void test_scores_alignment() {
    const int d = 4;
    const int n = 3;

    // Key 0: aligned with mu_prime (high score)
    // Key 1: orthogonal to mu_prime (medium score)
    // Key 2: anti-aligned with mu_prime (low score)
    std::vector<float> K = {
        1.0f, 0.0f, 0.0f, 0.0f,   // aligned with mu_prime[0]
        0.0f, 1.0f, 0.0f, 0.0f,   // orthogonal
        -1.0f, 0.0f, 0.0f, 0.0f,  // anti-aligned
    };
    std::vector<float> V((size_t)n * d, 1.0f);

    float mu_prime[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    ea_params params;
    params.use_covariance = false;
    params.use_vnorm = false;

    std::vector<float> scores(n);
    ea_compute_scores(K.data(), V.data(), mu_prime, nullptr,
                      n, d, params, scores.data());

    // score[0] = exp(1/sqrt(4)) = exp(0.5)
    // score[1] = exp(0) = 1
    // score[2] = exp(-0.5)
    ASSERT_TRUE(scores[0] > scores[1], "aligned key should score higher than orthogonal");
    ASSERT_TRUE(scores[1] > scores[2], "orthogonal key should score higher than anti-aligned");
}

// ============================================================================
// Test 8: Compression protects sink and local tokens
// ============================================================================

static void test_compress_protects_sink_local() {
    const int n = 100;
    std::vector<float> scores(n, 1.0f);
    // Make middle tokens have very low scores
    for (int i = 10; i < 90; i++) scores[i] = 0.01f;

    ea_params params;
    params.compression_ratio = 0.5f;
    params.n_sink = 4;
    params.n_local = 10;

    auto result = ea_compress(scores.data(), n, params);

    // Sink tokens (0-3) must be kept
    for (int i = 0; i < params.n_sink; i++) {
        ASSERT_TRUE(result.keep_mask[i] == 1, "sink token %d must be kept", i);
    }
    // Local tokens (90-99) must be kept
    for (int i = n - params.n_local; i < n; i++) {
        ASSERT_TRUE(result.keep_mask[i] == 1, "local token %d must be kept", i);
    }
    // Some middle tokens should be pruned
    ASSERT_TRUE(result.n_pruned > 0, "some tokens should be pruned");
    ASSERT_TRUE(result.n_kept + result.n_pruned == n, "kept + pruned == total");
}

// ============================================================================
// Test 9: Compression with ratio 0.0 prunes nothing
// ============================================================================

static void test_compress_zero_ratio() {
    const int n = 50;
    std::vector<float> scores(n, 1.0f);

    ea_params params;
    params.compression_ratio = 0.0f;

    auto result = ea_compress(scores.data(), n, params);
    ASSERT_TRUE(result.n_pruned == 0, "ratio=0 should prune nothing");
    ASSERT_TRUE(result.n_kept == n, "ratio=0 should keep all");
}

// ============================================================================
// Test 10: Full pipeline runs without crash
// ============================================================================

static void test_pipeline_smoke() {
    const int d = 8;
    const int n_queries = 32;
    const int n_tokens = 64;

    // Random-ish queries and keys
    std::vector<float> queries((size_t)n_queries * d);
    std::vector<float> keys((size_t)n_tokens * d);
    std::vector<float> values((size_t)n_tokens * d);

    for (int i = 0; i < n_queries * d; i++) queries[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < n_tokens  * d; i++) keys[i]    = (float)(i % 5) * 0.1f;
    for (int i = 0; i < n_tokens  * d; i++) values[i]  = (float)(i % 3) * 0.1f;

    ea_params params;
    params.compression_ratio = 0.3f;
    params.n_sink = 4;
    params.n_local = 8;
    params.use_covariance = true;
    params.use_vnorm = true;
    params.rolling_buffer_size = 32;

    auto result = ea_compress_pipeline(
        queries.data(), n_queries,
        keys.data(), values.data(),
        n_tokens, d,
        100.0f, 10000.0f,
        params);

    ASSERT_TRUE(result.n_kept + result.n_pruned == n_tokens, "pipeline: kept + pruned == total");
    ASSERT_TRUE(result.n_pruned > 0, "pipeline: should prune some tokens with ratio=0.3");

    // Sink tokens must be kept
    for (int i = 0; i < params.n_sink; i++) {
        ASSERT_TRUE(result.keep_mask[i] == 1, "pipeline: sink token %d kept", i);
    }
}

// ============================================================================
// Test 11: Mean-only vs covariance mode — mean-only should still produce valid scores
// ============================================================================

static void test_mean_only_mode() {
    const int d = 4;
    const int n = 5;

    std::vector<float> K = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    std::vector<float> V((size_t)n * d, 1.0f);

    float mu_prime[4] = {1.0f, 0.5f, 0.0f, 0.0f};

    ea_params params;
    params.use_covariance = false;
    params.use_vnorm = false;

    std::vector<float> scores(n);
    ea_compute_scores(K.data(), V.data(), mu_prime, nullptr,
                      n, d, params, scores.data());

    // All scores should be positive (exp is always positive)
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(scores[i] > 0.0f, "mean-only: score %d should be positive", i);
    }

    // Key 0 (aligned with mu) should have highest score
    ASSERT_TRUE(scores[0] > scores[2], "mean-only: key 0 > key 2");
    ASSERT_TRUE(scores[0] > scores[3], "mean-only: key 0 > key 3");
}

// ============================================================================
// Test 12: Vnorm rescaling changes ranking
// ============================================================================

static void test_vnorm_rescaling() {
    const int d = 2;
    const int n = 2;

    // Key 0: aligned with mu → high E(A), but V has small norm
    // Key 1: less aligned → lower E(A), but V has large norm
    std::vector<float> K = {
        1.0f, 0.0f,   // aligned
        0.1f, 0.0f,   // less aligned
    };
    std::vector<float> V = {
        0.01f, 0.01f,  // small norm
        10.0f, 10.0f,  // large norm
    };

    float mu_prime[2] = {1.0f, 0.0f};

    // Without vnorm
    ea_params params_no_vnorm;
    params_no_vnorm.use_covariance = false;
    params_no_vnorm.use_vnorm = false;

    std::vector<float> scores_no_vnorm(n);
    ea_compute_scores(K.data(), V.data(), mu_prime, nullptr,
                      n, d, params_no_vnorm, scores_no_vnorm.data());

    // With vnorm
    ea_params params_vnorm;
    params_vnorm.use_covariance = false;
    params_vnorm.use_vnorm = true;

    std::vector<float> scores_vnorm(n);
    ea_compute_scores(K.data(), V.data(), mu_prime, nullptr,
                      n, d, params_vnorm, scores_vnorm.data());

    // Without vnorm: key 0 should win (higher alignment)
    ASSERT_TRUE(scores_no_vnorm[0] > scores_no_vnorm[1],
                "without vnorm: aligned key should win");

    // With vnorm: key 1 should win (large V norm compensates)
    ASSERT_TRUE(scores_vnorm[1] > scores_vnorm[0],
                "with vnorm: large-V-norm key should win");
}

// ============================================================================
// Phase 3 Tests: NeoX RoPE mode, Q-Capture scoring, head aggregation
// ============================================================================

// Test: NeoX RoPE at position 0 should be identity
static void test_rope_neox_pos0() {
    const int d = 4;
    float rot[d * d];
    ea_compute_rope_matrix(0.0f, 10000.0f, d, rot, 1);  // rope_mode=1 (NeoX)

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            if (i == j) {
                ASSERT_NEAR(rot[(size_t)i * d + j], 1.0f, 1e-6f, "NeoX R[%d][%d] should be 1", i, j);
            } else {
                ASSERT_NEAR(rot[(size_t)i * d + j], 0.0f, 1e-6f, "NeoX R[%d][%d] should be 0", i, j);
            }
        }
    }
}

// Test: NeoX RoPE rotation matrix is orthogonal
static void test_rope_neox_orthogonal() {
    const int d = 8;
    float rot[d * d];
    ea_compute_rope_matrix(42.0f, 10000.0f, d, rot, 1);  // NeoX mode

    // Check R @ R^T = I
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += rot[(size_t)i * d + k] * rot[(size_t)j * d + k];
            }
            if (i == j) {
                ASSERT_NEAR(dot, 1.0f, 1e-4f, "NeoX R@R^T[%d][%d] should be 1", i, j);
            } else {
                ASSERT_NEAR(dot, 0.0f, 1e-4f, "NeoX R@R^T[%d][%d] should be 0", i, j);
            }
        }
    }
}

// Test: NeoX and interleaved modes produce different matrices (for same pos > 0)
static void test_rope_neox_differs_from_interleaved() {
    const int d = 4;
    float rot_neox[d * d];
    float rot_interleaved[d * d];

    ea_compute_rope_matrix(7.0f, 10000.0f, d, rot_neox, 1);       // NeoX
    ea_compute_rope_matrix(7.0f, 10000.0f, d, rot_interleaved, 0); // interleaved

    // They should differ (not identical)
    bool any_diff = false;
    for (int i = 0; i < d * d; i++) {
        if (std::fabs(rot_neox[i] - rot_interleaved[i]) > 1e-6f) {
            any_diff = true;
            break;
        }
    }
    ASSERT_TRUE(any_diff, "NeoX and interleaved RoPE should differ at pos=7");
}

// Test: NeoX RoPE pairs (i, i+d/2) — verify the rotation structure
static void test_rope_neox_pairing() {
    const int d = 4;
    float rot[d * d];
    ea_compute_rope_matrix(1.0f, 10000.0f, d, rot, 1);  // NeoX mode

    // In NeoX, pairs are (0, d/2), (1, d/2+1), etc.
    // For d=4: pairs are (0,2) and (1,3)
    // R[0][0] = cos(theta_0), R[0][2] = -sin(theta_0)
    // R[2][0] = sin(theta_0), R[2][2] = cos(theta_0)
    // R[1][1] = cos(theta_1), R[1][3] = -sin(theta_1)
    // R[3][1] = sin(theta_1), R[3][3] = cos(theta_1)
    // All other entries should be 0 (except identity for unpaired)

    // Check that R[0][1] = 0 (not paired in NeoX)
    ASSERT_NEAR(rot[0 * d + 1], 0.0f, 1e-6f, "NeoX R[0][1] should be 0");
    // Check that R[0][2] != 0 (paired in NeoX)
    ASSERT_TRUE(std::fabs(rot[0 * d + 2]) > 1e-6f, "NeoX R[0][2] should be non-zero (paired)");
}

// Test: EA scoring with rolling buffer — mean-only, no covariance
// Simulates the Phase 3 scoring path: build query buffer → compute mean →
// transform with RoPE → score keys
static void test_ea_scoring_mean_only() {
    const int d = 4;
    const int n_queries = 5;
    const int n_keys = 10;

    // Build a rolling buffer with queries that have a clear "preferred direction"
    ea_query_buffer buf(d, n_queries);
    for (int i = 0; i < n_queries; i++) {
        float q[d] = {1.0f, 0.0f, 0.0f, 0.0f};  // All queries point in dim 0
        buf.add(q);
    }
    ASSERT_TRUE(buf.count == n_queries, "buffer should have %d entries", n_queries);

    // Compute mean
    float mu[d];
    ea_compute_mean(buf, mu);
    ASSERT_NEAR(mu[0], 1.0f, 1e-6f, "mean[0] should be 1.0");
    ASSERT_NEAR(mu[1], 0.0f, 1e-6f, "mean[1] should be 0.0");

    // Average RoPE
    float rot_avg[d * d];
    ea_average_rope(10.0f, 8, 10000.0f, d, rot_avg);

    // Transform: mu' = R @ mu (mean-only, no covariance)
    float mu_prime[d];
    ea_transform_statistics(mu, nullptr, rot_avg, d, mu_prime, nullptr);

    // Keys: half aligned with dim 0, half orthogonal
    std::vector<float> K((size_t)n_keys * d);
    for (int t = 0; t < n_keys; t++) {
        if (t < n_keys / 2) {
            K[(size_t)t * d + 0] = 1.0f;  // aligned with query direction
            K[(size_t)t * d + 1] = 0.0f;
            K[(size_t)t * d + 2] = 0.0f;
            K[(size_t)t * d + 3] = 0.0f;
        } else {
            K[(size_t)t * d + 0] = 0.0f;  // orthogonal
            K[(size_t)t * d + 1] = 1.0f;
            K[(size_t)t * d + 2] = 0.0f;
            K[(size_t)t * d + 3] = 0.0f;
        }
    }

    // Compute scores
    std::vector<float> V((size_t)n_keys * d, 1.0f);
    float scores[n_keys];
    ea_params params;
    params.use_covariance = false;
    params.use_vnorm = false;
    ea_compute_scores(K.data(), V.data(), mu_prime, nullptr, n_keys, d, params, scores);

    // Aligned keys (first half) should have higher scores than orthogonal keys
    float avg_aligned = 0.0f, avg_orthogonal = 0.0f;
    for (int t = 0; t < n_keys / 2; t++) {
        avg_aligned += scores[t];
    }
    for (int t = n_keys / 2; t < n_keys; t++) {
        avg_orthogonal += scores[t];
    }
    avg_aligned /= (n_keys / 2);
    avg_orthogonal /= (n_keys / 2);

    ASSERT_TRUE(avg_aligned > avg_orthogonal,
                "aligned keys should score higher (aligned=%.4f, orthogonal=%.4f)",
                avg_aligned, avg_orthogonal);
}

// Test: EA scoring with NeoX RoPE mode
static void test_ea_scoring_neox_mode() {
    const int d = 4;
    const int n_queries = 4;

    ea_query_buffer buf(d, n_queries);
    for (int i = 0; i < n_queries; i++) {
        float q[d] = {1.0f, 0.5f, 0.0f, 0.3f};
        buf.add(q);
    }

    float mu[d];
    ea_compute_mean(buf, mu);

    // Use NeoX RoPE mode
    float rot_avg[d * d];
    ea_average_rope(5.0f, 4, 10000.0f, d, rot_avg, 1);  // rope_mode=1 (NeoX)

    float mu_prime[d];
    ea_transform_statistics(mu, nullptr, rot_avg, d, mu_prime, nullptr);

    // Verify mu_prime is not all zeros (transformation happened)
    bool any_nonzero = false;
    for (int i = 0; i < d; i++) {
        if (std::fabs(mu_prime[i]) > 1e-6f) {
            any_nonzero = true;
            break;
        }
    }
    ASSERT_TRUE(any_nonzero, "NeoX transformed mean should have non-zero entries");
}

// Test: Maximum head aggregation — verify that max over heads selects
// the most important tokens across heterogeneous heads
static void test_head_aggregation_max() {
    const int n_tokens = 10;
    const int n_heads = 3;

    // Simulate per-head scores: each head "cares about" different tokens
    // Head 0: high scores for tokens 0-2 (retrieval head)
    // Head 1: high scores for tokens 5-7 (reasoning head)
    // Head 2: uniform scores (background head)
    float head_scores[n_heads][n_tokens];
    for (int t = 0; t < n_tokens; t++) {
        head_scores[0][t] = (t < 3) ? 5.0f : 0.1f;
        head_scores[1][t] = (t >= 5 && t < 8) ? 5.0f : 0.1f;
        head_scores[2][t] = 1.0f;  // uniform
    }

    // Aggregate via maximum
    float token_scores[n_tokens];
    for (int t = 0; t < n_tokens; t++) {
        token_scores[t] = 0.0f;
        for (int h = 0; h < n_heads; h++) {
            token_scores[t] = std::max(token_scores[t], head_scores[h][t]);
        }
    }

    // Tokens 0-2 and 5-7 should have score 5.0 (protected by max aggregation)
    // Tokens 3-4 and 8-9 should have score 1.0 (only background head)
    ASSERT_NEAR(token_scores[0], 5.0f, 1e-6f, "token 0 should be protected by head 0");
    ASSERT_NEAR(token_scores[6], 5.0f, 1e-6f, "token 6 should be protected by head 1");
    ASSERT_NEAR(token_scores[3], 1.0f, 1e-6f, "token 3 should only have background score");
    ASSERT_NEAR(token_scores[9], 1.0f, 1e-6f, "token 9 should only have background score");
}

// ============================================================================
// TurboQuant WHT-Forward Tests
// ============================================================================

// Test: WHT-forward on a constant vector should produce a sparse output
// (Hadamard of constant = impulse at position 0)
static void test_wht_forward_identity() {
    std::vector<float> x(128, 1.0f);
    ea_wht_forward(x.data(), 128, nullptr);

    // After WHT of constant 1-vector: only position 0 should be non-zero
    // (H * 1 = sqrt(N) * e_0, but with signs it depends on s1/s2)
    // Actually with the sign arrays, the result is s2[0]*inv_sqrt*s1_sum * ...
    // Just check that the output is not all zeros and has finite values
    bool has_nonzero = false;
    bool all_finite = true;
    for (int i = 0; i < 128; i++) {
        if (std::fabs(x[i]) > 1e-6f) has_nonzero = true;
        if (!std::isfinite(x[i])) all_finite = false;
    }
    ASSERT_TRUE(has_nonzero, "WHT of constant should produce non-zero output");
    ASSERT_TRUE(all_finite, "WHT output should be finite");
}

// Reference WHT implementation (naive, independent of ea_wht_forward).
// Replicates the same sign arrays + butterfly + normalization so we can
// cross-check that ea_wht_forward matches exactly.
// The sign arrays are duplicated here deliberately (test-local copies) so a
// silent change in the production arrays would be caught.
static const float ref_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static const float ref_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};
static void ref_wht_forward(float * x, int n, const float * scale_inv) {
    const float inv_sqrt = (n == 128) ? 0.08838834764831845f : 0.125f;
    if (scale_inv) {
        for (int i = 0; i < n; i++) x[i] *= scale_inv[i % n];
    }
    for (int i = 0; i < n; i++) x[i] *= ref_wht_s1[i];
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < n; i++) x[i] *= inv_sqrt * ref_wht_s2[i];
}

// Test: ea_wht_forward matches the reference implementation exactly.
// This catches silent sign-array drift or butterfly-order bugs.
static void test_wht_forward_reference_equality() {
    // Use a non-trivial input with varied magnitudes and signs.
    std::vector<float> x_prod(128), x_ref(128);
    for (int i = 0; i < 128; i++) {
        const float v = std::sin((float)(i * 3 + 1)) * 0.7f + (float)(i % 5) * 0.03f;
        x_prod[i] = v;
        x_ref[i]  = v;
    }
    ea_wht_forward(x_prod.data(), 128, nullptr);
    ref_wht_forward(x_ref.data(), 128, nullptr);
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(x_prod[i], x_ref[i], 1e-6f, "WHT reference mismatch at %d", i);
    }
}

// Test: scale_inv path — ea_wht_forward with scale_inv must match reference
// with the same scale_inv. Exercises the per-channel scaling branch.
static void test_wht_forward_scale_inv() {
    std::vector<float> scale(128);
    for (int i = 0; i < 128; i++) {
        scale[i] = 0.5f + 0.01f * (float)((i * 7) % 100);  // 0.5 .. 1.49
    }
    std::vector<float> x_prod(128), x_ref(128);
    for (int i = 0; i < 128; i++) {
        const float v = std::cos((float)(i * 2 + 3)) * 0.5f;
        x_prod[i] = v;
        x_ref[i]  = v;
    }
    ea_wht_forward(x_prod.data(), 128, scale.data());
    ref_wht_forward(x_ref.data(), 128, scale.data());
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(x_prod[i], x_ref[i], 1e-5f, "WHT scale_inv mismatch at %d", i);
    }
}

// Test: WHT is linear — WHT(a*x) = a * WHT(x)
static void test_wht_forward_linearity() {
    std::vector<float> x(128);
    for (int i = 0; i < 128; i++) x[i] = (float)(i + 1) * 0.01f;

    std::vector<float> x_scaled = x;
    for (int i = 0; i < 128; i++) x_scaled[i] *= 3.0f;

    ea_wht_forward(x.data(), 128, nullptr);
    ea_wht_forward(x_scaled.data(), 128, nullptr);

    // x_scaled should be 3x x
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(x_scaled[i], 3.0f * x[i], 1e-4f, "WHT linearity failed at %d", i);
    }
}

// Test: WHT-forward should change the vector (not identity)
static void test_wht_forward_changes_vector() {
    std::vector<float> x(128);
    for (int i = 0; i < 128; i++) x[i] = (float)(i + 1) * 0.01f;

    std::vector<float> x_orig = x;
    ea_wht_forward(x.data(), 128, nullptr);

    // At least some elements should differ
    bool differs = false;
    for (int i = 0; i < 128; i++) {
        if (std::fabs(x[i] - x_orig[i]) > 1e-6f) {
            differs = true;
            break;
        }
    }
    ASSERT_TRUE(differs, "WHT-forward should change the vector");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Expected Attention Phase 1 + 3 Tests ===\n\n");

    test_query_buffer_circular();
    printf("  test_query_buffer_circular: done\n");

    test_mean();
    printf("  test_mean: done\n");

    test_covariance();
    printf("  test_covariance: done\n");

    test_rope_matrix_pos0();
    printf("  test_rope_matrix_pos0: done\n");

    test_rope_orthogonal();
    printf("  test_rope_orthogonal: done\n");

    test_scores_uniform();
    printf("  test_scores_uniform: done\n");

    test_scores_alignment();
    printf("  test_scores_alignment: done\n");

    test_compress_protects_sink_local();
    printf("  test_compress_protects_sink_local: done\n");

    test_compress_zero_ratio();
    printf("  test_compress_zero_ratio: done\n");

    test_pipeline_smoke();
    printf("  test_pipeline_smoke: done\n");

    test_mean_only_mode();
    printf("  test_mean_only_mode: done\n");

    test_vnorm_rescaling();
    printf("  test_vnorm_rescaling: done\n");

    // Phase 3 tests
    printf("\n--- Phase 3 Tests ---\n\n");

    test_rope_neox_pos0();
    printf("  test_rope_neox_pos0: done\n");

    test_rope_neox_orthogonal();
    printf("  test_rope_neox_orthogonal: done\n");

    test_rope_neox_differs_from_interleaved();
    printf("  test_rope_neox_differs_from_interleaved: done\n");

    test_rope_neox_pairing();
    printf("  test_rope_neox_pairing: done\n");

    test_ea_scoring_mean_only();
    printf("  test_ea_scoring_mean_only: done\n");

    test_ea_scoring_neox_mode();
    printf("  test_ea_scoring_neox_mode: done\n");

    test_head_aggregation_max();
    printf("  test_head_aggregation_max: done\n");

    // TurboQuant WHT tests
    printf("\n--- TurboQuant WHT Tests ---\n\n");

    test_wht_forward_identity();
    printf("  test_wht_forward_identity: done\n");

    test_wht_forward_reference_equality();
    printf("  test_wht_forward_reference_equality: done\n");

    test_wht_forward_scale_inv();
    printf("  test_wht_forward_scale_inv: done\n");

    test_wht_forward_linearity();
    printf("  test_wht_forward_linearity: done\n");

    test_wht_forward_changes_vector();
    printf("  test_wht_forward_changes_vector: done\n");

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
