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
// Main
// ============================================================================

int main() {
    printf("=== Expected Attention Phase 1 Tests ===\n\n");

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

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
