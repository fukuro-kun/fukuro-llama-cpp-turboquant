#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void quantize_row_turbo2_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo2_0(const void * x, float * y, long long k);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

/* Block sizes from ggml-common.h */
#define QK_TURBO2 128
#define QK_TURBO3 128
#define QK_TURBO4 128

#define BLOCK_SIZE_TURBO3 (sizeof(float) == 2 ? 0 : 0) /* placeholder, use sizeof */
/* Actual block sizes: turbo3_0 = 2+32+16 = 50, turbo4_0 = 2+2+64 = 68, turbo2_0 = 2+32 = 34 */
#define BYTES_PER_BLOCK_TURBO3 50
#define BYTES_PER_BLOCK_TURBO4 68
#define BYTES_PER_BLOCK_TURBO2 34

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        tests_failed++; \
    } else { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } \
} while(0)

/* Compute MSE, cosine similarity, and norms between input and output */
static void compute_metrics(const float * input, const float * output, int d,
                            float * mse, float * cosv, float * in_norm, float * out_norm) {
    float se = 0, dot = 0, ni = 0, no = 0;
    for (int i = 0; i < d; i++) {
        se   += (input[i] - output[i]) * (input[i] - output[i]);
        dot  += input[i] * output[i];
        ni   += input[i] * input[i];
        no   += output[i] * output[i];
    }
    *mse      = se / d;
    *cosv     = (ni > 0 && no > 0) ? dot / sqrtf(ni) / sqrtf(no) : 0;
    *in_norm  = sqrtf(ni);
    *out_norm = sqrtf(no);
}

/* Round-trip test helper: quantize → dequantize → inverse WHT → compare */
static void round_trip_test(const char * name, const float * input, int d,
                            void (*quant_fn)(const float *, void *, long long),
                            void (*dequant_fn)(const void *, float *, long long),
                            int block_bytes,
                            float * out_mse, float * out_cosv) {
    char * buf = (char *) calloc((d / 128) * block_bytes + 64, 1);
    float * output = (float *) calloc(d, sizeof(float));

    quant_fn(input, buf, d);
    dequant_fn(buf, output, d);
    /* Apply inverse WHT per 128-element group (matches quantization group size) */
    for (int off = 0; off + 128 <= d; off += 128) {
        turbo_cpu_fwht_inverse(output + off, 128);
    }

    float mse, cosv, in_norm, out_norm;
    compute_metrics(input, output, d, &mse, &cosv, &in_norm, &out_norm);

    printf("%s (d=%d): MSE=%.8f Cosine=%.6f InNorm=%.4f OutNorm=%.4f\n",
           name, d, mse, cosv, in_norm, out_norm);

    free(buf);
    free(output);
    *out_mse = mse;
    *out_cosv = cosv;
}

int main(void) {
    printf("=== TurboQuant C Round-Trip Test Suite ===\n\n");

    /* ===== Test Group 1: turbo3 with d=128 ===== */
    printf("--- turbo3_0 tests (d=128) ---\n");
    {
        const int d = 128;
        float input[128], mse, cosv;

        /* Test 1a: basis vector e0 */
        memset(input, 0, sizeof(input));
        input[0] = 1.0f;
        round_trip_test("turbo3 basis e0", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        ASSERT(cosv > 0.95f, "turbo3 basis e0 cosine > 0.95");
        ASSERT(mse < 0.01f, "turbo3 basis e0 MSE < 0.01");

        /* Test 1b: sin*10 (large norm) */
        for (int i = 0; i < d; i++) input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
        round_trip_test("turbo3 sin*10", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        ASSERT(cosv > 0.95f, "turbo3 sin*10 cosine > 0.95");

        /* Test 1c: random Gaussian (simulates real KV cache) */
        for (int i = 0; i < d; i++) input[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
        round_trip_test("turbo3 random", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        ASSERT(cosv > 0.90f, "turbo3 random cosine > 0.90");

        /* Test 1d: all zeros (edge case) */
        memset(input, 0, sizeof(input));
        round_trip_test("turbo3 zeros", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        ASSERT(mse < 0.001f, "turbo3 zeros MSE < 0.001");

        /* Test 1e: all ones (uniform) */
        for (int i = 0; i < d; i++) input[i] = 1.0f;
        round_trip_test("turbo3 ones", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        ASSERT(cosv > 0.90f, "turbo3 ones cosine > 0.90");
    }
    printf("\n");

    /* ===== Test Group 2: turbo4 with d=128 ===== */
    printf("--- turbo4_0 tests (d=128) ---\n");
    {
        const int d = 128;
        float input[128], mse, cosv;

        /* Test 2a: basis vector e0 */
        memset(input, 0, sizeof(input));
        input[0] = 1.0f;
        round_trip_test("turbo4 basis e0", input, d,
                        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0,
                        BYTES_PER_BLOCK_TURBO4, &mse, &cosv);
        ASSERT(cosv > 0.95f, "turbo4 basis e0 cosine > 0.95");

        /* Test 2b: cos*5 */
        for (int i = 0; i < d; i++) input[i] = cosf(i * 0.2f) * 5.0f;
        round_trip_test("turbo4 cos*5", input, d,
                        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0,
                        BYTES_PER_BLOCK_TURBO4, &mse, &cosv);
        ASSERT(cosv > 0.95f, "turbo4 cos*5 cosine > 0.95");

        /* Test 2c: random Gaussian */
        for (int i = 0; i < d; i++) input[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
        round_trip_test("turbo4 random", input, d,
                        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0,
                        BYTES_PER_BLOCK_TURBO4, &mse, &cosv);
        ASSERT(cosv > 0.92f, "turbo4 random cosine > 0.92");

        /* Test 2d: all zeros */
        memset(input, 0, sizeof(input));
        round_trip_test("turbo4 zeros", input, d,
                        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0,
                        BYTES_PER_BLOCK_TURBO4, &mse, &cosv);
        ASSERT(mse < 0.001f, "turbo4 zeros MSE < 0.001");
    }
    printf("\n");

    /* ===== Test Group 3: turbo2 with d=128 ===== */
    printf("--- turbo2_0 tests (d=128) ---\n");
    {
        const int d = 128;
        float input[128], mse, cosv;

        /* Test 3a: basis vector e0 */
        memset(input, 0, sizeof(input));
        input[0] = 1.0f;
        round_trip_test("turbo2 basis e0", input, d,
                        quantize_row_turbo2_0_ref, dequantize_row_turbo2_0,
                        BYTES_PER_BLOCK_TURBO2, &mse, &cosv);
        ASSERT(cosv > 0.85f, "turbo2 basis e0 cosine > 0.85");

        /* Test 3b: sin*10 */
        for (int i = 0; i < d; i++) input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
        round_trip_test("turbo2 sin*10", input, d,
                        quantize_row_turbo2_0_ref, dequantize_row_turbo2_0,
                        BYTES_PER_BLOCK_TURBO2, &mse, &cosv);
        ASSERT(cosv > 0.85f, "turbo2 sin*10 cosine > 0.85");

        /* Test 3c: random Gaussian */
        for (int i = 0; i < d; i++) input[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
        round_trip_test("turbo2 random", input, d,
                        quantize_row_turbo2_0_ref, dequantize_row_turbo2_0,
                        BYTES_PER_BLOCK_TURBO2, &mse, &cosv);
        ASSERT(cosv > 0.80f, "turbo2 random cosine > 0.80");
    }
    printf("\n");

    /* ===== Test Group 4: Multi-block (d=256, d=512) ===== */
    printf("--- multi-block tests (d=256, d=512) ---\n");
    {
        /* d=256 = 2 blocks of 128 */
        const int d = 256;
        float input[256], mse, cosv;
        for (int i = 0; i < d; i++) input[i] = sinf(i * 0.05f) * 3.0f;
        round_trip_test("turbo3 d=256", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        ASSERT(cosv > 0.90f, "turbo3 d=256 cosine > 0.90");

        /* d=512 = 4 blocks of 128 */
        const int d2 = 512;
        float input2[512], mse2, cosv2;
        for (int i = 0; i < d2; i++) input2[i] = cosf(i * 0.03f) * 2.0f;
        round_trip_test("turbo4 d=512", input2, d2,
                        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0,
                        BYTES_PER_BLOCK_TURBO4, &mse2, &cosv2);
        ASSERT(cosv2 > 0.90f, "turbo4 d=512 cosine > 0.90");
    }
    printf("\n");

    /* ===== Test Group 5: NaN/Inf edge cases ===== */
    printf("--- edge case tests ---\n");
    {
        const int d = 128;
        float input[128], mse, cosv;

        /* NaN input — should not crash, output should be finite or zero */
        for (int i = 0; i < d; i++) input[i] = (i == 0) ? NAN : 0.0f;
        round_trip_test("turbo3 NaN e0", input, d,
                        quantize_row_turbo3_0_ref, dequantize_row_turbo3_0,
                        BYTES_PER_BLOCK_TURBO3, &mse, &cosv);
        /* No assertion on values — just verify it doesn't crash */
        tests_run++; tests_passed++;
        printf("  PASS: turbo3 NaN did not crash\n");

        /* Inf input */
        for (int i = 0; i < d; i++) input[i] = (i == 0) ? INFINITY : 0.0f;
        round_trip_test("turbo4 Inf e0", input, d,
                        quantize_row_turbo4_0_ref, dequantize_row_turbo4_0,
                        BYTES_PER_BLOCK_TURBO4, &mse, &cosv);
        tests_run++; tests_passed++;
        printf("  PASS: turbo4 Inf did not crash\n");
    }
    printf("\n");

    /* ===== Summary ===== */
    printf("=== Test Summary ===\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("=== %s ===\n", tests_failed == 0 ? "ALL PASSED" : "FAILURES DETECTED");

    return tests_failed == 0 ? 0 : 1;
}
