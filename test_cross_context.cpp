#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// ---------------------------------------------------------------------------
// Helper: verify 10x10 result buffer
// ---------------------------------------------------------------------------
static bool verify_result(const float * data, const char * label) {
    bool copy_ok = true;
    bool rest_ok = true;
    int errors = 0;
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            float val = data[y * 10 + x];
            if (val != 99.0f) {
                copy_ok = false;
                if (errors < 5) {
                    printf("  MISMATCH at (%d,%d): expected 99.0, got %.2f\n", x, y, val);
                }
                ++errors;
            }
        }
    }
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            if (y < 5 && x < 5) continue;
            float val = data[y * 10 + x];
            if (val != 42.0f) {
                rest_ok = false;
                if (errors < 10) {
                    printf("  REST MISMATCH at (%d,%d): expected 42.0, got %.2f\n", x, y, val);
                }
                ++errors;
            }
        }
    }
    printf("  [%s] copy region OK: %s | rest OK: %s\n",
           label,
           copy_ok ? "YES" : "NO",
           rest_ok ? "YES" : "NO");
    return copy_ok && rest_ok;
}

// ---------------------------------------------------------------------------
// Test 1: simple ggml_graph_compute_with_ctx
// ---------------------------------------------------------------------------
static bool test_simple_path() {
    printf("--- Test 1: ggml_graph_compute_with_ctx ---\n");

    size_t mem_size_ctx1 = 1024 * 1024;
    struct ggml_init_params params1 = {
        .mem_size   = mem_size_ctx1,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx1 = ggml_init(params1);
    struct ggml_tensor  * t1   = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 10, 10);
    ggml_set_name(t1, "t1");

    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(ctx1, buft_cpu);
    if (!buf1) { fprintf(stderr, "alloc failed\n"); return false; }

    std::vector<float> data42(10 * 10, 42.0f);
    ggml_backend_tensor_set(t1, data42.data(), 0, data42.size() * sizeof(float));

    size_t mem_size_ctx2 = 1024 * 1024;
    struct ggml_init_params params2 = {
        .mem_size   = mem_size_ctx2,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    struct ggml_context * ctx2 = ggml_init(params2);
    struct ggml_tensor * v1 = ggml_view_2d(ctx2, t1, 5, 5, t1->nb[1], 0);
    ggml_set_name(v1, "v1");
    struct ggml_tensor * s1 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 5, 5);
    ggml_set_name(s1, "s1");
    std::vector<float> data99(5 * 5, 99.0f);
    memcpy(s1->data, data99.data(), data99.size() * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx2);
    struct ggml_tensor * cpy = ggml_cpy(ctx2, s1, v1);
    ggml_build_forward_expand(gf, cpy);

    enum ggml_status status = ggml_graph_compute_with_ctx(ctx2, gf, 1);
    printf("  status = %d (%s)\n", status,
           status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

    std::vector<float> result(10 * 10);
    ggml_backend_tensor_get(t1, result.data(), 0, result.size() * sizeof(float));

    bool ok = (status == GGML_STATUS_SUCCESS) && verify_result(result.data(), "simple");

    ggml_free(ctx2);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx1);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: backend scheduler path (used in llama.cpp)
// We create s1 on a backend buffer so ggml_backend_tensor_set works.
// The key question: does sched_alloc_graph reallocate v1 (the cross-ctx view)?
// ---------------------------------------------------------------------------
static bool test_scheduler_path() {
    printf("\n--- Test 2: ggml_backend_sched ---\n");

    // -- ctx1 with backend-allocated t1 --
    size_t mem_size_ctx1 = 1024 * 1024;
    struct ggml_init_params params1 = {
        .mem_size   = mem_size_ctx1,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx1 = ggml_init(params1);
    struct ggml_tensor  * t1   = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 10, 10);
    ggml_set_name(t1, "t1");

    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(ctx1, buft_cpu);
    if (!buf1) { fprintf(stderr, "alloc failed\n"); return false; }

    std::vector<float> data42(10 * 10, 42.0f);
    ggml_backend_tensor_set(t1, data42.data(), 0, data42.size() * sizeof(float));
    printf("  t1 filled with 42.0 (on backend buffer)\n");

    // -- ctx2: no_alloc=true so we can put s1 on a backend buffer too --
    size_t mem_size_ctx2 = 1024 * 1024;
    struct ggml_init_params params2 = {
        .mem_size   = mem_size_ctx2,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx2 = ggml_init(params2);

    struct ggml_tensor * v1 = ggml_view_2d(ctx2, t1, 5, 5, t1->nb[1], 0);
    ggml_set_name(v1, "v1");
    struct ggml_tensor * s1 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 5, 5);
    ggml_set_name(s1, "s1");

    // Allocate ctx2 tensors on backend buffer
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, buft_cpu);
    if (!buf2) { fprintf(stderr, "alloc ctx2 failed\n"); return false; }
    printf("  ctx2 tensors allocated on backend buffer\n");

    // Fill s1
    std::vector<float> data99(5 * 5, 99.0f);
    ggml_backend_tensor_set(s1, data99.data(), 0, data99.size() * sizeof(float));
    printf("  s1 filled with 99.0\n");

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx2);
    struct ggml_tensor * cpy = ggml_cpy(ctx2, s1, v1);
    ggml_build_forward_expand(gf, cpy);

    // -- scheduler --
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "backend init failed\n"); return false; }

    ggml_backend_buffer_type_t bufts[1] = { buft_cpu };
    ggml_backend_t backends[1] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, bufts, 1, 1024, /* parallel */ false, /* op_offload */ false);
    if (!sched) { fprintf(stderr, "sched init failed\n"); return false; }

    bool reserve_ok = ggml_backend_sched_reserve(sched, gf);
    printf("  sched_reserve = %s\n", reserve_ok ? "OK" : "FAIL");
    if (!reserve_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_backend_buffer_free(buf2);
        ggml_free(ctx2);
        ggml_backend_buffer_free(buf1);
        ggml_free(ctx1);
        return false;
    }

    bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    printf("  sched_alloc_graph = %s\n", alloc_ok ? "OK" : "FAIL");
    if (!alloc_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_backend_buffer_free(buf2);
        ggml_free(ctx2);
        ggml_backend_buffer_free(buf1);
        ggml_free(ctx1);
        return false;
    }

    // After sched_alloc_graph, refresh pointers and re-upload data
    v1 = ggml_get_tensor(ctx2, "v1");
    s1 = ggml_get_tensor(ctx2, "s1");

    printf("  v1->data after sched_alloc = %p\n", v1->data);
    printf("  t1->data after sched_alloc = %p\n", t1->data);
    printf("  v1->buffer after alloc = %s\n",
           v1->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(v1->buffer)) : "NULL");

    ggml_backend_tensor_set(s1, data99.data(), 0, data99.size() * sizeof(float));

    enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    printf("  sched_graph_compute status = %d (%s)\n", status,
           status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

    // read back t1
    std::vector<float> result(10 * 10);
    ggml_backend_tensor_get(t1, result.data(), 0, result.size() * sizeof(float));

    bool ok = (status == GGML_STATUS_SUCCESS) && verify_result(result.data(), "sched");

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buf2);
    ggml_free(ctx2);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx1);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: same as Test 2 but with the full llama.cpp pattern:
//         We do NOT pre-allocate ctx2 with alloc_ctx_tensors_from_buft;
//         we let the scheduler handle everything.
//         This is the pattern used in llama-graph.cpp where cross-context
//         views may be passed to the scheduler.
// ---------------------------------------------------------------------------
static bool test_scheduler_without_prealloc() {
    printf("\n--- Test 3: sched without prealloc (llama.cpp pattern) ---\n");

    // -- ctx1 with backend-allocated t1 --
    size_t mem_size_ctx1 = 1024 * 1024;
    struct ggml_init_params params1 = {
        .mem_size   = mem_size_ctx1,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx1 = ggml_init(params1);
    struct ggml_tensor  * t1   = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 10, 10);
    ggml_set_name(t1, "t1");

    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(ctx1, buft_cpu);
    if (!buf1) { fprintf(stderr, "alloc failed\n"); return false; }

    void * t1_data_before = t1->data;
    std::vector<float> data42(10 * 10, 42.0f);
    ggml_backend_tensor_set(t1, data42.data(), 0, data42.size() * sizeof(float));
    printf("  t1->data before = %p (buf base = %p)\n", t1_data_before, ggml_backend_buffer_get_base(buf1));

    // -- ctx2: normal alloc (tensors live in ctx mem, NOT backend buffer) --
    size_t mem_size_ctx2 = 1024 * 1024;
    struct ggml_init_params params2 = {
        .mem_size   = mem_size_ctx2,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    struct ggml_context * ctx2 = ggml_init(params2);

    struct ggml_tensor * v1 = ggml_view_2d(ctx2, t1, 5, 5, t1->nb[1], 0);
    ggml_set_name(v1, "v1");
    struct ggml_tensor * s1 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 5, 5);
    ggml_set_name(s1, "s1");

    // Fill s1 using memcpy (it's in ctx mem, no backend buffer yet)
    std::vector<float> data99(5 * 5, 99.0f);
    memcpy(s1->data, data99.data(), data99.size() * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf, ggml_cpy(ctx2, s1, v1));

    // -- scheduler --
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t backends[1] = { backend };
    ggml_backend_buffer_type_t bufts[1] = { buft_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, 1024, false, false);

    bool reserve_ok = ggml_backend_sched_reserve(sched, gf);
    printf("  sched_reserve = %s\n", reserve_ok ? "OK" : "FAIL");
    if (!reserve_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx2);
        ggml_backend_buffer_free(buf1);
        ggml_free(ctx1);
        return false;
    }

    bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    printf("  sched_alloc_graph = %s\n", alloc_ok ? "OK" : "FAIL");
    if (!alloc_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx2);
        ggml_backend_buffer_free(buf1);
        ggml_free(ctx1);
        return false;
    }

    // Refresh pointers
    v1 = ggml_get_tensor(ctx2, "v1");
    s1 = ggml_get_tensor(ctx2, "s1");

    void * v1_data_after = v1->data;
    printf("  v1->data after sched_alloc = %p\n", v1_data_after);
    printf("  t1->data after sched_alloc = %p\n", t1->data);
    printf("  v1->buffer after alloc     = %s\n",
           v1->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(v1->buffer)) : "NULL");

    bool pointer_unchanged = (v1_data_after == t1_data_before);
    printf("  View still points to t1 buffer: %s\n", pointer_unchanged ? "YES" : "NO");

    // After alloc_graph, check if s1 got a backend buffer.
    // In llama.cpp, input tensors are usually pre-allocated on backend buffers.
    // If the scheduler did NOT allocate one, we must use memcpy on ctx memory.
    if (s1->buffer) {
        printf("  s1 has backend buffer after alloc -> using ggml_backend_tensor_set\n");
        ggml_backend_tensor_set(s1, data99.data(), 0, data99.size() * sizeof(float));
    } else {
        printf("  s1 has NO backend buffer after alloc -> using memcpy on ctx mem\n");
        memcpy(s1->data, data99.data(), data99.size() * sizeof(float));
    }

    enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    printf("  sched_graph_compute status = %d (%s)\n", status,
           status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

    std::vector<float> result(10 * 10);
    ggml_backend_tensor_get(t1, result.data(), 0, result.size() * sizeof(float));

    bool ok = (status == GGML_STATUS_SUCCESS) && verify_result(result.data(), "no_prealloc");

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx2);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx1);
    return ok;
}

// ---------------------------------------------------------------------------
int main() {
    printf("=== Cross-Context ggml_cpy Test ===\n");

    bool ok1 = test_simple_path();
    bool ok2 = test_scheduler_path();
    bool ok3 = test_scheduler_without_prealloc();

    printf("\n=== SUMMARY ===\n");
    printf("Test 1 (simple ctx compute):       %s\n", ok1 ? "PASS" : "FAIL");
    printf("Test 2 (sched with prealloc):        %s\n", ok2 ? "PASS" : "FAIL");
    printf("Test 3 (sched without prealloc):     %s\n", ok3 ? "PASS" : "FAIL");

    if (ok1 && ok2 && ok3) {
        printf("\n>>> Cross-Context ggml_cpy WORKS in all tested paths <<<\n");
    } else {
        printf("\n>>> Cross-Context ggml_cpy HAS ISSUES in at least one path <<<\n");
    }

    return (ok1 && ok2 && ok3) ? 0 : 1;
}
