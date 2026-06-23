#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

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
// Test 1: basic ggml_cpy on CUDA (single context, no cross-context)
//         Verifies that CUDA cpy itself works.
// ---------------------------------------------------------------------------
static bool test_basic_cpy_cuda() {
    printf("--- Test 1: basic ggml_cpy on CUDA (single context) ---\n");

    int device = 0;
    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) { fprintf(stderr, "CUDA backend init failed\n"); return false; }

    size_t mem_size = 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 10);
    ggml_set_name(t1, "t1");
    struct ggml_tensor * s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5);
    ggml_set_name(s1, "s1");
    struct ggml_tensor * v1 = ggml_view_2d(ctx, t1, 5, 5, t1->nb[1], 0);
    ggml_set_name(v1, "v1");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return false; }

    std::vector<float> data42(10 * 10, 42.0f);
    ggml_backend_tensor_set(t1, data42.data(), 0, data42.size() * sizeof(float));

    std::vector<float> data99(5 * 5, 99.0f);
    ggml_backend_tensor_set(s1, data99.data(), 0, data99.size() * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, s1, v1));

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    printf("  status = %d (%s)\n", status,
           status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

    std::vector<float> result(10 * 10);
    ggml_backend_tensor_get(t1, result.data(), 0, result.size() * sizeof(float));

    bool ok = (status == GGML_STATUS_SUCCESS) && verify_result(result.data(), "basic_cuda");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: backend scheduler path on CUDA with preallocated cross-context view
// ---------------------------------------------------------------------------
static bool test_scheduler_path_cuda() {
    printf("\n--- Test 2: ggml_backend_sched (CUDA, prealloc cross-ctx view) ---\n");

    int device = 0;
    ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);

    // -- ctx1 with CUDA-allocated t1 --
    size_t mem_size_ctx1 = 1024 * 1024;
    struct ggml_init_params params1 = {
        .mem_size   = mem_size_ctx1,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx1 = ggml_init(params1);
    struct ggml_tensor  * t1   = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 10, 10);
    ggml_set_name(t1, "t1");

    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(ctx1, buft_cuda);
    if (!buf1) { fprintf(stderr, "alloc ctx1 failed\n"); return false; }

    std::vector<float> data42(10 * 10, 42.0f);
    ggml_backend_tensor_set(t1, data42.data(), 0, data42.size() * sizeof(float));
    printf("  t1 filled with 42.0 (on CUDA backend buffer)\n");

    // -- ctx2: no_alloc=true so we can put s1 on a CUDA backend buffer too --
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

    // Allocate ctx2 tensors on CUDA backend buffer
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, buft_cuda);
    if (!buf2) { fprintf(stderr, "alloc ctx2 failed\n"); return false; }
    printf("  ctx2 tensors allocated on CUDA backend buffer\n");

    // Fill s1
    std::vector<float> data99(5 * 5, 99.0f);
    ggml_backend_tensor_set(s1, data99.data(), 0, data99.size() * sizeof(float));
    printf("  s1 filled with 99.0\n");

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx2);
    struct ggml_tensor * cpy = ggml_cpy(ctx2, s1, v1);
    ggml_build_forward_expand(gf, cpy);

    // -- scheduler with CUDA backend + CPU fallback --
    ggml_backend_t backend_cuda = ggml_backend_cuda_init(device);
    if (!backend_cuda) { fprintf(stderr, "CUDA backend init failed\n"); return false; }
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    if (!backend_cpu) { fprintf(stderr, "CPU backend init failed\n"); return false; }
    printf("  CUDA backend initialized (device %d)\n", device);

    ggml_backend_buffer_type_t bufts[2] = { buft_cuda, ggml_backend_cpu_buffer_type() };
    ggml_backend_t backends[2] = { backend_cuda, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, bufts, 2, 1024, /* parallel */ false, /* op_offload */ false);
    if (!sched) { fprintf(stderr, "sched init failed\n"); return false; }

    bool reserve_ok = ggml_backend_sched_reserve(sched, gf);
    printf("  sched_reserve = %s\n", reserve_ok ? "OK" : "FAIL");
    if (!reserve_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_cuda);
        ggml_backend_free(backend_cpu);
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
        ggml_backend_free(backend_cuda);
        ggml_backend_free(backend_cpu);
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

    bool ok = (status == GGML_STATUS_SUCCESS) && verify_result(result.data(), "sched_cuda");

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cuda);
    ggml_backend_free(backend_cpu);
    ggml_backend_buffer_free(buf2);
    ggml_free(ctx2);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx1);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: llama.cpp pattern — ctx2 tensors NOT pre-allocated,
//         let the scheduler handle everything.
// ---------------------------------------------------------------------------
static bool test_scheduler_without_prealloc_cuda() {
    printf("\n--- Test 3: sched without prealloc (llama.cpp pattern, CUDA) ---\n");

    int device = 0;
    ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(device);

    // -- ctx1 with CUDA-allocated t1 --
    size_t mem_size_ctx1 = 1024 * 1024;
    struct ggml_init_params params1 = {
        .mem_size   = mem_size_ctx1,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx1 = ggml_init(params1);
    struct ggml_tensor  * t1   = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 10, 10);
    ggml_set_name(t1, "t1");

    ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors_from_buft(ctx1, buft_cuda);
    if (!buf1) { fprintf(stderr, "alloc ctx1 failed\n"); return false; }

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

    // -- scheduler with CUDA backend + CPU fallback --
    ggml_backend_t backend_cuda = ggml_backend_cuda_init(device);
    if (!backend_cuda) { fprintf(stderr, "CUDA backend init failed\n"); return false; }
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    if (!backend_cpu) { fprintf(stderr, "CPU backend init failed\n"); return false; }
    printf("  CUDA backend initialized (device %d)\n", device);

    ggml_backend_t backends[2] = { backend_cuda, backend_cpu };
    ggml_backend_buffer_type_t bufts[2] = { buft_cuda, ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 1024, false, false);

    bool reserve_ok = ggml_backend_sched_reserve(sched, gf);
    printf("  sched_reserve = %s\n", reserve_ok ? "OK" : "FAIL");
    if (!reserve_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_cuda);
        ggml_backend_free(backend_cpu);
        ggml_free(ctx2);
        ggml_backend_buffer_free(buf1);
        ggml_free(ctx1);
        return false;
    }

    bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    printf("  sched_alloc_graph = %s\n", alloc_ok ? "OK" : "FAIL");
    if (!alloc_ok) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_cuda);
        ggml_backend_free(backend_cpu);
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

    bool ok = (status == GGML_STATUS_SUCCESS) && verify_result(result.data(), "no_prealloc_cuda");

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_cuda);
    ggml_backend_free(backend_cpu);
    ggml_free(ctx2);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx1);
    return ok;
}

// ---------------------------------------------------------------------------
int main() {
    printf("=== Cross-Context ggml_cpy Test (CUDA) ===\n");

    int n_devices = ggml_backend_cuda_get_device_count();
    printf("CUDA devices available: %d\n", n_devices);
    if (n_devices == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    char desc[256];
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    printf("Using device 0: %s\n\n", desc);

    bool ok1 = test_basic_cpy_cuda();
    bool ok2 = test_scheduler_path_cuda();
    bool ok3 = test_scheduler_without_prealloc_cuda();

    printf("\n=== SUMMARY ===\n");
    printf("Test 1 (basic cpy on CUDA):         %s\n", ok1 ? "PASS" : "FAIL");
    printf("Test 2 (sched with prealloc):       %s\n", ok2 ? "PASS" : "FAIL");
    printf("Test 3 (sched without prealloc):    %s\n", ok3 ? "PASS" : "FAIL");

    if (ok1 && ok2 && ok3) {
        printf("\n>>> Cross-Context ggml_cpy WORKS on CUDA in all tested paths <<<\n");
    } else {
        printf("\n>>> Cross-Context ggml_cpy HAS ISSUES on CUDA in at least one path <<<\n");
    }

    return (ok1 && ok2 && ok3) ? 0 : 1;
}
