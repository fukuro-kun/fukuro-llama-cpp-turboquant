# Vulkan Commit-Kategorisierung (2026-06-21)

61 Vulkan-Commits zwischen unserem Stand und upstream/master.

## Kategorisierung

### 🔴 PERF — Performance-relevant (FA, MatMul, Buffer, Dispatch)

| Hash | PR | Beschreibung | AMD-System-Relevanz |
|------|-----|-------------|---------------|
| `75f3bc94e` | #20797 | FA DP4A shader for quantized KV cache | 🔴 Hoch — turbo3 FA Performance |
| `f9f33654a` | #21751 | Coalesce Q4_K/Q5_K scale loads | 🟡 Mittel — Q4_K_M Modell |
| `05e141a6b` | #21753 | Asymmetric FA in coopmat2 path | 🟡 Mittel — coopmat2 nicht auf AMD-System |
| `dd9280a66` | #22589 | Asymmetric FA in scalar/mmq/coopmat1 paths | 🔴 Hoch — AMD-System nutzt coopmat1! |
| `706fbd8ab` | #22693 | Check shared memory size for mmq shaders | 🟡 Mittel — Robustheit |
| `dbe7901ca` | #23005 | Fix matmul integer pipeline selection | 🔴 Hoch — könnte PP-Klippe erklären! |
| `c6e408837` | #22887 | MUL_MAT_VEC 4 K per iteration for F16/32 | 🟡 Mittel — TG-Performance |
| `19620004f` | #23056 | Block-load Q3_K/Q6_K block data | 🟡 Mittel — Q3_K/Q6_K Performance |
| `bef69f130` | #23376 | Reduce host memory lock contention | 🔴 Hoch — UMA lock contention! |
| `55ac0909e` | #23641 | Don't hold device mutex while compiling pipelines | 🔴 Hoch — Pipeline-Compile-Hang! |
| `b4e3dc613` | #24123 | v_dot2_f32_f16 in matmul and FA | 🟡 Mittel — coopmat1 Performance |
| `c74759a24` | #23991 | cm2 decode_vector for mul_mat_id B loads | 🟡 Niedrig — coopmat2 |
| `d6d0ce821` | #24287 | Reduce iq1 shared memory for mul_mm | 🟢 Niedrig — iq1 nicht primär |
| `fdc3db9b6` | #23973 | Fast path for contiguous buffer transfers | 🔴 Hoch — Buffer-Transfer-Performance! |
| `b36eefc1b` | #23541 | GL_NV_cooperative_matrix_decode_vector | 🟢 Niedrig — NVIDIA-spezifisch |

### 🔴 CORRECT — Korrektheits-Fixes

| Hash | PR | Beschreibung | AMD-System-Relevanz |
|------|-----|-------------|---------------|
| `19821178b` | #21865 | Add barrier after writetimestamp | 🟢 Niedrig |
| `7c48fb81c` | #23665 | Fix wrong index variable in inner loop | 🔴 Hoch — könnte Hang verursachen! |
| `91eb8f4fa` | #23667 | Fix memory logger unsafe iterator access | 🟡 Mittel — Stabilität |
| `6e093b80e` | #23420 | FA support for BFloat16 KV cache | 🟢 Niedrig — AMD-System hat kein BF16 |
| `3e7bd4f39` | #23770 | Pipeline barriers for memcpy read operations | 🔴 Hoch — UMA Korrektheit! |
| `558e221b7` | #24326 | Record actual memory properties during buffer creation | 🔴 Hoch — Korrektheit (getestet, kein Effekt) |
| `5a69c9743` | #24186 | Check coopmat2 features before reporting support | 🟡 Mittel — Stabilität |
| `6d57a49a7` | #22760 | Fix spv shadowing | 🟢 Niedrig |

### 🔴 UMA — UMA/APU-spezifisch

| Hash | PR | Beschreibung | AMD-System-Relevanz |
|------|-----|-------------|---------------|
| `4d8cc0c56` | #22455 | Avoid preferring transfer queue on AMD UMA | 🔴 Hoch — AMD-System ist AMD UMA! |
| `32120c10e` | #22930 | Prefer host-visible memory on UMA devices | 🔴 Hoch — AMD-System ist UMA! |
| `e95dae18d` | #24086 | Remove padding and multiple D2D copies for MTP | 🟡 Mittel — MTP Performance |

### 🟡 NEWOP — Neue Operationen

| Hash | PR | Beschreibung | AMD-System-Relevanz |
|------|-----|-------------|---------------|
| `edd4d9bca` | #21029 | FA dequant for q4_1, q5_0, q5_1, iq4_nl | 🟡 Mittel — iq4_nl ist unser Modell! |
| `7b6912533` | #21539 | Support Q1_0 | 🟢 Niedrig |
| `6a6780a23` | #21455 | Support GGML_TYPE_NVFP4 | 🟢 Niedrig |
| `1f30ac0ce` | #21572 | Programmatically add RoundingModeRTE | 🟡 Mittel — FP-Genauigkeit |
| `b3d758750` | #21713 | Optimize im2col | 🟢 Niedrig |
| `82209efb7` | #22177 | Support F16 OP_FILL | 🟢 Niedrig |
| `660b1b4bd` | #22514 | Get/set tensor 2d functions | 🟢 Niedrig |
| `3fbadb06d` | #22653 | Fuse SSM_CONV + BIAS + SILU | 🟢 Niedrig |
| `7ba22c6a0` | #22637 | Support unaligned tensors for ROPE | 🟡 Mittel — Robustheit |
| `fcae601e4` | #22677 | cpy bf16 -> f32 pipelines | 🟢 Niedrig |
| `acd604fb2` | #22685 | Optimize IM2COL shader | 🟢 Niedrig |
| `47c0eda9d` | #22855 | Fuse snake activation | 🟢 Niedrig |
| `7799d31e6` | #22855 | Optimize conv2d, coopmat1 support | 🟢 Niedrig |
| `837bb6b44` | #23298 | REPEAT op for f16 to f16 | 🟢 Niedrig |
| `48e7078ee` | #23687 | Fast path for walsh-hadamard transform | 🟡 Mittel — TurboQuant WHT! |
| `e82beaa60` | #23964 | fwht support for Intel with shmem reduction | 🟢 Niedrig — Intel |
| `1a7718b4c` | #24215 | Non-contig unary/glu ops | 🟢 Niedrig |
| `9dbc6621a` | #24579 | More CONCAT types | 🟢 Niedrig |
| `ad39ccaa1` | #24425 | col2im_1d op | 🟢 Niedrig |
| `d5fb10429` | #24581 | gated_delta_net with S_v=16 | 🟢 Niedrig |
| `255582687` | #22673 | MTP Support | 🟡 Mittel — wir haben eigenes MTP |

### 🟢 CI/BUILD — CI, Build-Fixes

| Hash | PR | Beschreibung |
|------|-----|-------------|
| `2a619f6fb` | #20904 | Output error string for errno on fork failure |
| `8a132faaa` | #21605 | Unify type macros to use Vx instead of _VECx |
| `698d19b93` | #21918 | Improve SPIR-V headers detection |
| `a6d6183db` | #22009 | CMake check for SPIRV-Headers |
| `604990613` | #23144 | Removed duplicate #include |
| `95405ac65` | #23215 | Fix Windows find_package of SPIRV-Headers |
| `f8c0a19d4` | #23175 | Removed unused functions |
| `1af154a76` | #24306 | Medium matmul tile on Asahi Linux |
| `4c6595503` | #24479 | ifdef eMesaHoneykrisp (build fix) |
| `d5376cf5d` | #24595 | Fix vulkan docker images |
| `6dcd824fc` | #22621 | Delete dead GGML_VK_MAX_NODES def |

### 🟢 OTHER — Andere

| Hash | PR | Beschreibung |
|------|-----|-------------|
| `ecce0087d` | #21549 | Detect streaming state in reasoning content blocks |
| `d6f303004` | #19378 | Backend-agnostic tensor parallelism |
| `ef93e98d0` | #22461 | Fix Windows performance regression on Intel GPU |

---

## Cherry-Pick-Empfehlung

### Gruppe A: UMA + Korrektheit (höchste Priorität)
1. `4d8cc0c56` #22455 — Avoid transfer queue on AMD UMA
2. `3e7bd4f39` #23770 — Pipeline barriers for memcpy on UMA
3. `558e221b7` #24326 — Record actual memory properties
4. `32120c10e` #22930 — Prefer host-visible on UMA

### Gruppe B: Bugfixes (hohe Priorität)
5. `7c48fb81c` #23665 — Fix wrong index variable in inner loop
6. `91eb8f4fa` #23667 — Fix memory logger unsafe iterator
7. `dbe7901ca` #23005 — Fix matmul integer pipeline selection

### Gruppe C: Performance (hohe Priorität)
8. `55ac0909e` #23641 — Don't hold mutex while compiling pipelines
9. `bef69f130` #23376 — Reduce host memory lock contention
10. `fdc3db9b6` #23973 — Fast path for contiguous buffer transfers

### Gruppe D: FA/MatMul (mittlere Priorität)
11. `dd9280a66` #22589 — Asymmetric FA in scalar/mmq/coopmat1
12. `75f3bc94e` #20797 — FA DP4A shader for quantized KV
13. `c6e408837` #22887 — MUL_MAT_VEC 4 K per iteration
14. `b4e3dc613` #24123 — v_dot2_f32_f16 in matmul and FA

### Gruppe E: TurboQuant-relevant
15. `48e7078ee` #23687 — Fast WHT path
16. `edd4d9bca` #21029 — FA dequant for iq4_nl
17. `19620004f` #23056 — Block-load Q3_K/Q6_K

**Strategie:** Gruppe A zuerst (UMA-spezifisch), dann Gruppe B (Bugfixes), dann Gruppe C (Performance). Nach jeder Gruppe: Build auf AMD-System, Cache löschen, Benchmark.
