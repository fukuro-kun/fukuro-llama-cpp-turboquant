# E4B + MTP auf 8GB Single-GPU: Crash-Kombination

**Datum:** 2026-07-11
**System:** Dev-Host (RTX 3070 Laptop, 8GB VRAM, CUDA)
**Modell:** `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (4.0GB) + `mtp-gemma-4-E4B-it-Q4_0.gguf` (57MB Draft)

## Problem

E4B mit MTP-Draft lässt sich auf 8GB Single-GPU (Ampere) nicht starten.
Drei verschiedene Crash-Pfade je nach Konfiguration:

### 1. FlashAttention + turbo4/turbo3 KV-Cache → `fattn.cu:110` fatal error

```
--flash-attn auto --cache-type-k turbo4 --cache-type-v turbo3 -ngl 99 -c 8192
```

```
cmd_child_to_router:error:/home/fukuro/git/fukuro-llama-cpp-turboquant/ggml/src/ggml-cuda/fattn.cu:110: fatal error
```

FlashAttention crasht beim Versuch die PLE-Architektur (Per-Layer Embeddings)
zu verarbeiten. E4B hat unterschiedlich große V-Embeddings pro Layer — FA
kommt damit nicht klar.

### 2. Ohne FlashAttention, f16 KV-Cache → OOM

```
--flash-attn off --cache-type-k f16 --cache-type-v f16 -ngl 99 -c 8192
```

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2493.32 MiB on device 0: cudaMalloc failed: out of memory
```

Ohne turbo-Kompression braucht der KV-Cache zu viel VRAM. 4.0GB Modell +
2.5GB KV-Cache + Draft übersteigen 8GB.

### 3. Ohne FlashAttention, turbo KV-Cache → `GGML_ASSERT` crash

```
--flash-attn off --cache-type-k turbo4 --cache-type-v turbo3 -ngl 60 -c 4096
```

turbo KV-Cache **erfordert** FlashAttention (wird automatisch enabled):
```
W llama_init_from_model: turbo cache types require flash_attn — enabling automatically
```
Dann crasht FA wie in Fall 1, oder bei `-ngl 60` (partieller CPU-Offload):
```
cmd_child_to_router:error:/home/fukuro/git/fukuro-llama-cpp-turboquant/ggml/src/ggml.c:3692: GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed
```

Layer-Sharing (PLE: Layer 0-3 teilen sich mit Layer 40-41) + turbo-FA
→ Tensor-Shape-Mismatch im CPU-Offload-Pfad.

## Root Cause

E4B's Full-Attention-Layer verwenden `head_dim=512` (`attention.key_length: 512`,
`attention.value_length: 512`). Die SWA-Layer verwenden `head_dim=256`.
Das SWA-Pattern ist 5 SWA + 1 full (jeder 6. Layer ist full-attention).

Der MMA FA-Kernel (`ggml_cuda_flash_attn_ext_mma_f16`) hat keine Template-Instanz
für DKQ=512 — `switch_ncols2<512, 512>` trifft `GGML_ABORT("fatal error")` bei
`fattn.cu:110` weil DKQ > 256 nicht im switch-case behandelt wird.

Der TILE-Kernel hatte ebenfalls eine Lücke: `launch_fattn_tile_switch_ncols2`
hatte keinen Fallback für DV > 256 wenn `use_gqa_opt=false` (kein Mask, z.B.
während Model-Loading/Warmup). Der `DV <= 256` Guard übersprang den
ncols2=1 Fallback → `GGML_ABORT`.

Zusätzlich fehlte eine TILE-Config für 512/512 bei ncols=2, was für den MTP-Draft
(gqa_ratio=2, 4 heads / 2 KV heads) benötigt wird.

## Fix (2026-07-13)

Drei Commits lösen das Problem:

1. **`fattn.cu`**: Route `head_dim=512` zu `BEST_FATTN_KERNEL_TILE` vor dem
   `turing_mma_available` Check. TILE hat bereits eine `dkq512-dv512` Instanz.

2. **`fattn-tile.cuh`**: Fallback für DV > 256 ohne Mask. GQA-Grouping ist eine
   strukturelle Eigenschaft (unabhängig vom Mask):
   - `gqa_ratio % 4 == 0` → ncols2=4 (E4B main model, gqa_ratio=4)
   - `gqa_ratio % 2 == 0` → ncols2=2 (MTP draft, gqa_ratio=2)
   - else → ncols2=1

3. **`fattn-tile.cuh`**: Neue TILE-Config für 512/512 bei ncols=2 in allen 4
   Config-Funktionen (nvidia fp16/fp32, amd, amd rdna). Werte: nthreads=128,
   occupancy=2, nbatch_fa=64, nbatch_K=64.

## Verifikation (Uranus, RTX 4060 Ti 16GB, 2026-07-13)

```
E4B + MTP, turbo4/turbo3 KV, -ngl 99 -fa 1:
  llama-cli: "The capital of France is" → "Paris" ✅
  Generation: 103.4 t/s mit MTP spec decoding

E4B + MTP, f16 KV, -ngl 99 -fa 1:
  llama-cli: "What is 2+2?" → "4" ✅
  Generation: 112.1 t/s

E4B only (no MTP), f16 KV:
  pp128: 3589 t/s, tg64: 72 t/s (keine Regression)

E4B only (no MTP), turbo4/turbo3 KV:
  pp128: 1692 t/s, tg64: 62 t/s (keine Regression)
```

## Was funktioniert

- **E4B+MTP auf 16GB (RTX 4060 Ti)**: Voll funktionsfähig mit FA + turbo KV.
  MTP spec decoding funktioniert, ~100 t/s Generation.
- **E4B+MTP auf 8GB (RTX 3070)**: Sollte mit f16 KV und minimalem Kontext
  funktionieren (4GB Modell + Draft ~200MB + KV-Cache). Nicht auf Hydra getestet
  (GPU-Regel: keine GPU-Prozesse auf Hydra).
- **Ollama** mit E4B: Stabil als Fallback.

## Ursprüngliche Fehldiagnose

Die ursprüngliche Analyse ging von "PLE-Architektur mit variablen V-Embedding-Sizes"
als Root Cause aus. Tatsächlich war es simpler: E4B's full-attention Layer haben
`head_dim=512`, und der MMA-Kernel unterstützt `DKQ > 256` nicht (nur 576/640
als Spezialfälle für Deepseek/GLM). PLE ist nicht direkt im Attention-Pfad.
