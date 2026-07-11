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

E4B's PLE-Architektur (Per-Layer Embeddings) erzeugt Tensoren mit
unterschiedlichen Embedding-Größen pro Layer. Das bricht drei Annahmen
gleichzeitig:

1. **FlashAttention** erwartet uniforme K/V-Shapes → crash bei variablen V-Embeddings
2. **turbo KV-Cache** benötigt FA → kann nicht ohne FA genutzt werden
3. **Partieller GPU-Offload** (`-ngl < max`) mit Layer-Sharing → Shape-Mismatch
   beim CPU↔GPU Transfer

Das ist verwandt mit dem bereits bekannten Multi-GPU-Crash
(`GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)`, siehe
InferenzQuelle MTP_INDEX.md "Technische Erkenntnis: E2B/E4B Multi-GPU Crash"),
trifft aber auch auf Single-GPU bei 8GB weil turbo-FA hier zwingend nötig ist
um überhaupt in den VRAM zu passen.

## Was funktioniert

- **Ollama** mit E4B: Stabil, 50s pro Inference, 30 tok/s, `num_ctx: 8192`.
  Ollama nutzt eigenen llama-server mit Standard-KV-Cache (kein turbo, kein
  custom FA-Pfad). Kein MTP-Speedup, aber fehlerfrei.
- **E2B** auf 8GB: Funktioniert mit Ollama (kleiner, weniger PLE-Tensoren).

## Offene Fragen / TODO

- [ ] Läuft E4B+MTP auf 16GB Single-GPU (z.B. RTX 4060 Ti)? Genug VRAM für
      f16 KV-Cache ohne turbo → FA nicht nötig → könnte funktionieren.
- [ ] Läuft E4B+MTP mit `-ngl 99` (volle GPU) + f16 + `-c 2048` (minimaler
      Kontext)? Vielleicht reicht minimaler KV-Cache um ohne turbo auszukommen.
- [ ] FA-Fix für variable V-Embedding-Sizes: PLE-Layer pad/truncate vor FA?
      (Upstream-Work, nicht trivial)

## Workaround

Bis FA PLE-kompatibel ist: **Ollama für E4B nutzen**, kein llama-server+MTP
auf 8GB-Systemen. MTP-Speedup ist nur für größere Modelle (12B, 26B) auf
Systemen mit mehr VRAM (16GB+) verfügbar.
