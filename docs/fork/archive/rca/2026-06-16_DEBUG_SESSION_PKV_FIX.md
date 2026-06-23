# Debug-Session: DiffusionGemma PREFILL→DECODE Bug auf CUDA

> Datum: 2026-06-15 (fortgesetzt)
> Modell: DiffusionGemma 26B-A4B (Q4_K_M)
> Hardware: Ampere-GPU (RTX 3070 Laptop, 8 GB VRAM)
> Repo: fukuro-llama-cpp-turboquant

---

## Symptom

DiffusionGemma PREFILL→DECODE Path lieferte bei partiellem GPU-Offload (`-ngl 8`) leeren Output (`cut=0`), funktionierte aber auf reiner CPU (`-ngl 0`) korrekt (`cut>0`).

| `-ngl` | Erwartet | Tatsächlich |
|--------|----------|-------------|
| 0 (CPU) | cut>0 | cut=8 ✅ |
| 8 (Split) | cut>0 | cut=0 ❌ |
| 99 (GPU) | cut>0 | OOM (16 GB nötig) |

---

## Analyse-Methodik

### Phase 1: Statisches Diffing (7 Subagents)
Vollständiger Datei-Diff zwischen Unsloth-PR und unserem Fork. Ergebnis: **Keine relevanten Unterschiede** in den verdächtigen Code-Pfaden.

### Phase 2: Hypothesen-Prüfung (empirisch)

| Hypothese | Test | Ergebnis | Bewertung |
|-----------|------|----------|-----------|
| **A:** Buffer-Overread in `ggml-alloc.c` | Fix angewendet, getestet | `cut=0` bleibt | ❌ Nebenbefund |
| **B:** Missing `uid`/CUDA-Graph Reuse | `GGML_CUDA_DISABLE_GRAPHS=1` | Keine Änderung | ❌ Ampere nutzt `uid` nicht |
| **C:** Missing `ggml_cuda_pdl_sync` | Code-Analyse (Hopper-only) | Relevant für Hopper | ❌ No-Op auf Ampere |
| **D:** `llama_encode()` vs `llama_decode()` | Trace | Identisch zum PR | ❌ Kein Unterschied |
| **E:** Cross-Backend `ggml_cpy` | **`-ngl 0`=OK, `-ngl 8`=FAIL** | **Bestätigt** | ✅ **Root Cause** |

### Phase 3: `-ngl`-Bisect (entscheidend)
Der `-ngl 0` vs `-ngl 8` Unterschied isolierte den Bug auf die **CPU/GPU-Split-Grenze**. Bei `-ngl 8` liegen Layer 0-7 auf CPU, Layer 8-29 auf CUDA.

---

## Root Cause

`dg_ensure_pkv_store()` in `src/models/diffusion-gemma.cpp` allokierte den **gesamten** Prompt-KV-Store (alle `pkv_k[il]`/`pkv_v[il]`) in **einem** `ggml_context` auf der Buffer-Type von `m.dev_layer(0)`.

- Bei `-ngl 8`: `dev_layer(0)` = CPU → PKV-Store komplett auf CPU
- Aber: `Kcur`/`Vcur` der GPU-Layer (ab Layer 8) sind CUDA-resident
- Das `ggml_cpy(Kcur, sk)` (PREFILL) und `ggml_concat(pk, Kcur)` (DECODE) sind damit **Cross-Backend** (CUDA → CPU)
- Cross-Backend `ggml_cpy` im Scheduler-Allokations-Pfad schlägt fehl → leerer PKV-Store → DECODE liert nichts → `cut=0`

**Wichtige Nuance:** Der Unsloth-PR teilt dieselbe globale PKV-Allokation (kommentiert als "Single-GPU only"), funktioniert aber bei `-ngl 8` dennoch. Unser TurboQuant-Fork hat stark modifiziertes `ggml-cuda`, das Cross-Backend-cpy anders handhabt (oder bricht).

---

## Fix

**Prinzip:** PKV pro Layer auf dem Buffer-Type des jeweiligen Layer-Device allokieren, sodass alle `ggml_cpy`/`ggml_concat` **intra-Backend** bleiben.

**Implementierung:**
1. Gruppiere Layer nach distinct `buft = ggml_backend_dev_buffer_type(dev_layer(il))`
2. Pro distinct `buft`: eigener `ggml_context` (no_alloc=true), darin `pkv_k[il]`/`pkv_v[il]` nur für die Layer dieser Gruppe
3. `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` pro Gruppe
4. Speichere alle ctx/buf in Vektoren (`pkv_ctxs`, `pkv_bufs`)

**Geänderte Dateien:**
- `src/llama-model.h`: `pkv_ctx`/`pkv_buf` (Einzel-Member) → `pkv_ctxs`/`pkv_bufs` (Vektoren)
- `src/models/diffusion-gemma.cpp`: `dg_ensure_pkv_store()` komplett umgebaut
- `src/llama-model.cpp`: Destruktor-Cleanup für Vektoren (nebenbei Memory-Leak behoben)

---

## Test-Ergebnisse (nach Fix)

| Test | Konfiguration | Ergebnis |
|------|--------------|----------|
| `-ngl 0` (CPU) | "What is the capital of France?" | cut=8, "The capital of France is is." ✅ |
| `-ngl 8` (Split) | "What is the capital of France?" | **cut=14, "The capital of France is Paris."** ✅ |
| `-ngl 99` (GPU) | — | OOM (16 GB nötig, 8 GB verfügbar) |

---

## Lehren

1. **Empirisch > Statisch:** Drei plausibel klingende Hypothesen (uid, CUDA-Graphs, PDL) waren auf der Ziel-Hardware (Ampere) **No-Ops**. Erst der empirische `-ngl`-Bisect brachte die Wahrheit.
2. **Cross-Backend ist fragil:** `ggml_cpy` über Backend-Grenzen hinweg ist der häufigste Stolperstein bei partiellen Offloads. Intra-Backend-Allokation ist die robuste Lösung.
3. **PR-Kommentare lesen:** Der PR-Autor hatte den richtigen Fix bereits als Kommentar notiert ("per-buft context map"), aber wir hatten ihn übersehen, weil wir im Diffing nach subtilen Code-Unterschieden suchten.
4. **Hardware-Spezifika beachten:** CUDA-Features wie `uid`/PDL sind generationsspezifisch. Was auf Hopper funktioniert, ist auf Ampere irrelevant.

---

## Verwandte Dateien

- `docs/fork/2026-06-15_DIFFUSION_GEMMA_STATUS.md` — Aktueller Projekt-Status
- `docs/fork/2026-06-16_VERGLEICH_UNSLOTH.md` — Vergleich mit Referenz-Implementierung
- Trilium-Note "Port-Bericht: DiffusionGemma Integration" — Vollständiger Port-Bericht mit Architekturentscheidung
