# QAT + MTP Q4_0 Draft Benchmark (26B-A4B)

**Datum:** 2026-07-10
**System:** Mars (AMD Ryzen 5 7640HS, Radeon 760M RDNA3, 30GB RAM, Vulkan 1.4.309, Mesa 25.0.7)
**Modell:** Gemma-4 26B-A4B IQ4_NL (14.7GB) vs QAT-UD-Q4_K_XL (14.2GB)
**Draft:** mtp-gemma-4-26B-A4B-it-Q4_0 (241MB, neu Jul 10)
**Backend:** Vulkan, -ngl 99, FlashAttention on, turbo3/turbo4 KV-Cache

## Hintergrund

Neue Modellvarianten auf Ganymed:
- **QAT-UD-Q4_K_XL** (14G): Quantization-Aware Training Variante von Unsloth. QAT ist eine Training-Methodik, kein eigenes GGUF-Format. Die Gewichte sind "baked in" — kein spezieller Loader-Code nötig.
- **MTP Q4_0 Draft** (241M): Neuer Draft (Jul 10), konvertiert mit upstream llama.cpp PR #23398. Nutzt `gemma4-assistant` (Bindestrich) als arch-Namen und `embedding_length_out` statt `n_embd_backbone`.

## Adapter für neue GGUF-Metadata-Keys

Der neue Q4_0 Draft konnte nicht geladen werden (3 Probleme):
1. `general.architecture = gemma4-assistant` (Bindestrich) — Fork suchte nach `gemma4_assistant` (Unterstrich) als Key-Prefix
2. `gemma4-assistant.embedding_length_out = 2816` — Fork suchte nach `gemma4_assistant.n_embd_backbone`
3. `nextn.post_projection.weight` Tensor-Name — bereits durch existierendes Alias-Mapping abgedeckt

**Fix (Commit `f9a3dfc62`):**
- `LLM_KV arch_name-Override`: Nutzt originalen arch_name aus GGUF als Key-Prefix (cherry-picked von `673629f4d` auf `archive/cherry-dflash`)
- Arch-Aliase vereinheitlicht: `gemma4-assistant`, `gemma4_assistant`, `gemma4_mtp` → `LLM_ARCH_GEMMA4_ASSISTANT`
- `embedding_length_out` als Fallback für `n_embd_backbone` (upstream PR #23398 nutzt diesen Key)

## Ergebnisse Mars

### QAT vs IQ4_NL (ohne MTP, turbo3/4, FA on, pp512, tg64)

| Modell | pp512 | tg64 | Größe |
|--------|-------|------|-------|
| **QAT-UD-Q4_K_XL** | **210.1 t/s** | **25.3 t/s** | 14.2G |
| IQ4_NL (Google) | 191.0 t/s | 21.7 t/s | 14.7G |
| **Diff** | **+10.0%** | **+16.6%** | -3.4% |

QAT ist bei pp und tg deutlich schneller — kleineres Modell (14.2G vs 14.7G) bei gleicher Parameterzahl.

### MTP Q4_0 Draft (QAT-Modell, turbo3/4, FA on)

| Test | tg | Acceptance | Drafts generated | Drafts accepted |
|------|----|------------|------------------|-----------------|
| 100 tokens | 14.2 t/s (server-log) | 65% | 100 | 65 |
| 200 tokens | 24.7 t/s | 57.5% | 219 | 126 |

**MTP bringt auf Mars KEINEN Speedup:**
- QAT ohne MTP: 25.3 t/s
- QAT mit MTP: 24.7 t/s (200 tokens) — **leicht langsamer**
- Grund: Auf AMD APU (shared memory) überwiegt der Draft-Forward-Pass den MTP-Vorteil. Die 57% Acceptance reicht nicht um den Overhead zu kompensieren.

## Ergebnisse Styx (GTX 1070, CUDA, MoE-Offload)

**System:** Intel i7-7700HQ, GTX 1070 8GB, 32GB RAM, CUDA 12.0, `--n-cpu-moe 20` (20 Experten auf CPU)

### QAT vs IQ4_NL (ohne MTP, turbo3/4, FA on, --n-cpu-moe 20, pp512, tg64)

| Modell | pp512 | tg64 | Größe |
|--------|-------|------|-------|
| **QAT-UD-Q4_K_XL** | 350.4 t/s | **24.9 t/s** | 14.2G |
| IQ4_NL (Google) | 359.3 t/s | 24.7 t/s | 14.7G |
| **Diff** | -2.5% | **+0.5%** | -3.4% |

Auf Styx sind QAT und IQ4_NL praktisch gleichauf (tg ±0.5%). Der pp-Unterschied liegt im Rauschen. Im Gegensatz zu Mars (wo QAT +10% pp/+16.6% tg schneller war) ist auf Styx kein QAT-Vorteil messbar — die GTX 1070 mit MoE-Offload ist CPU-limitiert (i7-7700HQ), da fällt der Größen-Unterschied (14.2 vs 14.7G) nicht ins Gewicht.

### MTP Q4_0 Draft (turbo3/4, FA on, --n-cpu-moe 20, 200 tokens)

| Modell | tg mit MTP | tg ohne MTP | Acceptance | Speedup |
|--------|-----------|-------------|------------|---------|
| QAT-UD-Q4_K_XL | 21.3 t/s | 24.9 t/s | 48.1% | **-14.5%** |
| IQ4_NL (Google) | 21.5 t/s | 24.7 t/s | 50.8% | **-12.9%** |

**MTP bringt auf Styx KEINEN Speedup — im Gegenteil, es ist 13-14% langsamer!**
- Beide Modelle: ~50% Acceptance, aber der Draft-Forward-Pass kostet mehr als die akzeptierten Tokens einsparen
- Grund: GTX 1070 mit 8GB VRAM muss den Draft (241MB) in VRAM laden, der Haupt-Modell-Forward ist aber CPU-limitiert (MoE-Offload) → Draft-Compute konkurriert mit Expert-Prefetch um CPU-Zeit

## Fazit

1. **QAT-UD-Q4_K_XL ist auf Mars schneller** (+10% pp, +16.6% tg), auf Styx gleichauf mit IQ4_NL. QAT ist kleiner (14.2G vs 14.7G) — auf VRAM-limitierten Systemen vorteilhaft.
2. **MTP Q4_0 Draft funktioniert auf beiden Systemen** (48-65% Acceptance), bringt aber **auf KEINEM System Speedup**:
   - Mars (AMD APU, shared memory): -2.4% (Draft-Overhead überwiegt)
   - Styx (GTX 1070, MoE-Offload): -14% (Draft konkurriert mit Expert-Prefetch um CPU)
3. **Empfehlung: MTP Q4_0 Draft AUS** auf beiden Systemen. Der Draft-Overhead überkompensiert die Acceptance-Rate.
4. **Adapter für neue GGUF-Metadata-Keys** ermöglicht Kompatibilität mit upstream PR #23398 konvertierten Modellen — funktioniert, aber MTP lohnt sich nicht.

## Kontextfenster-Erweiterung durch QAT

**Modell-Maximum: 256K (262144 tokens)** — `gemma4.context_length = 262144` in der GGUF. Kontexte über 256k sind sinnlos (RoPE-Positionen gehen nicht weiter). Quelle: [Google Gemma-4 26B-A4B Hugging Face](https://huggingface.co/google/gemma-4-26B-A4B), Trilium-Note `z6bNQ69yJmzc`.

QAT ist 0.5G kleiner als IQ4_NL → mehr VRAM/RAM für KV-Cache → größeres Kontextfenster.

### Mars (AMD APU, Vulkan, shared memory, 30GB RAM)

| Kontext | Lädt? | Bemerkung |
|---------|-------|-----------|
| 180224 (180k) | ✅ | Altes IQ4_NL Limit |
| 229376 (224k) | ✅ | **Neuer Produktiv-Standard** — 15k token Prompt: 139 t/s pp, 18.5 t/s tg |
| 262144 (256k) | ❌ | OOM-killed — KV-Cache übersteigt 30GB RAM |
| >256k | ❌ | Sinnlos — Modell-Maximum ist 256k |

**Limit: 224k** — 256k (Modell-Maximum) wäre möglich wenn mehr RAM verfügbar wäre. Bei ~180k token Prompts tritt die Vulkan-Performance-Klippe auf (siehe `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md`). Bis 15k token Prompts sind bei 224k voll nutzbar.

### Styx (GTX 1070, CUDA, MoE-Offload, 32GB RAM)

| Kontext | Lädt? | Bemerkung |
|---------|-------|-----------|
| 163840 (160k) | ✅ | Altes IQ4_NL Limit |
| 229376 (224k) | ✅ | **Neuer Produktiv-Standard** — 64 token tg: 25.85 t/s stabil |
| 245760 (240k) | ❌ | CUDA out of memory |

**Limit: 224k** — CUDA OOM bei 245k. 256k (Modell-Maximum) ist mit 8GB VRAM nicht erreichbar. +64k Kontext vs IQ4_NL.

### Produktiv-Server Verifikation (2026-07-10)

| System | Modell | ctx | tg (32 tok) | Status |
|--------|--------|-----|-------------|--------|
| Mars | QAT-UD-Q4_K_XL | 229376 (224k) | **25.85 t/s** | ✅ aktiv |
| Styx | QAT-UD-Q4_K_XL | 229376 (224k) | **26.02 t/s** | ✅ aktiv |

Beide Server laufen als systemd User-Services mit dem neuen QAT-Standard. 224k ist das Hardware-Limit — das Modell-Maximum von 256k ist auf Mars (30GB RAM) und Styx (8GB VRAM) nicht voll erreichbar.

## Technische Hinweise

- **QAT = normale GGUF:** Kein spezieller QAT-Loader nötig. QAT ist eine Training-Methodik, die Gewichte sind normal quantisiert. Der einzige relevante upstream-Fix ist PR #21451 (BF16 precision für Gemma 4 scale ops) für beste Qualität.
- **MTP Q4_0 Draft Metadaten:** `general.architecture = gemma4-assistant` (Bindestrich), `gemma4-assistant.embedding_length_out = 2816` (statt `n_embd_backbone`). Tensor-Namen: `nextn.post_projection.weight` (bereits durch Alias-Mapping abgedeckt).
- **Unsloth Dynamic (UD) Quants:** Standard GGUF-kompatibel, keine spezielle llama.cpp Version nötig.
