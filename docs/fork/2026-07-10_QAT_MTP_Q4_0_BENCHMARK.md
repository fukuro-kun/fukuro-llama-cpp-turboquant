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

## Fazit

1. **QAT-UD-Q4_K_XL ist schneller als IQ4_NL** (+10% pp, +16.6% tg) bei kleinerer Größe. QAT ist die bessere Wahl für Mars.
2. **MTP Q4_0 Draft funktioniert** (57-65% Acceptance), bringt aber auf Mars keinen Speedup — Draft-Overhead zu hoch für shared-memory GPUs.
3. **Adapter für neue GGUF-Metadata-Keys** ermöglicht Kompatibilität mit upstream PR #23398 konvertierten Modellen.

## Technische Hinweise

- **QAT = normale GGUF:** Kein spezieller QAT-Loader nötig. QAT ist eine Training-Methodik, die Gewichte sind normal quantisiert. Der einzige relevante upstream-Fix ist PR #21451 (BF16 precision für Gemma 4 scale ops) für beste Qualität.
- **MTP Q4_0 Draft Metadaten:** `general.architecture = gemma4-assistant` (Bindestrich), `gemma4-assistant.embedding_length_out = 2816` (statt `n_embd_backbone`). Tensor-Namen: `nextn.post_projection.weight` (bereits durch Alias-Mapping abgedeckt).
- **Unsloth Dynamic (UD) Quants:** Standard GGUF-kompatibel, keine spezielle llama.cpp Version nötig.
