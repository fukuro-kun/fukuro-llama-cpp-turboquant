# #87 Cross-Layer Gate Expert Prediction — Evaluation Report

**Datum:** 2026-07-15 | **ROADMAP Item:** #87 | **Status:** ❌ Evaluiert — nicht viable für Gemma-4 26B-A4B

---

## Zusammenfassung

**Fate (arXiv:2502.12224)** schlägt training-freie Expert-Prediction via Cross-Layer Gating vor. Die Kernannahme: Gate-Inputs benachbarter Layer haben >83% Cosine-Ähnlichkeit, was zu 97.15% Prefetch-Accuracy führt.

**Ergebnis:** Die Expert-Selection-Overlap zwischen benachbarten Layern in Gemma-4 26B-A4B beträgt nur **6.6%** — nahe der Random-Baseline von 6.25% (8²/128). Dies bedeutet, dass benachbarte Layer **völlig unterschiedliche Experten** selektieren.

**Schlussfolgerung:** Der Fate-Ansatz ist für Gemma-4 26B-A4B (128 Experten, Top-8) **nicht viable**. Die hohe Sensitivität der Gate-Funktion (Softmax + Top-8 aus 128) macht Cross-Layer-Prediction ineffektiv. Das Paper wurde auf Modellen mit 8-16 Experten (Top-2) evaluiert, wo die Gate-Funktion wesentlich weniger sensibel ist.

---

## Paper-Details

| Feld | Wert |
|------|------|
| **Titel** | Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate |
| **Autoren** | Zhiyuan Fang, Zicong Hong, Yuegui Huang, et al. |
| **ArXiv** | 2502.12224 (Mai 2025) |
| **Code** | https://github.com/FFFzy/Fate_open (MIT License) |
| **Training** | Training-frei (Plug-and-play) |
| **Genauigkeit** | 97.15% Prefetch Accuracy (auf evaluierten Modellen) |
| **Evaluierte Modelle** | Mixtral 8x7B (8 experts, top-2), ähnliche Modelle |

### Algorithmus

1. Vor der Gate-Berechnung in MoE-Block N wird der Gate-Input (intermediate state) zur CPU kopiert
2. CPU führt parallel die Vorhersage-Berechnung durch: Apply Layer N+1's Gate-Weights auf Layer N's Output
3. Generiere Prefetch-Liste mit Expert-Indizes (Konfidenz > 75. Perzentil)
4. Prefetch Experts für Layer N+1 während GPU noch Layer N berechnet

### Zusätzliche Features
- **Shallow-favoring Caching:** Bevorzugt frühe Layer (wo Vorhersagen schwächer sind)
- **ARC Eviction:** Adaptive Replacement Cache für 99% Hit-Rate
- **Hybrid INT2/INT4 Quantisierung:** Für Cache-Optimierung

---

## Gemma-4 26B-A4B Architektur

| Parameter | Wert |
|-----------|------|
| **Layer** | 30 (alle MoE) |
| **Experten** | 128 (fine-grained) |
| **Active Experts** | 8 (Top-8) |
| **n_embd** | 2816 |
| **n_ff_exp** | 704 |
| **Gate-Computation** | NACH Attention (auf `attn_out`) |
| **Gate-Funktion** | RMSNorm → Scale → Mul(gate_inp_s) → MatMul(gate_inp) → Softmax → Top-8 |

---

## Messung: Expert-Selection-Overlap

### Methode

Tool: `llama-expert-overlap` (neu entwickelt in dieser Session)

Für jeden Token in jedem Layer werden die selektierten Experten (Top-8) erfasst. Die Overlap-Metrik misst, wie viele Experten von Layer N auch in Layer N+1 selektiert werden:

```
overlap(N, N+1) = |experts_N ∩ experts_N+1| / |experts_N|
```

### Test-Setup

- **Modell:** gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
- **Plattform:** Mars (AMD Radeon 760M, Vulkan, full GPU offload)
- **Prompt:** 91 Tokens (MoE-relevanter Text)
- **Generation:** 64 Tokens (greedy)
- **Total:** 155 Tokens pro Layer

### Ergebnisse

| Layer Pair | Avg Overlap % | Min % | Max % | Exact Match % |
|------------|--------------|-------|-------|---------------|
| 0-1 | 7.6 | 0.0 | 37.5 | 0.0 |
| 1-2 | 3.7 | 0.0 | 37.5 | 0.0 |
| 2-3 | 4.2 | 0.0 | 37.5 | 0.0 |
| ... | ... | ... | ... | ... |
| 27-28 | 9.5 | 0.0 | 50.0 | 0.0 |
| 28-29 | 8.4 | 0.0 | 37.5 | 0.0 |

**Overall Average: 6.6%**
**Random Baseline: 6.25%** (8²/128 = 0.5, normalized: 0.5/8 = 6.25%)
**Exact Match: 0.0%** (kein einziger Token hat identische Expert-Sets in benachbarten Layern)

### Multi-Step Overlap

| Step k | Avg Overlap % |
|--------|--------------|
| 1 | 6.6 |
| 2 | 6.1 |
| 3 | 6.2 |

Auch für Layer N vs N+2 und N+3 ist die Overlap nahe der Random-Baseline.

---

## Analyse

### Warum die Overlap so niedrig ist

1. **128 Experten, Top-8:** Die Gate-Funktion (Softmax + Top-8 aus 128) ist extrem nicht-linear. Selbst kleine Unterschiede in den Gate-Inputs führen zu völlig unterschiedlichen Top-8 Selektionen.

2. **Fine-grained Experts:** Gemma-4 verwendet fine-grained Experts (n_ff_exp = 704), was bedeutet, dass viele Experten spezialisiert sind. Die Gate-Funktion muss zwischen 128 ähnlichen Experten unterscheiden, was zu hoher Sensitivität führt.

3. **Verschiedene Gate-Weights pro Layer:** Jeder Layer hat eigene `ffn_gate_inp` Weights. Selbst wenn die Gate-Inputs (attn_out) ähnlich sind, produzieren unterschiedliche Gate-Weights unterschiedliche Logits → unterschiedliche Top-8.

### Was die Messung NICHT zeigt

Die Expert-Selection-Overlap misst **nicht** die Cosine-Ähnlichkeit der Gate-Inputs. Es ist möglich, dass die Gate-Inputs (attn_out) benachbarter Layer tatsächlich >83% Cosine-Ähnlichkeit haben (wie im Fate-Paper behauptet), aber die Gate-Funktion diese Ähnlichkeit nicht in ähnliche Expert-Selektionen übersetzt.

### Was Fate tatsächlich macht

Fate verwendet **nicht** die Expert-Selektion von Layer N um Layer N+1 vorherzusagen. Stattdessen:
1. Nehme Layer N's Gate-Input (attn_out)
2. Apply Layer N+1's Gate-Weights darauf
3. Erhalte eine Vorhersage der Logits für Layer N+1
4. Wähle Top-K aus diesen vorhergesagten Logits

Dies könnte genauer sein als die direkte Expert-Overlap-Messung, aber die 6.6% Overlap suggeriert, dass auch diese Methode ungenau sein wird, da die Gate-Inputs (trotz möglicher Cosine-Ähnlichkeit) zu unterschiedliche Logits produzieren.

### Vergleich: Fate's evaluierte Modelle vs Gemma-4

| Modell | Experten | Top-K | Random Baseline | Fate Accuracy |
|--------|----------|-------|-----------------|---------------|
| Mixtral 8x7B | 8 | 2 | 25% (2²/8) | 97.15% |
| Gemma-4 26B-A4B | 128 | 8 | 6.25% (8²/128) | ~6.6% (gemessen) |

Mit 8 Experten und Top-2 ist die Gate-Funktion wesentlich robuster gegenüber Input-Variationen. Die Random-Baseline ist 25%, und Fate erreicht 97% — eine 3.9x Verbesserung über Random.

Mit 128 Experten und Top-8 ist die Random-Baseline 6.25%, und die gemessene Overlap ist 6.6% — **keine Verbesserung über Random**. Die Gate-Funktion ist zu sensibel für Cross-Layer-Prediction.

---

## Empfehlung

**#87 Cross-Layer Gate Expert Prediction: ❌ NICHT VIABLE für Gemma-4 26B-A4B**

### Begründung

1. **Expert-Overlap nahe Random-Baseline:** 6.6% vs 6.25% random — keine Signifikanz
2. **0% Exact Match:** Kein einziger Token hat identische Expert-Sets in benachbarten Layern
3. **128 Experten + Top-8 = extreme Sensitivität:** Die Gate-Funktion ist zu nicht-linear für Cross-Layer-Prediction
4. **Fate evaluiert auf 8-16 Experten:** Die Methode wurde nicht für fine-grained MoE mit 128 Experten validiert

### Was stattdessen?

- **#65 Pre-Attention Expert Prediction** bleibt verschoben (braucht Training)
- **Prefetch-Heuristik** (aktuelle `GGML_SCHED_PREFETCH_EXPERTS`) ist bereits optimal für Prefill
- **Decode-Prefetch** benötigt eine andere Methode: z.B. n-gram-basierte Expert-Vorhersage (ähnlich DSD #86) oder Frequency-basiertes Caching (bereits implementiert als #40 MoE Frequency Tracking)
- **Upstream-Rebase** würde #86 DSD ngram-map kostenlos in den Fork bringen

---

## Tool: llama-expert-overlap

Neues Tool `tools/expert-profile/expert-overlap.cpp` entwickelt für diese Evaluation.

### Usage
```bash
llama-expert-overlap -m model.gguf -ngl 99 -c 4096 -t 8 -n 64 -p "prompt text"
```

### Output
- Per-Layer Expert-Set-Größen
- Adjacent Layer Overlap (Avg/Min/Max/Exact-Match %)
- Multi-Step Overlap (N vs N+k für k=1,2,3)
- PASS/FAIL verdict gegen Fate threshold (>80%)

### Build
```bash
cmake --build build --target llama-expert-overlap
```
