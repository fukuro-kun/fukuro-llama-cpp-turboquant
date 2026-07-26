# SESSION_PLAN: #44 Alloc-MoE Phase 0 — Quality Benchmark

**Erstellt:** 2026-07-26 (Auto-Fortsetzung nach #71/#18)
**Status:** ❌ NO-GO (2026-07-26). Phase 0 Quality-Benchmark: K=6 +13.5% PPL, K=4 +22.9%, K=2 +66.5%. Selbst K=6 ist 2.7× über dem 5% Limit.
**ROADMAP-Item:** #44 Alloc-MoE: Budget-aware Expert Activation (Tier 3)
**Vorgänger:** #69 ❌, #71 ❌, #18 ❌ (MoE-Cache-Thema erschöpft)

## Session-Ziel

**Phase 0:** Quality-Benchmark für reduziertes K (aktive Experten pro Token). Gemma-4-26B-A4B nutzt K=8 (nicht K=2 wie ursprünglich angenommen). Reduktion auf K=6/K=4/K=2 gibt uns die Quality-Daten um zu entscheiden, ob Alloc-MoE (1.34× decode speedup) den Quality-Drop rechtfertigt.

**Hypothese:** Bei K=8→K=4 (50% Reduktion) ist der Quality-Drop <5% (akzeptabel). Bei K=8→K=2 (75% Reduktion) ist der Drop >10% (unakzeptabel für Produktiv-Server).

## Hintergrund

- **Modell:** Gemma-4-26B-A4B QAT, 128 Experten, **8 aktive** pro Token, 30 Layer
- **#44 Paper (arXiv:2604.08133):** 1.34× decode speedup durch Alloc-L (reduziertes K pro Layer)
- **Risiko:** 17% Quality-Drop bei K=2 (Subagent-Analyse, aber bezog sich auf K=2 als Default — unser Default ist K=8)
- **Ansatz:** `hparams.n_expert_used` zur Laufzeit überschbar machen via CLI flag `--moe-k N`

## Schritte

### Block A: CLI Flag + hparams Override (30min)
1. `--moe-k N` CLI flag in `common/arg.cpp` / `common/arg.h` hinzufügen
2. In `llama-context.cpp` oder `llama-model.cpp`: `hparams.n_expert_used = override_k` wenn gesetzt
3. Build testen

### Block B: Perplexity-Benchmark (60min)
4. Perplexity mit K=8 (Default) auf WikiText-2 oder ähnlichem Korpus
5. Perplexity mit K=6
6. Perplexity mit K=4
7. Perplexity mit K=2
8. Perplexity mit K=1 (Lower Bound)

### Block C: Geschwindigkeits-Benchmark (30min)
9. llama-bench tg512 mit K=8, K=6, K=4, K=2 auf hydra
10. Speedup vs Quality-Drop Tabelle

### Block D: Go/No-Go + Doku (30min)
11. Go/No-Go: Quality-Drop <5% bei K=4 → GO für Phase 1 (Alloc-L Implementierung)
12. ROADMAP #44 aktualisieren
13. CHANGELOG, TTT, Commit

## Defaults bei unklaren Entscheidungen

- **Perplexity-Korpus:** WikiText-2 (klein, schnell, Standard). Falls nicht lokal verfügbar: wikitext-2-raw-v1 von HuggingFace
- **Perplexity-Tool:** `build/bin/llama-perplexity` (bereits vorhanden)
- **K-Werte:** 8 (default), 6, 4, 2, 1
- **Benchmark-Modell:** 26B-A4B QAT (Produktivmodell)
- **GPU:** hydra (GPU-Ausnahme noch aktiv)

## Verifikations-Strategie

| Metrik | Vorher (K=8) | Nachher (K=X) | Go/No-Go |
|--------|-------------|--------------|----------|
| Perplexity (WikiText-2) | Baseline | Δ% | <5% = GO |
| tg512 t/s | Baseline | Speedup | >10% = GO |

**Go-Kriterium:** K=4 mit <5% Perplexity-Drop UND >10% tg Speedup
**No-Go-Kriterium:** K=4 mit >10% Perplexity-Drop ODER <5% tg Speedup

## Risiken + Abbruch-Kriterien

- **Risiko 1:** `--moe-k` Override funktioniert nicht (Graph ist zur Build-Zeit gebackt) → Abbruch nach Block A
- **Risiko 2:** Perplexity-Tool braucht zu lange auf 26B → Auf kleineres Korpus ausweichen
- **Risiko 3:** K=1 produziert NaN (keine Expert-Diversität) → Erwartet, als Lower Bound dokumentieren

## Referenzen

- **Paper:** arXiv:2604.08133 (Alloc-MoE)
- **Code:** `src/llama-graph.cpp:1482` (build_moe_ffn), `src/llama-hparams.h:55` (n_expert_used)
- **Modell-Meta:** 128 experts, 8 active, 30 layers, Q4_K_XL
