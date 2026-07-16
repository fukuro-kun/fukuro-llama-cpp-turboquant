# Expected Attention KV Cache Compression — Design-Dokument

**Datum:** 2026-07-15
**ROADMAP-Item:** #19
**Paper:** arXiv:2510.00636
**Referenz:** NVIDIA kvpress (Apache 2.0)
**Status:** Phase 3 (Intelligent EA Scoring) implementiert — 2026-07-15

## Zusammenfassung

Expected Attention ist eine training-freie KV-Cache-Compression-Methode, die KV-Paare durch Vorhersage ihrer zukünftigen Wichtigkeit prunt. Im Gegensatz zu TurboQuant (Quantisierung: weniger Bits pro KV-Paar) reduziert Expected Attention die **Anzahl** der KV-Paare (Pruning). Die Methoden sind **orthogonal** und kombinierbar.

## Kernkonzept

### Algorithmus

1. **Query-Statistik:** Berechne Mean (μ) und Covariance (Σ) der Queries vor RoPE aus einem Rolling-Buffer (default: 128 hidden states). Entferne erste n_sink Tokens (sink attention).

2. **RoPE-Vorhersage:** Berechne RoPE-Rotationsmatrix R für nächste n_future_positions (default: 512), mittlere über alle Positionen.

3. **Transformierte Statistik:** μ' = R @ μ, Σ' = R @ Σ @ R^T

4. **Expected Attention (geschlossene Form):**
   ```
   E(A) = exp(K @ μ'.T / sqrt(d) + 0.5 * K @ Σ' @ K.T / d)
   ```
   Mit `use_covariance=False`: Nur Mean-Term (schneller, weniger akkurat).

5. **Value-Norm Reskalierung (optional):** score = (E(A) + ε) * ||V||_2

6. **Pruning:** Rank KV-Paare nach Score, entferne bottom compression_ratio%. Schütze n_sink + n_local Tokens.

### Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| compression_ratio | 0.5 | Anteil zu entfernender KV-Paare |
| n_future_positions | 512 | RoPE-Vorhersage-Horizont |
| n_sink | 4 | Geschützte initiale Tokens |
| n_local | 128 | Geschützte recent Tokens |
| use_covariance | true | Covariance-Term (genauer, O(d²)) |
| use_vnorm | true | Value-Norm Reskalierung |

## Kompatibilität mit TurboQuant

**Voll orthogonal.** TurboQuant komprimiert Bits-pro-Paar, Expected Attention komprimiert Anzahl-Paare.

| Kombination | Kompression |
|-------------|-------------|
| TurboQuant allein | 5.1x (turbo3) / 3.8x (turbo4) |
| Expected Attention allein | 2x (50% pruning) |
| **TurboQuant + Expected Attention** | **~10.2x** |

### Integrations-Reihenfolge

**Option A (empfohlen): Pruning → Quantisierung**
1. Expected Attention Pruning auf FP16 KV-Paare
2. TurboQuant Quantisierung der verbleibenden Paare
- Vorteil: Weniger Quantisierungs-Overhead, Pruning auf unquantisierten Werten

## Implementierungsplan (4-6 Wochen)

### Phase 1 (Woche 1-2): Grundimplementierung CPU
- [x] Expected Attention Statistik (Mean/Covariance, Rolling-Buffer)
- [x] RoPE-Rotations-Vorhersage
- [x] Expected Attention Score-Berechnung
- [x] Unit-Tests für Mathematik

### Phase 2 (Woche 3-4): KV-Cache Integration
- [x] Integration in `llama_kv_cache`
- [x] Pruning-Logik (Ranking + Eviction)
- [x] Sink-Token und Local-Window Schutz
- [x] CLI-Parameter `--ea-ratio` (und 5 weitere)

### Phase 3 (Woche 5-6): Intelligent EA Scoring + Optimierung
- [x] Q-Capture in `build_qkv()` via `ggml_set_output`
- [x] Q-Extraction + Rolling Buffer im Context
- [x] Score-basiertes Pruning in `ea_compress()`
- [x] NeoX + Interleaved RoPE Konventionen
- [x] GQA Query-Head-Mapping (`h * n_gqa`)
- [x] Value-Norm Rescaling implementiert
- [x] Unit-Tests (614/614 grün) + Smoke-Test (Gemma 4 12B CPU)
- [x] **TurboQuant+EA Kombination** (Phase 3 — WHT-Forward auf μ' angewendet, Domain-Mismatch gelöst)
- [ ] CUDA/Vulkan Backend-Optimierung
- [ ] Benchmarks (Accuracy, Memory, Speed) — auf Styx (GPU)

## Zu ändernde Dateien

| Datei | Änderung | Aufwand |
|-------|----------|---------|
| `src/llama-kv-cache.h` | Neue Parameter für Expected Attention | Gering |
| `src/llama-kv-cache.cpp` | Expected Attention Logik | Hoch |
| `src/llama-context.cpp` | Parameter-Integration | Mittel |
| `tools/server/server-context.cpp` | CLI-Parameter | Gering |
| `tests/` | Neue Tests | Mittel |
| `ggml/src/ggml-turbo-quant.c` | Keine Änderungen (orthogonal) | - |

## Risiken

1. **RoPE-Variation:** RoPE unterscheidet sich zwischen Architekturen (Llama, Gemma, Qwen). Muss architektur-spezifisch unterstützt werden. ✅ NeoX + Interleaved implementiert.
2. **Covariance O(d²):** Bei head_dim=128 → 16K floats pro Head. Mit vielen Heads teuer. Mean-only-Modus als Fallback. ✅ Mean-only aktiv, Covariance deferred to Phase 4.
3. **Flash Attention:** Expected Attention benötigt Query-Zugriff. Flash Attention materialisiert Queries nicht. Workaround: Sampling-Buffer. ✅ Q-Capture via `ggml_set_output` in `build_qkv()`.
4. **Streaming:** Statistik muss während Decoding aktualisiert werden (rolling buffer). ✅ Rolling Buffer implementiert, `ea_clear_queries` in `clear()`/`seq_rm()`.

## Bekannte Limitationen (Phase 3)

1. **V-norm nur bei Flash Attention:** Value-Norm Rescaling ist nur aktiv wenn `v_trans == false` (d.h. Flash Attention enabled). Ohne Flash Attention ist V transposed und die K-Style-Stride-Leselogik greift nicht — vnorm wird deaktiviert.
2. **Covariance-Term:** Nur Mean-only-Modus aktiv (`use_covariance=false` hardcoded). Covariance O(d²) ist für Phase 4 geplant. **Recherche 2026-07-15 (arXiv:2510.00636 Ablation Table 4):** Für Gemma 4 (QK-Normalization) bringt Covariance <0.1% Accuracy-Gewinn bei 128× mehr Rechnung (O(d²)=16K vs O(d)=128 floats pro Head). Empfehlung: Mean-only beibehalten, Covariance nur optional für Llama-Modelle ohne QK-Norm. Phase 4 priorisiert CUDA/Vulkan Backend-Optimierung statt Covariance.
3. **RoPE-Parameter:** `freq_scale`, `freq_factors`, `ext_factor`, `attn_factor` und per-Layer `rope_freq_base` werden nicht an `ea_average_rope` übergeben. Nur `theta_base` aus `cparams.rope_freq_base`. Bei Modellen mit nicht-standard RoPE-Parametern kann die Vorhersage ungenau sein.
4. **RoPE-Mode:** Nur NEOX und Interleaved (NORM). `MROPE`, `IMROPE`, `VISION`, `NONE` fallen auf Interleaved zurück.
5. **EA-Buffer-Init:** Rolling Buffer wird mit Layer-0 Dimensionen (`n_head_kv`, `head_dim`) initialisiert. Layer mit abweichenden Dimensionen (SWA) werden stillschweigend übersprungen. Korrekt wäre lazy per-Layer Initialisierung.
6. **GPU-Tests:** CPU Smoke-Test bestanden (Gemma 4 12B). GPU-Benchmark auf Styx (GTX 1070) läuft — Baseline (EA disabled) zeigt pp512=371 t/s, tg128=24 t/s, pp4096=220 t/s. EA-enabled Vergleich ausstehend.

## Review-Fixes (Round 3, 2026-07-15)

Zwei kritische Bugs im K-Cache-Readout von `ea_compress()` wurden gefunden und behoben:

1. **cell_stride unpadded (P0):** `cell_stride` verwendete `hparams.n_embd_k_gqa(il)` (unpadded) statt `k_tensor->nb[1]` (padded). Bei TurboQuant mit `head_dim` ≠ Vielfaches von 128 wurden K-Rows mit falschem Stride gelesen.
2. **Stream-Dimension fehlt (P0):** Offset-Formel erweitert auf `s*stream_stride + cell_idx*cell_stride + h*head_stride` mit `stream_stride = k_tensor->nb[2]`. Gleicher Fix für V-Tensor (vnorm-Pfad).
3. **WHT-Tests gestärkt (P1):** Reference-Equality-Test (unabhängige Sign-Array-Kopien) und `scale_inv`-Pfad-Test hinzugefügt. Testzahl 358 → 614.

## Relevanz für Fork-Systeme

| System | Aktuelles Limit | Mit Expected Attention |
|--------|----------------|----------------------|
| Styx (GTX 1070, 8GB) | 224k Kontext (turbo3/4) | ~448k theoretisch |
| Mars (AMD APU, 30GB) | 256k (Modell-Limit) | Weniger RAM bei gleichem Kontext |
| Uranus (2x 4060 Ti, 32GB) | 128k (E4B) | ~256k theoretisch |

## Empfehlung

**Integration sinnvoll.** Phase 3 (Mean-only) ist implementiert und getestet (614 Unit-Tests). Covariance (Phase 4) ist depriorisiert — für Gemma 4 (QK-Norm) bringt es <0.1% Accuracy-Gewinn bei 128× mehr Rechnung. Nächster Schritt: GPU-Backend-Optimierung (CUDA/Vulkan) für den EA-Scoring-Pfad, der aktuell CPU-only single-threaded ist.

## Referenzen

- Paper: arXiv:2510.00636
- Referenzcode: NVIDIA/kvpress (Apache 2.0)
- ROADMAP-Item: #19
