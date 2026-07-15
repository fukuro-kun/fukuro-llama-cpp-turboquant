# Expected Attention KV Cache Compression — Design-Dokument

**Datum:** 2026-07-15
**ROADMAP-Item:** #19
**Paper:** arXiv:2510.00636
**Referenz:** NVIDIA kvpress (Apache 2.0)
**Status:** Phase 1 (Recherche + Design) — Solo-Session Regel 14

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
- [ ] Expected Attention Statistik (Mean/Covariance, Rolling-Buffer)
- [ ] RoPE-Rotations-Vorhersage
- [ ] Expected Attention Score-Berechnung
- [ ] Unit-Tests für Mathematik

### Phase 2 (Woche 3-4): KV-Cache Integration
- [ ] Integration in `llama_kv_cache`
- [ ] Pruning-Logik (Ranking + Eviction)
- [ ] Sink-Token und Local-Window Schutz
- [ ] CLI-Parameter `--expected-attention-ratio`

### Phase 3 (Woche 5-6): TurboQuant-Kombination + Optimierung
- [ ] Kombination Pruning → Quantisierung
- [ ] CUDA/Vulkan Backend-Optimierung
- [ ] Benchmarks (Accuracy, Memory, Speed)
- [ ] Dokumentation

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

1. **RoPE-Variation:** RoPE unterscheidet sich zwischen Architekturen (Llama, Gemma, Qwen). Muss architektur-spezifisch unterstützt werden.
2. **Covariance O(d²):** Bei head_dim=128 → 16K floats pro Head. Mit vielen Heads teuer. Mean-only-Modus als Fallback.
3. **Flash Attention:** Expected Attention benötigt Query-Zugriff. Flash Attention materialisiert Queries nicht. Workaround: Sampling-Buffer.
4. **Streaming:** Statistik muss während Decoding aktualisiert werden (rolling buffer).

## Relevanz für Fork-Systeme

| System | Aktuelles Limit | Mit Expected Attention |
|--------|----------------|----------------------|
| Styx (GTX 1070, 8GB) | 224k Kontext (turbo3/4) | ~448k theoretisch |
| Mars (AMD APU, 30GB) | 256k (Modell-Limit) | Weniger RAM bei gleichem Kontext |
| Uranus (2x 4060 Ti, 32GB) | 128k (E4B) | ~256k theoretisch |

## Empfehlung

**Integration sinnvoll.** Phase 1 auf CPU + Mean-only beschränken (niedrigstes Risiko), Phase 2 für GPU-Optimierung und Covariance. Die Orthogonalität zu TurboQuant macht es zur idealen Ergänzung — keine Konkurrenz, sondern Multiplikator.

## Referenzen

- Paper: arXiv:2510.00636
- Referenzcode: NVIDIA/kvpress (Apache 2.0)
- ROADMAP-Item: #19
