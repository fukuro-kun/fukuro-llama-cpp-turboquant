# FINALE EMPFEHLUNG — Maximales Modell, maximaler Kontext, maximale t/s

**Datum:** 08.07.2026
**Fazit aus:** allen Benchmarks, die je auf Styx, Mars, Hydra und Uranus durchgeführt wurden
**Trilium:** `h4n3eI0yRUlL` (unter Performance Benchmarks)

---

## Finale Konfiguration für Styx (GTX 1070, 8GB VRAM)

| Parameter | Wert |
|-----------|------|
| **Modell** | Gemma-4 26B-A4B IQ4_NL |
| **MTP** | **AUS** |
| **KV-Cache** | K=turbo3, V=turbo4 |
| **ctx** | 160000 (160k) |
| **MoE-Offload** | `--n-cpu-moe 20` |
| **Pinning** | `GGML_CUDA_REGISTER_HOST=1` |
| **Prefetch** | `GGML_SCHED_PREFETCH_EXPERTS=1` |
| **FlashAttn** | on |

**Ergebnis:** ~16-18 t/s, 160k Kontext, 8110 MiB VRAM, höchste Modellqualität (26B MoE, 30 Experten)

```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
llama-server -m 26B-A4B-IQ4_NL.gguf \
  -c 160000 -ctk turbo3 -ctv turbo4 \
  -ngl 999 --n-cpu-moe 20 -fa on \
  --temp 1.0 --top-p 0.95 --top-k 64
```

---

## Warum 26B-A4B und nicht 12B?

Beide Modelle haben **4B aktive Parameter pro Token** — aber der 26B-A4B hat 26B
Gesamt-Parameter (MoE mit 30 Experten). Der 12B ist ein dichtes Modell mit 12B
aktiven Parametern.

| Eigenschaft | 12B IQ4_NL | 26B-A4B IQ4_NL |
|-------------|-----------|----------------|
| Aktive Params/Token | **12B** | 4B |
| Gesamt-Params | 12B | 26B (MoE) |
| t/s (realistisch, lang) | 14-25 (mit MTP) | **16-18** (ohne MTP) |
| t/s (kurz, tg128) | ~19 (ohne MTP) | **~21** (ohne MTP) |
| Max. Kontext | **196k** | 160k |
| VRAM bei max ctx | 7922 MiB | 8110 MiB |
| MoE-Offloading | Nein | Ja (--n-cpu-moe 20) |
| MTP hilfreich? | Ja (bei kurzer Gen) | Nein (immer langsamer) |
| Modell-Qualität | mittel | **hoch** (mehr Experten) |

### Der entscheidende Vergleich: t/s über Generierungslänge

12B mit MTP fällt mit zunehmender Generierungslänge stark ab:

| Tokens generiert | 12B t/s (mit MTP) | 26B-A4B t/s (ohne MTP) | Sieger |
|-----------------|-------------------|----------------------|--------|
| 20 (Benchmark) | 28-33 | ~21 | 12B |
| 1500 | 25.2 | ~18 | 12B (knapp) |
| 3000 | 18.2 | ~17 | ca. gleich |
| 5000 | 16.2 | **~16-18** | **26B-A4B** |
| 10000 | 14.2 | **~16-18** | **26B-A4B** |

**Ab ca. 3000 Tokens ist der 26B-A4B ohne MTP gleich schnell oder schneller**
als der 12B mit MTP — und dabei das intelligentere Modell.

---

## Warum MTP aus?

1. **MTP bremst bei JEDEM Kontext** auf der GTX 1070 (-2% bis -28% t/s) fuer
   realistische Generierung. Getestet mit Essay-Prompts bei 8k, 16k, 32k, 48k,
   96k, 128k.
2. **VRAM-Ersparnis:** Ohne MTP fallen Draft-Gewichte (~300-400 MiB) und
   Draft-Compute-Buffer (0-400+ MiB) weg -> 300-830 MiB mehr fuer KV-Cache.
3. **64k (2^16) Bug:** ctx=65536 crasht reproduzierbar mit MTP
   (Integer-Overflow). Ohne MTP kein Problem.
4. **MTP hilft nur bei trivialen Prompts** ("Count 1-100": 31.9 t/s, ~100%
   Acceptance). Bei kreativen Texten werden Drafts meist abgelehnt.

---

## Cross-Platform Empfehlung

| System | GPU | Modell | Config | tg | ctx max | MTP? |
|--------|-----|--------|--------|-----|---------|------|
| **Styx** | GTX 1070 (8GB) | 26B-A4B IQ4_NL | no MTP, turbo3/4, Pinning+Prefetch | 16-18 t/s | 160k | nein |
| **Mars** | AMD 760M (~1GB) | 26B-A4B IQ4_NL | no MTP, turbo3/3 | ~22 t/s | 180k | nein |
| **Hydra** | RTX 3070 (8GB) | 12B IQ4_NL | MTP an (Q3_K_M), turbo4/3 | 56-70 t/s | ~128k | ja |
| **Uranus** | 2x 4060 Ti (32GB) | 26B-A4B oder 12B | MTP an, turbo4/3 | 45-65 t/s | 196k | ja |

**Pattern:** MTP hilft nur auf GPUs mit ausreichender Bandwidth (Ampere/Ada,
~900 GB/s). Auf Low-Bandwidth-GPUs (Pascal 256 GB/s, AMD APU shared) ueberwiegt
der Draft-Overhead den Nutzen.

---

## Wann welches Modell?

| Use-Case | Empfehlung | Begruendung |
|----------|-----------|-------------|
| **Produktion (Chat, RAG, Analyse)** | 26B-A4B, MTP aus | Beste Qualitaet, stabile 16-18 t/s, 160k ctx |
| **Kurze Antworten (Code-Snippets, Q&A)** | 12B, MTP an | 25-33 t/s bei <1500 Tokens, MTP hilft hier |
| **Maximale Geschwindigkeit, geringe Ansprueche** | E4B IQ4_XS | 45 t/s, aber nur 4B dicht — begrenzte Qualitaet |
| **Maximaler Kontext** | 12B, MTP an, ctx=196k | 196k moeglich, aber t/s faellt bei langer Gen |
