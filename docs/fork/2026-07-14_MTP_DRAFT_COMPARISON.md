# MTP Draft-Vergleich — Styx (CUDA, MoE-Offload) + Mars (Vulkan, voller GPU-Offload)

**Datum:** 2026-07-14 | **Systeme:** Styx (GTX 1070, 8GB VRAM, CUDA, MoE-Offload) + Mars/phobos (AMD Radeon 760M, Vulkan, voller GPU-Offload) | **Modell:** Gemma-4-26B-A4B QAT Q4_K_XL

## Fragestellung

Der 08.07. Solo-Session-Report dokumentierte 31.9 t/s mit MTP (+50% über Config-B-Baseline 21.3 t/s). Dieser Wert wurde in der Wochenrückschau als "+150% tg with MTP" zitiert — ein Vergleich zwischen Config A (alle Experten CPU, kein Pinning) und Config B + MTP (mit Pinning), was irreführend ist. Zudem wurde der Q4_K_M-Draft nie mit dem QAT-Modell getestet. Dieser Benchmark klärt:

1. Was ist die tatsächliche MTP-Performance mit QAT?
2. Welcher Draft ist besser — Q4_K_M (325 MB) oder Q4_0 (241 MB)?
3. Lohnt sich MTP in Produktion auf Styx?

## Setup

| Parameter | Styx | Mars/phobos |
|-----------|------|-------------|
| GPU | GTX 1070 (Pascal, 8GB VRAM, CUDA) | AMD Radeon 760M (RDNA3, Vulkan) |
| Modell | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14.2 GB) | gleicher |
| Kontext | 8192 (8k) | 8192 (8k) |
| MoE-Offload | `--n-cpu-moe 20` (obere 20 Layer auf CPU) | **kein** (voller GPU-Offload, `-ngl 999`) |
| KV-Cache | `-ctk turbo3 -ctv turbo4` | `-ctk turbo3 -ctv turbo4` |
| FlashAttn | on | on |
| Pinning+Prefetch | `GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_SLOTS=2` | `GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_SLOTS=2` |
| Test-Prompt | "Schreibe eine kurze Einführung in die Quantenmechanik mit drei Absätzen." | gleicher |
| Generation | 200 tokens, temp=1.0, top-p=0.95, top-k=64 | gleicher |
| Draft-Modelle | Q4_K_M (325 MB), Q4_0 (241 MB) | gleicher (auf `/jade/models/.../drafts/`) |

**Warum 8k Kontext:** Bei Produktiv-Kontext (Styx 224k, Mars 256k) ist `-ngld 999` nicht möglich (OOM — Draft braucht ~712 MB VRAM, die bei vollem Kontext nicht frei sind). Mit `-ngld 0` (Draft auf CPU) sinkt die Performance katastrophal (Styx: 14.89 t/s, -44%). Daher Benchmark mit 8k, wo der Draft auf GPU passt.

## Ergebnisse

### Styx (GTX 1070, CUDA, MoE-Offload `--n-cpu-moe 20`)

| Konfiguration | tg t/s | n_decode | tok/decode | ms/decode | vs Baseline |
|---------------|--------|----------|------------|-----------|-------------|
| **Baseline** (QAT, kein MTP) | **26.65** | 202 | 0.99 | 37.5 | — |
| **Q4_0 Draft** MTP (-ngld 999) | **24.29** | 75 | 2.67 | 41.2 | **-9%** |
| **Q4_K_M Draft** MTP (-ngld 999) | **20.97** | 87 | 2.30 | 47.7 | **-21%** |
| Q4_K_M Draft (-ngld 0, 224k ctx) | 14.89 | — | — | — | **-44%** |

### Mars/phobos (AMD Radeon 760M, Vulkan, voller GPU-Offload)

| Konfiguration | tg t/s | n_decode | tok/decode | ms/decode | vs Baseline |
|---------------|--------|----------|------------|-----------|-------------|
| **Baseline** (QAT, kein MTP) | **25.79** | 202 | 0.99 | 38.8 | — |
| **Q4_0 Draft** MTP (-ngld 999) | **26.70** | 74 | 2.70 | 37.5 | **+3.5%** (im Rauschen) |
| **Q4_K_M Draft** MTP (-ngld 999) | **21.83** | 87 | 2.30 | 45.8 | **-15%** |

### Systemvergleich

| Eigenschaft | Styx (MoE-Offload) | Mars (voller GPU-Offload) |
|-------------|--------------------|---------------------------|
| Baseline ms/decode | 37.5ms | 38.8ms |
| Q4_0 ms/decode | 41.2ms (+10%) | 37.5ms (-3%) |
| Q4_0 tok/decode | 2.67 | 2.70 |
| Q4_0 Netto-Effekt | **-9%** | **+3.5%** (neutral) |
| Q4_K_M Netto-Effekt | **-21%** | **-15%** |

**Beobachtung:** Auf Mars (voller GPU-Offload) ist der Q4_0-Draft-Overhead praktisch null — die Decode-Zeit bleibt gleich (38.8→37.5ms) trotz zusätzlichem Draft-Forward. Auf Styx (MoE-Offload) kostet der Draft +10% Decode-Zeit. Die Akzeptanz ist auf beiden Systemen ähnlich (2.67-2.70 tok/decode).

## Analyse

### MTP funktioniert technisch, bremst aber

`n_decode_total` sinkt von 202 (Baseline) auf 75-87 (MTP) — der Draft akzeptiert 2.3-2.7 Tokens pro Decode. Das ist eine echte Akzeptanzrate von ~57-63% (200 Tokens / 75-87 Decodes). Aber jeder Decode wird teurer:

- **Baseline:** 37.5ms/decode → 1 Token → 26.65 t/s
- **Q4_0:** 41.2ms/decode → 2.67 Tokens → 24.29 t/s (+10% Decode-Zeit für +170% Tokens → Netto -9%)
- **Q4_K_M:** 47.7ms/decode → 2.30 Tokens → 20.97 t/s (+27% Decode-Zeit für +130% Tokens → Netto -21%)

### Q4_0 ist der bessere Draft

| Eigenschaft | Q4_0 (241 MB) | Q4_K_M (325 MB) |
|-------------|---------------|-----------------|
| Akzeptanz (tok/decode) | **2.67** | 2.30 |
| ms/decode | **41.2** | 47.7 |
| tg t/s | **24.29** | 20.97 |
| VRAM-Bedarf | geringer | größer (+84 MB) |

Q4_0 hat höhere Akzeptanz UND geringeren Overhead. Der größere Q4_K_M-Draft braucht mehr Compute pro Forward, akzeptiert aber schlechter — ein doppelter Nachteil.

### Warum der 08.07. Report 31.9 t/s zeigte

Der 08.07. Wert ist nicht falsch, aber nicht auf QAT übertragbar:

1. **Anderes Modell:** IQ4_NL (nicht QAT) — Baseline 21.3 t/s (Config B). QAT-Baseline ist 26.65 t/s (+25%).
2. **Kurztest:** Nur 11 Tokens mit "100% Akzeptanz" — statistisch nicht aussagekräftig. Realistische Akzeptanz (200 Tokens) liegt bei 57-63%.
3. **Q4_K_M Draft:** Der größere, schlechtere Draft (wie dieser Benchmark bestätigt).
4. **+50% MTP-Boost** (21.3 → 31.9) war korrekt für IQ4_NL Config B — gilt aber nicht für QAT, wo die Baseline schon schneller ist.

### Warum MTP bei MoE-Offload (Styx) nicht funktioniert, bei vollem Offload (Mars) aber neutral ist

Styx betreibt 26B-A4B mit `--n-cpu-moe 20` (obere 20 Layer Experten auf CPU). Die Baseline-Decode-Zeit (37.5ms) wird dominiert von CPU-Expert-Fetch + H2D-Copy. Der Draft-Forward adds weiteren Compute auf die GPU, die bereits die Attention + untere 20 Layer Experten bedient. Der Draft-Overhead (+10% Decode-Zeit) überwiegt den Benefit der akzeptierten Tokens.

Auf Mars (voller GPU-Offload, kein MoE-Offload) ist der Draft-Overhead praktisch null — die Decode-Zeit bleibt gleich (38.8→37.5ms) trotz zusätzlichem Draft-Forward. Die GPU hat genug Parallelität, um Draft und Backbone gleichzeitig zu berechnen. Die Akzeptanz (2.70 tok/decode) reicht aus, um den Overhead zu kompensieren → Netto +3.5% (im Rauschen, aber zumindest neutral).

Q4_K_M ist auf beiden Systemen ein Netto-Nachteil (-15% bis -21%): der größere Draft braucht mehr Compute, akzeptiert aber schlechter (2.30 tok/decode).

## Fazit

1. **MTP mit Q4_0 Draft ist auf Styx (MoE-Offload) ein Netto-Nachteil (-9%)**, auf Mars (voller GPU-Offload) **praktisch neutral (+3.5%, im Rauschen)**.
2. **Q4_K_M Draft ist auf beiden Systemen ein Netto-Nachteil** (-15% bis -21%) — schlechtere Akzeptanz bei höherem Overhead.
3. **Q4_0 ist der bessere Draft** auf beiden Systemen (höhere Akzeptanz 2.67-2.70 vs 2.30, geringerer Overhead).
4. **Die Entscheidung vom 10.07., MTP abzuschalten, war korrekt** für Styx. Auf Mars ist Q4_0-MTP neutral — könnte aktiviert werden, bringt aber keinen nennenswerten Speedup.
5. **Der 31.9 t/s-Wert vom 08.07. gilt nur für IQ4_NL Config B und ist nicht auf QAT übertragbar.**
6. **Bei Produktiv-Kontext (Styx 224k, Mars 256k) ist MTP ohnehin unmöglich** (Draft-OOM auf GPU, CPU-Draft ist -44%).
7. **MTP könnte auf Mars bei kleinerem Kontext (z.B. 32k-64k) einen leichten Benefit bringen** — aber das Produktiv-Setup nutzt 256k, wo der Draft nicht auf GPU passt.

## Korrektur zur Wochenrückschau

Die Wochenrückschau `docs/fork/2026-07-14_WEEKLY_REVIEW.md` und der TTT-Eintrag vom 14.07. enthalten "+150% tg with MTP (12.7→31.9 t/s)". Das ist irreführend, weil es Config-A-Baseline (alle Experten CPU, kein Pinning) mit Config-B+MTP vergleicht. Der korrekte MTP-Boost ist +50% (21.3→31.9 t/s), bezogen auf die gleiche Config B — und gilt nur für IQ4_NL, nicht für QAT. Mit QAT ist MTP ein Netto-Nachteil (-9% bis -21%).
