# MTP Draft-Vergleich — Styx (GTX 1070, Pascal, MoE-Offload)

**Datum:** 2026-07-14 | **System:** Styx (GTX 1070, 8GB VRAM, CUDA) | **Modell:** Gemma-4-26B-A4B QAT Q4_K_XL

## Fragestellung

Der 08.07. Solo-Session-Report dokumentierte 31.9 t/s mit MTP (+50% über Config-B-Baseline 21.3 t/s). Dieser Wert wurde in der Wochenrückschau als "+150% tg with MTP" zitiert — ein Vergleich zwischen Config A (alle Experten CPU, kein Pinning) und Config B + MTP (mit Pinning), was irreführend ist. Zudem wurde der Q4_K_M-Draft nie mit dem QAT-Modell getestet. Dieser Benchmark klärt:

1. Was ist die tatsächliche MTP-Performance mit QAT?
2. Welcher Draft ist besser — Q4_K_M (325 MB) oder Q4_0 (241 MB)?
3. Lohnt sich MTP in Produktion auf Styx?

## Setup

| Parameter | Wert |
|-----------|------|
| Modell | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14.2 GB) |
| Kontext | 8192 (8k) — um `-ngld 999` (Draft auf GPU) zu ermöglichen |
| MoE-Offload | `--n-cpu-moe 20` (obere 20 Layer Experten auf CPU) |
| KV-Cache | `-ctk turbo3 -ctv turbo4` |
| FlashAttn | on |
| Pinning+Prefetch | `GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_SLOTS=2` |
| Test-Prompt | "Schreibe eine kurze Einführung in die Quantenmechanik mit drei Absätzen." |
| Generation | 200 tokens, temp=1.0, top-p=0.95, top-k=64 |
| Draft-Modelle | Q4_K_M: `gemma-4-26b-a4b-it-assistant.Q4_K_M.gguf` (325 MB); Q4_0: `mtp-gemma-4-26B-A4B-it-Q4_0.gguf` (241 MB) |

**Wichtig:** Bei Produktiv-Kontext (224k) ist `-ngld 999` nicht möglich (OOM — Draft braucht ~712 MB VRAM, die bei 224k nicht frei sind). Mit `-ngld 0` (Draft auf CPU) sinkt die Performance auf 14.89 t/s (-44% vs Baseline) — der CPU-Forward des Drafts pro Token ist katastrophal. Daher wurde der Benchmark mit 8k Kontext durchgeführt, wo der Draft auf GPU passt.

## Ergebnisse

| Konfiguration | tg t/s | n_decode | tok/decode | ms/decode | vs Baseline |
|---------------|--------|----------|------------|-----------|-------------|
| **Baseline** (QAT, kein MTP) | **26.65** | 202 | 0.99 | 37.5 | — |
| **Q4_0 Draft** MTP (-ngld 999) | **24.29** | 75 | 2.67 | 41.2 | **-9%** |
| **Q4_K_M Draft** MTP (-ngld 999) | **20.97** | 87 | 2.30 | 47.7 | **-21%** |
| Q4_K_M Draft (-ngld 0, 224k ctx) | 14.89 | — | — | — | **-44%** |

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

### Warum MTP bei MoE-Offload nicht funktioniert

Styx betreibt 26B-A4B mit `--n-cpu-moe 20` (obere 20 Layer Experten auf CPU). Die Baseline-Decode-Zeit (37.5ms) wird dominiert von CPU-Expert-Fetch + H2D-Copy. Der Draft-Forward adds weiteren Compute auf die GPU, die bereits die Attention + untere 20 Layer Experten bedient. Der Draft-Overhead (4-10ms/decode) überwiegt den Benefit der akzeptierten Tokens, weil die Baseline-Decode-Zeit durch den MoE-Offload ohnehin hoch ist.

Auf Systemen mit vollem GPU-Offload (kein MoE-Offload) wäre die Baseline-Decode-Zeit viel kürzer (~10-15ms), und der Draft-Overhead wäre relativ kleiner — dort könnte MTP helfen. Aber Styx hat nur 8GB VRAM und muss offloaden.

## Fazit

1. **MTP ist auf Styx (Pascal, MoE-Offload) ein Netto-Nachteil mit beiden Drafts.** Q4_0: -9%, Q4_K_M: -21%.
2. **Q4_0 ist der bessere Draft** (höhere Akzeptanz, geringerer Overhead).
3. **Die Entscheidung vom 10.07., MTP abzuschalten, war korrekt.**
4. **Der 31.9 t/s-Wert vom 08.07. gilt nur für IQ4_NL Config B und ist nicht auf QAT übertragbar.**
5. **Bei Produktiv-Kontext (224k) ist MTP ohnehin unmöglich** (Draft passt nicht auf GPU, CPU-Draft ist -44%).

## Korrektur zur Wochenrückschau

Die Wochenrückschau `docs/fork/2026-07-14_WEEKLY_REVIEW.md` und der TTT-Eintrag vom 14.07. enthalten "+150% tg with MTP (12.7→31.9 t/s)". Das ist irreführend, weil es Config-A-Baseline (alle Experten CPU, kein Pinning) mit Config-B+MTP vergleicht. Der korrekte MTP-Boost ist +50% (21.3→31.9 t/s), bezogen auf die gleiche Config B — und gilt nur für IQ4_NL, nicht für QAT. Mit QAT ist MTP ein Netto-Nachteil (-9% bis -21%).
