# K2-Horizon: Bekannte Bugs im MBZUAI-IFM Fork + Patches

Quelle: `vincespeed/K2-Horizon-MoVA-36B-A4B-APEX-GGUF` auf HuggingFace
(validiert gegen MBZUAI-IFM/llama.cpp Commit `35999d101`).
Patches in diesem Verzeichnis; beide Defekte sind im fukuro-Fork-Stand
von `feature/k2-horizon` **vorhanden** (verifiziert 2026-09-26).

## Bug 1 — Tokenizer-Regex crasht auf MSVC (Windows-only)

`src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_K2_HORIZON`: die Pre-Regex
enthält `\u200C|\u200D`-Alternation (Zero-Width (Non-)Joiner), die MSVCs
`std::regex` ablehnt → `Failed to process regex ... regex_error(error_escape)`
beim Modell-Load. Auf Linux irrelevant, aber trivial fixbar.

**Patch:** `0001-fix-k2-horizon-msvc-regex.patch` — entfernt die Alternation.
Vincespeed-Variante nutzt `(?i:'s|...)`-Stil; unser Fork hat bereits
`(?i:...)` (Zeile ~534) — nur `|\\u200C|\\u200D` entfernen.

## Bug 2 — Residual/Norm-Verdrahtung ≠ vLLM-Referenz (ALLE Plattformen)

`src/models/k2-horizon.cpp`, Decoder-Loop: das Modell nutzt
**Parallel-Norm** (GPT-NeoX/PaLM-Stil, vLLM `K2HorizonRMSNorm.forward(x,
residual)`), die MBZUAI-Implementierung verdrahtet den Residual-Pfad
klassisch sequenziell (`inpSA`-Pattern). Symptom: Modell lädt, generiert
aber Garbage/wiederholte Tokens (`> > >`), ggf. Crash bei
Graph-Reservation.

**Patch:** `0002-fix-k2-horizon-parallel-norm.patch` — ~40 Zeilen:
`residual`-Akkumulator über Layer hinweg, addiert VOR jeder Norm
(attn_norm, ffn_norm), Final-Norm auf `inpL + residual`.

**Wichtig:** Der Patch ist "validated fix, not proven diagnosis" — nach
Anwendung Smoke-Test mit echtem GGUF (kohärente Ausgabe, kein `> > >`).
Test-Prompt: `The capital of France is`, temp 0, `--jinja` Pflicht
(eigenes Chat-Template).

## Weitere Runtime-Hinweise aus den Quantisierer-Cards

- `--jinja` ist Pflicht (custom Chat-Template, `<ifm|think>`-Blöcke).
- Empfohlenes Sampling: temp 1.0, top_p 0.95, reasoning effort high.
- `attn_v_gate` (MoVA-Router) muss in Quants F32 bleiben — kein
  Runtime-Bug, aber relevant beim eigenen Quantisieren/Prüfen.
- Vulkan: `GGML_VK_DISABLE_ASYNC=1` auf schwachen iGPUs (NaN-Garbage).
