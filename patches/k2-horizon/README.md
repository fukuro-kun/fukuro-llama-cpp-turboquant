# K2-Horizon: Bekannte Bugs im MBZUAI-IFM Fork + Patches

Quelle: `vincespeed/K2-Horizon-MoVA-36B-A4B-APEX-GGUF` auf HuggingFace
(validiert gegen MBZUAI-IFM/llama.cpp Commit `35999d101`).
Patches in diesem Verzeichnis. Status im fukuro-Fork (`feature/k2-horizon`,
verifiziert 2026-09-27): **Bug 1 übernommen, Bug 2 verworfen** (siehe unten).

## Bug 1 — Tokenizer-Regex crasht auf MSVC (Windows-only) — ÜBERNOMMEN

`src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_K2_HORIZON`: die Pre-Regex
enthält `\u200C|\u200D`-Alternation (Zero-Width (Non-)Joiner), die MSVCs
`std::regex` ablehnt → `Failed to process regex ... regex_error(error_escape)`
beim Modell-Load. Auf Linux irrelevant, aber trivial fixbar.

**Patch:** `0001-fix-k2-horizon-msvc-regex.patch` — entfernt die Alternation.
Vincespeed-Variante nutzt `(?i:'s|...)`-Stil; unser Fork hat bereits
`(?i:...)` (Zeile ~534) — nur `|\\u200C|\\u200D` entfernen.

**Wichtig — Patch allein reicht nicht:** Der Regex-String ist zugleich der
Dispatch-Schlüssel in `src/unicode.cpp` (`unicode_regex_split_custom`
vergleicht wörtlich). Wird die Regex nur in `llama-vocab.cpp` gekürzt,
matcht sie keinen Dispatch-Eintrag mehr → Fallback auf `std::regex` →
stürzt auf MSVC weiterhin ab (und tokenisiert überall falsch, weil der
generische Pfad `(?i:...)` nicht kann). Korrekte Übernahme = **beides**:

1. `|\\u200C|\\u200D` in `src/llama-vocab.cpp` entfernen.
2. In `src/unicode.cpp` die gekürzten Varianten als weitere Alternativen
   im K2-Dispatch aufnehmen, damit weiterhin
   `unicode_regex_split_custom_k2_horizon` greift:
   - `(?i:'s|'t|...)`-Kurzform
   - `(?:'[sS]|'[tT]|...)`-Kurzform (explizit expandierte Groß/Klein-Schreibung)

Die Custom-Splitter-Logik (inkl. ZWNJ/ZWJ-Semantik) bleibt unverändert —
der Regex dient nur als Selektor.

## Bug 2 — Residual/Norm-Verdrahtung — VERWORFEN (Fehldiagnose)

`0002-fix-k2-horizon-parallel-norm.patch` behauptet, die MBZUAI-
Implementierung verdrahte Residuals klassisch sequenziell (`inpSA`-
Pattern), während die Referenz (vLLM/HF) Parallel-Norm nutze.

**Gegenprobe mit `modeling_k2_horizon.py` (HF-Referenz):**
`K2HorizonDecoderLayer.forward` macht klassisch sequenziell
`residual + hidden_states` nach Attention *und* MLP; `K2HorizonRMSNorm`
nimmt gar kein `residual`-Argument. Das `inpSA`-Pattern ist also korrekt.
Der Patch gruppiert dieselbe Rechnung nur assoziativ um (der akkumulierte
`residual` enthält identische Summanden, die Norm sieht identische Eingabe)
— keine Korrektur, nur andere Verdrahtung plus `build_cvec` an anderer
Stelle. **Nicht anwenden**, solange kein reproduzierbarer
Referenz-Mismatch nachgewiesen ist.

**Der eigentliche Garbage-Verursacher war ein anderer Bug:** MBZUAI-Commit
`d6b148a79` fehlte beim Cherry-Pick — die Group-Norm multiplizierte mit
`(1+w)` statt `w`. Nach dem Fix liefert `The capital of France is`
temp 0 + `--jinja` sauber `Paris.` (verifiziert, K2-Horizon-1B-Q4_K_M).

## Weitere Runtime-Hinweise aus den Quantisierer-Cards

- `--jinja` ist Pflicht (custom Chat-Template, `<ifm|think>`-Blöcke).
- Empfohlenes Sampling: temp 1.0, top_p 0.95, reasoning effort high.
- `attn_v_gate` (MoVA-Router) muss in Quants F32 bleiben — kein
  Runtime-Bug, aber relevant beim eigenen Quantisieren/Prüfen.
- Vulkan: `GGML_VK_DISABLE_ASYNC=1` auf schwachen iGPUs (NaN-Garbage).
