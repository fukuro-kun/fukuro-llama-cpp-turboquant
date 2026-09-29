# SESSION_PLAN — Vision-Eval: Qwen3.8-27B (swiftal + swiftunc) auf Eris

**Datum:** 2026-09-29 (Solo-Session) | **Autor:** Devin
**Referenz:** `inferenzquelle/docs/SESSION_PLAN_vision_eval.md` (Gemma-4 vs GLM-4.6V)

## Session-Ziel

Empirisch klären wie gut die beiden entschiedenen Qwen3.8-Kandidaten
(`swiftal` aligned + `swiftunc` uncensored) Vision-Aufgaben lösen —
mit GPU-offloaded mmproj auf Eris. Ergebnis: Vergleichstabelle
Qualität (0–3 gegen fixierter Ground-Truth) + Latenz + Empfehlung ob
Qwen3.8 als Vision-Modell auf Eris taugt (Vergleichs-Referenz: GLM-4.6V,
Gemma-4/phobos).

## Entscheidungen (mit User geklärt)

| Frage | Entscheidung |
|-------|--------------|
| Varianten | **beide** — swiftal + swiftunc (sequentiell, ein Server aktiv) |
| Gemma auf eris:18080 | **stoppen, nach Eval identisch wiederherstellen** (Cmdline sichern) |
| mmproj-Pfad | **nur GPU @64k** — ctx 65536, mmproj GPU-Offload (~1–2 s Encode statt ~150 s CPU) |
| Granularität | **3 Stufen** (--image-min/max-tokens 256/512/1024) auf ~4 Kern-Bildern |
| Korpus | bestehender 13-Bilder-Korpus `/tmp/vision-eval/corpus/` + `ground_truth.md` (fixiert, von Gemma/GLM-Eval) |
| mmproj-Datei | `mmproj-Qwen3.8-27B-F16.gguf` von ganymed (`shared-mmproj/`); Fallback Q8_0; letzter Fallback F32 @/tmp |
| Requests | Direkt an eris-Testport 18088 — kein Router-Umbau |
| Sampler | `temp=0.7 top_p=0.95 min_p=0.0 seed=-1`, `reasoning_effort="low"` (Vision-Beschreibung braucht kein medium; hält Laufzeit im Rahmen) |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 2` (Produktions-Realität, funktioniert mit mmproj laut Kampagnen-§5) |
| Scoring | Ich (Devin) score 0–3 gegen ground_truth.md + Flags (refusal/loop/hallucination/empty) |
| Output | Ledger `/tmp/vision-eval-qwen38/` auf hydra, RANKING.md-Update, Trilium, TTT |

## VRAM-Budget (16 384 MiB)

| Modell | Gewichte | KV@64k turbo4/3 | Recurrent+Compute+Draft | mmproj-F16 | Summe |
|--------|----------|-----------------|-------------------------|------------|-------|
| swiftal (11,3 GiB) | ~11 160 MiB | ~1 172 MiB | ~1 550 MiB | ~900 MiB | **~14,8 GiB** ✓ |
| swiftunc (13,1 GiB) | ~12 100 MiB | ~1 172 MiB | ~1 550 MiB | ~900 MiB | **~15,7 GiB** knapp |

Fallbacks bei OOM: ctx 57344 → 49152; mmproj Q8_0 (~0,5 GiB); Draft-KV bleibt
f16 (§7 RANKING: Quant bringt nichts unter ~130k).

## Schritte

| # | Schritt | Status |
|---|---------|--------|
| 1 | SESSION_PLAN + Korpus/Material sichern | ✓ |
| 2 | F16-mmproj ganymed→eris transferieren | ☐ |
| 3 | Gemma-Cmdline sichern, Gemma stoppen, VRAM-Freigabe prüfen | ☐ |
| 4 | swiftal-Server @65536 + GPU-mmproj auf :18088 | ☐ |
| 5 | Smoke: 1 Bild — GPU-Encode-Zeit + tatsächliche Bild-Token-Zahl verifizieren | ☐ |
| 6 | Granularitäts-Mechanik klären: per-request möglich? sonst Server-Restart pro Stufe | ☐ |
| 7 | Haupt-Eval swiftal: 13 Bilder × Standard/OCR-Prompt @ Default-Granularität (512) | ☐ |
| 8 | Sweep: 4 Kern-Bilder × 256/1024 (Restarts) | ☐ |
| 9 | Modellwechsel swiftunc (ctx adaptiv), Schritte 5–7 wiederholen | ☐ |
| 10 | Scoring gegen ground_truth.md, Ergebnis-Tabelle | ☐ |
| 11 | Vergleich vs GLM-4.6V/Gemma (Ledger `/tmp/vision-eval/results/`) | ☐ |
| 12 | Gemma wiederherstellen (exakte Cmdline, Port 18080, Health-Check) | ☐ |
| 13 | RANKING.md-Update + Trilium + TTT + Bericht | ☐ |

## Prompts (identisch Korpus-Definition)

- Standard: „Beschreibe detailliert was du auf diesem Bild siehst."
- OCR: „Transkribiere allen sichtbaren Text exakt."
- Spezial: aus `corpus.json` (kategorie-spezifisch, z.B. Chart-Werte)

## Defaults

- Timeout 300 s/Request; max_tokens 2048 (OCR-Vollseiten); Retry 1× bei
  leerem Content/HTTP-Fehler
- GPU-Exklusivität: immer nur EIN llama-server auf eris — Stop →
  `nvidia-smi` ~0 MiB → Start
- Bei Load-Fail: ctx in 8k-Schritten senken, erst dann mmproj-Quant kleiner

## Recherche-Fallbacks

| Problem | Fallback |
|---------|----------|
| F16-mmproj inkompatibel | Q8_0, dann F32 @/tmp/mmproj-F32-test.gguf |
| `--image-*-tokens` ignoriert (Qwen-Arch) | Mechanik ehrlich dokumentieren; Sweep per Bild-Resize |
| MTP+mmproj-Konflikt | ohne `--spec-type` weiterfahren, vermerken |
| 0-Content-Loop (bekannt ~8 %) | Retry seed=-1 einmal; als Failure-Flag zählen |

## Folge-Plan

Ergebnis entscheidet ob ein `eris-vlm`-Profil-Update sinnvoll ist
(Qwen3.8 statt GLM-4.6V als On-Demand-VLM) — Implementierung in separater
Session.
