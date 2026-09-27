# SESSION_PLAN: Qwen3.8-27B Evaluierung auf Eris

**Erstellt:** 2026-09-27 ~17:00 (User aktiv, kein Solo-Zwang — aber autark durchziehen)
**Auftrag:** Quantitative + qualitative Evaluierung von Qwen3.8-27B auf Eris (16 GB),
Fokus unsloth UD-IQ4_XS + MTP-Sidecar vs. Jackrong Q3_K_L (MTP embedded).
Workflow gemäß `llm-eval`-Skill. Vergleichbarkeit mit K2-Eval durch identischen
11-Prompt-Satz (`eris:/tmp/k2-qual-prompts/`).

## Assets (Download läuft → eris:~/modelle/)

- `Qwen3.8-27B-UD-IQ4_XS.gguf` (14,3 GB — unsloth Dynamic, ohne embedded MTP)
- `mtp-Qwen3.8-27B-Q4_0.gguf` (1,37 GB — MTP-Sidecar aus unsloth-Repo)
- `Qwen3.8-27B-MTP-Q3_K_L.gguf` (14,6 GB — Jackrong, MTP embedded als blk.64)

## Besonderheiten Qwen3.8

- Arch `qwen35` (hybrid: 48 DeltaNet + 16 Gated-Attention Layer) → KV nur auf
  ~1/4 der Layer → deutlich kleinerer KV-Cache als K2 → mehr Kontext pro VRAM
- MTP via `--spec-type draft-mtp --spec-draft-n-max 2`
  - embedded (Jackrong): Flag reicht
  - unsloth: zusätzlich `--spec-draft-model mtp-....gguf`
- Aktiv-Nachweis im Log: `creating MTP draft context` + draft-acceptance-Stats
  (Warnung "unused tensor blk.64" ist kosmetisch, upstream #26765)
- DeltaNet-CUDA-Bug in alten Builds → Smoke-Test auf Garbage-Output VOR Sweep
- Vision (mmproj) NICHT Teil dieser Eval — Text-only

## Phasen

### Phase 1: Smoke-Test (~10 min)
- UD-IQ4_XS laden (ngl 30, c8192), kurze Generation → prüfen auf Garbage
  (DeltaNet-Bug-Check) + ob `qwen35` sauber erkannt wird
- MTP-Variante laden → `creating MTP draft context` im Log verifizieren

### Phase 2: Frontier (~60-90 min)
- Bisection stabiles ngl pro ctx für: UD-IQ4_XS, Q3_K_L
- ctx: 8192, 32768, 65536, 131072 (+optional 262144 native!)
- Erfolg = komplette Generation (nicht nur Load); Marge ngl−1
- Zusatz-Frage: KV turbo4/3 greift auf DeltaNet-Hybrid wie erwartet?
  (nur Attention-Layer haben KV → Kosten deutlich < K2)

### Phase 3: Qual-Suite (~90-120 min)
- Identische 11 Prompts wie K2-Eval (temp 0.7, max_tokens 4092)
- Drei Läufe: UD-IQ4_XS (ohne MTP), UD-IQ4_XS+MTP-Sidecar, Q3_K_L (MTP embedded)
- Auswertung via analyze-outputs.py (tail_rep-Metrik, Klassifikation)
- MTP-Varianten: Acceptance-Stats aus Server-Log ziehen

### Phase 4: Doku + Restore
- Vergleichstabelle vs. K2-Ergebnisse (APEX-mini Referenz: 37,5 t/s @32k/ngl44)
- Trilium: neue/aktualisierte Modell-Notiz + GPU-Bench-Notiz + TTT
- eris-chat restaurieren + health prüfen
- Archiv-Hinweis: Modelle anschließend auf Ganymed-Verleiter ablegen (falls
  noch nicht geschehen) — kanonischer Ort für GGUFs

## Defaults
- Bei Garbage-Output/Gewichtsfehlern → Variante dokumentieren + andere testen
- Bei MTP-Crash → Baseline ohne spec ist trotzdem valides Ergebnis
- pkill nie mit Port-Pattern; Kill via PID; curl -o für Bodies
