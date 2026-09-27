# SESSION_PLAN: Qwen3.8-27B Evaluierung auf Eris

**Erstellt:** 2026-09-27 ~17:00 (User aktiv, kein Solo-Zwang — aber autark durchziehen)
**Auftrag:** Quantitative + qualitative Evaluierung von Qwen3.8-27B auf Eris (16 GB),
Fokus unsloth UD-IQ4_XS + MTP-Sidecar vs. Jackrong Q3_K_L (MTP embedded).
Workflow gemäß `llm-eval`-Skill. Vergleichbarkeit mit K2-Eval durch identischen
11-Prompt-Satz (`eris:/tmp/k2-qual-prompts/`).

## Assets (Download läuft → eris:~/modelle/)

- `Qwen3.8-27B-UD-IQ4_XS.gguf` (14,3 GB — unsloth Dynamic, ohne embedded MTP)
- `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13,1 GB — unsloth Dynamic; **Voll-Offload-Kandidat:**
  13,1 + ~0,6 GB KV@32k (nur 16 Attention-Layer) + ~1 GB Compute ≈ 15 GB → passt
  potenziell komplett in 16 GB VRAM; MTP-Sidecar (+1,4 GB) macht es eng → separat)
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

## Interim-Ergebnisse (~18:10)

- Smoke: **kein DeltaNet-Garbage** — saubere dt. Antwort, arch=qwen35, nextn=1 in allen Files
- Partial (ngl30, IQ4_XS, 8k): 3,5 t/s — dense 27B ist CPU-Limit-dominiert
- **Voll-Offload Q3_K_XL (ngl99): 22,7 t/s @8k, 22,6 @131k, 22,4 @160k** — KV auf nur
  16 Attention-Layern → Kontext kostet fast keine Speed/VRAM
- Frontier: 160k stabil (15,6 GB), 192k OOM Compute-Buffer, 262k OOM
- **MTP n=2 Sidecar: 28,3 t/s @65k** (Acceptance ~49 %), n=3 schlechter (23,1)
- Jackrong Q3_K_L embedded-MTP @65k: **21,7 t/s** — langsamer (größere Datei), Acceptance identisch
- **Dauerlast-Throttle:** 89 °C → SM 255 MHz, sustained tg fällt auf ~9–14 t/s
  (Burst ≠ Dauerlast — wichtiger Produktiv-Faktor auf Eris)
- Qual (UD-Q3_K_XL+MTP, 11 Prompts, 4092): **0 Loops** (alle rep=1, vs K2-DERISKED!).
  6× finish=length mit leerem content = denkt >4092 Token im Thinking → 8k-Retest
  läuft für a1/a2/a4/c1/c2. Alignment: c3–c6 saubere Refusals MIT defensiven
  Alternativen (besser strukturiert als K2-APEX); aligned base ≠ uncensored.
- Neue Downloads (Subagent-Recherche): HauhauCS-Aggressive Q3_K_P 13,4 GB
  (uncensored, MTP embedded, 0/465-Refusals-Claim) + agentionai AP-IQ4_XS 13,27 GiB
  (beste KLD-Fidelity der Klasse, MTP embedded) — Jackrong raus (zu groß/langsam)
- WICHTIG: Embedded-MTP ≠ langsam — Jackrongs Problem war Dateigröße, nicht
  Einbettung. Embedded bei ~13,3 GB sollte ~unsloth+Sidecar-Niveau liefern.

### Eris-Performance @65k, MTP n=2 (2500 tok, Telemetrie-Protokoll)

**Thermal-Befund:** Eris (ROG Zephyrus S17) drosselte unter
`platform_profile=balanced` innerhalb von <10 s (89 °C → SM 210 MHz, ~64 W).
Fix: `platform_profile=performance` + `asus-nb-wmi fan_boost_mode=1` →
**~2× Speed**, Peak 83 °C, keine Dauer-Throttle. Lüfter laufen EC-gesteuert.

| Config | balanced | performance | acc |
|---|---|---|---|
| unsloth Q3_K_XL, kein MTP | — | 18,2 t/s | — |
| unsloth + Q4_0-Sidecar | 10,9 t/s | 21,2 t/s | 60 % |
| HauhauCS Q3_K_P (embedded MTP) | 14,6 t/s | **29,2 t/s** | 58 % |
| agentionai AP-IQ4_XS (embedded) | 10,0 t/s | **29,6 t/s** | 59 % |

- Embedded MTP ≈ +60 % über no-MTP-Baseline; Sidecar nur +16 % (eigener
  Draft-Forward + 1,4 GB Extra-VRAM). Jackrong-Einbettung war unschuldig.
- Frontier Eris: HauhauCS 65k ok (14,7 GB), 131k **OOM** (KV passt nicht).
  AP @65k = 15,5 GB → Frontier niedriger. unsloth+Sidecar 15,2 GB.
- Qual-Eval läuft auf Uranus (2×4060Ti, 29–33 t/s sustained, kein Throttle):
  unsloth 11/11 komplett (a2/a4/c1/c2 mit 32k nachgeholt, alle CONTENT);
  HauhauCS + AP-Suiten folgen — dort liegt der Uncensored-/Qualitäts-Befund.
- Ganymed-Archiv befüllt: `Qwen/Qwen3.8-27B/{unsloth,HauhauCS,agentionai,drafts}/`
  — hatte Teile schon seit Aug. (Skill-Lektion: Archiv vor HF prüfen).

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
