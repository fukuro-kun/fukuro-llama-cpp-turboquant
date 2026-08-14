# RADV GTT-Spill-Heuristik: 262144+mmproj Pathologie — Recherche & Lösung

**Datum:** 2026-08-14 bis 2026-08-15
**Status:** GELÖST — `RADV_PERFTEST=nogttspill`
**Betroffen:** AMD gfx1103 (Radeon 760M, RDNA3 Phoenix APU, UMA) mit Vulkan/RADV
**Trilium:** `SWumEN7WOXBI` §5.14–§5.16, `o6jGT8Qwqm4y` (LXC 240: phobos)
**Production-Commit:** `0311122e6`

---

## Zusammenfassung

Bei n_ctx=262144 (2×128k Slots) mit mmproj auf gfx1103 kompilierte RADV FlashAttention-Shader pathologisch langsam (0.07 t/s, ~110s pro 8 Tokens). Drei Erkenntnisstufen führten zur Lösung:

1. **Mesa 25.0.7 → 26.1.2:** Kein Fix
2. **Mesa 26.1.2 → 26.1.6** (nir_opt_dead_write_vars Bug, MR !42038): Fixt 163840+mmproj, aber 262144+mmproj bleibt pathologisch
3. **`RADV_PERFTEST=nogttspill`:** Fixt 262144+mmproj — 0.07 t/s → 23 t/s

---

## Ursprungsproblem

### Beobachtung (seit Juni 2026)

Auf phobos (AMD 760M, RDNA3 Phoenix APU, Vulkan/RADV) trat eine extrem scharfe Performance-Klippe auf:

| n_ctx | mmproj | t/s | Status |
|-------|--------|-----|--------|
| ≤161792 | ja | 32 | ✅ Normal |
| ≥163840 | ja | 0.07 | ❌ Pathologisch (Faktor 457x langsamer) |
| 262144 | nein | 21 | ✅ Normal |

Die Grenze lag exakt bei 163840 = 160 × 1024. Mit mmproj war kein Kontext ≥ 163840 nutzbar.

### Erste falsche Hypothese: Vulkan-Code-Pfad-Wechsel

Ursprünglich vermutet in `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md`:
- Buffer-Platzierungs-Strategie wechselt bei bestimmter Größe
- FlashAttention-Tuning-Parameter-Wechsel
- Integer-Overflow in Buffer-Größen-Berechnung

**Widerlegt:** RCA am 12.07.2026 zeigte dass die ursprüngliche 188k-Klippe ein OOM-Artefakt bei konkurrierenden GPU-Prozessen war (zwei Server gleichzeitig). Aber die 163840-Grenze mit mmproj war real und reproduzierbar auch mit nur einem Server.

---

## Erkenntnisstufe 1: Mesa-Upgrade 25.0.7 → 26.1.2 (Fehlschlag)

**Datum:** 2026-08-14

Mesa 25.0.7 → 26.1.2 (trixie-backports) brachte keine Besserung. Der 163840-Schwellwert mit mmproj blieb.

### Recherche: nir_opt_dead_write_vars Bug

Web-Recherche fand den relevanten Mesa Bug:
- **Mesa MR !42038:** Fix für `nir_opt_dead_write_vars` infinite loop
- **llama.cpp Issue #23755:** "Hang/infinite loop loading Gemma 4 on Vulkan backend with RADV"
- Gemeldet auf RX 7900 XT (dGPU), aber gleicher Bug betrifft alle RADV-GPUs mit Gemma 4

**Hypothese:** Der `nir_opt_dead_write_vars` Pass verursacht einen infinite loop im NIR-Compiler bei Gemma-4-FlashAttention-Shadern. Mesa 26.1.3+ enthält den Fix.

**Problem:** trixie-backports hatte nur Mesa 26.1.2, nicht 26.1.3+.

---

## Erkenntnisstufe 2: Mesa 26.1.6 (Teilerfolg)

**Datum:** 2026-08-15

### Vorgehen

- Temporär Debian `sid`-Repository eingebunden
- Mesa 26.1.6-1 installiert (enthält nir_opt_dead_write_vars-Fix)
- sid-Repository wieder entfernt
- Mesa-Shader-Caches geleert (alte Caches sind an Mesa-Version gebunden)

### Test-Matrix mit Mesa 26.1.6

| n_ctx | mmproj | Startup | t/s | Status |
|-------|--------|---------|-----|--------|
| 163840 | ja | 60s | 20 | ✅ Behoben (war 0.07 t/s) |
| 262144 | ja | 220s | 0.07 | ❌ Weiterhin pathologisch |
| 262144 | nein | 290s | 21 | ✅ Funktioniert |

**Ergebnis:** Der alte 163840-Schwellwert ist behoben. Aber 262144+mmproj bleibt pathologisch. Der nir_opt_dead_write_vars-Fix war notwendig aber nicht hinreichend.

### Diagnose-Korrektur

Die frühere Annahme, `nir_opt_dead_write_vars` sei die alleinige Ursache, war falsch:
- Der NIR-Bug war real und für den 163840-Schwellwert verantwortlich
- Für 262144+mmproj existiert ein **zusätzliches, separates Problem**
- `RADV_DEBUG=nooptimizer` half nicht → Problem ist nicht in NIR-Optimierungspässen

### Untersuchte Upstream-PRs

| PR | Beschreibung | Status im Fork | Relevanz für 262144+mmproj |
|----|-------------|----------------|---------------------------|
| **#19625** | Vulkan FlashAttention Refactor (row splitting, shmem staging, Q caching, vendor-specific Br) | ✅ Bereits integriert (Commit `66e999ecc`) | Keine — bereits vorhanden, Problem persists |
| **#19075** | Vulkan Cooperative Matrix (Coopmat2) Support | ✅ Bereits integriert | Keine — bereits vorhanden, Problem persists |
| **#12087** | Wave32/Wave64 Subgroup Size Tuning | ✅ Evaluiert (#84 in ROADMAP) — Wave64 ist bereits optimal für RDNA3, Wave32 bricht Coopmat-Shader | Keine — Wave64-Default ist korrekt |

**Fazit:** Die relevanten Upstream-PRs waren bereits im Fork. Das Problem lag nicht in fehlenden llama.cpp-Optimierungen, sondern in RADV's Compiler-Heuristik.

---

## Erkenntnisstufe 3: RADV_PERFTEST=nogttspill (Breakthrough)

**Datum:** 2026-08-15

### Recherche: RADV GTT-Spill-Heuristik

Subagent-Recherche (Web + Mesa-Source) klärte den Mechanismus auf:

#### Was ist GTT-Spilling?

Auf AMD GPUs gibt es drei Speichertypen:
- **VRAM** (Device Memory) — dedizierter GPU-Speicher, am schnellsten
- **GTT** (Graphics Translation Table) — System-RAM den die GPU über PCIe adressiert
- **System-RAM** — nur CPU direkt

RADV's GTT-Spill-Heuristik entscheidet beim **Kompilieren** welche Buffer in den GTT ausgelagert werden. Bei großen Buffern (wie KV-Cache bei 262144 Context) aktiviert die Heuristik einen anderen, komplexeren Compiler-Pfad mit Spilling-Code-Generierung.

#### Warum ist das auf dGPUs sinnvoll?

Auf einer dedizierten GPU (z.B. RX 7900 XT, 20GB VRAM) ist GTT = System-RAM über PCIe. Spilling ist langsam (PCIe 4.0 = ~32 GB/s vs. VRAM = ~900 GB/s), aber besser als Absturz. Die Heuristik generiert effizienten Spilling-Code.

#### Warum ist das auf APUs (gfx1103) falsch?

Auf einer UMA-APU wie der Radeon 760M gibt es keinen echten VRAM. Die "GPU" nutzt System-RAM direkt — VRAM und GTT sind **derselbe physikalische Speicher**. Die 27GB die Vulkan meldet sind alles System-RAM über GTT. Spilling ist **kostenlos** — es gibt keinen Geschwindigkeitsunterschied.

Aber: Die GTT-Spill-Heuristik **weiß das nicht**. Sie aktiviert bei großen Buffern den komplexeren Compiler-Pfad, der auf dieser APU völlig unnötig ist. Bei 262144+mmproj wird dieser Pfad **pathologisch** — ACO (der AMD-Compiler-Backend) braucht ~110s pro 8 Tokens.

#### Warum nur mit mmproj?

mmproj fügt vision-spezifische FlashAttention-Pipeline-Varianten hinzu. Die Kombination aus großen KV-Cache-Buffern (262144) + zusätzlichen mmproj-Pipelines überschreitet den Schwellwert der GTT-Spill-Heuristik. Ohne mmproj reicht der KV-Cache allein nicht um die Heuristik zu triggern.

### Test-Matrix: RADV_PERFTEST-Varianten

| RADV_PERFTEST | n_ctx | mmproj | Startup | t/s | Status |
|---------------|-------|--------|---------|-----|--------|
| nircache (Default) | 262144 | ja | 220s | 0.07 | ❌ Pathologisch |
| nircache,nogttspill | 262144 | ja | 80s | 23 | ✅ Behoben |
| nircache,cswave32 | 262144 | ja | 270s+ | — | ❌ Hängt |
| nircache,cswave32,nogttspill | 262144 | ja | 110s | 23 | ✅ Behoben |
| nircache,nogttspill | 163840 | ja | 60s | 20 | ✅ (auch ohne nogttspill) |
| nircache (Default) | 262144 | nein | 290s | 21 | ✅ (ohne mmproj) |

### Schlussfolgerungen

1. **nogttspill ist der entscheidende Flag** — cswave32 allein hängt, nogttspill allein funktioniert
2. **cswave32 ist nicht nötig** — die Recherche-Hypothese dass cswave32 (Wave32 auf RDNA3) der Fix sei war falsch. Wave64 ist der Default auf RDNA3 und korrekt für Coopmat2-Shader
3. **nogttspill ist safe auf UMA-APUs** — GTT == System-RAM, Spilling ist kostenlos, der Flag korrigiert eine falsche Heuristik
4. **Die Heuristik ist ein Mesa/RADV-Bug**, kein llama.cpp-Bug — der richtige Ort für einen Bug-Report wäre `gitlab.freedesktop.org/mesa/mesa`

### cswave32 auf gfx1103 (RDNA3)

Recherche klärte auf:
- Compute-Shader Default auf RDNA3: **wave64** (wegen dual-issue benefits)
- `RADV_PERFTEST=cswave32` schaltet auf wave32 um
- ROCm/HIP unterstützt auf gfx11 nur wave32, aber RADV nutzt wave64 als Default
- Für llama.cpp FA-Shader mit Coopmat2 ist wave64 korrekt — Wave32 bricht Coopmat-Shader (bestätigt in ROADMAP #84)

### nogttspill auf UMA/APUs

Recherche-Hypothese war dass nogttspill auf APUs irrelevant sei (GTT == System-RAM, Spilling kostenlos). **Diese Hypothese war technisch korrekt über die Speicher-Hierarchie, aber falsch in der Schlussfolgerung** — genau weil Spilling kostenlos ist, ist die Heuristik die es aktiviert unnötig und pathologisch.

---

## Lösung

### Production-Umstellung

**Start-Skript:** `scripts/start-mars-26b-server.sh`
**Commit:** `0311122e6`

```bash
# Vorher:
export RADV_PERFTEST="${RADV_PERFTEST:-nircache}"
# -c 161792

# Nachher:
export RADV_PERFTEST="${RADV_PERFTEST:-nircache,nogttspill}"
# -c 262144
```

### Production-Status (seit 15.08.2026)

| Parameter | Wert |
|-----------|------|
| n_ctx | 262144 (2×128k pro Slot) |
| mmproj | ✅ Aktiv (Q6_K, 806MB) |
| KV-Cache | turbo3/3 |
| RADV_PERFTEST | nircache,nogttspill |
| Mesa | 26.1.6-1 (sid-Paket) |
| Startup | ~80s |
| Performance | ~23 t/s (tg), stabil |
| FlashAttention | on |

---

## Drei-Erkenntnisstufen-Zeitleiste

| Datum | Erkenntnis | Ergebnis |
|-------|-----------|----------|
| 2026-06-20 | Performance-Klippe bei ~188k identifiziert | Workaround: 180k Limit |
| 2026-07-12 | RCA: OOM-Artefakt bei konkurrierenden Prozessen | 224k funktioniert solo |
| 2026-08-13 | mmproj reaktiviert, 163840-Schwellwert mit mmproj entdeckt | Workaround: 161792 |
| 2026-08-14 | nir_opt_dead_write_vars Bug identifiziert (Mesa MR !42038) | Mesa-Upgrade nötig |
| 2026-08-15 | Mesa 26.1.6 installiert — 163840 behoben, 262144+mmproj nicht | Teilerfolg |
| 2026-08-15 | RADV_PERFTEST-Varianten getestet — nogttspill ist der Fix | **Breakthrough** |
| 2026-08-15 | Production auf 262144+mmproj+nogttspill umgestellt | **Ziel erreicht** |

---

## Offene Posten

| # | Was | Status | Priorität |
|---|-----|--------|-----------|
| 1 | Mesa Bug-Report für GTT-Spill-Heuristik auf UMA-APUs | Nicht erstellt (sollte ins Mesa GitLab, nicht llama.cpp) | Niedrig |
| 2 | Frischer Shader-Cache-Backup mit Mesa 26.1.6 + nogttspill | Ausstehend | Niedrig |
| 3 | turbo4 V + mmproj + 262144 mit nogttspill testen | Nicht getestet (turbo4 V ist separates Problem) | Niedrig |
| 4 | Mesa 26.1.6 durch Trixie-Backport-Version ersetzen sobald verfügbar | Warten auf Backport | Niedrig |

---

## Referenzen

- **Trilium SWumEN7WOXBI** §5.14 (nir_opt_dead_write_vars Analyse), §5.15 (Mesa 26.1.6 Upgrade), §5.16 (nogttspill Breakthrough)
- **Trilium o6jGT8Qwqm4y** — LXC 240: phobos System-Note
- **llama.cpp Issue #23755** — Hang/infinite loop loading Gemma 4 on Vulkan (geschlossen, Mesa-Fix)
- **Mesa MR !42038** — nir_opt_dead_write_vars Fix
- **`docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md`** — Ursprüngliche Dokumentation der Performance-Klippe
- **`docs/fork/ROADMAP.md`** — #7 (PR #19625 ✅), #12 (PR #19075 ✅), #84 (Wave32/64 ✅)
- **`docs/LLAMA_SERVER_LAN.md`** — Production-Konfiguration phobos
