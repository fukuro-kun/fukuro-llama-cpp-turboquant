# Masterplan: Vulkan Performance-Klippe auf Mars — Systematische RCA

**Datum:** 2026-06-21
**Ziel:** Root Cause für PP-Klippe (~16k) und TG-Klippe (~188k) auf Mars (AMD 760M, RADV PHOENIX) finden und beheben.
**Dauer:** 12h KI-Solo-Schicht
**Branch:** `feature/vulkan-perf-rca` (neu, von `master`)

---

## Ausgangslage

### Systeme
| System | GPU | Rolle |
|--------|-----|-------|
| **Mars** | AMD 760M (RDNA3, RADV PHOENIX, UMA) | Test-System — hier tritt das Problem auf |
| **Hydra** | NVIDIA RTX 3070 (CUDA) | Build-System, git-Operationen, CUDA-Baseline |

### Fork-Struktur
- Fork von Fork von Fork: `ggml-org → TheTom → AtomicBot → fukuro`
- **1057 Commits hinter upstream**, davon **61 Vulkan-spezifisch**
- Vollständiger Rebase nicht praktikabel (würde TurboQuant/MTP/NextN/DiffusionGemma zerstören)
- Strategie: Selektiv Vulkan-Commits cherry-picken, TurboQuant intakt halten

### Bekannte Probleme
1. **TG-Klippe bei ~188k:** 24 t/s → 0.09 t/s (Faktor 243x) — scharfer Knick
2. **PP-Klippe bei ~16k:** pp8192=150 t/s ✅, pp16384=HANG ❌
3. **KV-Cache-Hang bei ctx ≥ 186k:** Modell laden hängt bei "......" dots
4. **Pipeline-Cache-Korruption:** Nach Code-Änderungen muss Cache gelöscht werden

### Bereits ausgeschlossene Hypothesen
- ❌ `maxStorageBufferRange` (128 MiB) — ist 4 GiB auf Mars
- ❌ `shader_64bit_indexing` fehlt — wird nicht benötigt
- ❌ FA-Tuning-Parameter-Wechsel — Parameter bleiben konstant
- ❌ uint32_t Overflow in Push Constants — Overflow bei ~21M Tokens
- ❌ `maxComputeWorkGroupCount[0]` — wird bei N=1 nicht überschritten
- ❌ UMA HostCached-Preference (PR #23762) — System-RAM ist langsamer als GTT für GPU-Compute

---

## Phase 1: Baseline etablieren (0-2h)

### 1.1 Branch erstellen
```bash
cd ~/git/fukuro-llama-cpp-turboquant
git checkout master
git checkout -b feature/vulkan-perf-rca
```

### 1.2 Mars: Post-Reboot Verification
**Skript:** `/tmp/mars_post_reboot.sh` (liegt bereit auf Mars)

Reproduziert die Venus-Verification auf Mars:
1. Pipeline-Cache löschen
2. Backend-Op-Tests: `SET_ROWS_TURBO3`, `FLASH_ATTN_EXT turbo3`
3. Smoke-Test: `llama-cli` mit turbo3+FA, ctx=32k
4. Benchmark: turbo3 vs f16, mit/ohne FA (pp256, tg64)
5. PP-Scaling: pp512, pp4096, pp8192

**Wichtig:** Nach jedem Testlauf Pipeline-Cache löschen!

### 1.3 Baseline-Dokumentation
Alle Ergebnisse in `docs/fork/2026-06-21_VULKAN_PERF_RCA_BASELINE.md` dokumentieren:
- Gerätedaten (vulkaninfo)
- Memory-Types
- Backend-Op-Test-Ergebnisse
- Benchmark-Ergebnisse (pp/tg für turbo3, f16, mit/ohne FA)
- PP-Scaling-Ergebnisse

### Erfolgskriterium Phase 1
- turbo3+FA Korrektheit bestätigt (wie auf Venus)
- Baseline pp/tg Zahlen für Mars dokumentiert
- PP-Klippe reproduziert (oder nicht — nach Reboot + clean Cache)

---

## Phase 2: Systematische Untersuchung (2-8h)

### 2.1 PP-Scaling Matrix (Mars)
Mit cleanem Pipeline-Cache nach jedem Lauf:

| pp | ctx=n+1 | Erwartung |
|----|---------|-----------|
| 512 | 513 | ~205 t/s |
| 4096 | 4097 | ~168 t/s |
| 8192 | 8193 | ~150 t/s |
| 10240 | 10241 | ? |
| 12288 | 12289 | ? |
| 14336 | 14337 | ? |
| 16384 | 16385 | HANG? |

Ziel: Exakte Klippe lokalisieren.

### 2.2 TG-Scaling Matrix (Mars)
Mit `llama-server` + API, kurzer Prompt (100 Tokens), ctx variiert:

| ctx | tg8 | Erwartung |
|-----|-----|-----------|
| 180000 | ? | ~24 t/s |
| 184000 | ? | ~24 t/s |
| 186000 | ? | HANG? |
| 188000 | ? | 0.09 t/s? |

**Wichtig:** Server mit `--no-warmup` starten, kurzen Prompt senden, nur tg messen.

### 2.3 Vulkan Commit-Bisect (Hydra + Mars)

**Strategie:** Nicht 1057 Commits bisecten, sondern gezielt die 61 Vulkan-Commits.

#### Schritt 1: Vulkan-Commit-Liste erstellen (Hydra)
```bash
git log --oneline HEAD..upstream/master -- ggml/src/ggml-vulkan/ | tac
```
→ Chronologische Liste der 61 Vulkan-Commits.

#### Schritt 2: Kategorisierung
Commits einteilen in:
- **🔴 Performance-relevant:** FA, MatMul, Buffer, Memory, Dispatch
- **🟡 Korrektheit:** Bugfixes, Pipeline-Barriers
- **🟢 Irrelevant:** Neue Ops, CI, Build-Fixes, andere Architekturen

#### Schritt 3: Selektive Cherry-Picks auf Branch
Relevanten Commits in Gruppen cherry-picken:
1. **Gruppe A: Memory/Buffer** (PR #24326, #23770, #22930, #23762)
2. **Gruppe B: FA/MatMul** (#23420 BF16 FA, #24123 v_dot2, #22887 MUL_MAT_VEC)
3. **Gruppe C: Performance** (#23973 fast path, #23376 lock contention, #23641 pipeline mutex)
4. **Gruppe D: UMA-spezifisch** (#22455 transfer queue, #22930 host-visible)

Nach jeder Gruppe: Auf Mars builden, Pipeline-Cache löschen, Benchmark.

#### Schritt 4: Binary Search innerhalb Gruppen
Wenn eine Gruppe das Problem löst/verschärft: Innerhalb der Gruppe bisecten.

### 2.4 Mesa/RADV-Vergleich (Mars LXC)

**Mars ist Proxmox-Host** → LXC mit anderer Mesa-Version erstellen.

#### Option 1: Neuere Mesa (z.B. Mesa 25.2 oder 26.x)
```bash
# Auf Mars: LXC erstellen mit neuerer Mesa
pct create ...
# Mesa aus PPA oder build-from-source
```

#### Option 2: Ältere Mesa (bekannt stabiler Stand)
Falls neuere Mesa das Problem löst → ältere als Referenz.

**Vorsicht:** Nur in LXC/VM, niemals auf dem Host-System!

### 2.5 GPU-Debug-Logging (Mars)

Wenn PP/TG-Klippe reproduziert:
- `VK_LAYER_KHRONOS_validation` aktivieren
- `GGML_VK_DEBUG=1` oder ähnliche Debug-Variablen
- `strace -f -e trace=ioctl` auf den Prozess
- Vulkan-Dispatch-Größen loggen (Code-Instrumentierung)

---

## Phase 3: Root Cause & Fix (8-12h)

### 3.1 Root Cause identifizieren
Basierend auf Phase 2 Ergebnissen:
- Welcher Commit/PR löst das Problem?
- Oder: Welcher Code-Pfad verursacht den Hang?

### 3.2 Fix auf Branch anwenden
- Fix auf `feature/vulkan-perf-rca` committen
- Build auf Mars
- Vollständige Verification (wie Phase 1)

### 3.3 Dokumentation
- `docs/fork/2026-06-21_VULKAN_PERF_RCA_ROOT_CAUSE.md`
- FORKS.md aktualisieren (cherry-picked commits)
- AGENTS.md aktualisieren (falls Konventionen geändert)
- Trilium-Notiz aktualisieren

### 3.4 Merge-Entscheidung
- Wenn Fix stabil: PR an `master` oder direkt mergen
- Wenn Risk: auf Branch belassen, User-Review einholen

---

## Subagent-Strategie

### Parallelisierungsmöglichkeiten

| Task | System | Subagent? | Dauer |
|------|--------|-----------|-------|
| Git-Operationen, Cherry-Picks | Hydra | Direkt | ~30min |
| Build auf Mars | Mars | Background | ~10min pro Build |
| Benchmark pp-Scaling | Mars | Background | ~30min pro Serie |
| Benchmark tg-Scaling | Mars | Background | ~60min pro Serie |
| Web-Recherche Vulkan-Issues | Hydra | Subagent | ~15min |
| Code-Analyse spezifischer Commits | Hydra | Subagent | ~10min pro Commit |
| Mesa LXC erstellen | Mars | Background | ~30min |

### Subagent-Profile
- `subagent_explore`: Code-Analyse, Web-Recherche, Commit-Untersuchung
- `subagent_general`: Builds, Cherry-Picks, Datei-Änderungen

---

## Pipeline-Cache-Disziplin

**REGEL:** Nach JEDEM Code-Änderung oder JEDEM Commit-Wechsel:
```bash
rm -f ~/.cache/llama.cpp/vulkan-pipeline-cache.bin
```

Alternativ: Pro-Test eigenen Cache-Pfad:
```bash
GGML_VK_PIPELINE_CACHE_DIR=/tmp/vk_cache_test1
```

---

## Erfolgskriterien

### Minimum (12h)
- [ ] Baseline auf Mars etabliert und dokumentiert
- [ ] PP-Klippe exakt lokalisiert (zwischen welchen pp-Werten)
- [ ] TG-Klippe reproduziert
- [ ] 3+ Vulkan-Commit-Gruppen getestet

### Optimum
- [ ] Root Cause identifiziert
- [ ] Fix auf Branch angewendet und verifiziert
- [ ] PP-Klippe behoben (pp16384 funktioniert)
- [ ] TG-Klippe behoben oder zumindest verschoben (>188k)

### Stretch
- [ ] Mesa-Vergleich in LXC durchgeführt
- [ ] Vollständige Upstream-Vulkan-Sync evaluiert
- [ ] PR an AtomicBot vorbereitet
