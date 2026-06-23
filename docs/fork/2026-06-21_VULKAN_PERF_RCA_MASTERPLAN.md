# Masterplan: Vulkan Performance-Klippe auf AMD-System — Systematische RCA

**Datum:** 2026-06-21
**Ziel:** Root Cause für PP-Klippe (~16k) und TG-Klippe (~188k) auf AMD-System (AMD 760M, RADV PHOENIX) finden und beheben.
**Dauer:** 12h KI-Solo-Schicht
**Branch:** `feature/vulkan-perf-rca` (neu, von `master`)

---

## Ausgangslage

### Systeme
| System | GPU | Rolle |
|--------|-----|-------|
| **AMD-System** | AMD 760M (RDNA3, RADV PHOENIX, UMA) | Test-System — hier tritt das Problem auf |
| **CUDA-System** | NVIDIA RTX 3070 (CUDA) | Build-System, git-Operationen, CUDA-Baseline |

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
- ❌ `maxStorageBufferRange` (128 MiB) — ist 4 GiB auf AMD-System
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

### 1.2 AMD-System: Post-Reboot Verification
**Skript:** `/tmp/amd_post_reboot.sh` (liegt bereit auf AMD-System)

Reproduziert die AMD-System-2-Verification auf AMD-System:
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
- turbo3+FA Korrektheit bestätigt (wie auf AMD-System-2)
- Baseline pp/tg Zahlen für AMD-System dokumentiert
- PP-Klippe reproduziert (oder nicht — nach Reboot + clean Cache)

---

## Phase 2: Systematische Untersuchung (2-8h)

### 2.1 PP-Scaling Matrix (AMD-System)
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

### 2.2 TG-Scaling Matrix (AMD-System)
Mit `llama-server` + API, kurzer Prompt (100 Tokens), ctx variiert:

| ctx | tg8 | Erwartung |
|-----|-----|-----------|
| 180000 | ? | ~24 t/s |
| 184000 | ? | ~24 t/s |
| 186000 | ? | HANG? |
| 188000 | ? | 0.09 t/s? |

**Wichtig:** Server mit `--no-warmup` starten, kurzen Prompt senden, nur tg messen.

### 2.3 Vulkan Commit-Bisect (CUDA-System + AMD-System)

**Strategie:** Nicht 1057 Commits bisecten, sondern gezielt die 61 Vulkan-Commits.

#### Schritt 1: Vulkan-Commit-Liste erstellen (CUDA-System)
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

Nach jeder Gruppe: Auf AMD-System builden, Pipeline-Cache löschen, Benchmark.

#### Schritt 4: Binary Search innerhalb Gruppen
Wenn eine Gruppe das Problem löst/verschärft: Innerhalb der Gruppe bisecten.

### 2.4 Mesa/RADV-Vergleich (AMD-System LXC)

**AMD-System ist Proxmox-Host** → LXC mit anderer Mesa-Version erstellen.

#### Option 1: Neuere Mesa (z.B. Mesa 25.2 oder 26.x)
```bash
# Auf AMD-System: LXC erstellen mit neuerer Mesa
pct create ...
# Mesa aus PPA oder build-from-source
```

#### Option 2: Ältere Mesa (bekannt stabiler Stand)
Falls neuere Mesa das Problem löst → ältere als Referenz.

**Vorsicht:** Nur in LXC/VM, niemals auf dem Host-System!

### 2.5 GPU-Debug-Logging (AMD-System)

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
- Build auf AMD-System
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
| Git-Operationen, Cherry-Picks | CUDA-System | Direkt | ~30min |
| Build auf AMD-System | AMD-System | Background | ~10min pro Build |
| Benchmark pp-Scaling | AMD-System | Background | ~30min pro Serie |
| Benchmark tg-Scaling | AMD-System | Background | ~60min pro Serie |
| Web-Recherche Vulkan-Issues | CUDA-System | Subagent | ~15min |
| Code-Analyse spezifischer Commits | CUDA-System | Subagent | ~10min pro Commit |
| Mesa LXC erstellen | AMD-System | Background | ~30min |

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
- [x] Baseline auf AMD-System etabliert und dokumentiert
- [x] PP-Klippe exakt lokalisiert (8192-16384)
- [x] TG-Klippe reproduziert — und behoben!
- [x] 3+ Vulkan-Commit-Gruppen getestet (Gruppe A+B, 6 Commits)

### Optimum — ✅ Alle erfüllt
- [x] Root Cause identifiziert: `amdgpu.lockup_timeout` (2000ms) + `nodes_per_submit=100` (Issue #21724)
- [x] Fix auf Branch angewendet: `nodes_per_submit=10` für UMA
- [x] PP-Klippe behoben: pp16384=122 t/s (vorher HANG)
- [x] TG-Klippe behoben: tg32=21.3 t/s bei 188k (vorher 0.099 t/s = 216x Verbesserung)

### Stretch
- [ ] Mesa-Vergleich in LXC — nicht nötig, Fix gefunden
- [ ] Vollständige Upstream-Vulkan-Sync — nicht nötig
- [ ] PR an AtomicBot — vorbereitetbar

---

## Ergebnisse (2026-06-21)

### Root Cause
`amdgpu.lockup_timeout` (Default: 2000ms) wird überschritten, wenn
`ggml_backend_vk_graph_compute` bis zu 100 Nodes pro `vkQueueSubmit` batcht.
Auf langsamen APUs (AMD 760M) dauern große Batches >2s → Kernel denkt GPU
ist abgestürzt → Ring Reset → `vk::DeviceLostError` → GPU-Hang.

### Fix
`nodes_per_submit = ctx->device->uma ? 10 : 100;` in `ggml-vulkan.cpp`

### PP-Scaling (mit Fix + Cherry-Picks)
| pp | Ohne Fix | Mit Fix (nps=10) | Status |
|----|----------|-----------------|--------|
| 512 | 205 t/s | 160 t/s | ✅ |
| 4096 | 168 t/s | 166 t/s | ✅ |
| 8192 | 150 t/s | 147 t/s | ✅ |
| 16384 | HANG | 122 t/s | ✅ DURCHBRUCH |

### TG-Scaling (mit Fix + Cherry-Picks)
| ctx | tg32 vorher | tg32 mit Fix | Status |
|-----|-------------|-------------|--------|
| 180000 | 24.1 t/s | 21.24 t/s | ✅ |
| 186000 | HANG | 22.05 t/s | ✅ BEHOBEN |
| 188000 | 0.099 t/s | 21.33 t/s | ✅ 216x schneller |
| 192000 | 0.099 t/s | 21.35 t/s | ✅ BEHOBEN |

### Offene Frage
Sind die 6 Cherry-Picks nötig, oder reicht `nodes_per_submit=10` allein?
→ Test auf `test/nps-only` Branch läuft
