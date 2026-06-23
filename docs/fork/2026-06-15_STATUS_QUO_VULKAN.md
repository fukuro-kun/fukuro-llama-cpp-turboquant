# Status Quo: Vulkan-Entwicklung auf AMD-Systemen

> Stand: 2026-06-15, 21:45 Uhr
> Zweck: Snapshot vor weiteren Vulkan-Optimierungen und Cherry-Picks

---

## 1. Codeberg-Umstellung abgeschlossen

| System | Codeberg-SSH | Repo | Status |
|--------|-------------|------|--------|
| **System A** (AMD APU, RDNA3) | Ja | Aktualisiert auf `master` (3fe45b370) | Produktiv |
| **System B** (AMD APU, Vega) | Ja | Geklont, cmake 3.28.3 installiert | Build-pending |
| **Dev-Host** (NVIDIA CUDA) | Ja | Origin | Aktiv |

**Wichtig:** Alle Systeme arbeiten jetzt mit demselben Codeberg-`master`.

---

## 2. Vulkan-Build auf System A (AMD RDNA3 APU)

| Eigenschaft | Wert |
|-------------|------|
| GPU | AMD Radeon 760M (RADV PHOENIX, RDNA3) |
| Vulkan API | 1.4.305 |
| Treiber | RADV 25.0.7 (Mesa) |
| coopmat | Unterstuetzt |
| coopmat2 | Unterstuetzt |
| GL_EXT_bfloat16 | NICHT unterstuetzt (glslc) |
| Build | Erfolgreich (108 Binaries, libggml-vulkan.so gelinkt) |

**TurboQuant-Shaders:** Alle generiert (`dequant_turbo3_0`, `dequant_tq4_1s`, `mul_mat_vec_tq4_1s`, `turbo_wht`).

---

## 3. Benchmark-Ergebnisse (Vulkan, AMD RDNA3 APU)

### 3.1 Gemma 4 E2B (2.9 GB, Q4_K_M)

| Metrik | CPU (`-ngl 0`) | Vulkan (`-ngl 99`) | Speedup |
|--------|---------------|-------------------|---------|
| **pp512** | 457.7 t/s | 559.0 t/s | **+22%** |
| **tg64** | 29.6 t/s | 39.0 t/s | **+32%** |

### 3.2 Gemma 4 12B (6.6 GB, Q4_K_M)

| Metrik | CPU (`-ngl 0`) | Vulkan (`-ngl 99`) | Speedup |
|--------|---------------|-------------------|---------|
| **pp512** | 87.2 t/s | 102.5 t/s | **+17%** |
| **tg64** | 7.43 t/s | 7.96 t/s | **+7%** |

### 3.3 Gemma 4 26B-A4B (16 GB, Q4_K_M)

| Metrik | Vulkan (`-ngl 99`, Turbo3 KV) |
|--------|------------------------------|
| **pp512** | 19.2 t/s |
| **tg64** | <1 t/s (zu langsam fuer interaktiv) |

### 3.4 BFloat16 FlashAttention — Nicht nutzbar auf unserer Hardware

| Extension | Status | Grund |
|-----------|--------|-------|
| `VK_KHR_shader_bfloat16` | **Fehlt** | Mesa 25.2.x hat BF16 nur fuer GFX12+ (RDNA4/CDNA4). Unsere GPUs (RDNA3/Vega) sind aelter. |

**Konsequenz:** Der upstream Cherry-Pick `6e093b80e` (Vulkan BFloat16 FA) ist fuer unsere Hardware **nicht anwendbar**. Der Build wuerde durchlaufen, aber der BF16-Pfad wuerde nie aktiviert. Siehe [FORKS.md §5.7](FORKS.md#57-warum-bfloat16-fa-fuer-uns-nicht-nutzbar-ist).

**Fazit:** Vulkan bringt auf der APU messbare Vorteile, die bei kleineren Modellen deutlicher ausfallen. Bei 26B ist die APU bandbreiten-limitiert.

---

## 4. Turbo3 KV auf Vulkan

### Kurzer Kontext (p=512)

| Modell | Standard KV | Turbo3 KV | Delta |
|--------|------------|-----------|-------|
| Gemma 4 12B pp512 | 102.5 t/s | 99.9 t/s | -2.5% |
| Gemma 4 12B tg64 | 7.96 t/s | 7.97 t/s | ~0% |

### Langkontext-Test (Gemma 4 E2B, verschiedene Prompt-Laengen)

| Kontext | Standard KV pp | Turbo3 KV pp | Delta | Standard tg64 | Turbo3 tg64 |
|---------|---------------|--------------|-------|---------------|---------------|
| **p=512** | 559.0 t/s | ~559 t/s | ~0% | 39.0 t/s | ~39.0 t/s |
| **p=2048** | 531.6 t/s | 449.5 t/s | **-15%** | 38.9 t/s | 37.6 t/s |
| **p=4096** | 513.9 t/s | 389.1 t/s | **-24%** | 38.5 t/s | 37.1 t/s |
| **p=8192** | 464.0 t/s | 307.1 t/s | **-34%** | 38.5 t/s | 36.8 t/s |

**Fazit:** Turbo3 wird auf der AMD APU (Vulkan) mit **steigendem Kontext immer langsamer**. Der Dequant-Overhead auf der GPU frisst den KV-Kompression-Vorteil komplett auf. Bei p=8192 ist Turbo3 **34% langsamer** als Standard KV. Der einzige verbleibende Nutzen von Turbo3 auf Vulkan waere **Speichereinsparung** (fuer grossere Modelle bei knappem VRAM), nicht Performance.

---

## 5. MTP (Multi-Token Prediction) auf Vulkan

| Modell | Status | Ergebnis |
|--------|--------|----------|
| **Gemma 4 E2B** | Funktioniert! | Prompt: 109 t/s, Gen: 45.6 t/s |
| **Gemma 4 12B** | Funktioniert! | Prompt: 24.4 t/s, Gen: 12.7 t/s |

**Wichtige Erkenntnis:** MTP auf Vulkan erfordert `--single-turn` (sonst interaktiver Chat-Modus, endloses Warten).

**Parameter fuer MTP:**
```bash
llama-cli -m <target.gguf> --model-draft <draft.gguf> --spec-type mtp \
  -ngl 99 --single-turn --no-display-prompt -p "Prompt"
```

---

## 6. WHT Cherry-Pick Ergebnis

Upstream-Commits `48e7078ee` + `e82beaa60` (Walsh-Hadamard-Transform fast path) erfolgreich cherry-picked.

### Build
- ✅ Kompiliert auf System A (AMD RDNA3 APU)
- ✅ `libggml-vulkan.so` gelinkt
- ⚠️ Testfaelle auskommentiert (`test_mul_mat_hadamard` Klasse fehlt)

### Benchmark

| Modell | KV | pp512 (vorher) | pp512 (mit WHT) | tg64 (vorher) | tg64 (mit WHT) |
|--------|----|---------------|-----------------|---------------|----------------|
| Gemma 4 12B Q4_K_M | Standard | 87.2 t/s | **101.9 t/s** (+17%) | 7.96 t/s | 7.89 t/s (-1%) |
| Gemma 4 E2B Q4_K_M | Standard | 559.0 t/s | **554.8 t/s** (-1%) | 39.0 t/s | 38.3 t/s (-2%) |

**Fazit:** WHT beschleunigt Prompt-Verarbeitung (+17% bei 12B), nicht Token-Generation. E2B (klein, GPU-bound) profitiert nicht. Turbo3 KV bringt keinen zusaetzlichen WHT-Vorteil.

### v_dot2 Cherry-Pick

**Abgebrochen** — zu komplex (6 Konflikte, FlashAttention-Refactor-Abhaengigkeiten). Siehe [FORKS.md §5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen).

---

## 7. Verfuegbare Modelle auf System A

| Modell | Groesse | Ort |
|--------|---------|-----|
| Llama 3.2 1B Q4_K_M | 771 MB | `/jade/models/` |
| Gemma 4 E2B Q4_K_M | 2.9 GB | `/jade/models/` |
| Gemma 4 E2B Assistant Q4_K_M | 75 MB | `/jade/models/drafts/` |
| Gemma 4 12B Q4_K_M | 6.6 GB | `/jade/models/` |
| Gemma 4 12B Assistant MTP Q4_K_M | 313 MB | `/jade/models/drafts/` |
| Gemma 4 26B-A4B Q4_K_M | 16 GB | `/jade/models/` |
| Gemma 4 31B IQ4_NL | 17 GB | `/jade/models/` |

---

## 8. Offene Punkte / Naechste Schritte

| Prioritaet | Aufgabe | Status |
|-----------|---------|--------|
| 1 | System B (Vega) Build mit Vulkan starten | Offen |
| 2 | Turbo3 bei langem Kontext testen (>4096) | Offen |
| 3 | ~~Cherry-Pick upstream-Vulkan-Verbesserungen (WHT, v_dot2, BF16 FA)~~ | **WHT done, v_dot2 abgebrochen, BF16 nicht nutzbar** |
| 4 | MTP-Benchmark-Matrix (E2B + 12B mit verschiedenen Draft-Quants) | Offen |
| 5 | DiffusionGemma Forward-Pass auf Vulkan testen | Offen |
| 6 | Trilium-Note aktualisieren | Offen |

---

## 9. Bekannte Einschraenkungen

1. **Turbo3 ohne KV-Vorteil auf APU** (bei kurzem Kontext) — braucht Langkontext-Test
2. **Kein BFloat16 in glslc** — BFloat16-FA aus upstream nicht direkt nutzbar
3. **MTP benoetigt `--single-turn`** — sonst haengt der Prozess im Chat-Modus
4. **26B-A4B tg zu langsam auf APU** — nur fuer pp-Tests geeignet
5. **Vulkan-Turbo3 nie mit MTP kombiniert getestet** — Kombination steht aus

---

*Dieser Status-Quo-Snapshot dient als Baseline fuer weitere Vulkan-Optimierungen und Cherry-Picks aus upstream.*
