# DiffusionGemma Integration — Status Report

> Datum: 2026-06-15, 15:45 Uhr
> Stand: Inferenz funktioniert! Build erfolgreich.

---

## Was funktioniert

### Build-System
| Komponente | Status |
|------------|--------|
| `libllama.so` | ✅ Baut fehlerfrei |
| `llama-server` | ✅ Baut fehlerfrei |
| `llama-cli` | ✅ Baut fehlerfrei |
| `test-llama-archs` | ✅ Baut fehlerfrei |

### Modell-Loading
| Schritt | Status | Details |
|---------|--------|---------|
| GGUF-Parsing | ✅ | 44 KV-Paare, 692 Tensoren |
| Architektur-Erkennung | ✅ | `LLM_ARCH_DIFFUSION_GEMMA` |
| HParams | ✅ | canvas_length=256, n_layer=30, n_embd=2816, n_expert=128 |
| Tensor-Mapping | ✅ | Alle Tensoren geladen |
| Graph-Reserve | ✅ | 2800 Nodes, 507 Splits, 15.17 ms |

### Inferenz (getestet)
| Test | Prompt | Tokens | Ergebnis | Zeit |
|------|--------|--------|----------|------|
| CPU-only (ngl=0) | "Hello" | 2 | Next: ' own' (logit=26.51) | ~2-3 min |
| GPU+8 Layers | "What is the meaning of life?" | 8 | Next: '.' (logit=22.58) | 3.1 s |

**Berechnete Performance:**
- CPU-only (i7-12700H, 20 Threads): ~0.02 t/s (sehr langsam, aber funktioniert)
- GPU+8 Layers (RTX 3070 Laptop): ~2.5 t/s (8 Tokens / 3.1 s)

---

## Was noch fehlt

### 1. Diffusion-spezifische Features (nicht implementiert)
| Feature | Status | Prioritaet |
|---------|--------|----------|
| **Block-Diffusion-Decoding** (multi-step denoise) | ❌ Nicht implementiert | Hoch |
| **Entropy-bound Decoder** (EB) | ❌ Nicht implementiert | Hoch |
| **Visual Diffusion Mode** | ❌ Nicht implementiert | Mittel |
| **llama-diffusion-cli** (spezialisiertes CLI) | ❌ Nicht gebaut | Hoch |
| **Self-Conditioning (SC)** im Graph | ⚠️ Code vorhanden, nicht getestet | Mittel |

### 2. C-API Funktionen
| Funktion | Status | Nutzung |
|----------|--------|---------|
| `llama_diffusion_set_sc` | ✅ Implementiert | SC Parameter setzen |
| `llama_diffusion_set_device_sc` | ✅ Implementiert | Device-resident SC |
| `llama_diffusion_device_sample` | ✅ Implementiert | CUDA Sampling |
| `llama_diffusion_set_phase` | ✅ Implementiert | PKV Phase (Prefill/Decode/Unified) |

### 3. Beispiel-Anwendungen (upstream)
| Tool | Status | Grund |
|------|--------|-------|
| `diffusion-gemma-eval` | ❌ Nicht gebaut | Baut nicht, `common.h` fehlt |
| `diffusion-gemma-server` | ❌ Nicht gebaut | Baut nicht, `common.h` fehlt |
| `diffusion-gemma-visual-server` | ❌ Nicht gebaut | Baut nicht, `common.h` fehlt |

### 4. CUDA-Kernels
| Datei | Status |
|-------|--------|
| `diffusion-sampling.cu` | ✅ Kopiert (ungetestet) |

---

## Unterschiede zum upstream PR #24423

### Architektur
| Aspekt | Upstream PR | Unser Fork |
|--------|-------------|------------|
| **Modell-Struktur** | `llama_model_diffusion_gemma : public llama_model_base` | Monolithische `llama_model` mit Diffusion-Feldern |
| **load_hparams** | In `diffusion-gemma.cpp` als Klassenmethode | In `llama-model.cpp` als Case |
| **load_tensors** | In `diffusion-gemma.cpp` als Klassenmethode | In `llama-model.cpp` als Case |
| **build_graph** | `std::make_unique<llm_build_diffusion_gemma>(*this, params)` | Identisch (graph-builder Klasse) |

### Funktionalitaet
| Feature | Upstream PR | Unser Fork |
|---------|-------------|------------|
| Modell-Loading | ✅ | ✅ |
| Einzelner Forward-Pass | ✅ | ✅ |
| Block-Diffusion (multi-step) | ✅ | ❌ (nur einzelner Step) |
| Entropy-bound Decoding | ✅ | ❌ |
| Visual Mode | ✅ | ❌ |
| SC (Self-Conditioning) | ✅ | ⚠️ Code da, nicht getestet |

---

## Naechste Schritte

1. **llama-diffusion-cli erstellen** — Ein minimales CLI-Tool fuer DiffusionGemma-Inferenz
2. **Block-Diffusion-Loop implementieren** — Multi-step Denoising (Prefill → Decode × N)
3. **GPU-Offloading optimieren** — Mehr Layers auf GPU (aktuell nur 8 moeglich)
4. **Performance testen** — Mit `-ngl 15` auf besserer Hardware (Ampere/Ada)
5. **Entropy-bound Decoder** — Implementieren wenn Core-Inferenz stabil ist

---

## Ergebnis-Zusammenfassung

> **"Hello" → 2 Token → Forward-Pass → Next Token: ' own' (logit=26.51)**
> 
> **"What is the meaning of life?" → 8 Token → Forward-Pass → Next Token: '.' (logit=22.58)**

Der Graph-Builder funktioniert. Die Token-Vorhersage ist deterministisch (greedy decode). Die Logits sind im erwarteten Bereich (20-30).

**Was wir haben:** Ein funktionierender DiffusionGemma-Forward-Pass in unserem Fork.
**Was wir noch brauchen:** Den vollstaendigen Diffusion-Decoding-Loop (Prefill → mehrere Decode-Steps).
