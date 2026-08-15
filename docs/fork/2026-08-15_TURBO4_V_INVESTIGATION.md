# turbo4 V + mmproj + 262144 Untersuchung (15.08.2026)

**Status:** OPEN — turbo3/3 ist Production-Default, turbo4 V ist Ziel aber benötigt Precompile
**Betroffen:** AMD gfx1103 (Radeon 760M, RDNA3 Phoenix APU, UMA) mit Vulkan/RADV
**Production-Commit (turbo3/3):** `0311122e6`
**Cache-Flush-Fix:** `d795404be`

---

## Zusammenfassung

turbo3/4 (K=turbo3, V=turbo4) + mmproj + 262144 Kontext ist das **Ziel-Setup** — höhere V-Cache-Präzision (4.25 bit vs 3.125 bit). Zwei separate Probleme verhindern derzeit den Production-Betrieb:

1. **OOM** — turbo3/4 braucht 1.8 GB mehr KV-Speicher als turbo3/3. Mit `--cache-ram 6144` wird das 28GB RAM-Limit überschritten → OOM-Kill.
2. **ACO Pipeline-Kompilierung pathologisch langsam** — Jede turbo4 V FA-Pipeline-Variante braucht ~3min zu kompilieren. Bei 262144+mmproj gibt es viele Varianten → Startup 30-60min.

Beide Probleme sind verstanden und haben Workarounds. Ein einmaliger Precompile-Lauf (im Wartungsfenster) sollte den Pipeline-Cache füllen, danach ist der Restart schnell.

---

## Problem 1: OOM

### Diagnose

Der OOM-Killer killte den turbo4 V Server-Prozess beim Startup. systemd-journal bestätigt:

```
llama-server.service: A process of this unit has been killed by the OOM killer.
```

### Memory-Bedarf bei 262144 (2×128k Slots)

| Config | KV-Cache | Modell | cache-ram | Gesamt | Status |
|--------|----------|--------|-----------|--------|--------|
| turbo3/3 | 10.1 GB | 13.7 GB | 6 GB | 23.8 + 6 = 29.8 GB | ❌ knapp über 28GB |
| turbo3/4 | 11.9 GB | 13.7 GB | 6 GB | 25.6 + 6 = 31.6 GB | ❌ OOM |
| turbo3/3 | 10.1 GB | 13.7 GB | 0 | 23.8 GB | ✅ passt |
| turbo3/4 | 11.9 GB | 13.7 GB | 0 | 25.6 GB | ✅ passt (27GB free) |

**Hinweis:** turbo3/3 mit `cache-ram=6144` war bisher knapp aber funktionierte weil der Compiler-Speicher während des laufenden Betriebs kleiner ist als beim Startup (Pipelines bereits kompiliert). turbo3/4 braucht beim Startup zusätzlichen Compiler-Speicher → OOM.

### Fix

`--cache-ram 0` für turbo4 V. Der Prompt-Cache ist ein Nice-to-have aber nicht essential — der Router kann KV-States auch ohne `--cache-ram` cachen (über die slot-cache-key-Validierung). Wenn turbo4 V im Production-Betrieb ist und der Pipeline-Cache vollständig, kann `--cache-ram` schrittweise wieder erhöht werden (z.B. 2048).

---

## Problem 2: ACO Pipeline-Kompilierung pathologisch langsam

### Diagnose

| Config | mmproj | n_ctx | Startup | t/s (erster Request) | Status |
|--------|--------|-------|---------|---------------------|--------|
| turbo3/3 | ✅ | 262144 | 80s | 23 | ✅ Production |
| turbo3/4 | ❌ | 262144 | 60s | 23 | ✅ Funktioniert |
| turbo3/4 | ✅ | 161792 | 250s | 23 | ⚠️ Langsam aber funktioniert |
| turbo3/4 | ✅ | 262144 | 450s+ | 0.04 (Pipeline-Compile) | ⚠️ Super-langsam |

### Was passiert beim Startup

1. **0-60s:** Modell laden, KV-Cache allokiieren
2. **60-450s:** mmproj laden, Slots initialisieren
3. **450s+:** Server listening aber **Pipelines nicht kompiliert** (`--no-warmup`)
4. **Erster Request:** Jede Pipeline-Variante wird lazy kompiliert — ~3min pro Variante bei turbo4 V

### Warum turbo4 V langsamer als turbo3 V

Die Subagent-Analyse klärte auf: **turbo4 V Shader sind arithmetisch EINFACHER als turbo3 V** (keine Sign-Byte-Verarbeitung, keine OR-Verknüpfungen). Die Centroid-Tabelle ist 2× größer (16 vs 8 Floats) aber das ist kein Compile-Zeit-Treiber.

Die Ursache ist **RADV's GTT-Spill-Heuristik** (gleicher Mechanismus wie bei turbo3/3 + mmproj + 262144, aber turbo4 V triggert ihn stärker):
- turbo4 V Block = 68 Bytes (norm + rnorm + qs[64]) vs turbo3 V Block = 50 Bytes
- Größere KV-Buffer → Heuristik triggert bei mehr Varianten
- ACO's Spilling-Code-Generierung ist auf UMA-APUs pathologisch langsam

`nogttspill` ist bereits aktiv aber für turbo4 V allein nicht ausreichend — die Kombination aus turbo4 V's komplexerem FA-Shader + mmproj's zusätzlichen Vision-Pipelines + 262144's großem KV-Buffer überschreitet einen weiteren Schwellwert.

### Getestete RADV/ACO Flags — alle fehlgeschlagen

| Flag | Ergebnis | Ursache |
|------|----------|---------|
| `ACO_DEBUG=nosched` | ❌ Crash nach 40s | ACO-Bug bei deaktiviertem Scheduling |
| `ACO_DEBUG=noopt` | ❌ Crash nach 40s | ACO-Bug bei deaktivierten Optimierungen |
| `RADV_PERFTEST=cswave32` | ❌ Crash nach 20s | Bricht Coopmat-Shader auf RDNA3 (bekannt, ROADMAP #84) |
| `small_cache`-Schwelle | ❌ Existiert nicht | Falsche Spur — keine n_ctx-abhängige Schwelle in ggml-vulkan.cpp |

### Warum turbo3/3 nicht betroffen ist

- turbo3 V FA ist **deaktiviert** (`vulkan-shaders-gen.cpp:692-704`) → fällt auf scalar Attention zurück → keine turbo3-spezifischen FA-Pipelines
- turbo3/3 KV-Buffer (1.0 GB) liegt unter dem Heuristik-Schwellwert
- Weniger Pipeline-Varianten (turbo3 V FA existiert nicht als Shader)

---

## Problem 3: Cache-Verlust bei OOM-Kill

### Diagnose

Der Subagent fand: Bei OOM-Kill (SIGKILL vom Kernel) wird weder der VK Pipeline Cache noch der Mesa Shader Cache geflusht:

- `ggml_vk_save_pipeline_cache()` wird nur bei `ggml_backend_vk_free()` und `~vk_device_struct()` aufgerufen — beide laufen nur bei normalem Shutdown (SIGTERM → C++ Destruktoren)
- Mesa's asynchroner Cache-Writer wird bei SIGKILL abrupt beendet → ungeschriebene Pages verloren
- Keine Signal-Handler oder `atexit`-Hooks im Code

### Fix: Periodischer Pipeline-Cache-Flush (Commit `d795404be`)

Nach jeder Pipeline-Kompilierung wird der VK Pipeline Cache auf Disk gespeichert:

```cpp
// ggml-vulkan.cpp:2634-2642 (nach pipeline->compiled = true)
{
    std::lock_guard<std::recursive_mutex> lock(device->mutex);
    device->pipeline_cache_saved = false;  // Reset für wiederholte Saves
}
ggml_vk_save_pipeline_cache(device.get());
```

**Verifikation:** Der turbo4 V Precompile-Lauf zeigte Cache-Wachstum von 0 → 113K → 177K → 201K → 213K während des Startups und der ersten Requests. Beim alten Build wäre der Cache bis zum Shutdown bei 0 geblieben.

**Limitation:** Der Mesa Shader Cache kann nicht aus App-Code periodisch geflusht werden — nur `vkDestroyDevice` flusht ihn. Bei OOM-Kill geht der Mesa Cache verloren. Der VK Pipeline Cache ist aber der wichtigere (er speichert die fertigen Pipeline-Objekte).

---

## Lösungsstrategie

### Wartungsfenster-Plan

1. **Production stoppen** (`systemctl --user stop llama-server`)
2. **turbo4 V Precompile laufen lassen** (`KV_CACHE_VARIANT=turbo4 bash scripts/precompile-vulkan-shaders.sh`)
   - Dauer: 30-60min (alle Pipeline-Varianten kompilieren)
   - Cache wird nach jeder Pipeline gesichert (periodischer Flush)
   - Backup wird automatisch erstellt
3. **turbo4 V Production starten** (`bash scripts/start-mars-26b-server-turbo4.sh`)
   - Beim zweiten Start: Pipeline-Cache-Hit → Startup deutlich schneller
   - `--cache-ram 0` bis Pipeline-Cache vollständig verifiziert
4. **Wenn stabil:** `--cache-ram` schrittweise erhöhen (2048, dann 4096)
5. **Wenn instabil:** Zurück zu turbo3/3 (`systemctl --user start llama-server`)

### Was nicht funktioniert hat

- ACO_DEBUG-Flags crashen den Prozess
- cswave32 crasht Coopmat-Shader
- small_cache-Schwelle existiert nicht in ggml-vulkan.cpp
- Parallelbetrieb von turbo3/3 + turbo4 V → OOM (zwei Server auf einer GPU)

---

## Dateien

| Datei | Zweck |
|-------|------|
| `scripts/start-mars-26b-server-turbo4.sh` | turbo4 V Start-Skript (cache-ram=0, separater Cache) |
| `scripts/start-mars-26b-server.sh` | turbo3/3 Production-Start-Skript (unverändert) |
| `scripts/precompile-vulkan-shaders.sh` | Precompile-Skript (unterstützt turbo4 V via `KV_CACHE_VARIANT=turbo4`) |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp:2634-2642` | Periodischer Pipeline-Cache-Flush |

---

## Offene Fragen

| # | Frage | Status |
|---|-------|--------|
| 1 | Wie lange dauert der turbo4 V Startup mit vollständigem Pipeline-Cache? | ❌ GETESTET — Cache bringt KEINEN Vorteil (411s vs 440s ohne Cache) |
| 2 | Kann `--cache-ram` nach Pipeline-Cache-Füllung erhöht werden? | ❌ N/A — Cache funktioniert nicht für FA-Pipelines |
| 3 | Ist turbo4 V Quality merklich besser als turbo3/3? | Nicht getestet — Quality-Benchmark ausstehend |
| 4 | Gibt es einen Mesa Bug-Report für ACO's pathologische Compile-Zeit? | Nicht erstellt — niedrige Priorität |
| 5 | Warum nutzt der VK Pipeline Cache die turbo4 V FA-Pipelines nicht? | ❌ OFFEN — Cache wächst (209K→848K) aber Restart ist nicht schneller |

---

## Wartungsfenster-Test (15.08.2026, 17:23-18:30)

### Durchgeführt

1. Production (turbo3/3) gestoppt
2. turbo4 V Precompile mit `KV_CACHE_VARIANT=turbo4` gestartet
3. Server healthy nach 440s (7.3min)
4. 6 systematische Requests mit `max_tokens=1` (Decode, Small/Medium/Large/VeryLarge Prompt, Vision)
5. Cache wuchs von 209K → 848K (VK Pipeline Cache) + 2.4M → 3.4M (Mesa Shader Cache)
6. Server mit SIGTERM heruntergefahren (sauberer Cache-Flush)
7. turbo4 V mit Cache neu gestartet

### Ergebnis: NEGATIV

| Metric | Ohne Cache | Mit Cache | Verbesserung |
|--------|-----------|-----------|-------------|
| Startup bis /health | 440s | 411s | -7% (Rauschen) |
| Erster Request (Hi, max_tokens=1) | 5min 34s | 4min 56s | -11% (Rauschen) |
| PP Rate erster Request | 0.09 t/s | 0.1 t/s | minimal |

**Der VK Pipeline Cache wird geladen (848K) aber nicht für turbo4 V FA-Pipelines genutzt.** Der Cache wächst während des Precompiles (periodischer Flush funktioniert), aber beim Restart werden die FA-Pipelines neu kompiliert.

### Mögliche Ursachen

1. **FA-Pipeline-Cache-Key-Mismatch:** Die Pipeline-Cache-Keys könnten sich zwischen Runs ändern (z.B. durch Pointer-Adressen, Thread-IDs, oder andere nicht-deterministische Werte)
2. **Mesa Shader Cache nicht effektiv:** Mesa's NIR→ISA Cache könnte geladen werden aber nicht gematcht werden (z.B. verschiedene Shader-Hashes durch Spec-Constant-Änderungen)
3. **VK Pipeline Cache ignoriert FA-Pipelines:** Möglicherweise nutzt der FA-Pipeline-Erstellungspfad `vkCreateComputePipelines` nicht mit dem `pipeline_cache` Parameter
4. **Shader-Module werden neu erstellt:** Wenn die SPIR-V-Shader-Module bei jedem Start neu geladen werden, ändert sich der Shader-Module-Hash und der Pipeline-Cache kann nicht matchen

### Fazit

**turbo4 V + mmproj + 262144 ist NICHT production-tauglich auf Phobos.** Der Precompile-Ansatz funktioniert nicht weil der VK Pipeline Cache die FA-Pipelines nicht cached. Jeder Restart erfordert 5+min pro Pipeline-Variante, was bei 15+ Varianten 75+min bedeutet.

**turbo3/3 bleibt Production-Standard.** Die turbo3 V FA-Pipelines sind deaktiviert (scalar Attention fallback), daher tritt das Problem dort nicht auf.

### Nächste Schritte (falls turbo4 V angestrebt wird)

1. **Root Cause Analysis:** Warum nutzt der VK Pipeline Cache die FA-Pipelines nicht? Code-Inspektion von `vkCreateComputePipelines` Aufruf mit `pipeline_cache` Parameter im FA-Pipeline-Erstellungspfad.
2. **VK_EXT_pipeline_creation_cache_control:** Non-blocking Pipeline-Kompilierung — erlaubt der App, Pipeline-Kompilierung asynchron durchzuführen.
3. **Mesa Bug Report:** ACO's pathologische Compile-Zeit für turbo4 V FA-Shader auf RDNA3/UMA melden.
4. **Alternative:** turbo3/4 ohne mmproj testen (weniger Pipeline-Varianten, möglicherweise schnellerer Startup).
