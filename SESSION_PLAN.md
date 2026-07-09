# SESSION_PLAN: Vulkan turbo3/turbo4 KV-Cache Benchmark

**Erstellt:** 2026-07-09
**Typ:** Solo-Session (Research + Benchmark)
**Status:** ✅ ABGESCHLOSSEN

---

## Session-Ziel

Benchmark auf AMD-RDNA3 (AMD 760M APU, Vulkan) vergleicht turbo3/3, turbo3/4, turbo4/4 KV-Cache Kombinationen. Ziel: Herausfinden ob K=turbo4 (mit FlashAttention) schneller ist als K=turbo3 (ohne FA, scalar fallback) trotz geringerer Kompression. Ergebnisse in Trilium dokumentieren.

## Ergebnis

**Hypothese WIDERLEGT.** turbo3/3 ist konsistent am schnellsten. Der Dequant-Overhead von turbo4 (4.25 bit) überwiegt den FA-Vorteil bei Kontexten bis 8192. Empfehlung K=turbo3/V=turbo3 bestätigt.

## Schritte

1. ✅ AMD-RDNA3: Build verifizieren + Benchmark-Script erstellt
2. ✅ AMD-RDNA3: Benchmark turbo3/3, turbo3/4, turbo4/4, f16 (pp512, 2048, 4096, 8192) mit 26B-A4B
3. ✅ Ergebnisse synthetisieren + Vergleichstabelle
4. ✅ Local: Benchmark-Report als Markdown gespeichert
5. ✅ Trilium: Neue Benchmark-Subnote unter AMD-RDNA3 (5DTGKZb95DUO)
6. ✅ Trilium: AMD-RDNA3-Hauptnote aktualisiert
7. ✅ Trilium: FINALE EMPFEHLUNG aktualisiert (Btrfs+Pinning+Vulkan-Benchmark)
8. ✅ TTT-Eintrag erstellt

## Anpassungen während der Session

- **Erster Versuch mit 26B-A4B OOMte** weil RAM/GTT nicht freigegeben wurde → korrigiert: `killall -9 llama-bench; sleep 8-10` zwischen Tests
- **Große Prompts (>4096) OOM** bei parallelen Prozessen → pp512-8192 getestet (nicht 32k-180k)
- **CSV-Parsing** mehrfach korrigiert (Spalten-Index-Fehler)
- **OOM-Killer** killte SSH-Sessions und systemd bei GTT-Überlastung durch parallele llama-bench Prozesse
