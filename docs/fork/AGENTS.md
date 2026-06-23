# docs/fork/ — AGENTS.md

**Zweck:** Fork-spezifische Dokumentation, die nicht in der upstream-Doku (`docs/`) existiert.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** `docs/fork/` und Unterverzeichnisse.

---

## Lokale Vertraege

### Struktur

| Verzeichnis | Inhalt |
|-------------|--------|
| `docs/fork/` | Aktive Fork-Dokumentation (Status, Vergleiche, Snapshots) |
| `docs/fork/archive/` | Historische RCAs und Debug-Sessions |
| `docs/fork/archive/rca/` | Root Cause Analyses (RCA) |

### Dateien

| Datei | Zweck | Datum |
|-------|-------|-------|
| `2026-06-15_DIFFUSION_GEMMA_STATUS.md` | Aktueller Feature-Status DiffusionGemma | 2026-06-15 |
| `2026-06-16_VERGLEICH_UNSLOTH.md` | Vergleich Unsere Implementierung vs. Unsloth-Referenz | 2026-06-16 |
| `2026-06-15_STATUS_QUO_VULKAN.md` | Vulkan-Snapshot AMD RDNA3/Vega | 2026-06-15 |
| `archive/rca/2026-06-16_DEBUG_SESSION_PKV_FIX.md` | RCA: PKV Cross-Backend Bug | 2026-06-16 |

### Naming Convention

- **Aktive Doku:** `YYYY-MM-DD_<Thema>.md`
- **Archiv:** `archive/rca/YYYY-MM-DD_<Beschreibung>.md`
- Wenn eine Datei historisch wird: Ins `archive/` verschieben und Header als `[HISTORISCH — ARCHIV]` kennzeichnen.

---

## Verifikation

- [ ] Keine Verweise auf geloeschte `pocs/`-Pfade
- [ ] Alle internen Links auf `docs/fork/` aktualisiert

---

## Child-DOX-Index

*Keine Children.*
