# SESSION_PLAN — Forgejo-Self-Hosting im Eulenhorst-LAN

**Erstellt:** 2026-09-29 | **Session:** Forgejo-Self-Hosting | **Vorgänger:** HANDOFF-forgejo-selfhost.md (wird am Ende gelöscht)
**Status-Legende:** `[ ]` offen | `[~]` in Arbeit | `[x]` fertig | `[!]` blockiert/Problem

**Resume-Anleitung:** Bei Abbruch dieses Dokument lesen — jeder Schritt trägt seinen Status + konkrete Kommandos/Details. Weiter ab dem ersten `[~]`/`[ ]`-Schritt.

---

## Entscheidungen (mit fukuro getroffen)

| Element | Entscheidung |
|---|---|
| Software | **Forgejo v15 LTS** (15.0.x, Support bis 07/2027), Upstream-Binary, SQLite |
| Host | **pluto** (192.168.1.206) — Repos liegen auf ZFS titan-Storage (Backup-Pipeline gratis) |
| LXC-Name | **europa** (Jupiter-Mond auf pluto — Präzedenz: ganymed; Pluto-Monde alle durch Geräte belegt: charon/nix/kerberos/hydra/styx) |
| VMID / IP | **233** / **192.168.1.233** (pluto-Range .232-.239; .232=gany-test belegt) |
| Sizing | 2 vCPU, 2 GB RAM, 20 GB rootfs auf `local-zfs` |
| Edelstein | **`jaspis/titan/pyrit`** — war in Edelstein-Note `RfmB2282Nz2c` bereits semantisch als "Cluster-speicher" reserviert, Dataset existierte noch nicht |
| Quota | **200G** auf allen 3 Hosts (Daten heute <2GB → >100× Headroom; wie rauchquarz-Präzedenz) |
| DNS | `europa.eulenhorst.lan` → .233 + `git.eulenhorst.lan`-CNAME auf pan (dnsmasq) |
| Codeberg | Voll abkoppeln, ABER erst nach Verifikation des LAN-Systems; `codeberg`-Remote + weekly-Cron bleiben bis dahin |
| Install-Methode | Upstream-Binary manuell (NICHT community-script — kollidiert mit lxc-build-Konventionen) |

## Gemessene Datenmengen (Quota-Begründung)

- Codeberg `fukuro`: 17 Repos, ~167 MB API-Size
- GitHub `fukuro-kun`: 25 Repos (meist Forks), ~975 MB
- hydra lokal: fork 500 MB, inferenzquelle 112 MB, deepseek-harness 264 MB, augmentor-agent 17 MB
- phobos `llama-fpd`: Größe beim Push prüfen (llama.cpp-Fork, vermutlich ~400-500 MB)
- **→ pyrit 200G Quota = ~100-130× Headroom inkl. Forgejo-Dumps + LFS**

---

## Phase 0 — Vorbereitung [x]

- [x] Handoff gelesen, Session-Titel "Forgejo-Self-Hosting" gesetzt
- [x] Recherche (2 Subagents): Forgejo vs. Alternativen + LXC-Deployment-Best-Practices
- [x] Host-Ressourcen verglichen (jupiter/saturn/pluto/mars)
- [x] Entscheidungen mit fukuro getroffen (siehe Tabelle)
- [x] Namenskonvention-Note `eyXf6ISr7J1B` korrigiert: Pluto-Monde-Sperrliste (alle 5 als Geräte belegt) + Konsequenz dokumentiert
- [x] Volumen-Analyse + Quota festgelegt (200G)

## Phase 1 — Pre-existing Bug: ZFS-Snapshots ohne `-r` auf pluto + jupiter

**Bug:** Seit 28.06.2026 snapshotten auf pluto+jupiter nur die Parents `jaspis/titan`, keine Sub-Datasets — identischer Mechanismus wie der am 20.08. auf saturn gefixte Bug (fehlendes `-r`-Flag). Subdataset-Snapshots enden 28.06.2026 00:45. Saturn-Referenz (korrekter Zustand): 5 crontab-Jobs mit `-r` (weekly keep=8, monthly keep=16, daily keep=10, hourly keep=20, frequent */15 keep=8) + cron.d frequent */15 keep=4 `-r`.

- [x] **pluto:** `-r` in alle 5 crontab-Jobs + `/etc/cron.d/zfs-auto-snapshot` ergänzt (Backups: `/tmp/cron.bak-20260929`, `/etc/cron.d/zfs-auto-snapshot.bak-20260929`)
- [x] **jupiter:** cron.d `-r` ergänzt + komplette Kadenz in root-crontab wiederhergestellt (weekly keep=8, monthly keep=16, daily keep=10, hourly keep=20, frequent */15 keep=8 — saturn-Werte)
- [x] Verifiziert: manueller `zfs-auto-snapshot -r`-Lauf erzeugte **27 Subdataset-Snapshots pro Host** (Test-Snaps danach via `zfs destroy -r` entfernt)
- [x] Mars: unbetroffen (`//`-Aufruf in cron.d) — dokumentiert, kein Eingriff

## Phase 2 — ZFS `pyrit` auf allen 3 Hosts [x]

- [x] pluto: `jaspis/titan/pyrit` (200G) — Quelle
- [x] jupiter: `jaspis/titan/pyrit` (200G) — Backup-Ziel tethys
- [x] saturn: `jaspis/titan/pyrit` (200G) — Backup-Ziel rhea
- [x] Ownership pluto: `/titan/pyrit` root:root; `/titan/pyrit/forgejo` → **100102:100105** (CT-uid 102/gid 105 des git-Users)

## Phase 3 — LXC `europa` (VMID 233) auf pluto [x]

- [x] Template `debian-13-standard_13.6-1_amd64.tar.zst` gedownloadet
- [x] `pct create 233` — **Fallstrick:** vorab angelegtes+chown'ed subvol → `tar: failed to return to initial working directory: Permission denied`. Lösung: subvol löschen, Proxmox selbst anlegen lassen: `--rootfs lokales_zfs:20`. Erfolgreich.
- [x] MAC `BC:24:11:DD:3F:8E`; DNS auf pan: `address=/europa.eulenhorst.lan/192.168.1.233` + ptr + `dhcp-host=BC:24:11:DD:3F:8E,europa,192.168.1.233,infinite`
- [x] **DNS-Fix nachträglich:** `cname=git.eulenhorst.lan,europa.eulenhorst.lan` lieferte CNAME ohne A-Record → phobos-Resolver schlug fehl. Ersetzt durch `address=/git.eulenhorst.lan/192.168.1.233` (dnsmasq.conf:191)
- [x] Basis: fukuro-User+SSH-Key+sudo, Pakete (git git-lfs curl openssh-server u.a.), locales
- [x] git-Systemuser: `adduser --system --group --home /var/lib/forgejo git` → **uid 102 / gid 105**; Shell später auf `/bin/bash` gesetzt (nologin blockierte SSH-forced-commands)

## Phase 4 — Mounts + Backup-Pipeline [x]

- [x] ganymed (222): `mp27 /jaspis/titan/pyrit,mp=/titan/pyrit` — aktiv
- [x] tethys (102): `mp27` — aktiv
- [x] rhea (242): `mp27` — aktiv
- [x] europa (233): `mp0 /titan/pyrit` (backup=0, replicate=0)
- [x] `MOUNTPOINT_NOT_ACTIVE.lock` in den Basispfaden unterhalb der Datasets auf pluto/jupiter/saturn angelegt (mp-Bind entfernt → unmount → lock → remount → Bind zurück)
- [x] `sicherung_ganymed.sh` v0.5.5 auf tethys+rhea: `pyrit` in SUBVOLUMES (zirkon-Archivkopie = v0.3.2 historisch, bewusst unangetastet)

## Phase 5 — Forgejo-Installation auf europa [x]

- [x] **forgejo-15.0.9-linux-amd64** LTS — sha256 verifiziert → `/usr/local/bin/forgejo` (`15.0.9+gitea-1.22.0`)
- [x] `/etc/forgejo/app.ini` — WORK_PATH=`/titan/pyrit/forgejo` (Repos+SQLite+LFS+Avatare alles auf pyrit → Backup); git:git-Eigentümer (INTERNAL_TOKEN wird hineingeschrieben)
- [x] systemd-Unit — **Fix:** `ProtectSystem=strict` → `/var/lib/forgejo` + `/etc/forgejo` zu `ReadWritePaths` (git-Home/authorized_keys war read-only)
- [x] `systemctl enable --now forgejo` → active, HTTP 200, INSTALL_LOCK
- [x] Admin `fukuro` via CLI angelegt; `--must-change-password` blockierte API (403) → via `forgejo admin user change-password --must-change-password=false` aufgehoben (Bootstrap-PW in `/root/forgejo-admin-bootstrap.txt` 600 auf europa)
- [x] **Token-Sicherheit:** initialer API-Token war im Session-Output sichtbar → **rotiert**: `admin-session`(id1) gelöscht via Basic-Auth-DELETE, neuer `admin-cli2`(id2) in `/root/.forgejo-admin-token` (600, nie ausgegeben)

## Phase 6 — Forgejo-Befüllung [x]

- [x] **8 SSH-Keys** via API registriert (HTTP 201): codeberg-shared (alle Hosts teilen denselben Key!), github-shared, hydra/eris/styx/venus-default, janus-router, phobos-default
- [x] sshd: git-User-Shell nologin→bash gefixt; SSH-Auth verifiziert (Forgejo-Forced-Command antwortet)
- [x] Repos angelegt: `fukuro-llama-cpp-turboquant`, `inferenzquelle`, `llama-fpd` (⚠ llama-fpd leer — phobos' Verzeichnis ist ein **worktree des Forks**, kein eigenes Repo → leeres Repo noch löschen)
- [x] phobos-Stash geprüft: **2 widersprüchliche Stashes** — stash@{1} fügt `--mmproj`+Vision-Doku hinzu, stash@{0} entfernt `--mmproj` wieder → **Entscheidung offen an fukuro** (Vision an oder aus?)

## Phase 7 — Push-Backlog [x]

- [x] **hydra:** `forgejo`-Remote + `push --all`: 8 Branches (master, feat/k2-horizon-mtp [5 Commits, mehr als Handoff-2], feature/qwen4exp-model-family-port, backup/eris-…, +4 weitere). Tags: `--tags` schlug fehl ("send-pack disconnect") → **Repo ist shallow clone** (`.git/shallow`, 1 Boundary). Tags in 200er-Batches gepusht; ~alle ab Boundary-Reichweite OK, Tags jenseits der Grenze `shallow update not allowed` (→ später `git fetch --unshallow` von github/ggml-org, dann Rest pushen)
- [x] **eris:** forgejo-Remote + push `feature/qwen4exp-model-family-port` → `ecf850af7..aa1976da9` (12 Commits)
- [x] **phobos:** DNS-Fix + hostkey + push `fix/k2-horizon-seqrm-probe` (98f359911) — Achtung: llama-fpd = worktree von `~/git/fukuro-llama-cpp-turboquant`
- [x] **uranus:** forgejo-Remote + fetch — **keine unique Commits** (alle in hydra-Push enthalten); 2 lokale Backup-Branches (`test-cherry-pick-13683`, `uranus-local-backup-20260613`) mitgepusht
- [x] **InferenzQuelle:** `forgejo`-Remote, `main`+`develop` gepusht. Topologie: `origin`=codeberg (weekly-cron-Mirror, bleibt vorerst), `janus`=LAN-Deploy-Pfad (bleibt unangetastet)
- [x] ssh-config-Eintrag `git.eulenhorst.lan` auf hydra/eris/phobos/uranus (IdentityFile id_ed25519_codeberg)

## Phase 8 — Verifikation + Codeberg-Umstellung [x]

- [x] `git push`/`fetch` von hydra, eris, uranus, phobos funktioniert
- [x] **Klon-Test:** `git clone git@git.eulenhorst.lan:fukuro/inferenzquelle` auf **styx** — sauber (main+develop, volle History)
- [x] Web-UI läuft (HTTP 200); Admin `fukuro` existiert; API verifiziert
- [x] Tag-Push fertig: **6172 gepusht, 228 rejected** (shallow-Boundary — Upstream-Release-Tags, via `git fetch --unshallow` von github/ggml-org heilbar)
- [x] Leeres `llama-fpd`-Repo gelöscht (phobos' Verzeichnis ist worktree des Forks)
- [x] Backup-Pfad verifiziert: tethys Mountpoint-Check PASS, pyrit in SUBVOLUMES, Dataset vorhanden (erster echter Lauf heute Nacht 04:45)
- [x] Snapshot-Verifikation: 27 Subdataset-Snaps pro Host nach Fix bestätigt
- [x] **Alle 17 Codeberg-Repos migriert** via `POST /api/v1/repos/migrate` (HTTP 201 ×15 + 2 manuell + LLM_Selector-Pilot → **18 Repos auf Forgejo**)
- [x] **phobos-Stash:** gelöst ohne Eingriff — `stash@{0}` (mmproj raus) war längst als Commit `6c54299ac` in master (Vision DEAKTIVIERT = Production-Standard, RADV-Bug); `stash@{1}` veraltet. Beide liegen gelassen.
- [x] **origin-Flip auf allen Hosts:** hydra/eris/uranus/phobos (Fork) + IQ: `origin`→forgejo, `codeberg`→Remote behalten, Upstream-Tracking repariert. Cron-Tag `codeberg-sync`→`forgejo-sync` (pusht jetzt `origin`=forgejo)
- [x] Weekly `forgejo dump` Cron auf europa (So 04:17, Rotation 4, `/usr/local/sbin/forgejo-dump.sh` → `/titan/pyrit/forgejo/dumps/`)

## Phase 9 — Doku + Abschluss [~]

- [x] Trilium: Europa-System-Note `uDttn4lAvZK7` (unter pluto), Netzwerkplan `1IYS31C9CrSu` (.233+git-A-Record+Aktualisierung), Edelstein `RfmB2282Nz2c` (pyrit→Git-Repos + topas-Stale-Fix 2.72T), Namenskonvention `eyXf6ISr7J1B` (europa in Pluto-Tabelle), Port-Plan `WFXb5URFtYoU` (europa:3000)
- [x] `PORT_PLAN.md`: europa 3000 eingetragen
- [x] hardware-Projekt: `ip-plan.md` + `AGENTS.md` (Hosts-Tabelle + Offene-Punkte-Eintrag)
- [x] InferenzQuelle `AGENTS.md`: Remote-Topologie statt codeberg-origin
- [x] `LOKAL.md`: Git-Remote-Topologie-Sektion + clone-URL gefixt
- [x] zirkon-Archiv: `sicherung_ganymed.sh` v0.3.2-Museumsstück → **Blaupause v0.6.0** (generisches Deployment-Rezept, kanonische SUBVOLUMES-Liste)
- [x] DOX-Durchlauf Fork-Repo: `AGENTS.md` (Primary Remote) + `FORKS.md` (Remote-Tabelle/Branch-Strategie) auf Forgejo umgestellt; IQ `AGENTS.md` committed+gepusht (`6646ac4`)
- [x] TTT-Eintrag (Tag 29.09, ~17:00–23:06)
- [x] `docs/HANDOFF-forgejo-selfhost.md` gelöscht (Implementation abgeschlossen)
- [ ] **Restpunkt heute Nacht:** erster echter sicherung_ganymed-Lauf (tethys 04:45 / rhea 06:45) — pyrit-Inhalt muss auf beiden ankommen → morgen verifizieren (`ssh root@jupiter 'pct exec 102 -- du -sh /titan/pyrit'`)
- [ ] `git fetch --unshallow` auf hydra + 228 Rest-Tags pushen (optional, Upstream-Tags auch via github erreichbar)
- [ ] Diese Plan-Datei am Ende löschen + committen — nach Backup-Verifikation

## Bekannte Fallstricke (aus Recherche)

- Unpriv. LXC: git-User uid im CT → Host-uid = 100000+uid → Dataset-Ownership darauf mappen
- authorized_keys nur via Forgejo-UI pflegen (Validierung!) — nicht manuell editieren
- `FORGEJO_WORK_DIR` (nicht `GITEA_WORK_DIR`) in systemd-Unit
- HTTP ohne TLS: keine Token-/Passwort-Auth über HTTP — git-Ops via SSH
- pct create: nicht `/etc/pve/lxc/*.conf` direkt editieren (Cluster-managed)
- Git-Identität auf Remote-Hosts: `git -c user.name=... -c user.email=...` (nie `git config` ändern!)
- Fork-Repo hat viele Remotes — nur `origin` umstellen
