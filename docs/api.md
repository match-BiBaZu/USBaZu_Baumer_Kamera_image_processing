# Kamera-API v1

Die lokale API startet mit der Kamera-GUI, auch wenn die Kamera noch nicht
verbunden ist. Standard: `127.0.0.1:8766`; Port über `BIBAZU_VISION_API_PORT` ändern.
`BIBAZU_API_DISABLED=1` deaktiviert den Server. `Setup.ps1` aktualisiert die
Abhängigkeiten. Bei belegtem Port läuft die GUI weiter und protokolliert den Fehler.

OpenAPI: `http://127.0.0.1:8766/openapi.json`, interaktive Ansicht: `/docs`.
Alle folgenden Endpunkte tragen den Präfix `/api/v1`.

## Sitzung und Aufnahme

- `GET /health`, `/capabilities`, `/state`: Verbindung, unterstützte Funktionen,
  aktuelle Messreferenz und Aufnahmekonfiguration lesen. Für Messungen muss die
  Kamera oder Demo verbunden sein und bereits ein Bild geliefert haben.
- `POST /sessions` mit `{"client":"Mein Programm"}`: Exklusive Sitzung beginnen.
  Eine manuelle Aufnahme muss vorher beendet und entschärft sein.
- Schreibaufrufe benötigen `X-Session-Id` aus der Antwort. Alle fünf Sekunden
  `POST /sessions/heartbeat`; die Sitzung verfällt nach 15 Sekunden.
- `POST /trials` mit `run_id`, eindeutiger `trial_id`, optionalem `context` und
  `timeout_seconds` (Standard 60, zulässig 1–3600): genau eine Aufnahme vorbereiten.
  HTTP 202 bedeutet angenommen; `GET /trials/{trial_id}` zeigt den weiteren Zustand.
- Aufnahmeeinstellungen werden für jeden Versuch aus der GUI übernommen, auch wenn
  diese nach dem Verbinden der Kamera geändert wurden. Messreferenz und
  Aufnahmeeinstellungen bleiben bis zum Versuchsende gesperrt.
- Im Startmodus `manual` den Versuch über `POST /trials/{trial_id}/trigger`
  auslösen. Bei `motion` oder `hardware` erfolgt der Start entsprechend der GUI.
- `POST /trials/{trial_id}/abort`: entschärfen beziehungsweise laufende Aufnahme
  mit einem ausdrücklich als abgebrochen markierten Manifest speichern.
- `DELETE /sessions/current`: Sitzung freigeben; ein aktiver Versuch wird beendet.

Die Versuchskennung wird beim Beginn fest mit den Aufnahme-Metadaten verbunden.
Nach dem ersten Durchlauf wird automatisch entschärft. Ein weiterer Lichtschranken-
oder Bewegungsimpuls erzeugt keinen zweiten Clip mit derselben Kennung.

## Ergebnisse und Livewerte

`GET /trials/{trial_id}` liefert zunächst `pending`, `armed`, `recording` oder
`saving`. Endzustände sind `completed`, `failed` und `aborted`. Erst ein erfolgreich
gespeicherter Durchlauf mit Endgrund `part_left` wird `completed`. Ein stehendes
Bauteil, Zeitlimit oder Speicherfehler gilt nicht als vollständiger Durchlauf.
Auch `completed` ist keine Aussage über die erreichte Roadmap-Zielpose.

Ein gespeichertes Ergebnis enthält `clip_path`, Manifest und vollständige
`measurements`. Ergebnisse bleiben nach dem Trennen der Kamera abfragbar, solange
die GUI läuft. Originalbilder bleiben im vorhandenen Aufnahmeordner. API und
CaptureManager halten höchstens 256 externe Versuche pro Pipeline vor.

`GET /measurements?after=0&limit=2048` liefert kleine Messdatensätze ohne Bilder:
Sequenznummer, monotonen Host-Abrufzeitstempel, Frame-ID, Winkel, Qualitätsmaß,
Gültigkeit, Randkontakt, Anwesenheit, Aufnahmezustand und Versuchskennung.
`angle_deg=null` ist ein ungültiger oder uneindeutiger Winkel. `confidence` ist ein
geometrisches Qualitätsmaß und keine kalibrierte Messunsicherheit.

Den zurückgegebenen `next_cursor` bei der nächsten Abfrage als `after` verwenden.
`latest_cursor` bezeichnet den neuesten verfügbaren Datensatz. Der Livepuffer hält
8192 Datensätze; bei `overflow=true` fehlen ältere Livewerte. Die gespeicherte
Messreihe des Versuchs ist davon unabhängig. Zum Anzeigen reichen 200-ms-Abfragen;
für vollständige Versuchsanalysen das fertige Ergebnis verwenden.

Zeitstempel sind monotone Host-Abrufzeiten, keine Hardware-Exposure-Zeitstempel.
SPS- und Kamerazeitstempel werden nicht ohne zusätzliche Synchronisierung als
identische Zeitbasis interpretiert. Die Zuordnung erfolgt über `run_id`/`trial_id`.
Der Winkel beschreibt weiterhin eine 2D-Achse modulo 180°.

## Fehler und Laufzeit

Während **Kameraeinstellungen anwenden** läuft, wird `POST /trials` mit HTTP 409
abgelehnt. Nach Abschluss kann derselbe Versuch erneut angefragt werden; die
abgelehnte Anfrage reserviert keine Versuchskennung.

409: Sitzung oder Ressource belegt/veraltet; 422: ungültige Anfrage; 503: Kamera
nicht verfügbar; 504: Qt-Antwort ausgeblieben; 404: Versuch unbekannt.
Ein Verbindungsverlust oder Sitzungsablauf beendet den laufenden Versuch. Es gibt
keine automatische Wiederaufnahme nach erneutem Verbinden. Förderband und SPS
werden durch diese API nicht angesprochen.

`local_api_runtime.py` entspricht der gleichnamigen Datei im Pressure-Control-
Projekt. Diese bewusst kleine gemeinsame Laufzeit bei Änderungen synchron halten.
