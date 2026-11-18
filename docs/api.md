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
- Optionales `start_mode` überschreibt ausschließlich für diesen Versuch den
  Aufnahmestart (`presence`, `motion`, `hardware`, `manual`). Capability
  `trial_start_mode_override=true` kennzeichnet die Unterstützung. `presence`
  beginnt mit erkanntem Bauteil und endet nach dessen Verschwinden und der
  eingestellten Ruhezeit; ein Drehwinkel, Bewegungssignal oder Hardwaretrigger
  ist dafür nicht erforderlich. Danach wird die vorherige Konfiguration wiederhergestellt.
- Aufnahmeeinstellungen werden für jeden Versuch aus der GUI übernommen, auch wenn
  diese nach dem Verbinden der Kamera geändert wurden. Messreferenz und
  Aufnahmeeinstellungen bleiben bis zum Versuchsende gesperrt.
- Bei Capability `qualified_presence_start=true` kann ein Versuch zusätzlich
  `presence_confirm_seconds` (0,05–0,5 Sekunden) anfordern. Dies ist nur im
  effektiven Startmodus `presence` zulässig. Der Start benötigt dann mindestens
  drei aufeinanderfolgende Bilder über diese Dauer mit gültigem Winkel, Kontur
  und Bounding Box ohne Randkontakt, Qualitätswert mindestens 0,2, Winkelspanne
  höchstens 5° sowie Fläche und Boxseiten jeweils innerhalb ±25 % des ersten
  Kandidatenbilds. Zeitlücken über 50 ms und ungültige Bilder setzen den Kandidaten
  zurück; Positionsänderung durch das Förderband ist erlaubt. Eine verworfene
  Kandidatur beendet den externen Versuch nicht und erzeugt keinen Clip.
  Der Vorlauf wird mindestens auf Bestätigungsdauer plus 50 ms erweitert.
  Antwort und Manifest enthalten die effektive Konfiguration. Nach dem Versuch
  wird die vorherige Konfiguration wiederhergestellt. Ohne dieses Feld gilt das
  bisherige Startverhalten. Der Filter ist keine semantische Hand-/Bauteilerkennung.
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
die GUI läuft. Ohne Speicherprofil bleiben Originalbilder im bisherigen Aufnahmeordner. API und
CaptureManager halten höchstens 256 externe Versuche pro Pipeline vor.

Ein Wechsel des Kameraprofils in der GUI macht die bisherige externe Sitzung ungültig.
Nach Verbinden der anderen Kamera muss der Client eine neue Sitzung anfordern und
seinen Live-Messcursor zurücksetzen. Abgeschlossene Versuchsergebnisse bleiben über
ihre bisherigen Kennungen abfragbar. Endpunkte und Antwortformate bleiben unverändert.

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

## Kompakte Prüfvideo-Aufnahmen

Die optionalen Fähigkeiten `review_video_v1=true` und `direct_trial_output=true`
erlauben ein kompaktes Video direkt im Versuchsordner des Clients. Dazu vor
`POST /trials` einen neuen absoluten Zielordner anlegen und darin
`capture-request.json` mit genau diesen Werten schreiben:

```json
{"format":"review_video_v1","run_id":"kampagne","trial_id":"versuch"}
```

Der Trial-Aufruf enthält dieselben Kennungen sowie
`"storage_profile":"review_video_v1"` und
`"output_directory":"C:/Daten/kampagne/trials/versuch"`. Der Ordner muss für den
Kameraprozess schreibbar sein und darf noch kein `clip` oder `clip.tmp` enthalten.
Unter 1 GiB freiem Speicher wird die Anfrage abgelehnt. Ohne diese Felder bleibt
das bisherige NPY-Aufnahmeformat unverändert.

Während der Aufnahme schreibt die Kamera `clip.tmp/review.mp4` (`mp4v`) und ein
fortlaufend geflushtes `measurements.jsonl`. Das Video enthält jedes verarbeitete
Bild, zugeschnitten auf den Messbereich und verkleinert auf maximal 960 Pixel an
der längsten Seite. `capture.json` enthält die zugehörigen Metadaten für eine
Wiederherstellung nach Unterbrechung. Bilder werden nach dem Schreiben aus dem
begrenzten RAM-Puffer freigegeben. Es gibt keine zweite Bildkopie im Kamera-Repo
und keinen automatischen Rückfall auf NPY bei Encoderfehlern.

Zum Abschluss entsteht `manifest.json` mit `schema_version=2`. Dessen `frames`
enthalten die ursprünglichen Messwerte, Zeitstempel, Frame-IDs und
`video_frame_index`; die Videokomprimierung verändert diese Messwerte nicht.
`media` beschreibt `file`, `format`, `state`, `error`, `crop_xywh`, `scale_xy`,
`image_size`, `content_size` und `fps`. Bildrate und Videozeit ersetzen keine
Messzeitstempel. Overlays werden beim Anzeigen in die Videokoordinaten umgerechnet.
Bei Encoderfehlern bleiben Messwerte erhalten, `media.state` ist `failed`.

Erst nach Schließen aller Dateien wird `clip.tmp` in `clip` umbenannt.
Windows-Zugriffssperren werden begrenzt erneut versucht. Bleibt eine Sperre
bestehen, verweist `clip_path` zunächst auf `clip.tmp`; spätere Ergebnisabfragen
versuchen den Abschluss erneut. Ein unterbrochenes Journal wird als abgebrochene
Aufnahme mit Endgrund `write_error` wiederhergestellt. Die Kamera-GUI liest
beide Manifestversionen und lädt aus MP4 nur einen kleinen Bildpuffer.
Fehlende oder vom Client nach Prüfung gelöschte Videos verhindern nicht den
Zugriff auf Messwerte und deren CSV-Export.

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
