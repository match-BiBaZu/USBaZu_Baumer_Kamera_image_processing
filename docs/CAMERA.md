# Baumer USB-Kamera

Der Adapter `src/bibazu_vision/camera.py` verwendet Harvesters/GenICam mit dem
Baumer USB GenTL-Producer. `find_cti_paths()` bevorzugt USB gegenüber GigE.
Ein üblicher Installationspfad des Producers lautet
`C:\Program Files\Baumer Camera Explorer\bgapi2_usb.cti`.

```powershell
python scripts/probe_camera.py --line Line0
```

Die Probe speichert ein unverändertes Kamera-Frame in
`.diagnostics/current_camera.png` und lesbare Gerätedaten in
`.diagnostics/camera_diagnostics.json`. Auch fehlgeschlagene Verbindungen werden
dort dokumentiert. Der Camera Explorer muss seine exklusive Verbindung freigeben,
bevor ein zweites Programm auf die USB-Kamera zugreift.

## Trigger-Modi

- `live`: kontinuierliche Aufnahme. `input_monitor="auto"` reserviert bevorzugt
  einen nachweislich unbenutzten Hardware-Zähler (`CounterEventSource=Off`, keine
  aktive Zähler-Triggerquelle und `CounterResetSource=Off`). Der Zähler zählt
  die ausgewählte Flanke des Eingangs auch zwischen Frames. Eine Änderung von
  `CounterValue` markiert `CameraFrame.triggered`. Der Adapter setzt die Dauer auf
  das Zählermaximum und konfiguriert eine automatische Rücksetzung durch das
  eigene `CounterEnd`-Signal. Damit sättigt der Zähler nicht dauerhaft. Es wird
  kein Software-Reset ausgeführt. Vorhandene Zählerwerte dienen als Startwert.
  Ist kein sicher nutzbarer Zähler vorhanden, fällt `auto` auf `LineStatus` zurück.
  `input_monitor="counter"` verlangt einen solchen Zähler und meldet sonst einen
  Fehler; `input_monitor="line_status"` erzwingt die einfachere Software-Abfrage.
  Bei dieser Abfrage kann ein kürzerer Puls als der Frame-Abstand fehlen.
- `frame_trigger`: `TriggerSelector=FrameStart`, externe Quelle und Flanke.
  Jeder Puls erzeugt **ein** Bild. Dieser Modus zeichnet keine vollständige
  Durchfahrt auf, wenn die Lichtschranke nur einen Puls je Bauteil erzeugt.
- `acquisition_trigger`: setzt ausschließlich einen tatsächlich verfügbaren
  `TriggerSelector=AcquisitionStart` zusammen mit kontinuierlicher Aufnahme.
  Nach der ersten Aufnahme muss `rearm_acquisition()` zwischen Durchfahrten
  aufgerufen werden. Fehlt dieser Selector, meldet der Adapter einen Fehler;
  `FrameStart` wird nicht als Start einer Folge behandelt.

`diagnostics` enthält vorhandene Trigger-Selectoren und Quellen sowie verfügbare
Linien-, Event-, Chunk- und Zähler-Selectoren. `hardware_trigger_monitor` bezeichnet
den tatsächlichen gewählten Erfassungsweg. Der Live-Zähler wird über das GenICam-
Merkmal abgefragt; Events/Chunks werden dafür nicht benötigt. Mehrere Zählimpulse
zwischen zwei Frames werden diagnostisch gezählt, ergeben aber nur ein
Triggerflag in diesem Frame. Eine reale Passage ist nötig, um die Verdrahtung,
Polarität und Erfassung kurzer Lichtschranken-Pulse zu prüfen.

## Daten und Rücksetzen

`CameraFrame.image` ist zusammenhängendes `uint8` in Mono oder BGR. Der Adapter
kopiert das Bild innerhalb des GenTL-Fetch-Kontexts: Nach der Rückgabe bleibt es
gültig, auch wenn der Kamera-Producer seinen Puffer erneut verwendet. Gepackte
Formate werden mit einer klaren Meldung abgelehnt. Mono8 spart Debayering und
Speicherbandbreite. 16 GenTL-Puffer stehen für die Aufnahme bereit.

`timestamp` verwendet die monotone Host-Uhr zum Empfang des Frames und ist kein
kalibrierter Kamera-Belichtungszeitpunkt. `frame_id` übernimmt, soweit verfügbar,
die Kamera-/Producer-Frame-ID. Kamera-Einstellungen bleiben flüchtig: `close()`
stellt ursprüngliche Trigger-, Belichtungs- und Bildratenwerte wieder her. Für den
reservierten Zähler werden Quelle, Flanke, Dauer, Reset-Quelle und, falls
schreibbar, Zählerstand im korrekten Selektorkontext restauriert. Danach wird der
ursprüngliche `CounterSelector` wiederhergestellt. Andere Zähler bleiben erhalten.
Der Adapter bedient keine weiteren Hardware-Aktoren.

Bei expliziter manueller Belichtung werden auch vorhandene `GainAuto`,
`BalanceWhiteAuto` und `ColorTransformationAuto` temporär auf `Off` gestellt.
Gain, Farbkoeffizienten und Gamma werden dabei nicht gesetzt; ihre aktuellen
Werte bleiben erhalten. Damit können automatische Helligkeits- und Farbkorrekturen
die feste Segmentierungsschwelle während einer Messung nicht fortlaufend
verändern. `close()` stellt die ursprünglichen Auto-Modi wieder her. Eine Probe
ohne explizite Belichtungszeit verändert diese Auto-Modi nicht.

## Untersuchung am 14. Oktober 2026

Windows und der USB-Producer haben eine **Baumer VCXU-02C** mit aktivem
Baumer USB-Treiber identifiziert, Gerätestatus OK. Der geöffnete Baumer Camera
Explorer belegte zunächst das Gerät. GenTL meldete beim Öffnen `AccessDeniedException`
und Gerätezugang `NOACCESS`. Die angehängte Fehlermeldung „Unsupported info
parameter“ stammt aus einem vorherigen, nicht unterstützten Informationsabruf
und beschreibt den Zugangsfehler nur unzureichend.

Ein Screenshot des kurz wiederhergestellten Explorer-Fensters zeigte ein liegendes,
dunkelgraues Bauteil mit Langloch vor gelochtem Metall und auf einer strukturierten
Rutsche. Der Explorer zeigte **640×480**, **49,08 FPS** und automatische
Belichtung mit **20.331 µs**. Der Screenshot ist skaliert und kein Rohbild der
Kamera. Keine Kameraparameter wurden dabei geändert. Das zuvor minimierte
Explorer-Fenster wurde anschließend wieder minimiert.

Nach ausdrücklicher Genehmigung wurde der Explorer regulär geschlossen und
sein eigener Schließen-Dialog bestätigt. Danach konnte der Adapter echte
640×480-Frames aufnehmen. Die Kamera unterstützt ausschließlich den Trigger-
Selector `FrameStart`; ein echtes `AcquisitionStart` als Trigger-Selector fehlt.
`Line0` ist ein Eingang. `Counter1` und `Counter2` waren unbenutzt und unterstützen
`Line0`, `Line1`, `Line2` sowie steigende, fallende oder beide Flanken. Auch
`Line0RisingEdge` als Event und `CounterValue` als Chunk sind vorhanden.

Mit 1.000 µs Belichtung und 200 FPS als temporärer Sollwert wurden 408 Frames in
2,003 Sekunden gelesen (203,69 Host-FPS), Kamera-IDs 0 bis 407 ohne Lücke und ohne
doppelte Host-Zeitstempel. Bei der vorhandenen Beleuchtung ist dieses schnelle
Bild sehr dunkel. Der Baseline-Snapshot mit 20.331 µs Belichtung ist gut sichtbar.
Das ist ein Stream-Test bei stehendem Bauteil, kein Test einer schnellen Passage.

Die Hardware-Zählerkonfiguration wurde anschließend direkt am Gerät angenommen:
`Counter1`, Quelle `Line0`, steigende Flanke, Dauer `65535`, automatische
Reset-Quelle `Counter1End`. Bei unverändertem Eingang wurden 50 Frames in 1,011
Sekunden ohne falsches Triggerflag gelesen. Ein neuer Verbindungsabruf nach
`close()` bestätigte die Wiederherstellung: Quelle `Off`, Reset-Quelle `Off`,
Dauer `1`, Zählerstand `0` und ursprünglicher Zähler-Selector. Damit sind die
Native-Konfiguration und Rücksetzung geprüft. Ein physischer Lichtschranken-Puls
und eine tatsächlich bewegte Passage wurden noch nicht getestet.

Am Gerät waren `GainAuto`, `BalanceWhiteAuto` und `ColorTransformationAuto`
zusätzlich zu `ExposureAuto` ursprünglich auf `Continuous` gestellt. Ein
separater Native-Test mit 20.331 µs manueller Belichtung bestätigte alle vier
Auto-Modi als `Off`, fünf empfangene Frames sowie unveränderte Werte für Gain
(`1,0`) und Gamma (`1,203125`). Nach dem Schließen und erneuten Öffnen standen
alle vier Auto-Modi wieder auf `Continuous`; Gain und Gamma waren unverändert.

Ein langer Belichtungszeitpunkt kann bei schnellen Passagen Bewegungsunschärfe
erzeugen. Für den Betrieb kurze manuelle Belichtung mit ausreichender Beleuchtung
und die tatsächlich erreichbare Bildrate prüfen. Die softwareseitige Performance
allein bestätigt weder scharfe Kamera-Frames noch lückenlose Hardware-Aufnahme.

Referenz: [Baumer VCXG/VCXU Bedienungsanleitung, Version 4.8](https://media.baumer.com/Baumer_UsersGuide-VCXG_VCXU-v48_EN_20241025_MNL.pdf).
