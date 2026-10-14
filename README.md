# BiBaZu · Baumer-Kamera und Winkelmessung

Desktop-GUI für eine Baumer USB3-Vision-Kamera: Livebild mit Kontur, orientiertem
Rechteck (OBB) und Hauptachse, Winkel gegenüber der Rutsche, verlustlose
Durchlaufaufzeichnung, Einzelbildkontrolle, Zeitlupe und Messwerte als CSV.
Es werden keine trainierten Modelle oder Trainingsdaten benötigt.

## Start unter Windows

Auf diesem PC ist eine eigene Python-Umgebung im Repository eingerichtet.
`Start.cmd` öffnet die GUI; `Start.cmd -Demo` startet die synthetische Demo.

Der Button **English / Deutsch** oben rechts schaltet die Sprache direkt im
laufenden Fenster um. Auswahl, Messreferenz und Aufnahme bleiben erhalten.
Die Sprache wird für den nächsten Start gespeichert, auch im Demo-Modus.
Mit `bibazu-vision --language en` bzw. `--language de` kann sie beim Start
vorgegeben werden. Beschriftungen, Statusmeldungen, Log und Dialoge sind übersetzt;
Messdateien und Kamera-Parameter verwenden weiterhin ihre ursprünglichen Werte.

Für einen neuen Rechner Python 3.11+ und den **Baumer USB GenTL-Producer**
(64 Bit passend zu Python, beispielsweise mit Baumer Camera Explorer) installieren:

```powershell
./Setup.ps1
./Start.ps1
# Ohne Hardware:
./Start.ps1 -Demo
```

Alternativ mit uv:

```powershell
uv sync --extra dev
uv run bibazu-vision
uv run bibazu-vision --demo
```

Der Camera Explorer muss die Kamera freigeben, bevor die GUI sie öffnet.
Beim Verbinden werden Belichtung, Bildrate und Triggerkonfiguration nur temporär
gesetzt und beim regulären Trennen wiederhergestellt. Es werden keine UserSets
dauerhaft in der Kamera gespeichert. Ein Prozessabbruch kann die Wiederherstellung
verhindern; dann die gewünschten Einstellungen im Explorer erneut setzen.

## Erster Versuch mit liegendem Bauteil

1. Kamera verbinden. Der Standardpfad lautet
   `C:\Program Files\Baumer Camera Explorer\bgapi2_usb.cti`.
2. Einen Messbereich wählen: zwei gegenüberliegende Ecken im Livebild anklicken.
   Nur die tatsächlich durchfahrene Rutsche einschließen; Randstrukturen auslassen.
3. Das Bauteil kurz entfernen und **Leere Rutsche als Hintergrund aufnehmen**
   wählen. Dieses Referenzbild bleibt fest. Kamera, Rutsche, Fokus und Beleuchtung
   danach unverändert lassen. Ein liegendes Bauteil im Hintergrundbild würde
   anschließend herausgerechnet.
4. Bauteil wieder auflegen. Die Differenzschwelle so einstellen, dass die grüne
   Kontur und das OBB zum Bauteil passen. Die Hintergrundlöcher werden durch die
   feste Referenz weitgehend ausgeblendet.
5. **Rutsche markieren**: zwei Punkte entlang ihrer Richtung wählen.
6. **Liegendes Bauteil = 0°** wählen. Jetzt sollte das ruhende Teil ungefähr 0°
   anzeigen. Dieser Offset berücksichtigt, dass die Hauptachse eines unregelmäßigen
   Teils nicht parallel zu seiner Auflagefläche sein muss. Nach Teilewechsel,
   Methodenwechsel OBB/PCA oder anderer Ruhelage erneut nullen.

Ohne Hintergrundbild ist eine einstellbare Dunkel/Hell-Segmentierung vorhanden.
Bei einem gelochten Metallhintergrund ist dieser Einstieg erheblich weniger robust.
Die Hauptachse wird bei annähernd quadratischen oder runden Silhouetten als
uneindeutig markiert; dann erscheint kein scheinpräziser Winkel.

Das lokale Startprofil verwendet aktuell PCA. Beim statischen Bauteil ergab eine
separate Aufnahme mit 64 Frames etwa −0,026° bis +0,069° gegenüber der Ruhe-Referenz.
Die alte OBB-Achse wechselte stärker zwischen äußersten Konturpunkten.
Diese Wiederholbarkeit ersetzt keine Prüfung der Winkelgenauigkeit bei Bewegung.

**Für den nächsten Versuch vorgemerkt (14.10.2026):** Das Bauteil kann aktuell
nicht entfernt werden. Deshalb ist noch kein leeres Hintergrundbild vorhanden.
Sobald die Rutsche frei ist, bei unverändertem Kamerabild und gleicher Beleuchtung
die Hintergrundaufnahme nachholen, Kontur prüfen und das wieder aufgelegte Teil
erneut auf 0° setzen. Das aktuelle Bild mit Bauteil darf nicht als Hintergrund
aufgenommen werden.

## Messbedeutung und Geschwindigkeit

Die Messung ist eine **Rotation in der Bildebene**. Positive Werte bedeuten gegen
den Uhrzeigersinn, Wertebereich −90° bis +90° (ungerichtete Achse modulo 180°).
Eine halbe Umdrehung ist anhand der Achse nicht unterscheidbar. Es werden keine
3D-Höhe oder verdeckte räumliche Kippwinkel bestimmt. Bei Randkontakt/abgeschnittenen
Teilen wird kein gültiger Winkel ausgegeben. Die angezeigte Qualität ist ein geometrisches
Heuristikmaß, keine kalibrierte Winkelunsicherheit.

Aufnahme und Verarbeitung laufen in einem eigenen Thread. Die Oberfläche zeigt
nur das jeweils neueste Bild mit maximal etwa 30 Hz; erfasste Bilder werden während
einer Aufzeichnung unabhängig davon verarbeitet und behalten. Vorschauauslassungen
sind daher kein Aufnahmeverlust. Frame-ID-Lücken werden angezeigt und pro Clip
gezählt, soweit die Kamera echte Frame-IDs liefert. Jeder gespeicherte Frame hat
einen monotonen **Host-Abrufzeitstempel**; das ist kein Exposure-Hardwarezeitstempel.

ROI und halbe/viertel Verarbeitungsauflösung senken die CPU-Last, während die
Aufzeichnung Originalbilder behält. Für schnelle Bauteile kurze Belichtungszeit
und passende Beleuchtung verwenden. Kamera-FPS, USB-Transfer, Verarbeitung und
Speicherbudget müssen gemeinsam ausreichen. Die GUI zeigt gemessene Abruf-FPS
und Verarbeitungsdauer. Ein größerer FPS-Sollwert garantiert diesen Durchsatz nicht.

## Lichtschranke und Aufzeichnung

Die Aufnahme wird zuerst scharfgeschaltet. Verfügbare Starts:

- **Bewegung**: Start bei sichtbarer Änderung, mit kurzem Vorlauf.
- **Lichtschranke**: Start bei einem vom Kameraadapter gemeldeten Eingangssignal.
- **Manuell**: über den Auslöseknopf.

Kamerabetriebsarten sind getrennt vom Aufzeichnungsstart:

- **Live + Eingang beobachten**: kontinuierliches Bild. Ein freier kamerainterner
  Hardwarezähler zählt die Lichtschrankenflanken; sein Wert wird zwischen den
  Frames gelesen. Damit bleibt auch ein zwischen den Abfragen beendeter kurzer
  Impuls erkennbar. Auf der VCXU-02C wurde diese Konfiguration mit `Counter1`
  und automatischer Rücksetzung bei 65535 tatsächlich geöffnet und anschließend
  vollständig zurückgesetzt. Ein realer Lichtschrankenimpuls muss noch geprüft werden.
  Fehlt ein sicher nutzbarer Zähler, erfolgt ein ausdrücklich gemeldeter Rückfall
  auf `LineStatus`-Polling, das kurze Impulse verpassen kann.
- **Hardware startet Bildfolge**: benötigt kameraseitiges `AcquisitionStart` als
  TriggerSelector. Nach dem Signal läuft eine Bildfolge; nach abgeschlossenem
  Durchlauf wird neu scharfgeschaltet. Nicht unterstützte Modelle melden einen Fehler.
- **Hardware: Einzelbild pro Impuls**: `FrameStart`; ein Signal ergibt ein Bild.
  Diese Betriebsart kann mit nur einem Lichtschrankenimpuls keinen kompletten
  Durchlauf filmen. Sie dient der Einzelbilddiagnose.

Ein einzelner `FrameStart`-Impuls ist keine Bildfolge. Ein Burst-/Sequencer-Modus
ist je Kameramodell gesondert zu konfigurieren und wird hier nicht vorausgesetzt.
Die angeschlossene VCXU-02C bietet tatsächlich nur `FrameStart`; der Modus
`Hardware startet Bildfolge` wird für dieses Gerät daher abgelehnt. Für eine
Durchfahrt **Live + Eingang beobachten**, Start **Lichtschranke** und Eingang
**Line0** wählen und scharfschalten. Die Polarität am echten Signal prüfen.
Siehe [Baumer Hardware-Trigger](https://www.baumer.com/us/en/service-support/technical-information-industrial-cameras/hardware-trigger/a/hardware-trigger-with-baumer-gapi)
und [Baumer: mehrere Bilder pro Trigger](https://www.baumer.com/in/en/service-support/technical-information-industrial-cameras/capture-3-images-with-one-trigger/a/baumer-gapi-sequencer-vcxg-multiple-acquisitions-on-one-incoming-trigger).

Nach dem Start wartet die Erfassung zuerst auf Bewegung. Ein Durchlauf endet
erst, wenn Bewegung stattgefunden hat, der Messbereich keine sichtbare Komponente
mehr enthält und die Nachlaufzeit ohne Bewegung abgelaufen ist. Ein liegengebliebenes
Teil beendet keinen erfolgreichen Durchlauf. Wartezeit, maximale Dauer und ein
gemeinsames RAM-Limit begrenzen alle Aufzeichnungen; Abbrüche und Überlast werden
im Manifest ausgewiesen. Die Bilddaten werden nach dem Durchlauf von einem
separaten Writer gespeichert. Währenddessen bleiben Erfassung und Vorschau aktiv.

## Daten und Kontrolle

`data/camera` und `data/demo` sind getrennt. Einstellungen und Hintergrund bleiben
lokal und sind von Git ausgeschlossen. Ein Durchlauf-Ordner enthält ein Manifest,
Originalbilder als `.npy` und eine CSV mit Zeitstempeln, Frame-IDs und Winkeln.
Im Manifest stehen Endgrund, Abbrüche, fehlende Frames, Aufnahmeeinstellungen,
Kamera-Diagnose und Messreferenz. Ordner mit laufenden Schreibvorgängen sind verborgen;
erst ein vollständig geschriebener Durchlauf erscheint in der Liste.

Der Speicherordner für echte Durchläufe kann vor dem Verbinden gewählt werden,
beispielsweise auf einem Laufwerk mit mehr Platz. Hintergrund und Einstellungen
bleiben im lokalen Datenordner. Speicherfehler werden ausdrücklich angezeigt;
ein fehlgeschlagener Clip erscheint nicht als erfolgreich gespeicherter Durchlauf.

Doppelklick auf einen Durchlauf öffnet die Kontrolle. Zeitlupe, Schieberegler und
Einzelbildschritte verwenden gespeicherte Zeitstempel. Die beim Aufnehmen gemessene
Box und Achse werden eingezeichnet. CSV und Standbilder lassen sich exportieren.

## Diagnose und Entwicklung

```powershell
./.venv/Scripts/python.exe scripts/probe_camera.py --help
./.venv/Scripts/python.exe scripts/probe_camera.py --output .diagnostics/current_camera.png
./.venv/Scripts/python.exe -m pytest --basetemp .diagnostics/pytest
./.venv/Scripts/ruff.exe check src tests scripts
```

Für die Segmentierung werden OpenCV-Konturen, `minAreaRect` und flächenbasierte
PCA verwendet; siehe [OpenCV Konturmerkmale](https://docs.opencv.org/4.11.0/dd/d49/tutorial_py_contour_features.html).
Die Tests prüfen Winkel/Mehrdeutigkeit, feste Referenz, ROI, Aufzeichnungsablauf,
Speichergrenzen und Replay sowie die GUI mit einer synthetischen Kamera.

## Stand der Hardwareprüfung am 14.10.2026

Die angeschlossene **VCXU-02C** ist direkt über den
USB-Producer geöffnet und bei 640×480 geprüft. Das liegende längliche Bauteil
auf gelochter Rutsche wurde als Ruhe-0° kalibriert; die Liveanzeige ergab danach
etwa +0,03°. Das prüft die Kalibrierung, keine unabhängige Winkelgenauigkeit.
65 echte Bilder wurden manuell aufgezeichnet, gespeichert, als Replay geöffnet
und als CSV exportiert. Diese stationäre Aufnahme prüft noch keinen bewegten Durchlauf.

Beim Rohabruf mit 1000 µs Belichtung und 200 FPS Sollwert wurden 408 Frames in
2,003 s (etwa 204 FPS) ohne Frame-ID-Lücken abgerufen. Das Bild war dabei stark
unterbelichtet. Mit etwa 20331 µs liefert die Szene ein brauchbares Bild bei rund
50 FPS. Die Einzelbildverarbeitung am aktuellen Rohbild benötigte etwa 1,7 ms
Median (ROI, volle Verarbeitungsauflösung, ohne Erfassung/Overlay/GUI).
Der Hardwarezähler wurde mit kontinuierlicher Erfassung erfolgreich konfiguriert;
alle ursprünglichen Zählereinstellungen wurden nach Trennen geprüft zurückgesetzt.
Ein leeres Hintergrundbild, echte Lichtschrankenpulse und bewegte Durchläufe mit
passender Beleuchtung stehen noch aus.
