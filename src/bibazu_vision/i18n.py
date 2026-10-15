"""German source text and English presentation; acquisition data stays unchanged."""

from __future__ import annotations

import re
from functools import cache
from string import Formatter

ENGLISH = {
    "BiBaZu · Kamera & Winkelmessung": "BiBaZu · Camera & angle measurement",
    "BiBaZu  /  Bildverarbeitung": "BiBaZu  /  Image processing",
    "Kamera verbinden oder Demo starten": "Connect the camera or start the demo",
    "Bauteil: noch kein Bild": "Part: no image yet",
    "Kontur, OBB und Messachse anzeigen": "Show contour, OBB and measurement axis",
    "Segmentierung: noch kein Bild": "Segmentation: no image yet",
    "Winkelverlauf der Vorschau · ±90°": "Preview angle history · ±90°",
    "Live & Erkennung": "Live & detection",
    "Durchlauf · Zeitlupe": "Passage · slow motion",
    "Gespeicherten Durchlauf rechts auswählen.": "Select a saved passage on the right.",
    "◀ Bild": "◀ Frame",
    "Bild ▶": "Frame ▶",
    "▶ Abspielen": "▶ Play",
    "Ⅱ Pause": "Ⅱ Pause",
    "CSV exportieren": "Export CSV",
    "Standbild speichern": "Save still image",
    "Bereit · Kamera noch nicht verbunden": "Ready · camera not connected yet",
    "1 · Kamera": "1 · Camera",
    "CTI auswählen": "Select CTI",
    "Seriennummer (optional)": "Serial number (optional)",
    "Belichtungszeit": "Exposure time",
    "Gewünschte Bildrate": "Requested frame rate",
    "Kameraeinstellungen anwenden": "Apply camera settings",
    "Kameraeinstellungen werden angewendet; Versuch später starten.": (
        "Camera settings are being applied; start the trial later."
    ),
    (
        "Belichtung / Bildrate beim Verbinden oder mit Anwenden setzen. "
        "Aufzeichnung vorher beenden. Andere Kameraeinstellungen benötigen erneutes "
        "Verbinden. Tatsächliche FPS siehe Liveanzeige."
    ): (
        "Set exposure and frame rate when connecting or with Apply. Stop recording first. "
        "Other camera settings require reconnecting. See the live display for actual FPS."
    ),
    (
        "Kameraeinstellungen angewendet: {exposure:g} µs · Bildrate-Sollwert {fps:g} FPS."
    ): "Camera settings applied: {exposure:g} µs · Requested frame rate {fps:g} FPS.",
    "Kameraeinstellungen nicht angewendet: {error}": "Camera settings not applied: {error}",
    (
        "Nach Belichtungsänderung vorhandenen Hintergrund bei leerer Rutsche neu aufnehmen."
    ): "After changing exposure, recapture any existing background with the chute empty.",
    (
        "Kameraeinstellungen nur bei verbundener Kamera und beendeter Aufzeichnung "
        "ohne AutoTune-Versuch anwenden."
    ): (
        "Apply camera settings only with the camera connected, recording stopped "
        "and no AutoTune trial."
    ),
    (
        "Belichtung und Bildrate müssen positiv und endlich sein."
    ): "Exposure and frame rate must be positive and finite.",
    "Kameraeinstellungen fehlgeschlagen: {error}; Wiederherstellung fehlgeschlagen: {recovery}": (
        "Camera settings failed: {error}; Recovery failed: {recovery}"
    ),
    "Kameramerkmal {name} ist nicht lesbar.": "Camera feature {name} is not readable.",
    "Live + Eingang beobachten": "Live + monitor input",
    "Hardware startet Bildfolge": "Hardware starts image sequence",
    "Hardware: Einzelbild pro Impuls": "Hardware: one frame per pulse",
    "Kamerabetrieb": "Camera mode",
    "Triggereingang": "Trigger input",
    "Triggerflanke": "Trigger edge",
    "Kamera verbinden": "Connect camera",
    "Trennen": "Disconnect",
    "Demo starten": "Start demo",
    ("Belichtung / Bildrate werden beim Verbinden gesetzt. Tatsächliche FPS siehe Liveanzeige."): (
        "Exposure and frame rate are applied when connecting. See the live display for actual FPS."
    ),
    "2 · Erkennung & Referenz": "2 · Detection & reference",
    "Schwelle (0–255)": "Threshold (0–255)",
    "Mindestfläche": "Minimum area",
    "Dunkles Bauteil": "Dark part",
    "Helles Bauteil": "Bright part",
    "Ohne Hintergrund": "Without background",
    "OBB · orientiertes Rechteck": "OBB · oriented rectangle",
    "PCA · Hauptachse": "PCA · principal axis",
    "Messachse": "Measurement axis",
    "Volle Auflösung": "Full resolution",
    "½ Auflösung": "½ resolution",
    "¼ Auflösung": "¼ resolution",
    "Verarbeitung": "Processing",
    "Rutschenwinkel im Bild": "Chute angle in image",
    "0°-Offset: 0,00°": "0° offset: 0.00°",
    "0°-Offset: {offset:.2f}°": "0° offset: {offset:.2f}°",
    "Leere Rutsche als Hintergrund aufnehmen": "Capture empty chute as background",
    "Hintergrund entfernen": "Clear background",
    "Rutsche markieren · zwei Punkte": "Mark chute · two points",
    "Messbereich wählen · zwei Ecken": "Select ROI · two corners",
    "Gesamtes Bild als Messbereich": "Use entire image as ROI",
    "Liegendes Bauteil = 0°": "Resting part = 0°",
    "0°-Offset zurücksetzen": "Reset 0° offset",
    (
        "Mit Hintergrund: Differenzschwelle. Ohne Hintergrund: absolute "
        "Helligkeit. Positiv = gegen Uhrzeigersinn. Die 0°-Referenz gilt "
        "für das aktuelle Bauteil; runde/quadratische Formen liefern "
        "keinen eindeutigen Achswinkel."
    ): (
        "With a background: difference threshold. Without a background: "
        "absolute brightness. Positive = counterclockwise. The 0° "
        "reference applies to the current part; round/square shapes do not "
        "have an unambiguous axis angle."
    ),
    "3 · Durchlauf aufzeichnen": "3 · Record passage",
    "Bewegung": "Motion",
    "Lichtschranke": "Light barrier",
    "Manuell": "Manual",
    "Nachlauf ohne Bewegung": "Quiet time after motion",
    "Maximale Dauer": "Maximum duration",
    "RAM-Limit gesamt": "Total RAM limit",
    "Bewegungsschwelle im ROI": "Motion threshold in ROI",
    "Speicherordner": "Output folder",
    "Speicherordner auswählen": "Select output folder",
    "Aufzeichnung scharf schalten": "Arm recording",
    "Manuell auslösen": "Trigger manually",
    "Aufzeichnung beenden": "Stop recording",
    (
        "Ende nach erkannter Bewegung, ruhigem Bild und leerem "
        "Messbereich. Dauer- und RAM-Limit begrenzen jeden Durchlauf. Im "
        "Livebetrieb wird ein freier Hardwarezähler für die Lichtschranke "
        "verwendet. Ein Rückfall auf Eingangspolling wird im Protokoll "
        "angezeigt; dabei können kurze Impulse verloren gehen."
    ): (
        "Ends after detected motion, a quiet image and an empty ROI. "
        "Duration and RAM limits bound each passage. Live mode uses an "
        "unused hardware counter for the light barrier. Fallback to input "
        "polling is reported in the log; short pulses may be missed."
    ),
    "4 · Gespeicherte Durchläufe": "4 · Saved passages",
    "Liste aktualisieren": "Refresh list",
    "Durchlauf-Ordner öffnen": "Open passage folder",
    "GenTL-Producer": "GenTL producer",
    "Speicherordner für Durchläufe": "Output folder for passages",
    "Bitte die laufende Kamera zuerst trennen.": "Please disconnect the current camera first.",
    "Bitte einen Speicherordner auswählen.": "Please select an output folder.",
    "Verbinde Demo …": "Connecting demo …",
    "Verbinde Baumer-USB-Kamera …": "Connecting Baumer USB camera …",
    "Kamerafehler · {error}": "Camera error · {error}",
    "Aufzeichnung fehlgeschlagen · {error}": "Recording failed · {error}",
    "Aufzeichnung fehlgeschlagen: {error}": "Recording failed: {error}",
    "Kamera getrennt · Aufzeichnung abgeschlossen.": "Camera disconnected · recording finished.",
    "Kamera wird getrennt; Aufzeichnung wird gespeichert …": (
        "Disconnecting camera; saving recording …"
    ),
    "Zuerst Kamera oder Demo verbinden.": "Connect the camera or demo first.",
    "Zwei Punkte auf der Rutschenkante anklicken.": "Click two points on the chute edge.",
    "Zwei gegenüberliegende Ecken des Messbereichs anklicken.": (
        "Click two opposite corners of the ROI."
    ),
    "Punkte liegen zu nah beieinander.": "The points are too close together.",
    "Rutsche: {angle:.2f}° im Bild. Nullreferenz zurückgesetzt.": (
        "Chute: {angle:.2f}° in the image. Zero reference reset."
    ),
    "Messbereich muss mindestens 5 × 5 Pixel groß sein.": "ROI must be at least 5 × 5 pixels.",
    "Messbereich: {roi}": "ROI: {roi}",
    ("Aufzeichnung: {state} · {reason} · {frames} Bilder · {mib:.1f} MiB{error}{stale}"): (
        "Recording: {state} · {reason} · {frames} frames · {mib:.1f} MiB{error}{stale}"
    ),
    " · Kein neues Bild seit {seconds:.1f}s": " · No new frame for {seconds:.1f}s",
    "Durchlauf gespeichert: {path}": "Passage saved: {path}",
    "Kein Bauteil": "No part",
    "Qualität {confidence:.0%}{border}": "Quality {confidence:.0%}{border}",
    " · Randkontakt": " · Border contact",
    " · {error}": " · {error}",
    "{fps:.0f} FPS · {ms:.1f} ms\nID {frame_id} · ID-Lücken {gaps}": (
        "{fps:.0f} FPS · {ms:.1f} ms\nID {frame_id} · ID gaps {gaps}"
    ),
    "Segmentierung: fester Hintergrundabzug": "Segmentation: fixed background subtraction",
    "Segmentierung: absolute Helligkeitsschwelle · kein Hintergrund": (
        "Segmentation: absolute brightness threshold · no background"
    ),
    "Gespeicherten Durchlauf auswählen": "Select a saved passage",
    "Durchlauf enthält keine Bilder.": "Passage contains no frames.",
    "Durchlauf konnte nicht geladen werden: {error}": "Could not load passage: {error}",
    ("Bild {index}/{count} · t = {milliseconds:.1f} ms · Winkel {angle} · Ende: {reason}"): (
        "Frame {index}/{count} · t = {milliseconds:.1f} ms · Angle {angle} · End: {reason}"
    ),
    "Messwerte exportieren": "Export measurements",
    "CSV gespeichert: {path}": "CSV saved: {path}",
    "CSV-Export fehlgeschlagen: {error}": "CSV export failed: {error}",
    "Standbild mit Merkmalen": "Still image with features",
    "Standbild gespeichert: {path}": "Still image saved: {path}",
    "Standbild konnte nicht gespeichert werden.": "Could not save still image.",
    "Einstellungen konnten nicht gespeichert werden: {error}": "Could not save settings: {error}",
    "Sprache konnte nicht gespeichert werden: {error}": "Could not save language: {error}",
    ("Aufzeichnung wird noch gespeichert. Fenster danach erneut schließen."): (
        "Recording is still being saved. Close the window again afterwards."
    ),
    "Zu viele Einstellungen: kurz warten und erneut anwenden.": (
        "Too many settings: wait briefly and apply again."
    ),
    "Verbunden: {model} · Triggersignal: {monitor}": (
        "Connected: {model} · Trigger signal: {monitor}"
    ),
    "Messreferenz erst nach dem laufenden Durchlauf ändern.": (
        "Change the measurement reference after the current passage."
    ),
    "Hintergrund konnte nicht gespeichert werden.": "Could not save background.",
    "Hintergrund aufgenommen. Nur bei leerer Rutsche verwenden.": (
        "Background captured. Only use this with an empty chute."
    ),
    "Hintergrund entfernt; absolute Helligkeitsschwelle aktiv.": (
        "Background cleared; absolute brightness threshold active."
    ),
    "Liegendes Bauteil als 0° gesetzt; gilt für diese Geometrie.": (
        "Resting part set to 0°; applies to this geometry."
    ),
    "Keine eindeutige Bauteilachse: 0° konnte nicht gesetzt werden.": (
        "No unambiguous part axis: could not set 0°."
    ),
    "Aufzeichnungseinstellungen: {error}": "Recording settings: {error}",
    "Kamera schließen: {error}": "Closing camera: {error}",
    "Aufzeichnung abschließen: {error}": "Finishing recording: {error}",
    ("Gepacktes Pixelformat {fmt} wird nicht unterstützt; Mono8/Bayer8 auswählen."): (
        "Packed pixel format {fmt} is not supported; select Mono8/Bayer8."
    ),
    "Unerwartete Puffergröße für {fmt}: {size}.": "Unexpected buffer size for {fmt}: {size}.",
    ("Nicht unterstütztes Pixelformat: {fmt}; Mono8 oder Bayer8 auswählen."): (
        "Unsupported pixel format: {fmt}; select Mono8 or Bayer8."
    ),
    "Kameramerkmal {name} ist nicht schreibbar.": "Camera feature {name} is not writable.",
    "Unbekannter Kameramodus: {mode}": "Unknown camera mode: {mode}",
    "Unbekannte Eingangsüberwachung: {mode}": "Unknown input monitor: {mode}",
    ("Kein Baumer GenTL-Producer (.cti) gefunden. Baumer Camera Explorer/SDK installieren."): (
        "No Baumer GenTL producer (.cti) found. Install Baumer Camera Explorer/SDK."
    ),
    ("Keine eindeutige USB-Kamera gefunden. CTI-Pfad und Seriennummer prüfen."): (
        "Could not identify a unique USB camera. Check the CTI path and serial number."
    ),
    ("{name} ist weiterhin {mode}; Helligkeit/Farbe können die Konturerkennung verändern."): (
        "{name} is still {mode}; brightness/color may affect contour detection."
    ),
    (
        "{requested} wird nicht unterstützt (verfügbar: {supported}). "
        "FrameStart erzeugt nur ein Bild je Puls; für eine Folge Live + "
        "Lichtschranken-Erfassung verwenden."
    ): (
        "{requested} is not supported (available: {supported}). FrameStart "
        "produces only one image per pulse; use live mode with "
        "light-barrier recording for a sequence."
    ),
    "Trigger-Eingang {line} nicht verfügbar: {sources}.": (
        "Trigger input {line} is not available: {sources}."
    ),
    "Kein sicher nutzbarer Hardware-Zähler für den Eingang.": (
        "No safely usable hardware counter for the input."
    ),
    (" Baumer Camera Explorer oder andere Kameraprogramme schließen und erneut verbinden."): (
        " Close Baumer Camera Explorer or other camera applications and reconnect."
    ),
    (
        "Hardware-Zähler erfasst Eingangsflanken zwischen Frames. "
        "Verdrahtung, Polarität und reale kurze Lichtschranken-Pulse "
        "prüfen."
    ): (
        "Hardware counter captures input edges between frames. "
        "Check wiring, polarity and actual short light-barrier "
        "pulses."
    ),
    "Hardware-Zähler nicht nutzbar: {error}": "Hardware counter unavailable: {error}",
    "Zähler konnte nicht restauriert werden: {error}": "Could not restore counter: {error}",
    (
        "LineStatus wird zwischen Frames abgefragt. Sehr kurze "
        "Lichtschranken-Pulse können fehlen; Aufnahmen auf realen Pulsen "
        "prüfen."
    ): (
        "LineStatus is polled between frames. Very short "
        "light-barrier pulses may be missed; check recording with "
        "actual pulses."
    ),
    ("Lichtschranken-Eingang {line} nicht lesbar; verfügbare Linien: {lines}."): (
        "Light-barrier input {line} is not readable; available lines: {lines}."
    ),
    "Kamera ist nicht geöffnet.": "Camera is not open.",
    ("Trigger-Einstellungen konnten nicht wiederhergestellt werden: {error}"): (
        "Could not restore trigger settings: {error}"
    ),
    "{name} konnte nicht wiederhergestellt werden: {error}": "Could not restore {name}: {error}",
    "keine": "none",
    "AutoTune-Versuch aktiv; Einstellungen sind gesperrt.": (
        "AutoTune trial active; settings are locked."
    ),
    "Kamera-API: {url}": "Camera API: {url}",
    "Kamera-API konnte nicht starten: {error}": "Could not start camera API: {error}",
}

_HUMAN_FIELDS = {"error", "stale", "border", "supported"}


def translate(source: str, language: str, **values) -> str:
    text = ENGLISH.get(source, source) if language == "en" else source
    if not values:
        return text
    localized = {
        name: translate_message(value, language)
        if name in _HUMAN_FIELDS and isinstance(value, str)
        else value
        for name, value in values.items()
    }
    return text.format(**localized)


@cache
def _message_patterns():
    patterns = []
    for source, target in ENGLISH.items():
        if "{" not in source:
            continue
        expression, fields = "", []
        for literal, name, _, _ in Formatter().parse(source):
            expression += re.escape(literal)
            if name is not None:
                expression += f"(?P<p{len(fields)}>.*?)"
                fields.append(name)
        patterns.append((re.compile(expression, re.DOTALL), fields, target))
    return patterns


def translate_message(source: str, language: str) -> str:
    """Translate formatted worker messages while retaining opaque paths and identifiers."""
    if language != "en" or not source:
        return source
    if source in ENGLISH:
        return ENGLISH[source]
    exception = re.fullmatch(r"([\w.]+(?:Error|Exception)): (.*)", source, re.DOTALL)
    if exception:
        return exception[1] + ": " + translate_message(exception[2], language)
    for pattern, fields, target in _message_patterns():
        match = pattern.fullmatch(source)
        if match is None:
            continue
        values = {name: match[f"p{i}"] for i, name in enumerate(fields)}
        result = ""
        for literal, name, _, _ in Formatter().parse(target):
            result += literal
            if name is not None:
                value = values[name]
                result += translate_message(value, language) if name in _HUMAN_FIELDS else value
        return result
    # GenTL may append our connection advice to an opaque SDK error.
    advice = " Baumer Camera Explorer oder andere Kameraprogramme schließen und erneut verbinden."
    if source.endswith(advice):
        return source[: -len(advice)] + ENGLISH[advice]
    return source
