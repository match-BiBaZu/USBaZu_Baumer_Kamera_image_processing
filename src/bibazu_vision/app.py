from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QLibraryInfo,
    QLocale,
    QPointF,
    QSignalBlocker,
    Qt,
    QTimer,
    QTranslator,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .api import VisionApi
from .camera import CameraConfig
from .i18n import translate, translate_message
from .processing import ProcessingConfig, render_overlay
from .recording import CaptureConfig, load_clip
from .worker import PipelineWorker

ROOT = Path(__file__).resolve().parents[2]


class ImageView(QLabel):
    selected = pyqtSignal(str, object)

    def __init__(self):
        super().__init__("Kamera verbinden oder Demo starten")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 280)
        self.setStyleSheet("background:#0b1220;color:#94a3b8;border-radius:8px")
        self.source = None
        self.mode = None
        self.first = None
        self.points = None

    def set_image(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.source = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.refresh()

    def refresh(self):
        if self.source is not None:
            pixmap = QPixmap.fromImage(self.source).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()

    def image_point(self, position):
        if self.source is None or self.pixmap() is None:
            return None
        pixmap = self.pixmap()
        left = (self.width() - pixmap.width()) / 2
        top = (self.height() - pixmap.height()) / 2
        x = (position.x() - left) * self.source.width() / pixmap.width()
        y = (position.y() - top) * self.source.height() / pixmap.height()
        if 0 <= x < self.source.width() and 0 <= y < self.source.height():
            return (int(x), int(y))
        return None

    def mousePressEvent(self, event):
        point = self.image_point(event.position())
        if not self.mode or point is None:
            return
        if self.first is None:
            self.first = point
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            mode = self.mode
            self.points = (self.first, point)
            self.first = None
            self.mode = None
            self.unsetCursor()
            self.selected.emit(mode, self.points)

    def select(self, mode):
        self.mode = mode
        self.first = None
        self.setCursor(Qt.CursorShape.CrossCursor)


class AngleTrace(QWidget):
    def __init__(self):
        super().__init__()
        self.values = []
        self.caption = "Winkelverlauf der Vorschau · ±90°"
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)

    def add(self, angle):
        self.values.append(angle)
        self.values = self.values[-250:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111c30"))
        painter.setPen(QColor("#64748b"))
        painter.drawText(8, 16, self.caption)
        middle = self.height() / 2 + 8
        painter.drawLine(0, int(middle), self.width(), int(middle))
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        previous = None
        for i, value in enumerate(self.values):
            if value is None:
                previous = None
                continue
            point = QPointF(
                i * self.width() / max(1, len(self.values) - 1),
                middle - value / 90 * (self.height() / 2 - 18),
            )
            if previous is not None and abs(value - self.values[i - 1]) < 90:
                painter.drawLine(previous, point)
            previous = point


def spin(value, minimum, maximum, decimals=1, suffix=""):
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    return widget


class MainWindow(QMainWindow):
    def __init__(self, data_dir=None, demo=False, language=None):
        super().__init__()
        self.setWindowTitle("BiBaZu · Kamera & Winkelmessung")
        self.resize(1420, 920)
        self.data_dir = Path(data_dir or ROOT / "data")
        self.profile = {}
        profile_path = self.data_dir / "settings.json"
        if profile_path.exists():
            try:
                self.profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        preferred = language if language is not None else self.profile.get("language", "de")
        self.language = preferred if preferred in {"de", "en"} else "de"
        self._text_bindings = {}
        self._log_sources = []
        self._last_note = None
        self._qt_translator = QTranslator(self)
        self.worker = None
        self.camera_settings_pending = False
        self.last_snapshot = None
        self.last_frame = None
        self.clip = None
        self.review_index = 0
        self.review_position = 0.0
        self.demo_requested = demo
        self.last_saved = None
        self.pipeline_error = None
        self.recording_error_text = None
        self.build_ui()
        self._register_texts()
        self.set_language(self.language, persist=language is not None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_live)
        self.timer.start(33)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_review)
        self.play_timer.setInterval(16)
        self.refresh_recordings()
        self.control_api = VisionApi(self)
        self.control_api.start()
        if demo:
            QTimer.singleShot(0, lambda: self.connect_camera(True))

    def button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def t(self, source, **values):
        return translate(source, self.language, **values)

    def ui_text(self, widget, source, **values):
        self._text_bindings[widget] = (source, values)
        widget.setText(self.t(source, **values))

    def _register_texts(self):
        for kind in (QLabel, QPushButton, QCheckBox):
            for widget in self.findChildren(kind):
                if widget is not self.language_button:
                    self._text_bindings[widget] = (widget.text(), {})
        self._group_texts = [(widget, widget.title()) for widget in self.findChildren(QGroupBox)]
        self._combo_texts = [
            (widget, [widget.itemText(i) for i in range(widget.count())])
            for widget in self.findChildren(QComboBox)
        ]
        self._tab_texts = [self.tabs.tabText(i) for i in range(self.tabs.count())]

    def toggle_language(self):
        self.set_language("en" if self.language == "de" else "de")

    def set_language(self, language, persist=True):
        if language not in {"de", "en"}:
            raise ValueError("language must be 'de' or 'en'")
        self.language = language
        application = QApplication.instance()
        application.removeTranslator(self._qt_translator)
        if language == "de" and self._qt_translator.load(
            "qtbase_de", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        ):
            application.installTranslator(self._qt_translator)
        locale = QLocale("en_US" if language == "en" else "de_DE")
        blockers = [QSignalBlocker(widget) for widget in self.findChildren(QDoubleSpinBox)]
        self.setLocale(locale)
        del blockers
        self.setWindowTitle(self.t("BiBaZu · Kamera & Winkelmessung"))
        for widget, (source, values) in self._text_bindings.items():
            if isinstance(widget, ImageView) and widget.source is not None:
                continue
            widget.setText(self.t(source, **values))
        for widget, source in self._group_texts:
            widget.setTitle(self.t(source))
        for widget, sources in self._combo_texts:
            with QSignalBlocker(widget):
                for index, source in enumerate(sources):
                    widget.setItemText(index, self.t(source))
        for index, source in enumerate(self._tab_texts):
            self.tabs.setTabText(index, self.t(source))
        self.trace.caption = self.t("Winkelverlauf der Vorschau · ±90°")
        self.trace.update()
        self.language_button.setText("Deutsch" if language == "en" else "English")
        self.language_button.setToolTip(
            "Switch to German" if language == "en" else "Auf Englisch umschalten"
        )
        scrollbar = self.log.verticalScrollBar()
        previous = scrollbar.value()
        at_bottom = previous == scrollbar.maximum()
        self.log.setPlainText(
            "\n".join(translate_message(source, language) for source in self._log_sources)
        )
        scrollbar.setValue(scrollbar.maximum() if at_bottom else previous)
        if self._last_note:
            self.statusBar().showMessage(translate_message(self._last_note, language), 10000)
        if persist:
            self.persist_language()

    def persist_language(self):
        # Write only the preference, including in demo mode. Do not replace the
        # native camera calibration with the current synthetic scene.
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            target = self.data_dir / "settings.json"
            profile = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
            profile["language"] = self.language
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            temporary.replace(target)
            self.profile["language"] = self.language
        except (OSError, ValueError) as exc:
            self.note(f"Sprache konnte nicht gespeichert werden: {exc}")

    def build_ui(self):
        self.setStyleSheet("""
            QWidget {background:#101827;color:#e2e8f0;font-size:13px;}
            QGroupBox {border:1px solid #334155;border-radius:7px;margin-top:15px;padding:10px;}
            QGroupBox::title {subcontrol-origin:margin;left:12px;color:#7dd3fc;}
            QPushButton {background:#24354d;border:1px solid #425774;border-radius:5px;padding:8px;}
            QPushButton:hover {background:#334c6c;} QPushButton:disabled {color:#64748b;}
            QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QListWidget,QTextEdit {
                background:#172338;border:1px solid #334155;border-radius:4px;padding:4px;}
            QTabBar::tab {background:#24354d;padding:10px 20px;}
            QTabBar::tab:selected {background:#0369a1;}
        """)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        left = QVBoxLayout()
        title = QLabel("BiBaZu  /  Bildverarbeitung")
        title.setStyleSheet("font-size:23px;font-weight:600;color:#f8fafc;padding:6px")
        heading = QHBoxLayout()
        heading.addWidget(title, 1)
        self.language_button = self.button("English", self.toggle_language)
        self.language_button.setObjectName("language_toggle")
        heading.addWidget(self.language_button)
        left.addLayout(heading)
        indicators = QHBoxLayout()
        self.angle_label = QLabel("— °")
        self.angle_label.setMinimumWidth(260)
        self.angle_label.setStyleSheet("font-size:40px;font-weight:600;color:#38bdf8")
        self.quality_label = QLabel("Bauteil: noch kein Bild")
        self.rate_label = QLabel("0 FPS")
        indicators.addWidget(self.angle_label)
        indicators.addStretch()
        indicators.addWidget(self.quality_label)
        indicators.addWidget(self.rate_label)
        left.addLayout(indicators)
        self.tabs = QTabWidget()
        live = QWidget()
        live_layout = QVBoxLayout(live)
        self.live_view = ImageView()
        self.live_view.selected.connect(self.image_selection)
        live_layout.addWidget(self.live_view, 1)
        self.overlay_check = QCheckBox("Kontur, OBB und Messachse anzeigen")
        self.overlay_check.setChecked(True)
        live_layout.addWidget(self.overlay_check)
        self.segmentation_label = QLabel("Segmentierung: noch kein Bild")
        live_layout.addWidget(self.segmentation_label)
        self.trace = AngleTrace()
        live_layout.addWidget(self.trace)
        self.tabs.addTab(live, "Live & Erkennung")
        review = QWidget()
        review_layout = QVBoxLayout(review)
        self.review_view = ImageView()
        review_layout.addWidget(self.review_view, 1)
        self.review_info = QLabel("Gespeicherten Durchlauf rechts auswählen.")
        self.review_info.setWordWrap(True)
        review_layout.addWidget(self.review_info)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_review)
        review_layout.addWidget(self.slider)
        playback = QHBoxLayout()
        playback.addWidget(
            self.button("◀ Bild", lambda: self.slider.setValue(self.slider.value() - 1))
        )
        self.play_button = self.button("▶ Abspielen", self.toggle_play)
        playback.addWidget(self.play_button)
        playback.addWidget(
            self.button("Bild ▶", lambda: self.slider.setValue(self.slider.value() + 1))
        )
        self.speed = QComboBox()
        for label, value in [("1×", 1), ("½×", 0.5), ("¼×", 0.25), ("¹⁄₁₀×", 0.1), ("¹⁄₂₀×", 0.05)]:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(3)
        playback.addWidget(self.speed)
        playback.addWidget(self.button("CSV exportieren", self.export_csv))
        playback.addWidget(self.button("Standbild speichern", self.export_still))
        review_layout.addLayout(playback)
        self.tabs.addTab(review, "Durchlauf · Zeitlupe")
        left.addWidget(self.tabs, 1)
        self.state_label = QLabel("Bereit · Kamera noch nicht verbunden")
        self.state_label.setWordWrap(True)
        left.addWidget(self.state_label)
        layout.addLayout(left, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(460)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        scroll.setWidget(panel)
        layout.addWidget(scroll)

        group = QGroupBox("1 · Kamera")
        form = QFormLayout(group)
        default_cti = r"C:\Program Files\Baumer Camera Explorer\bgapi2_usb.cti"
        self.cti = QLineEdit(self.profile.get("cti", default_cti))
        form.addRow("USB GenTL (.cti)", self.cti)
        form.addRow(self.button("CTI auswählen", self.choose_cti))
        self.serial = QLineEdit(self.profile.get("serial", ""))
        form.addRow("Seriennummer (optional)", self.serial)
        self.exposure = spin(self.profile.get("exposure", 1000), 10, 1000000, 0, " µs")
        self.fps = spin(self.profile.get("fps", 200), 1, 2000, 1, " FPS")
        form.addRow("Belichtungszeit", self.exposure)
        form.addRow("Gewünschte Bildrate", self.fps)
        self.apply_camera_button = self.button(
            "Kameraeinstellungen anwenden", self.apply_camera_settings
        )
        self.apply_camera_button.setEnabled(False)
        form.addRow(self.apply_camera_button)
        self.camera_mode = QComboBox()
        for text, mode in [
            ("Live + Eingang beobachten", "live"),
            ("Hardware startet Bildfolge", "acquisition_trigger"),
            ("Hardware: Einzelbild pro Impuls", "frame_trigger"),
        ]:
            self.camera_mode.addItem(text, mode)
        camera_profile = self.profile.get("camera", {})
        self.camera_mode.setCurrentIndex(
            max(0, self.camera_mode.findData(camera_profile.get("mode", "live")))
        )
        form.addRow("Kamerabetrieb", self.camera_mode)
        self.line = QComboBox()
        self.line.addItems(["Line0", "Line1", "Line2", "Line3", "Line4"])
        self.line.setCurrentText(camera_profile.get("trigger_line", "Line0"))
        form.addRow("Triggereingang", self.line)
        self.edge = QComboBox()
        self.edge.addItems(["RisingEdge", "FallingEdge"])
        self.edge.setCurrentText(camera_profile.get("trigger_activation", "RisingEdge"))
        form.addRow("Triggerflanke", self.edge)
        row = QHBoxLayout()
        self.connect_button = self.button("Kamera verbinden", lambda: self.connect_camera(False))
        row.addWidget(self.connect_button)
        row.addWidget(self.button("Trennen", self.disconnect_camera))
        form.addRow(row)
        form.addRow(self.button("Demo starten", lambda: self.connect_camera(True)))
        hint = QLabel(
            "Belichtung / Bildrate beim Verbinden oder mit Anwenden setzen. "
            "Aufzeichnung vorher beenden. Andere Kameraeinstellungen benötigen erneutes "
            "Verbinden. Tatsächliche FPS siehe Liveanzeige."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        panel_layout.addWidget(group)

        group = QGroupBox("2 · Erkennung & Referenz")
        self.processing_group = group
        form = QFormLayout(group)
        self.threshold = QSpinBox()
        self.threshold.setRange(1, 254)
        self.threshold.setValue(self.profile.get("processing", {}).get("threshold", 25))
        form.addRow("Schwelle (0–255)", self.threshold)
        self.area = spin(
            self.profile.get("processing", {}).get("min_area", 150), 1, 10000000, 0, " px²"
        )
        form.addRow("Mindestfläche", self.area)
        self.foreground = QComboBox()
        self.foreground.addItem("Dunkles Bauteil", "dark")
        self.foreground.addItem("Helles Bauteil", "light")
        self.foreground.setCurrentIndex(
            max(
                0,
                self.foreground.findData(
                    self.profile.get("processing", {}).get("foreground", "dark")
                ),
            )
        )
        form.addRow("Ohne Hintergrund", self.foreground)
        self.method = QComboBox()
        self.method.addItem("OBB · orientiertes Rechteck", "obb")
        self.method.addItem("PCA · Hauptachse", "pca")
        self.method.setCurrentIndex(
            max(0, self.method.findData(self.profile.get("processing", {}).get("method", "obb")))
        )
        form.addRow("Messachse", self.method)
        self.scale = QComboBox()
        for label, value in [("Volle Auflösung", 1.0), ("½ Auflösung", 0.5), ("¼ Auflösung", 0.25)]:
            self.scale.addItem(label, value)
        self.scale.setCurrentIndex(
            max(0, self.scale.findData(self.profile.get("processing", {}).get("scale", 1.0)))
        )
        form.addRow("Verarbeitung", self.scale)
        self.chute_angle = spin(
            self.profile.get("processing", {}).get("chute_angle_deg", 0), -180, 180, 2, " °"
        )
        form.addRow("Rutschenwinkel im Bild", self.chute_angle)
        self.reference_label = QLabel("0°-Offset: 0,00°")
        form.addRow(self.reference_label)
        self.threshold.valueChanged.connect(self.apply_processing)
        self.area.valueChanged.connect(self.apply_processing)
        self.foreground.currentIndexChanged.connect(self.apply_processing)
        self.method.currentIndexChanged.connect(self.apply_processing)
        self.scale.currentIndexChanged.connect(self.apply_processing)
        self.chute_angle.valueChanged.connect(self.apply_processing)
        form.addRow(
            self.button(
                "Leere Rutsche als Hintergrund aufnehmen", lambda: self.command("background")
            )
        )
        form.addRow(self.button("Hintergrund entfernen", lambda: self.command("clear_background")))
        form.addRow(
            self.button("Rutsche markieren · zwei Punkte", lambda: self.select_image("chute"))
        )
        form.addRow(
            self.button("Messbereich wählen · zwei Ecken", lambda: self.select_image("roi"))
        )
        form.addRow(
            self.button(
                "Gesamtes Bild als Messbereich", lambda: self.command("config", {"roi": None})
            )
        )
        form.addRow(self.button("Liegendes Bauteil = 0°", lambda: self.command("zero")))
        form.addRow(
            self.button(
                "0°-Offset zurücksetzen", lambda: self.command("config", {"zero_offset_deg": 0.0})
            )
        )
        hint = QLabel(
            "Mit Hintergrund: Differenzschwelle. Ohne Hintergrund: absolute Helligkeit. "
            "Positiv = gegen Uhrzeigersinn. Die 0°-Referenz gilt für das aktuelle Bauteil; "
            "runde/quadratische Formen liefern keinen eindeutigen Achswinkel."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        panel_layout.addWidget(group)

        group = QGroupBox("3 · Durchlauf aufzeichnen")
        self.capture_group = group
        form = QFormLayout(group)
        self.start_mode = QComboBox()
        for text, mode in [
            ("Bewegung", "motion"),
            ("Lichtschranke", "hardware"),
            ("Manuell", "manual"),
        ]:
            self.start_mode.addItem(text, mode)
        self.start_mode.setCurrentIndex(
            max(
                0,
                self.start_mode.findData(
                    self.profile.get("capture", {}).get("start_mode", "hardware")
                ),
            )
        )
        form.addRow("Start", self.start_mode)
        capture_profile = self.profile.get("capture", {})
        self.quiet = spin(capture_profile.get("quiet_seconds", 0.15), 0.02, 5, 2, " s")
        self.max_duration = spin(capture_profile.get("max_seconds", 5), 0.1, 60, 1, " s")
        self.memory = spin(capture_profile.get("max_memory_mb", 256), 16, 4096, 0, " MB")
        self.motion_percent = spin(
            capture_profile.get("motion_threshold", 0.001) * 100, 0.001, 100, 3, " %"
        )
        form.addRow("Nachlauf ohne Bewegung", self.quiet)
        form.addRow("Maximale Dauer", self.max_duration)
        form.addRow("RAM-Limit gesamt", self.memory)
        form.addRow("Bewegungsschwelle im ROI", self.motion_percent)
        self.output_path = QLineEdit(
            self.profile.get("output_dir", str(self.data_dir / "camera" / "recordings"))
        )
        form.addRow("Speicherordner", self.output_path)
        self.output_button = self.button("Speicherordner auswählen", self.choose_output)
        form.addRow(self.output_button)
        self.arm_button = self.button("Aufzeichnung scharf schalten", self.arm)
        form.addRow(self.arm_button)
        form.addRow(self.button("Manuell auslösen", lambda: self.command("trigger")))
        form.addRow(self.button("Aufzeichnung beenden", lambda: self.command("disarm")))
        hint = QLabel(
            "Ende nach erkannter Bewegung, ruhigem Bild und leerem Messbereich. "
            "Dauer- und RAM-Limit begrenzen jeden Durchlauf. Im Livebetrieb wird ein freier "
            "Hardwarezähler für die Lichtschranke verwendet. Ein Rückfall auf Eingangspolling "
            "wird im Protokoll angezeigt; dabei können kurze Impulse verloren gehen."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        panel_layout.addWidget(group)

        group = QGroupBox("4 · Gespeicherte Durchläufe")
        vbox = QVBoxLayout(group)
        self.recordings = QListWidget()
        self.recordings.setMaximumHeight(140)
        self.recordings.itemDoubleClicked.connect(
            lambda item: self.open_clip(Path(item.data(Qt.ItemDataRole.UserRole)))
        )
        vbox.addWidget(self.recordings)
        vbox.addWidget(self.button("Liste aktualisieren", self.refresh_recordings))
        vbox.addWidget(self.button("Durchlauf-Ordner öffnen", self.choose_clip))
        panel_layout.addWidget(group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        panel_layout.addWidget(self.log)
        panel_layout.addStretch()

    def note(self, text):
        self._log_sources.append(text)
        self._last_note = text
        localized = translate_message(text, self.language)
        self.log.append(localized)
        self.statusBar().showMessage(localized, 10000)

    def choose_cti(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("GenTL-Producer"),
            self.cti.text(),
            "GenTL (*.cti)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.cti.setText(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            self.t("Speicherordner für Durchläufe"),
            self.output_path.text(),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.output_path.setText(path)
            self.refresh_recordings()

    def processing_config(self):
        previous = (
            self.last_snapshot["config"]
            if self.last_snapshot
            else self.profile.get("processing", {})
        )
        return ProcessingConfig(
            threshold=self.threshold.value(),
            min_area=self.area.value(),
            roi=previous.get("roi"),
            chute_angle_deg=self.chute_angle.value(),
            zero_offset_deg=previous.get("zero_offset_deg", 0),
            method=self.method.currentData(),
            foreground=self.foreground.currentData(),
            scale=self.scale.currentData(),
        )

    def capture_config(self):
        return CaptureConfig(
            start_mode=self.start_mode.currentData(),
            quiet_seconds=self.quiet.value(),
            max_seconds=self.max_duration.value(),
            max_memory_mb=self.memory.value(),
            motion_threshold=self.motion_percent.value() / 100,
        )

    def connect_camera(self, demo):
        if self.worker is not None and self.worker.isRunning():
            self.note("Bitte die laufende Kamera zuerst trennen.")
            return
        if not demo and not self.output_path.text().strip():
            self.note("Bitte einen Speicherordner auswählen.")
            return
        if demo:
            self.last_snapshot = None
            self.threshold.setValue(25)
            self.area.setValue(150)
            self.chute_angle.setValue(2.86)
            self.method.setCurrentIndex(0)
            self.foreground.setCurrentIndex(0)
            self.scale.setCurrentIndex(0)
            self.start_mode.setCurrentIndex(self.start_mode.findData("motion"))
        elif self.worker is not None and self.worker.demo:
            saved = self.profile.get("processing", {})
            self.last_snapshot = None
            self.threshold.setValue(saved.get("threshold", 25))
            self.area.setValue(saved.get("min_area", 150))
            self.chute_angle.setValue(saved.get("chute_angle_deg", 0))
            self.method.setCurrentIndex(max(0, self.method.findData(saved.get("method", "obb"))))
            self.foreground.setCurrentIndex(
                max(0, self.foreground.findData(saved.get("foreground", "dark")))
            )
            self.scale.setCurrentIndex(max(0, self.scale.findData(saved.get("scale", 1.0))))
            self.start_mode.setCurrentIndex(
                max(
                    0,
                    self.start_mode.findData(
                        self.profile.get("capture", {}).get("start_mode", "hardware")
                    ),
                )
            )
        self.pipeline_error = None
        self.recording_error_text = None
        camera = CameraConfig(
            cti_path=self.cti.text() or None,
            serial=self.serial.text() or None,
            exposure_us=self.exposure.value(),
            fps=self.fps.value(),
            mode=self.camera_mode.currentData(),
            trigger_line=self.line.currentText(),
            trigger_activation=self.edge.currentText(),
        )
        if not demo and camera.mode == "acquisition_trigger":
            self.start_mode.setCurrentIndex(self.start_mode.findData("hardware"))
        self.worker = PipelineWorker(
            camera,
            (ProcessingConfig(chute_angle_deg=2.86) if demo else self.processing_config()),
            self.capture_config(),
            self.data_dir / ("demo" if demo else "camera"),
            demo,
            output_dir=None if demo else Path(self.output_path.text()),
        )
        self.worker.message.connect(self.note)
        self.worker.failed.connect(self.camera_error)
        self.worker.recording_failed.connect(self.recording_error)
        self.worker.camera_settings_finished.connect(self.camera_settings_finished)
        self.worker.finished.connect(self.worker_finished)
        self.last_frame = None
        self.last_snapshot = None
        self.camera_settings_pending = False
        self.apply_camera_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.output_path.setEnabled(False)
        self.output_button.setEnabled(False)
        self.ui_text(
            self.state_label, "Verbinde Demo …" if demo else "Verbinde Baumer-USB-Kamera …"
        )
        self.worker.start()

    def worker_finished(self):
        self.camera_settings_pending = False
        self.apply_camera_button.setEnabled(False)
        self.connect_button.setEnabled(True)
        self.processing_group.setEnabled(True)
        self.output_path.setEnabled(True)
        self.output_button.setEnabled(True)
        if self.pipeline_error:
            self.ui_text(self.state_label, "Kamerafehler · {error}", error=self.pipeline_error)
        elif self.recording_error_text:
            self.ui_text(
                self.state_label,
                "Aufzeichnung fehlgeschlagen · {error}",
                error=self.recording_error_text,
            )
        else:
            self.ui_text(self.state_label, "Kamera getrennt · Aufzeichnung abgeschlossen.")
        self.refresh_recordings()
        self.save_profile()

    def camera_error(self, text):
        self.pipeline_error = text
        self.ui_text(self.state_label, "Kamerafehler · {error}", error=text)
        self.note(text)

    def recording_error(self, text):
        self.recording_error_text = text
        self.note("Aufzeichnung fehlgeschlagen: " + text)

    def disconnect_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.apply_camera_button.setEnabled(False)
            self.ui_text(self.state_label, "Kamera wird getrennt; Aufzeichnung wird gespeichert …")

    def refresh_camera_settings_button(self):
        self.apply_camera_button.setEnabled(
            self.worker is not None and self.worker.isRunning()
            and not self.camera_settings_pending and self.worker.camera_settings_available()
        )

    def apply_camera_settings(self):
        self.refresh_camera_settings_button()
        if not self.apply_camera_button.isEnabled():
            return
        self.camera_settings_pending = True
        self.apply_camera_button.setEnabled(False)
        if not self.worker.command(
            "camera_settings", {"exposure_us": self.exposure.value(), "fps": self.fps.value()}
        ):
            self.camera_settings_pending = False
            self.refresh_camera_settings_button()

    def camera_settings_finished(self, success, applied, error):
        self.camera_settings_pending = False
        if success:
            self.note(
                f"Kameraeinstellungen angewendet: {applied['exposure_us']:g} µs · "
                f"Bildrate-Sollwert {applied['fps']:g} FPS."
            )
            self.note(
                "Nach Belichtungsänderung vorhandenen Hintergrund bei leerer Rutsche "
                "neu aufnehmen."
            )
        else:
            self.note(f"Kameraeinstellungen nicht angewendet: {error}")
        self.refresh_camera_settings_button()

    def command(self, name, value=None):
        if self.worker is None or not self.worker.isRunning():
            self.note("Zuerst Kamera oder Demo verbinden.")
            return
        self.worker.command(name, value)

    def apply_processing(self, *args):
        if self.worker is not None and self.worker.isRunning():
            config = asdict(self.processing_config())
            # Calibration/ROI are owned by the worker and must not be reverted by stale previews.
            config.pop("zero_offset_deg")
            config.pop("roi")
            self.command("config", config)

    def select_image(self, mode):
        self.tabs.setCurrentIndex(0)
        self.live_view.select(mode)
        self.note(
            "Zwei Punkte auf der Rutschenkante anklicken."
            if mode == "chute"
            else "Zwei gegenüberliegende Ecken des Messbereichs anklicken."
        )

    def image_selection(self, mode, points):
        (x1, y1), (x2, y2) = points
        if math.hypot(x2 - x1, y2 - y1) < 5:
            self.note("Punkte liegen zu nah beieinander.")
            return
        if mode == "chute":
            angle = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
            angle = (angle + 90) % 180 - 90
            self.chute_angle.setValue(angle)
            self.command("config", {"chute_angle_deg": angle, "zero_offset_deg": 0.0})
            self.note(f"Rutsche: {angle:.2f}° im Bild. Nullreferenz zurückgesetzt.")
        else:
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                self.note("Messbereich muss mindestens 5 × 5 Pixel groß sein.")
                return
            roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.command("config", {"roi": roi})
            self.note(f"Messbereich: {roi}")

    def arm(self):
        if self.worker is None or not self.worker.isRunning():
            self.note("Zuerst Kamera oder Demo verbinden.")
            return
        self.command("capture_config", asdict(self.capture_config()))
        self.command("arm")

    def refresh_live(self):
        self.refresh_camera_settings_button()
        if self.worker is None or not self.worker.isRunning():
            return
        snapshot = self.worker.snapshot()
        if snapshot is None:
            return
        self.last_snapshot = snapshot
        status = snapshot["status"]
        state = getattr(status, "state", "?")
        self.processing_group.setEnabled(
            state not in {"waiting", "moving"} and not self.worker.api_capture.busy()
        )
        self.capture_group.setEnabled(not self.worker.api_capture.busy())
        reason = getattr(status, "reason", "") or ""
        error = getattr(status, "error", "")
        self.ui_text(
            self.state_label,
            "Aufzeichnung: {state} · {reason} · {frames} Bilder · {mib:.1f} MiB{error}{stale}",
            state=state,
            reason=reason,
            frames=getattr(status, "buffered_frames", 0),
            mib=getattr(status, "buffered_bytes", 0) / 1024**2,
            error=f" · {error}" if error else "",
            stale=(
                f" · Kein neues Bild seit {snapshot['stale_seconds']:.1f}s"
                if snapshot["stale_seconds"] > 1
                else ""
            ),
        )
        if snapshot["last_clip"] and snapshot["last_clip"] != self.last_saved:
            self.recording_error_text = None
            self.last_saved = snapshot["last_clip"]
            self.note(f"Durchlauf gespeichert: {self.last_saved}")
            self.refresh_recordings()
        if snapshot["frame_id"] == self.last_frame:
            return
        self.last_frame = snapshot["frame_id"]
        detection = snapshot["detection"]
        angle = detection.angle_deg
        self.ui_text(
            self.angle_label,
            "— °" if angle is None else "{angle:+.2f} °",
            **({} if angle is None else {"angle": angle}),
        )
        if detection.center is None:
            self.ui_text(self.quality_label, "Kein Bauteil")
        else:
            self.ui_text(
                self.quality_label,
                "Qualität {confidence:.0%}{border}",
                confidence=detection.confidence,
                border=" · Randkontakt" if detection.touching_border else "",
            )
        self.ui_text(
            self.rate_label,
            "{fps:.0f} FPS · {ms:.1f} ms\nID {frame_id} · ID-Lücken {gaps}",
            fps=snapshot["fps"],
            ms=detection.processing_ms,
            frame_id=snapshot["frame_id"],
            gaps=snapshot["frame_gaps"],
        )
        config = ProcessingConfig(**snapshot["config"])
        self.ui_text(
            self.segmentation_label,
            "Segmentierung: fester Hintergrundabzug"
            if snapshot.get("has_background")
            else "Segmentierung: absolute Helligkeitsschwelle · kein Hintergrund",
        )
        self.ui_text(
            self.reference_label, "0°-Offset: {offset:.2f}°", offset=config.zero_offset_deg
        )
        image = (
            render_overlay(snapshot["image"], detection, config)
            if self.overlay_check.isChecked()
            else snapshot["image"]
        )
        self.live_view.set_image(image)
        self.trace.add(angle)

    def refresh_recordings(self):
        self.recordings.clear()
        if not self.data_dir.exists():
            return
        manifests = set(self.data_dir.glob("*/recordings/*/manifest.json"))
        manifests.update(Path(self.output_path.text()).glob("*/manifest.json"))
        for manifest in sorted(manifests, reverse=True)[:100]:
            self.recordings.addItem(manifest.parent.name)
            self.recordings.item(self.recordings.count() - 1).setData(
                Qt.ItemDataRole.UserRole, str(manifest.parent)
            )

    def choose_clip(self):
        path = QFileDialog.getExistingDirectory(
            self,
            self.t("Gespeicherten Durchlauf auswählen"),
            str(self.data_dir),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.open_clip(Path(path))

    def open_clip(self, path):
        self.play_timer.stop()
        try:
            self.clip = load_clip(path)
            if not len(self.clip.frames):
                raise ValueError("Durchlauf enthält keine Bilder.")
        except Exception as exc:
            self.note(f"Durchlauf konnte nicht geladen werden: {exc}")
            return
        self.slider.setRange(0, len(self.clip.frames) - 1)
        self.slider.setValue(0)
        self.review_position = 0.0
        self.show_review(0)
        self.tabs.setCurrentIndex(1)
        self.ui_text(self.play_button, "▶ Abspielen")

    def review_image(self, index):
        image = self.clip.frames[index]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        measurement = self.clip.measurements[index]
        box = measurement.get("box")
        if box is not None:
            cv2.polylines(image, [np.asarray(box, dtype=np.int32)], True, (0, 220, 100), 2)
        center = measurement.get("center")
        axis = measurement.get("axis_deg")
        if center is not None and axis is not None:
            x, y = center
            dx, dy = 70 * math.cos(math.radians(axis)), -70 * math.sin(math.radians(axis))
            cv2.line(
                image, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (255, 190, 0), 2
            )
        angle = measurement.get("angle_deg")
        cv2.putText(
            image,
            "angle --" if angle is None else f"angle {angle:+.2f} deg",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 210, 30),
            2,
        )
        return image

    def show_review(self, index):
        if self.clip is None or index >= len(self.clip.frames):
            return
        self.review_index = index
        if not self.play_timer.isActive():
            self.review_position = self.clip.timestamps[index] - self.clip.timestamps[0]
        self.review_view.set_image(self.review_image(index))
        measurement = self.clip.measurements[index]
        angle = measurement.get("angle_deg")
        self.ui_text(
            self.review_info,
            "Bild {index}/{count} · t = {milliseconds:.1f} ms · Winkel {angle} · Ende: {reason}",
            index=index + 1,
            count=len(self.clip.frames),
            milliseconds=(self.clip.timestamps[index] - self.clip.timestamps[0]) * 1000,
            angle="—" if angle is None else f"{angle:+.2f}°",
            reason=self.clip.manifest.get("stop_reason", "?"),
        )

    def toggle_play(self):
        if self.clip is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.ui_text(self.play_button, "▶ Abspielen")
        else:
            if self.slider.value() == self.slider.maximum():
                self.slider.setValue(0)
            self.play_timer.start()
            self.ui_text(self.play_button, "Ⅱ Pause")

    def advance_review(self):
        if self.clip is None:
            return
        self.review_position += 0.016 * self.speed.currentData()
        target = self.clip.timestamps[0] + self.review_position
        index = int(np.searchsorted(self.clip.timestamps, target, side="right")) - 1
        self.slider.setValue(max(0, min(index, len(self.clip.frames) - 1)))
        if target >= self.clip.timestamps[-1]:
            self.play_timer.stop()
            self.ui_text(self.play_button, "▶ Abspielen")

    def export_csv(self):
        if self.clip is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.t("Messwerte exportieren"),
            "angles.csv",
            "CSV (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            try:
                self.clip.export_csv(Path(path))
                self.note(f"CSV gespeichert: {path}")
            except Exception as exc:
                self.note(f"CSV-Export fehlgeschlagen: {exc}")

    def export_still(self):
        if self.clip is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.t("Standbild mit Merkmalen"),
            "frame.png",
            "PNG (*.png)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            if cv2.imwrite(path, self.review_image(self.review_index)):
                self.note(f"Standbild gespeichert: {path}")
            else:
                self.note("Standbild konnte nicht gespeichert werden.")

    def save_profile(self):
        if self.worker is not None and self.worker.demo:
            return
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            profile = {
                "cti": self.cti.text(),
                "language": self.language,
                "serial": self.serial.text(),
                "exposure": self.exposure.value(),
                "fps": self.fps.value(),
                "capture": asdict(self.capture_config()),
                "output_dir": self.output_path.text(),
                "camera": {
                    "mode": self.camera_mode.currentData(),
                    "trigger_line": self.line.currentText(),
                    "trigger_activation": self.edge.currentText(),
                },
                "processing": self.last_snapshot["config"]
                if self.last_snapshot
                else asdict(self.processing_config()),
            }
            self.profile = profile
            target = self.data_dir / "settings.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            self.note(f"Einstellungen konnten nicht gespeichert werden: {exc}")

    def closeEvent(self, event):
        if hasattr(self, "control_api"):
            self.control_api.lease.invalidate()
            self.control_api.expire()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(3000):
                self.note("Aufzeichnung wird noch gespeichert. Fenster danach erneut schließen.")
                event.ignore()
                return
        self.save_profile()
        if hasattr(self, "control_api"):
            self.control_api.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="BiBaZu Baumer-USB Bildverarbeitung")
    parser.add_argument("--demo", action="store_true", help="Synthetischen Durchlauf starten")
    parser.add_argument("--connect", action="store_true", help="Kamera beim Start verbinden")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--language", choices=("de", "en"), help="Sprache: Deutsch / Englisch")
    args = parser.parse_args()
    application = QApplication(sys.argv[:1])
    window = MainWindow(args.data_dir, demo=args.demo, language=args.language)
    window.show()
    if args.connect and not args.demo:
        QTimer.singleShot(0, lambda: window.connect_camera(False))
    return application.exec()
