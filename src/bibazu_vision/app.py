from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QLibraryInfo,
    QLocale,
    QSignalBlocker,
    Qt,
    QTimer,
    QTranslator,
    pyqtSignal,
)
from PyQt6.QtGui import QImage, QPixmap
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
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .api import VisionApi
from .camera import CameraConfig
from .history import PreviewArchive
from .history_trace import AngleTrace
from .i18n import translate, translate_message
from .processing import ProcessingConfig, render_overlay
from .profiles import CAMERA_LABELS, VLXT_PROFILE, camera_data_dir, migrate_settings
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


def spin(value, minimum, maximum, decimals=1, suffix=""):
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    return widget


class CollapsibleSection(QWidget):
    """A disclosure button hides content without changing any setting values."""

    def __init__(self, title, expanded=False):
        super().__init__()
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.body = QWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.header.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def title(self):
        return self.header.text()

    def setTitle(self, title):
        self.header.setText(title)

    def set_expanded(self, expanded):
        with QSignalBlocker(self.header):
            self.header.setChecked(expanded)
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


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
        self.settings = migrate_settings(self.profile)
        self.camera_profiles = self.settings["camera_profiles"]
        self.selected_camera = self.settings["selected_camera"]
        self.profile = self.camera_profiles[self.selected_camera]
        preferred = language if language is not None else self.settings["language"]
        self.language = preferred if preferred in {"de", "en"} else "de"
        self._text_bindings = {}
        self._log_sources = []
        self._last_note = None
        self._qt_translator = QTranslator(self)
        self.worker = None
        self._connection_pending = False
        self._demo_controls = False
        self.camera_settings_pending = False
        self.last_snapshot = None
        self.last_frame = None
        self.history_archive = None
        self._history_archives = []
        self.history_selection = None
        self._last_preview_key = None
        self._preview_cache_key = None
        self._preview_cache_image = None
        self.clip = None
        self.review_index = 0
        self.review_position = 0.0
        self.demo_requested = demo
        self.last_saved = None
        self.pipeline_error = None
        self.recording_error_text = None
        self.build_ui()
        self._register_texts()
        self.ui_text(
            self.reference_label, "0°-Offset: {offset:.2f}°",
            offset=self.profile["processing"]["zero_offset_deg"],
        )
        self.set_language(self.language, persist=language is not None)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.refresh_live)
        self.timer.start(17)
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
        self._group_texts = [
            (widget, widget.title())
            for kind in (QGroupBox, CollapsibleSection)
            for widget in self.findChildren(kind)
        ]
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
        self.trace.language = language
        self.trace.caption = self.t("Fortlaufender Achswinkel · Lücken markiert")
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
            self.settings["language"] = self.language
        except (OSError, ValueError) as exc:
            self.note(f"Sprache konnte nicht gespeichert werden: {exc}")

    def build_ui(self):
        self.setStyleSheet("""
            QWidget {background:#101827;color:#e2e8f0;font-size:13px;}
            QGroupBox {border:1px solid #334155;border-radius:7px;margin-top:15px;padding:10px;}
            QGroupBox::title {subcontrol-origin:margin;left:12px;color:#7dd3fc;}
            QPushButton,QToolButton {background:#24354d;border:1px solid #425774;
                border-radius:5px;padding:8px;}
            QPushButton:hover,QToolButton:hover {background:#334c6c;}
            QPushButton:disabled,QToolButton:disabled {color:#64748b;}
            QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QListWidget,QTextEdit {
                background:#172338;border:1px solid #334155;border-radius:4px;padding:4px;}
            QTabBar::tab {background:#24354d;padding:10px 20px;}
            QTabBar::tab:selected {background:#0369a1;}
        """)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        heading = QHBoxLayout()
        title = QLabel("BiBaZu  /  Bildverarbeitung")
        title.setStyleSheet("font-size:21px;font-weight:600;color:#f8fafc;padding:6px")
        heading.addWidget(title)
        heading.addStretch()
        self.connection_label = QLabel("Nicht verbunden")
        heading.addWidget(self.connection_label)
        self.connect_button = self.button("Verbinden", lambda: self.connect_camera(False))
        self.disconnect_button = self.button("Trennen", self.disconnect_camera)
        self.disconnect_button.setEnabled(False)
        self.demo_button = self.button("Demo", lambda: self.connect_camera(True))
        self.language_button = self.button("English", self.toggle_language)
        self.language_button.setObjectName("language_toggle")
        for button in (
            self.connect_button,
            self.disconnect_button,
            self.demo_button,
            self.language_button,
        ):
            heading.addWidget(button)
        layout.addLayout(heading)
        self.tabs = QTabWidget()
        self.build_live_page()
        self.build_capture_page()
        self.build_review_page()
        layout.addWidget(self.tabs, 1)
        footer = QHBoxLayout()
        self.state_label = QLabel("Bereit · Kamera noch nicht verbunden")
        self.state_label.setWordWrap(True)
        footer.addWidget(self.state_label, 1)
        self.details_button = self.button("Protokoll & Details", self.toggle_details)
        self.details_button.setCheckable(True)
        footer.addWidget(self.details_button)
        layout.addLayout(footer)
        self.details_panel = QWidget()
        details = QVBoxLayout(self.details_panel)
        details.setContentsMargins(0, 0, 0, 0)
        self.diagnostics_label = QLabel("Noch keine Bilddaten")
        self.diagnostics_label.setWordWrap(True)
        details.addWidget(self.diagnostics_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(40)
        self.log.setMaximumHeight(100)
        details.addWidget(self.log)
        layout.addWidget(self.details_panel)
        self.details_panel.hide()
        self.tabs.currentChanged.connect(self.refresh_current_page)
        self.overlay_check.toggled.connect(self.refresh_current_page)

    def toggle_details(self):
        self.details_panel.setVisible(self.details_button.isChecked())

    def create_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        main = QVBoxLayout()
        layout.addLayout(main, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        sidebar = QVBoxLayout(panel)
        sidebar.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(panel)
        layout.addWidget(scroll)
        return page, main, sidebar, scroll

    @staticmethod
    def form(parent):
        form = QFormLayout(parent)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        return form

    @staticmethod
    def measurement_header(layout):
        indicators = QHBoxLayout()
        angle = QLabel("— °")
        angle.setMinimumWidth(220)
        angle.setStyleSheet("font-size:34px;font-weight:600;color:#38bdf8")
        quality = QLabel("Bauteil: noch kein Bild")
        quality.setWordWrap(True)
        rate = QLabel("0 FPS")
        indicators.addWidget(angle)
        indicators.addStretch()
        indicators.addWidget(quality)
        indicators.addWidget(rate)
        layout.addLayout(indicators)
        return angle, quality, rate

    def build_live_page(self):
        self.live_page, live_layout, sidebar, self.live_sidebar = self.create_page()
        self.angle_label, self.quality_label, self.rate_label = self.measurement_header(live_layout)
        view_heading = QHBoxLayout()
        self.history_status = QLabel("Livebild")
        self.history_status.setWordWrap(True)
        view_heading.addWidget(self.history_status, 1)
        self.live_button = self.button("Live", self.return_to_live)
        view_heading.addWidget(self.live_button)
        live_layout.addLayout(view_heading)
        self.live_view = ImageView()
        self.live_view.setMinimumSize(320, 160)
        self.live_view.selected.connect(self.image_selection)
        live_layout.addWidget(self.live_view, 1)
        self.overlay_check = QCheckBox("Kontur, OBB und Messachse anzeigen")
        self.overlay_check.setChecked(True)
        live_layout.addWidget(self.overlay_check)
        self.segmentation_label = QLabel("Segmentierung: noch kein Bild")
        self.segmentation_label.setWordWrap(True)
        live_layout.addWidget(self.segmentation_label)
        self.trace = AngleTrace()
        self.trace.selected.connect(self.select_history)
        self.trace.selection_expired.connect(self.history_expired)
        trace_controls = QHBoxLayout()
        trace_controls.addWidget(QLabel("Sichtbarer Zeitraum"))
        self.history_span = QComboBox()
        for seconds in (1, 5, 10, 30, 60, 600):
            self.history_span.addItem(f"{seconds} s", seconds)
        self.history_span.setCurrentIndex(2)
        self.history_span.currentIndexChanged.connect(
            lambda: self.trace.set_span(self.history_span.currentData())
        )
        trace_controls.addWidget(self.history_span)
        trace_controls.addStretch()
        live_layout.addLayout(trace_controls)
        live_layout.addWidget(self.trace)
        self.build_processing_controls(sidebar)
        self.build_camera_controls(sidebar)
        sidebar.addStretch()
        self.tabs.addTab(self.live_page, "Live & Erkennung")

    def build_capture_page(self):
        self.capture_page, main, sidebar, self.capture_sidebar = self.create_page()
        (self.capture_angle_label, self.capture_quality_label, self.capture_rate_label) = (
            self.measurement_header(main)
        )
        main.addWidget(QLabel("Livebild · Aktuelle Messung"))
        self.capture_view = ImageView()
        main.addWidget(self.capture_view, 1)
        self.capture_status = QLabel("Keine Aufzeichnung aktiv")
        self.capture_status.setWordWrap(True)
        main.addWidget(self.capture_status)
        saved = QHBoxLayout()
        self.last_saved_label = QLabel("Noch kein Durchlauf gespeichert.")
        self.last_saved_label.setWordWrap(True)
        saved.addWidget(self.last_saved_label, 1)
        self.open_saved_button = self.button("In Auswertung öffnen", self.open_last_saved)
        self.open_saved_button.setEnabled(False)
        saved.addWidget(self.open_saved_button)
        main.addLayout(saved)
        self.build_capture_controls(sidebar)
        sidebar.addStretch()
        self.tabs.addTab(self.capture_page, "Durchläufe aufnehmen")

    def build_review_page(self):
        self.review_page, review_layout, panel_layout, self.review_sidebar = self.create_page()
        self.review_angle_label = QLabel("— °")
        self.review_angle_label.setStyleSheet("font-size:34px;font-weight:600;color:#38bdf8")
        review_layout.addWidget(self.review_angle_label)
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
        self.build_recordings_controls(panel_layout)
        self.tabs.addTab(self.review_page, "Auswertung")

    def build_camera_controls(self, panel_layout):
        self.camera_section = group = CollapsibleSection("Kamera")
        form = self.form(group.body)
        self.camera_selector = QComboBox()
        for camera_id, label in CAMERA_LABELS.items():
            self.camera_selector.addItem(label, camera_id)
        self.camera_selector.setCurrentIndex(self.camera_selector.findData(self.selected_camera))
        self.camera_selector.currentIndexChanged.connect(self.change_camera_profile)
        form.addRow("Kameraprofil", self.camera_selector)
        default_cti = r"C:\Program Files\Baumer Camera Explorer\bgapi2_usb.cti"
        self.cti = QLineEdit(self.profile.get("cti", default_cti))
        form.addRow("GenTL-Producer (.cti)", self.cti)
        self.cti_button = self.button("CTI auswählen", self.choose_cti)
        form.addRow(self.cti_button)
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
        hint = QLabel(
            "Belichtung / Bildrate beim Verbinden oder mit Anwenden setzen. "
            "Aufzeichnung vorher beenden. Andere Kameraeinstellungen benötigen erneutes "
            "Verbinden. Tatsächliche FPS siehe Liveanzeige."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        panel_layout.addWidget(group)

    def build_processing_controls(self, panel_layout):
        self.processing_group = group = CollapsibleSection("Erkennung & Referenz", True)
        form = self.form(group.body)
        self.processing_advanced = CollapsibleSection("Weitere Erkennungseinstellungen")
        advanced = self.form(self.processing_advanced.body)
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
        advanced.addRow("Ohne Hintergrund", self.foreground)
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
        advanced.addRow("Verarbeitung", self.scale)
        self.chute_angle = spin(
            self.profile.get("processing", {}).get("chute_angle_deg", 0), -180, 180, 2, " °"
        )
        advanced.addRow("Rutschenwinkel im Bild", self.chute_angle)
        self.ignore_border_touch = QCheckBox("Randprüfung deaktivieren")
        self.ignore_border_touch.setChecked(
            self.profile.get("processing", {}).get("ignore_border_touch", False)
        )
        advanced.addRow(self.ignore_border_touch)
        self.reference_label = QLabel("0°-Offset: 0,00°")
        form.addRow(self.reference_label)
        self.threshold.valueChanged.connect(self.apply_processing)
        self.area.valueChanged.connect(self.apply_processing)
        self.foreground.currentIndexChanged.connect(self.apply_processing)
        self.method.currentIndexChanged.connect(self.apply_processing)
        self.scale.currentIndexChanged.connect(self.apply_processing)
        self.chute_angle.valueChanged.connect(self.apply_processing)
        self.ignore_border_touch.toggled.connect(self.apply_processing)
        form.addRow(
            self.button(
                "Leere Rutsche als Hintergrund aufnehmen", lambda: self.command("background")
            )
        )
        advanced.addRow(
            self.button("Hintergrund entfernen", lambda: self.command("clear_background"))
        )
        form.addRow(
            self.button("Rutsche markieren · zwei Punkte", lambda: self.select_image("chute"))
        )
        form.addRow(
            self.button("Messbereich wählen · zwei Ecken", lambda: self.select_image("roi"))
        )
        advanced.addRow(
            self.button(
                "Gesamtes Bild als Messbereich", lambda: self.command("config", {"roi": None})
            )
        )
        form.addRow(self.button("Liegendes Bauteil = 0°", lambda: self.command("zero")))
        advanced.addRow(
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
        advanced.addRow(hint)
        form.addRow(self.processing_advanced)
        panel_layout.addWidget(group)

    def build_capture_controls(self, panel_layout):
        self.capture_group = group = QGroupBox("Aufnahme")
        form = self.form(group)
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
        self.arm_button = self.button("Aufzeichnung scharf schalten", self.arm)
        form.addRow(self.arm_button)
        self.trigger_button = self.button("Manuell auslösen", lambda: self.command("trigger"))
        form.addRow(self.trigger_button)
        self.stop_button = self.button("Aufzeichnung beenden", lambda: self.command("disarm"))
        form.addRow(self.stop_button)
        self.output_path = QLineEdit(
            self.settings.get("output_dir", str(self.data_dir / "camera" / "recordings"))
        )
        form.addRow("Speicherordner", self.output_path)
        self.output_button = self.button("Speicherordner auswählen", self.choose_output)
        form.addRow(self.output_button)
        self.capture_advanced = CollapsibleSection("Weitere Aufnahmeeinstellungen")
        advanced = self.form(self.capture_advanced.body)
        capture_profile = self.profile.get("capture", {})
        self.quiet = spin(capture_profile.get("quiet_seconds", 0.15), 0.02, 5, 2, " s")
        self.max_duration = spin(capture_profile.get("max_seconds", 5), 0.1, 60, 1, " s")
        self.memory = spin(capture_profile.get("max_memory_mb", 256), 16, 4096, 0, " MB")
        self.motion_percent = spin(
            capture_profile.get("motion_threshold", 0.001) * 100, 0.001, 100, 3, " %"
        )
        advanced.addRow("Nachlauf ohne Bewegung", self.quiet)
        advanced.addRow("Maximale Dauer", self.max_duration)
        advanced.addRow("RAM-Limit gesamt", self.memory)
        advanced.addRow("Bewegungsschwelle im ROI", self.motion_percent)
        hint = QLabel(
            "Ende nach erkannter Bewegung, ruhigem Bild und leerem Messbereich. "
            "Dauer- und RAM-Limit begrenzen jeden Durchlauf. Im Livebetrieb wird ein freier "
            "Hardwarezähler für die Lichtschranke verwendet. Ein Rückfall auf Eingangspolling "
            "wird im Protokoll angezeigt; dabei können kurze Impulse verloren gehen."
        )
        hint.setWordWrap(True)
        advanced.addRow(hint)
        form.addRow(self.capture_advanced)
        panel_layout.addWidget(group)

    def build_recordings_controls(self, panel_layout):
        group = QGroupBox("Gespeicherte Durchläufe")
        vbox = QVBoxLayout(group)
        self.recordings = QListWidget()
        self.recordings.setMinimumHeight(240)
        self.recordings.itemDoubleClicked.connect(
            lambda item: self.open_clip(Path(item.data(Qt.ItemDataRole.UserRole)))
        )
        vbox.addWidget(self.recordings, 1)
        vbox.addWidget(self.button("Liste aktualisieren", self.refresh_recordings))
        vbox.addWidget(self.button("Durchlauf-Ordner öffnen", self.choose_clip))
        panel_layout.addWidget(group, 1)

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
            ignore_border_touch=self.ignore_border_touch.isChecked(),
        )

    def remember_camera_controls(self):
        self.profile = {
            "cti": self.cti.text(),
            "serial": self.serial.text(),
            "exposure": self.exposure.value(),
            "fps": self.fps.value(),
            "capture": asdict(self.capture_config()),
            "camera": {
                "mode": self.camera_mode.currentData(),
                "trigger_line": self.line.currentText(),
                "trigger_activation": self.edge.currentText(),
            },
            "processing": asdict(self.processing_config()),
        }
        self.camera_profiles[self.selected_camera] = self.profile

    def load_camera_controls(self):
        self.last_snapshot = None
        self.profile = self.camera_profiles[self.selected_camera]
        camera = self.profile["camera"]
        processing = self.profile["processing"]
        capture = self.profile["capture"]
        values = [
            (self.cti, self.profile["cti"]), (self.serial, self.profile["serial"]),
            (self.exposure, self.profile["exposure"]), (self.fps, self.profile["fps"]),
            (self.threshold, processing["threshold"]), (self.area, processing["min_area"]),
            (self.chute_angle, processing["chute_angle_deg"]),
            (self.quiet, capture["quiet_seconds"]), (self.max_duration, capture["max_seconds"]),
            (self.memory, capture["max_memory_mb"]),
            (self.motion_percent, capture["motion_threshold"] * 100),
        ]
        for widget, value in values:
            with QSignalBlocker(widget):
                if isinstance(widget, QLineEdit):
                    widget.setText(value)
                else:
                    widget.setValue(value)
        for widget, value in [
            (self.camera_mode, camera["mode"]), (self.line, camera["trigger_line"]),
            (self.edge, camera["trigger_activation"]), (self.method, processing["method"]),
            (self.foreground, processing["foreground"]), (self.scale, processing["scale"]),
            (self.start_mode, capture["start_mode"]),
        ]:
            with QSignalBlocker(widget):
                index = widget.findData(value)
                widget.setCurrentIndex(max(0, index if index >= 0 else widget.findText(value)))
        with QSignalBlocker(self.ignore_border_touch):
            self.ignore_border_touch.setChecked(processing["ignore_border_touch"])
        self.ui_text(
            self.reference_label, "0°-Offset: {offset:.2f}°",
            offset=processing["zero_offset_deg"],
        )

    def change_camera_profile(self, *_):
        selected = self.camera_selector.currentData()
        if selected == self.selected_camera:
            return
        if self._connection_pending or (self.worker is not None and self.worker.isRunning()):
            with QSignalBlocker(self.camera_selector):
                self.camera_selector.setCurrentIndex(
                    self.camera_selector.findData(self.selected_camera)
                )
            self.note("Bitte die laufende Kamera zuerst trennen.")
            return
        if self._demo_controls:
            self._demo_controls = False
            self.load_camera_controls()
        if not self.save_profile():
            with QSignalBlocker(self.camera_selector):
                self.camera_selector.setCurrentIndex(
                    self.camera_selector.findData(self.selected_camera)
                )
            return
        self.control_api.lease.invalidate()
        self.control_api.expire()
        self.worker = None
        self.selected_camera = selected
        self.load_camera_controls()
        self.reset_live_state()
        self.ui_text(self.state_label, "Bereit · Kamera noch nicht verbunden")
        self.ui_text(self.connection_label, "Nicht verbunden")
        self.ui_text(self.capture_status, "Bereit · Kamera noch nicht verbunden")
        self.save_profile()

    def set_connection_controls(self, enabled):
        for widget in (self.camera_selector, self.cti, self.serial, self.cti_button):
            widget.setEnabled(enabled)

    def capture_config(self):
        return CaptureConfig(
            start_mode=self.start_mode.currentData(),
            quiet_seconds=self.quiet.value(),
            max_seconds=self.max_duration.value(),
            max_memory_mb=self.memory.value(),
            motion_threshold=self.motion_percent.value() / 100,
        )

    def connect_camera(self, demo):
        if self._connection_pending or (self.worker is not None and self.worker.isRunning()):
            self.note("Bitte die laufende Kamera zuerst trennen.")
            return
        if not demo and not self.output_path.text().strip():
            self.note("Bitte einen Speicherordner auswählen.")
            return
        if demo:
            if not self._demo_controls:
                self.remember_camera_controls()
            self._demo_controls = True
            self.last_snapshot = None
            self.threshold.setValue(25)
            self.area.setValue(150)
            self.chute_angle.setValue(2.86)
            self.method.setCurrentIndex(0)
            self.foreground.setCurrentIndex(0)
            self.scale.setCurrentIndex(0)
            self.start_mode.setCurrentIndex(self.start_mode.findData("motion"))
        elif self._demo_controls:
            self._demo_controls = False
            self.load_camera_controls()
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
            pixel_format="Mono8" if self.selected_camera == VLXT_PROFILE else None,
        )
        if not demo and camera.mode == "acquisition_trigger":
            self.start_mode.setCurrentIndex(self.start_mode.findData("hardware"))
        self.worker = PipelineWorker(
            camera,
            (ProcessingConfig(chute_angle_deg=2.86) if demo else self.processing_config()),
            self.capture_config(),
            (
                self.data_dir / "demo" if demo
                else camera_data_dir(self.data_dir, self.selected_camera)
            ),
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
        self.demo_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.output_path.setEnabled(False)
        self.output_button.setEnabled(False)
        self._connection_pending = True
        self.set_connection_controls(False)
        self.ui_text(
            self.state_label, "Verbinde Demo …" if demo else "Verbinde Baumer-Kamera …"
        )
        self.ui_text(self.connection_label, "Verbinde …")
        self.start_history()
        self.worker.start()

    def worker_finished(self):
        # Up to one full telemetry buffer may remain after the last preview tick.
        for _ in range(4):
            if not self.drain_measurements():
                break
        self.camera_settings_pending = False
        self._connection_pending = False
        self.set_connection_controls(True)
        self.apply_camera_button.setEnabled(False)
        self.connect_button.setEnabled(True)
        self.demo_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.ui_text(self.connection_label, "Nicht verbunden")
        self.processing_group.setEnabled(True)
        self.capture_group.setEnabled(True)
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
        source, values = self._text_bindings[self.state_label]
        self.ui_text(self.capture_status, source, **values)
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
            self.disconnect_button.setEnabled(False)
            self.ui_text(self.connection_label, "Wird getrennt …")
            self.ui_text(self.state_label, "Kamera wird getrennt; Aufzeichnung wird gespeichert …")

    def refresh_camera_settings_button(self):
        self.apply_camera_button.setEnabled(
            self.worker is not None
            and self.worker.isRunning()
            and not self.camera_settings_pending
            and self.worker.camera_settings_available()
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
                "Nach Belichtungsänderung vorhandenen Hintergrund bei leerer Rutsche neu aufnehmen."
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
        self.tabs.setCurrentWidget(self.live_page)
        self.return_to_live()
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
        cutoff = time.perf_counter() - 600
        self.trace.expire_before(cutoff)
        if self.history_archive is not None:
            self.history_archive.expire_before(cutoff)
        snapshot = self.worker.snapshot()
        if snapshot is not None or self.worker.camera_idle.is_set():
            if self.worker.demo:
                self.ui_text(self.connection_label, "Demo verbunden")
            else:
                self.ui_text(
                    self.connection_label, "Verbunden: {camera}",
                    camera=getattr(self.worker, "camera_identity", "Baumer"),
                )
        if snapshot is None:
            return
        self.last_snapshot = snapshot
        self.drain_measurements()
        status = snapshot["status"]
        state = getattr(status, "state", "?")
        self.processing_group.setEnabled(
            state not in {"waiting", "moving"} and not self.worker.api_capture.busy()
        )
        self.capture_group.setEnabled(not self.worker.api_capture.busy())
        reason = getattr(status, "reason", "") or ""
        error = getattr(status, "error", "")
        state_text = {
            "idle": "Bereit",
            "armed": "Scharfgeschaltet",
            "waiting": "Wartet auf Bewegung",
            "moving": "Aufnahme läuft",
            "saving": "Wird gespeichert",
        }.get(state, state)
        stale = (
            f" · Kein neues Bild seit {snapshot['stale_seconds']:.1f}s"
            if snapshot["stale_seconds"] > 1
            else ""
        )
        self.ui_text(
            self.state_label,
            "Aufzeichnung: {state}{error}{stale}",
            state=state_text,
            error=f" · {error}" if error else "",
            stale=stale,
        )
        self.ui_text(
            self.capture_status,
            "Aufzeichnung: {state} · {frames} Bilder · {mib:.1f} MiB{error}{stale}",
            state=state_text,
            frames=getattr(status, "buffered_frames", 0),
            mib=getattr(status, "buffered_bytes", 0) / 1024**2,
            error=f" · {error}" if error else "",
            stale=stale,
        )
        measured, previewed, valid = self.trace.rates(time.perf_counter())
        self.ui_text(
            self.diagnostics_label,
            "ID {frame_id} · ID-Lücken {gaps} · {ms:.1f} ms · Ende: {reason}\n"
            "Messrate {measured}/s · Vorschau {previewed}/s · Gültig {valid:.0%} · "
            "Pufferüberläufe {overflows} · Ohne Vorschau {skipped} (Messwerte vorhanden)",
            frame_id=snapshot["frame_id"],
            gaps=snapshot["frame_gaps"],
            ms=snapshot["detection"].processing_ms,
            reason=reason or "—",
            measured=measured,
            previewed=previewed,
            valid=valid,
            overflows=self.trace.measurements.overflow_count,
            skipped=max(0, self.trace.measurements.total_count - self.trace.preview_count),
        )
        if snapshot["last_clip"] and snapshot["last_clip"] != self.last_saved:
            self.recording_error_text = None
            self.last_saved = snapshot["last_clip"]
            self.ui_text(
                self.last_saved_label, "Gespeichert: {name}", name=Path(self.last_saved).name
            )
            self.open_saved_button.setEnabled(True)
            self.note(f"Durchlauf gespeichert: {self.last_saved}")
            self.refresh_recordings()
        preview_key = snapshot["timestamp"], snapshot["frame_id"]
        if preview_key == self._last_preview_key:
            return
        self._last_preview_key = preview_key
        self.last_frame = snapshot["frame_id"]
        detection = snapshot["detection"]
        angle = detection.angle_deg
        if self.history_selection is None:
            self.show_live_measurement(snapshot)
        self.show_capture_measurement(snapshot)
        self.ui_text(
            self.rate_label,
            "{fps:.0f} FPS",
            fps=snapshot["fps"],
        )
        config = ProcessingConfig(**snapshot["config"])
        self.ui_text(
            self.reference_label, "0°-Offset: {offset:.2f}°", offset=config.zero_offset_deg
        )
        image = self.preview_image(snapshot)
        page = self.tabs.currentWidget()
        if page is self.live_page and self.history_selection is None:
            self.live_view.set_image(image)
        elif page is self.capture_page:
            self.capture_view.set_image(image)
        timestamp = snapshot["timestamp"]
        sample = self.trace.add(timestamp, snapshot["frame_id"], angle)
        if self.history_archive is not None:
            self.history_archive.submit(sample, image)

    def drain_measurements(self):
        if self.worker is None:
            return 0
        batch = self.worker.api_capture.history_measurements(self.trace.measurements.cursor, 2048)
        added = self.trace.add_measurements(batch)
        if added and self.history_selection is not None:
            self.show_history_position(self.history_selection)
        return added

    def preview_image(self, snapshot):
        key = snapshot["timestamp"], snapshot["frame_id"], self.overlay_check.isChecked()
        if key != self._preview_cache_key:
            self._preview_cache_image = (
                render_overlay(
                    snapshot["image"], snapshot["detection"], ProcessingConfig(**snapshot["config"])
                )
                if self.overlay_check.isChecked()
                else snapshot["image"]
            )
            self._preview_cache_key = key
        return self._preview_cache_image

    def refresh_current_page(self, *_):
        """Refresh on navigation even if acquisition has not produced a new frame."""
        snapshot = self.last_snapshot
        if snapshot is None:
            return
        page = self.tabs.currentWidget()
        if page is self.review_page or (
            page is self.live_page and self.history_selection is not None
        ):
            return
        image = self.preview_image(snapshot)
        if page is self.capture_page:
            self.show_capture_measurement(snapshot)
            self.capture_view.set_image(image)
        else:
            self.show_live_measurement(snapshot)
            self.live_view.set_image(image)

    def show_capture_measurement(self, snapshot):
        detection = snapshot["detection"]
        angle = detection.angle_deg
        self.ui_text(
            self.capture_angle_label,
            "— °" if angle is None else "{angle:+.2f} °",
            **({} if angle is None else {"angle": angle}),
        )
        if detection.center is None:
            self.ui_text(self.capture_quality_label, "Kein Bauteil")
        else:
            self.ui_text(
                self.capture_quality_label,
                "Qualität {confidence:.0%}{border}",
                confidence=detection.confidence,
                border=" · Randkontakt" if detection.touching_border else "",
            )
        self.ui_text(self.capture_rate_label, "{fps:.0f} FPS", fps=snapshot["fps"])

    def open_last_saved(self):
        if self.last_saved is not None:
            self.open_clip(Path(self.last_saved))

    def reset_live_state(self):
        if self.history_archive is not None:
            self.history_archive.stop()
        self.history_archive = None
        self.last_snapshot = None
        self.last_frame = None
        self.pipeline_error = None
        self.recording_error_text = None
        self.history_selection = None
        self._last_preview_key = None
        self._preview_cache_key = None
        self._preview_cache_image = None
        self.trace.reset(time.perf_counter())
        self.overlay_check.setEnabled(True)
        self.ui_text(self.history_status, "Livebild")
        self.ui_text(self.angle_label, "— °")
        self.ui_text(self.quality_label, "Bauteil: noch kein Bild")
        self.ui_text(self.capture_angle_label, "— °")
        self.ui_text(self.capture_quality_label, "Bauteil: noch kein Bild")
        self.ui_text(self.rate_label, "0 FPS")
        self.ui_text(self.capture_rate_label, "0 FPS")
        self.ui_text(self.diagnostics_label, "Noch keine Bilddaten")
        self.ui_text(self.segmentation_label, "Segmentierung: noch kein Bild")
        self.live_view.source = None
        self.live_view.clear()
        self.ui_text(self.live_view, "Kamera verbinden oder Demo starten")
        self.capture_view.source = None
        self.capture_view.clear()
        self.ui_text(self.capture_view, "Kamera verbinden oder Demo starten")
        self.live_view.mode = None
        self.live_view.first = None
        self.live_view.unsetCursor()
        self.capture_view.mode = None
        self.capture_view.first = None
        self.capture_view.unsetCursor()

    def start_history(self):
        self.reset_live_state()
        self.history_archive = archive = PreviewArchive(self)
        self._history_archives.append(archive)
        archive.image_loaded.connect(
            lambda key, image: self.history_image_loaded(archive, key, image)
        )
        archive.start()

    def select_history(self, sample):
        self.history_selection = sample
        self.live_view.mode = None
        self.live_view.first = None
        self.live_view.unsetCursor()
        self.overlay_check.setEnabled(False)
        self.show_history_position(sample)
        self.ui_text(
            self.angle_label,
            "— ° · Historie" if sample.angle is None else "{angle:+.2f} ° · Historie",
            **({} if sample.angle is None else {"angle": sample.angle}),
        )
        self.ui_text(
            self.quality_label,
            "Kein gültiger Winkel" if sample.angle is None else "Historischer Messwert",
        )
        self.ui_text(
            self.segmentation_label,
            "Historisches Vorschaubild · Messmarkierungen vom Aufnahmezeitpunkt",
        )
        self.live_view.source = None
        self.live_view.clear()
        self.ui_text(self.live_view, "Historisches Bild wird geladen …")
        if self.history_archive is not None:
            self.history_archive.request(sample.key)

    def show_history_position(self, sample):
        row = self.trace.measurements.for_preview(sample)
        angle = row.continuous_angle if row is not None else None
        self.ui_text(
            self.history_status,
            "Historie · t = {seconds:.3f} s · Messwinkel {raw} · Fortlaufend {angle}",
            seconds=sample.timestamp - self.trace.origin,
            raw="—" if sample.angle is None else f"{sample.angle:+.2f}°",
            angle="—" if angle is None else f"{angle:+.2f}°",
        )

    def history_image_loaded(self, archive, key, image):
        if (
            archive is not self.history_archive
            or self.history_selection is None
            or self.history_selection.key != key
        ):
            return
        if image is None:
            self.live_view.source = None
            self.live_view.clear()
            self.ui_text(self.live_view, "Historisches Bild nicht verfügbar.")
        else:
            self.live_view.set_image(image)

    def history_expired(self):
        self.return_to_live()
        self.note("Der ausgewählte Zeitpunkt liegt außerhalb der zehnminütigen Historie.")

    def show_live_measurement(self, snapshot):
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
            self.segmentation_label,
            "Segmentierung: fester Hintergrundabzug"
            if snapshot.get("has_background")
            else "Segmentierung: absolute Helligkeitsschwelle · kein Hintergrund",
        )

    def return_to_live(self):
        self.history_selection = None
        self.trace.live()
        self.overlay_check.setEnabled(True)
        self.ui_text(self.history_status, "Livebild")
        if self.last_snapshot is not None:
            snapshot = self.last_snapshot
            self.show_live_measurement(snapshot)
            image = self.preview_image(snapshot)
            self.live_view.set_image(image)

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
        self.tabs.setCurrentWidget(self.review_page)
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
            self.review_angle_label,
            "— °" if angle is None else "{angle:+.2f} °",
            **({} if angle is None else {"angle": angle}),
        )
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
        if self._demo_controls:
            return True
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if (
                self.worker is not None and not self.worker.demo
                and not self.worker.isRunning()
            ):
                snapshot = self.worker.snapshot()
                if snapshot is not None:
                    self.last_snapshot = snapshot
            self.remember_camera_controls()
            self.settings = {
                "version": 2,
                "selected_camera": self.selected_camera,
                "camera_profiles": self.camera_profiles,
                "language": self.language,
                "output_dir": self.output_path.text(),
            }
            target = self.data_dir / "settings.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            temporary.replace(target)
            return True
        except OSError as exc:
            self.note(f"Einstellungen konnten nicht gespeichert werden: {exc}")
            return False

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
        for archive in self._history_archives:
            archive.stop()
        for archive in self._history_archives:
            if not archive.wait(3000):
                self.note("Bildhistorie wird noch geschlossen. Fenster erneut schließen.")
                event.ignore()
                return
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="BiBaZu Baumer USB/GigE Bildverarbeitung")
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
