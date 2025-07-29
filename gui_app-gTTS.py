# gui_app_gTTS_styled.py

import sys
import os
import uuid
import time
import cv2
import threading
from gtts import gTTS
from playsound import playsound
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QSpinBox, QProgressBar,
    QGridLayout, QGraphicsOpacityEffect, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont

from app2_words_core import SignLanguageRecognizer

class SignLanguageGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Sign Language Recognition with Voice")
        self.setGeometry(100, 100, 1400, 800)

        self.recognizer = SignLanguageRecognizer()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detection_enabled = True
        self.dark_theme = True
        self.start_time = None
        self.session_word_count = 0

        self.heading_font = QFont("Segoe UI", 18, QFont.Bold)
        self.label_font = QFont("Consolas", 14)
        self.button_font = QFont("Segoe UI", 12, QFont.Medium)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(880, 660)
        self.video_label.setStyleSheet("border: 3px solid #00b4d8; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.detection_label = QLabel("Detected Sign: -")
        self.detection_label.setFont(self.label_font)
        self.current_word_label = QLabel("Current Word: -")
        self.current_word_label.setFont(self.label_font)

        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setFont(QFont('Courier New', 14))
        self.sentence_box.setMaximumHeight(400)
        self.sentence_box.setStyleSheet("background-color: #1e1e1e; color: #00ff88; border: 2px solid #00ff88; border-radius: 8px;")

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(10)
        self.confidence_bar.setStyleSheet("QProgressBar {border: 1px solid grey; border-radius: 5px;} QProgressBar::chunk {background-color: #00ff00; width: 10px;}")

        self.start_btn = QPushButton("🔴 Start")
        self.stop_btn = QPushButton("⏹ Stop")
        self.detect_btn = QPushButton("❌ Disable Detection")
        self.clear_btn = QPushButton("🧼 Clear")
        self.speak_btn = QPushButton("🎤 Speak")
        self.snapshot_btn = QPushButton("📸")
        self.theme_btn = QPushButton("🌗")
        self.theme_btn.setToolTip("Switch theme")

        for btn in [self.start_btn, self.stop_btn, self.detect_btn, self.clear_btn, self.speak_btn, self.snapshot_btn, self.theme_btn]:
            btn.setFont(self.button_font)
            btn.setStyleSheet(self.button_style())

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(5, 100)
        self.threshold_spin.setValue(30)
        self.threshold_spin.setStyleSheet("background-color: #222; color: white; padding: 5px; border-radius: 4px;")

        # Adjusted layout
        top_right_widget_layout = QHBoxLayout()
        top_right_widget_layout.addStretch()
        top_right_widget_layout.addWidget(self.snapshot_btn)
        top_right_widget_layout.addWidget(self.theme_btn)

        right_layout = QVBoxLayout()
        right_layout.addLayout(top_right_widget_layout)
        right_layout.addWidget(self.detection_label)
        right_layout.addWidget(self.confidence_bar)
        right_layout.addWidget(self.current_word_label)
        right_layout.addWidget(QLabel("Sentence:"))
        right_layout.addWidget(self.sentence_box)
        right_layout.addWidget(QLabel("Threshold:"))
        right_layout.addWidget(self.threshold_spin)

        self.stats_label = QLabel("Words: 0 | Time: 00:00 | Stability: 0%")
        self.stats_label.setFont(QFont("Segoe UI", 11))
        right_layout.addWidget(self.stats_label)

        button_layout = QHBoxLayout()
        for b in [self.start_btn, self.stop_btn, self.detect_btn, self.clear_btn, self.speak_btn]:
            button_layout.addWidget(b)

        right_layout.addLayout(button_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

        self.start_btn.clicked.connect(self.start_recognition)
        self.stop_btn.clicked.connect(self.stop_recognition)
        self.detect_btn.clicked.connect(self.toggle_detection)
        self.clear_btn.clicked.connect(self.clear_output)
        self.speak_btn.clicked.connect(self.speak_sentence)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.threshold_spin.valueChanged.connect(self.update_threshold)

        self.setFocusPolicy(Qt.StrongFocus)
        self.apply_dark_theme()

    def apply_dark_theme(self):
        self.setStyleSheet("QWidget { background-color: #121212; color: white; }")
        self.theme_btn.setText("🌙")

    def apply_light_theme(self):
        self.setStyleSheet("QWidget { background-color: #f0f0f0; color: black; }")
        self.theme_btn.setText("☀️")

    def toggle_theme(self):
        self.dark_theme = not self.dark_theme
        if self.dark_theme:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def button_style(self):
        return """
            QPushButton {
                background-color: #303030;
                border: 2px solid #00b4d8;
                border-radius: 10px;
                padding: 10px 20px;
                color: white;
            }
            QPushButton:hover {
                background-color: #00b4d8;
                color: black;
            }
        """

    def start_recognition(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open the camera.")
            # Display an error message to the user (e.g., QMessageBox)
            QMessageBox.critical(self, "Camera Error",
                                 "Could not access the camera. Please ensure it's connected and not in use.")
            self.cap = None  # Cap is None if it failed
            return
        self.start_time = time.time()
        self.session_word_count = 0
        self.timer.start(30)
        self.setFocus()

    def stop_recognition(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.setFocus()

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        self.detect_btn.setText("✅ Enable Detection" if not self.detection_enabled else "❌ Disable Detection")

    def clear_output(self):
        self.detection_label.setText("Detected Sign: -")
        self.current_word_label.setText("Current Word: -")
        self.sentence_box.clear()
        self.session_word_count = 0
        self.recognizer.current_word = ""
        self.recognizer.completed_words = []
        self.setFocus()

    def _play_sound_threaded(self, filename):
        try:
            playsound(filename)
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def speak_sentence(self):
        sentence = self.recognizer.get_full_sentence()
        if sentence.strip():
            print("Speaking:", sentence)
            try:
                filename = f"tts_{uuid.uuid4().hex}.mp3"
                tts = gTTS(text=sentence, lang='en')
                tts.save(filename)
                playsound(filename)
                os.remove(filename)
                threading.Thread(target=self._play_sound_threaded, args=(filename,)).start()
            except Exception as e:
                print("TTS Error:", e)
                QMessageBox.warning(self, "TTS Error", f"Failed to generate or play speech: {e}")
    def take_snapshot(self):
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            os.makedirs("screenshots", exist_ok=True)
            filename = os.path.join("screenshots", f"snapshot_{int(time.time())}.png")
            cv2.imwrite(filename, self.last_frame)
            print(f"Snapshot saved: {filename}")

    def update_threshold(self):
        self.recognizer.letter_confirmation_threshold = self.threshold_spin.value()

    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)

        if self.detection_enabled:
            detected, current, sentence = self.recognizer.process_frame(frame)
            threshold = self.threshold_spin.value()
            max_count = threshold
            current_count = self.recognizer.letter_stability_count
            confidence = int(min(100, (current_count / max_count) * 100))

        else:
            detected, current, sentence = "-", self.recognizer.current_word, self.recognizer.completed_words
            confidence = 0

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.detection_label.setText(f"Detected Sign: {detected}")
        self.current_word_label.setText(f"Current Word: {current}")
        self.sentence_box.setText(self.recognizer.get_full_sentence())
        self.confidence_bar.setValue(confidence)

        if isinstance(sentence, list):
            sentence = " ".join(sentence)

        self.last_frame = frame.copy()
        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        minutes = elapsed // 60
        seconds = elapsed % 60
        acc = min(100, int((confidence / 100) * 90))
        self.session_word_count = len(self.recognizer.completed_words)
        self.stats_label.setText(f"Words: {self.session_word_count} | Time: {minutes:02}:{seconds:02} | Stability: {confidence}%")

    def keyPressEvent(self, event):
        if self.detection_enabled:
            if event.key() == Qt.Key_Space:
                self.recognizer.handle_space()
                self.update_frame()
            elif event.key() == Qt.Key_Backspace:
                self.recognizer.handle_backspace()
                self.update_frame()
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.recognizer.handle_enter()
                self.update_frame()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.stop_recognition()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageGUI()
    window.show()
    sys.exit(app.exec_())
