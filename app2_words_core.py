#app2_words_core.py

import cv2 as cv
import numpy as np
import mediapipe as mp
import itertools
import copy
import csv
import time

from model import KeyPointClassifier


class SignLanguageRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.keypoint_classifier = KeyPointClassifier()
        self.keypoint_classifier_labels = self._load_labels()

        # State management
        self.current_word = ""
        self.completed_words = []
        self.last_detected_letter = ""
        self.letter_stability_count = 0
        self.letter_confirmation_threshold = 30
        self.cooldown_counter = 0
        self.detection_enabled = True

    def _load_labels(self):
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            return [row[0] for row in csv.reader(f)]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_landmark_list[index][0] = point[0] - base_x
            temp_landmark_list[index][1] = point[1] - base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value == 0:
            max_value = 1
        return [n / max_value for n in temp_landmark_list]

    def handle_space(self):
        if self.current_word.strip():
            self.completed_words.append(self.current_word.strip())
            self.current_word = ""
        self.cooldown_counter = 30

    def handle_backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.completed_words:
            self.current_word = self.completed_words.pop()
        self.cooldown_counter = 30

    def handle_enter(self):
        if self.current_word.strip():
            self.completed_words.append(self.current_word.strip())
            self.current_word = ""
        self.cooldown_counter = 30

    def process_frame(self, frame):
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(image)
        detected_letter = ""

        if results.multi_hand_landmarks and self.detection_enabled:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                pre_processed = self.pre_process_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(pre_processed)
                detected_letter = self.keypoint_classifier_labels[hand_sign_id]
                confidence = 1.0
                self.update_word_building(hand_sign_id, confidence)

        return detected_letter, self.current_word, " ".join(self.completed_words)

    def get_full_sentence(self):
        return " ".join(self.completed_words + ([self.current_word] if self.current_word else []))

    def update_word_building(self, hand_sign_id, confidence):
        detected_letter = self.keypoint_classifier_labels[hand_sign_id]
        current_time = time.time()

        # Confidence check
        if confidence < 0.8:
            return

        # Cooldown handling
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # Stability tracking
        if detected_letter == self.last_detected_letter:
            self.letter_stability_count += 1
        else:
            self.letter_stability_count = 1
            self.last_detected_letter = detected_letter

        if self.letter_stability_count >= self.letter_confirmation_threshold:
            self.current_word += detected_letter
            self.letter_stability_count = 0
            self.cooldown_counter = 30
