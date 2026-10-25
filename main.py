"""Headless MediaPipe-to-keyboard controller for Kool Sk8erz.

From the repository root: python src/main.py --dry-run --verbose
Quit with Ctrl+C in the terminal. Camera frames are processed, never displayed.
"""
import argparse
import csv
import itertools
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / 'hand-gesture-recognition-mediapipe'
if not (MODEL_ROOT / 'model').exists():
    MODEL_ROOT = ROOT
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
from model import KeyPointClassifier

MODEL = MODEL_ROOT / 'model/keypoint_classifier/keypoint_classifier.tflite'
LABELS = MODEL_ROOT / 'model/keypoint_classifier/keypoint_classifier_label.csv'
# These are provisional bindings: match them to the actual game's controls.
GESTURE_KEYS = {
    'Crouch': Key.down,
    'Jump': Key.space,
    'LeftSpin': Key.left,
    'RightSpin': Key.right,
}
HOLD_GESTURES = {'Crouch'}
TAP_GESTURES = {'Jump', 'LeftSpin', 'RightSpin'}


def parse_args():
    p = argparse.ArgumentParser(description='Run gesture control without video preview')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    p.add_argument('--min-detection-confidence', type=float, default=0.7)
    p.add_argument('--min-tracking-confidence', type=float, default=0.5)
    p.add_argument('--min-classification-confidence', type=float, default=0.65)
    p.add_argument('--window-size', type=int, default=5)
    p.add_argument('--min-votes', type=int, default=3)
    p.add_argument('--tap-cooldown', type=float, default=0.3)
    p.add_argument('--dry-run', action='store_true', help='Only print controls, do not press keys')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def preprocess_keypoints(image, hand_landmarks):
    height, width = image.shape[:2]
    points = [[min(max(int(pt.x * width), 0), width - 1),
               min(max(int(pt.y * height), 0), height - 1)]
              for pt in hand_landmarks.landmark]
    base_x, base_y = points[0]
    relative = [[x - base_x, y - base_y] for x, y in points]
    flattened = list(itertools.chain.from_iterable(relative))
    scale = max(map(abs, flattened), default=0)
    if not scale:
        return [0.0] * len(flattened)
    return [coordinate / scale for coordinate in flattened]


class GestureStabilizer:
    def __init__(self, window_size, min_votes):
        if window_size < 1 or not (1 <= min_votes <= window_size):
            raise ValueError('Require window_size >= min_votes >= 1')
        self.samples = deque(maxlen=window_size)
        self.min_votes = min_votes
        self.current = 'Neutral'

    def update(self, predicted):
        self.samples.append(predicted)
        value, count = Counter(self.samples).most_common(1)[0]
        if count >= self.min_votes:
            self.current = value
        return self.current

    def reset(self):
        self.samples.clear()
        self.current = 'Neutral'


class GameInputController:
    def __init__(self, dry_run=False, tap_cooldown=0.3):
        self.keyboard = None if dry_run else Controller()
        self.held = None
        self.last_gesture = 'Neutral'
        self.last_tap_at = -float('inf')
        self.tap_cooldown = tap_cooldown

    def key_down(self, key):
        if self.keyboard:
            self.keyboard.press(key)
        else:
            print(f'PRESS {key}')

    def key_up(self, key):
        if self.keyboard:
            self.keyboard.release(key)
        else:
            print(f'RELEASE {key}')

    def release_held(self):
        if self.held is not None:
            self.key_up(GESTURE_KEYS[self.held])
            self.held = None

    def update(self, gesture):
        if gesture in HOLD_GESTURES:
            if gesture != self.held:
                self.release_held()
                self.key_down(GESTURE_KEYS[gesture])
                self.held = gesture
        else:
            self.release_held()

        now = time.monotonic()
        if gesture in TAP_GESTURES and gesture != self.last_gesture:
            if now - self.last_tap_at >= self.tap_cooldown:
                key = GESTURE_KEYS[gesture]
                self.key_down(key)
                try:
                    time.sleep(0.04)
                finally:
                    self.key_up(key)
                self.last_tap_at = now
        self.last_gesture = gesture

    def stop(self):
        self.release_held()


def main():
    print('STARTING HAND GESTURE CONTROLLER', flush=True)
    args = parse_args()
    if not MODEL.exists():
        raise FileNotFoundError(f'Model not found: {MODEL}')
    if not LABELS.exists():
        raise FileNotFoundError(f'Labels not found: {LABELS}')
    with LABELS.open(encoding='utf-8-sig', newline='') as f:
        labels = [row[0].strip() for row in csv.reader(f) if row]
    if len(labels) != 5 or set(labels) != {'Crouch', 'Jump', 'Neutral', 'LeftSpin', 'RightSpin'}:
        raise ValueError(f'Unexpected labels: {labels}')
    classifier = KeyPointClassifier(model_path=str(MODEL))
    if int(classifier.output_details[0]['shape'][-1]) != len(labels):
        raise ValueError('Model output count does not match label CSV. Retrain/export the matching model.')
    stabilizer = GestureStabilizer(args.window_size, args.min_votes)
    controller = GameInputController(args.dry_run, args.tap_cooldown)
    cap = cv.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open camera {args.device}')
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    print('Camera running without preview. Ctrl+C here to quit.')
    last_printed = None
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=0,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as hands:
            while True:
                ok, image = cap.read()
                if not ok:
                    stabilizer.reset()
                    controller.update('Neutral')
                    time.sleep(0.02)
                    continue
                # Match the horizontally mirrored preprocessing used by app.py during data collection.
                image = cv.flip(image, 1)
                rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                if not result.multi_hand_landmarks:
                    stabilizer.reset()
                    controller.update('Neutral')
                    continue
                features = preprocess_keypoints(image, result.multi_hand_landmarks[0])
                # Invoke TFLite directly; compatible with both old and new classifier wrappers.
                sample = np.asarray(features, dtype=np.float32).reshape(1, 42)
                input_info = classifier.input_details[0]
                output_info = classifier.output_details[0]
                classifier.interpreter.set_tensor(input_info['index'], sample)
                classifier.interpreter.invoke()
                scores = classifier.interpreter.get_tensor(output_info['index']).reshape(-1)
                label_id = int(np.argmax(scores))
                confidence = float(scores[label_id])
                predicted = labels[label_id] if confidence >= args.min_classification_confidence else 'Neutral'
                stable = stabilizer.update(predicted)
                controller.update(stable)
                if args.verbose and (stable != last_printed):
                    print(f'gesture={stable:<10} predicted={predicted:<10} confidence={confidence:.2f}')
                    last_printed = stable
    except KeyboardInterrupt:
        print('\nStopping controller.')
    finally:
        controller.stop()
        cap.release()

if __name__ == '__main__':
    main()