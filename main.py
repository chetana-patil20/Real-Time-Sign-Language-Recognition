# dev/creator=tubakhxn

import os
import sys
import math
import time
import json
import queue
import ctypes
import random
import shutil
import logging
import threading
import subprocess
import importlib
import collections
import datetime

REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
]


def ensure_dependencies():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    for module_name, _ in REQUIRED_PACKAGES:
        importlib.import_module(module_name)


ensure_dependencies()

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
RECORDING_DIR = os.path.join(OUTPUT_DIR, "recordings")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_classifier.pt")
README_PATH = os.path.join(BASE_DIR, "README.md")

for directory in (MODEL_DIR, OUTPUT_DIR, SCREENSHOT_DIR, RECORDING_DIR, LOG_DIR):
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "session.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LABELS = [chr(ord("A") + i) for i in range(26)] + ["SPACE", "DELETE", "NOTHING", "UNKNOWN", "IDLE"]
NUM_CLASSES = len(LABELS)

FINGER_CURL_TEMPLATES = {
    "A": [1.0, 1.0, 1.0, 1.0, 1.0],
    "B": [1.0, 0.0, 0.0, 0.0, 0.0],
    "C": [0.4, 0.4, 0.4, 0.4, 0.4],
    "D": [0.6, 0.0, 1.0, 1.0, 1.0],
    "E": [1.0, 1.0, 1.0, 1.0, 1.0],
    "F": [0.2, 0.2, 0.0, 0.0, 0.0],
    "G": [0.3, 0.1, 1.0, 1.0, 1.0],
    "H": [0.3, 0.1, 0.1, 1.0, 1.0],
    "I": [1.0, 1.0, 1.0, 1.0, 0.0],
    "J": [1.0, 1.0, 1.0, 1.0, 0.0],
    "K": [0.2, 0.0, 0.0, 1.0, 1.0],
    "L": [0.0, 0.0, 1.0, 1.0, 1.0],
    "M": [0.5, 1.0, 1.0, 1.0, 0.9],
    "N": [0.5, 1.0, 1.0, 0.9, 1.0],
    "O": [0.5, 0.5, 0.5, 0.5, 0.5],
    "P": [0.2, 0.1, 0.1, 1.0, 1.0],
    "Q": [0.2, 0.1, 1.0, 1.0, 1.0],
    "R": [1.0, 0.05, 0.05, 1.0, 1.0],
    "S": [1.0, 1.0, 1.0, 1.0, 1.0],
    "T": [0.3, 1.0, 1.0, 1.0, 1.0],
    "U": [1.0, 0.0, 0.0, 1.0, 1.0],
    "V": [1.0, 0.0, 0.0, 1.0, 1.0],
    "W": [1.0, 0.0, 0.0, 0.0, 1.0],
    "X": [0.4, 0.5, 1.0, 1.0, 1.0],
    "Y": [0.0, 1.0, 1.0, 1.0, 0.0],
    "Z": [0.4, 0.0, 1.0, 1.0, 1.0],
    "SPACE": [0.0, 0.0, 0.0, 0.0, 0.0],
    "DELETE": [0.5, 0.0, 1.0, 1.0, 1.0],
    "NOTHING": [1.0, 1.0, 1.0, 1.0, 1.0],
    "UNKNOWN": [0.5, 0.5, 0.5, 0.5, 0.5],
    "IDLE": [0.8, 0.8, 0.8, 0.8, 0.8],
}

FINGER_SPREAD_TEMPLATES = {
    "B": 0.6, "F": 0.3, "K": 0.7, "U": 0.15, "V": 0.65, "W": 0.55,
    "Y": 0.5, "H": 0.35, "N": 0.1, "M": 0.1, "R": 0.05,
}

FINGER_BASE_OFFSETS = np.array([
    [-0.09, -0.02, 0.0],
    [-0.045, -0.02, 0.0],
    [0.0, -0.02, 0.0],
    [0.045, -0.02, 0.0],
    [0.085, -0.02, 0.0],
])

FINGER_SEGMENT_LENGTHS = [
    [0.05, 0.04, 0.03],
    [0.11, 0.06, 0.045],
    [0.12, 0.065, 0.05],
    [0.11, 0.06, 0.045],
    [0.09, 0.05, 0.04],
]

FINGER_BASE_ANGLES = [-0.55, -0.12, 0.0, 0.12, 0.28]


def synthesize_hand_landmarks(curl_vector, spread_factor=0.0, rotation=0.0, scale=1.0,
                               translation=(0.5, 0.6), noise_std=0.01):
    wrist = np.array([0.0, 0.0, 0.0])
    landmarks = [wrist]
    for finger_index in range(5):
        base = FINGER_BASE_OFFSETS[finger_index].copy()
        direction_angle = FINGER_BASE_ANGLES[finger_index]
        spread = spread_factor * (0.4 if finger_index in (0, 4) else 1.0)
        angle = direction_angle - spread * (1 if finger_index < 2 else -1 if finger_index > 2 else 0)
        curl = curl_vector[finger_index]
        pos = base.copy()
        landmarks.append(pos.copy())
        cumulative_angle = angle
        for segment_index, segment_length in enumerate(FINGER_SEGMENT_LENGTHS[finger_index]):
            bend = curl * (math.pi / 2.05) * (0.55 if segment_index == 0 else 1.0)
            cumulative_angle += bend
            dx = math.sin(cumulative_angle) * segment_length
            dy = -math.cos(cumulative_angle) * segment_length
            dz = curl * 0.015 * (segment_index + 1)
            pos = pos + np.array([dx, dy, dz])
            landmarks.append(pos.copy())
    landmarks = np.array(landmarks, dtype=np.float32)
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    landmarks[:, :2] = landmarks[:, :2] @ rotation_matrix.T
    landmarks *= scale
    landmarks[:, 0] += translation[0]
    landmarks[:, 1] += translation[1]
    if noise_std > 0:
        landmarks += np.random.normal(0, noise_std, landmarks.shape)
    return landmarks


def compute_finger_states(landmarks):
    landmarks = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = np.linalg.norm(middle_mcp - wrist) + 1e-6
    palm_normal_ref = landmarks[5] - landmarks[17]

    def joint_angle(a, b, c):
        v1 = landmarks[a] - landmarks[b]
        v2 = landmarks[c] - landmarks[b]
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return math.degrees(math.acos(cosang))

    finger_joint_ids = {
        "thumb": (2, 3, 4),
        "index": (5, 6, 8),
        "middle": (9, 10, 12),
        "ring": (13, 14, 16),
        "pinky": (17, 18, 20),
    }
    extension = {}
    for name, (mcp, pip, tip) in finger_joint_ids.items():
        angle = joint_angle(mcp, pip, tip)
        extension[name] = 1.0 if angle > 150 else 0.0 if angle < 100 else (angle - 100) / 50.0

    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    thumb_to_index_mcp = np.linalg.norm(thumb_tip - index_mcp) / palm_size
    thumb_across_palm = thumb_to_index_mcp < 0.55

    tip_ids = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    tips_up = {
        name: (landmarks[tip_ids[name]][1] < landmarks[0][1]) for name in tip_ids
    }

    spread_index_middle = np.linalg.norm(landmarks[8] - landmarks[12]) / palm_size
    spread_middle_ring = np.linalg.norm(landmarks[12] - landmarks[16]) / palm_size
    spread_ring_pinky = np.linalg.norm(landmarks[16] - landmarks[20]) / palm_size
    thumb_index_tip_distance = np.linalg.norm(landmarks[4] - landmarks[8]) / palm_size

    return {
        "extension": extension,
        "thumb_across_palm": thumb_across_palm,
        "tips_up": tips_up,
        "spread_index_middle": spread_index_middle,
        "spread_middle_ring": spread_middle_ring,
        "spread_ring_pinky": spread_ring_pinky,
        "thumb_index_tip_distance": thumb_index_tip_distance,
        "palm_size": palm_size,
    }


def classify_gesture_rule_based(landmarks):
    state = compute_finger_states(landmarks)
    ext = state["extension"]
    thumb_ext = ext["thumb"] > 0.55
    index_ext = ext["index"] > 0.55
    middle_ext = ext["middle"] > 0.55
    ring_ext = ext["ring"] > 0.55
    pinky_ext = ext["pinky"] > 0.55
    index_curl = ext["index"] < 0.45
    middle_curl = ext["middle"] < 0.45
    ring_curl = ext["ring"] < 0.45
    pinky_curl = ext["pinky"] < 0.45
    thumb_curl = ext["thumb"] < 0.45
    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
    curled_count = sum([index_curl, middle_curl, ring_curl, pinky_curl])
    pinch = state["thumb_index_tip_distance"] < 0.35
    thumb_over_fingers = state["thumb_across_palm"]

    candidates = []

    if index_ext and middle_curl and ring_curl and pinky_curl and not thumb_ext:
        candidates.append(("D", 0.92))
    if index_ext and middle_curl and ring_curl and pinky_curl and thumb_ext:
        candidates.append(("L", 0.9))
    if index_ext and middle_ext and ring_curl and pinky_curl:
        gap = state["spread_index_middle"]
        if gap > 0.32:
            candidates.append(("V", 0.88))
        elif thumb_ext:
            candidates.append(("K", 0.8))
        elif gap < 0.15:
            candidates.append(("R", 0.75))
        else:
            candidates.append(("U", 0.85))
    if index_ext and middle_ext and ring_ext and not pinky_ext:
        candidates.append(("W", 0.88))
    if extended_count == 4 and not thumb_ext:
        candidates.append(("B", 0.92))
    if curled_count == 4:
        if thumb_over_fingers and not thumb_ext:
            candidates.append(("S", 0.88))
        elif thumb_ext and not thumb_over_fingers:
            candidates.append(("A", 0.88))
        elif thumb_ext and thumb_over_fingers and pinch:
            candidates.append(("T", 0.7))
        elif thumb_curl and thumb_over_fingers:
            candidates.append(("N", 0.55))
        elif thumb_curl:
            candidates.append(("M", 0.5))
    if pinky_ext and thumb_ext and index_curl and middle_curl and ring_curl:
        candidates.append(("Y", 0.9))
    if index_ext and pinky_ext and not middle_ext and not ring_ext and not thumb_ext:
        candidates.append(("H", 0.55))
    if not index_ext and pinky_ext and middle_curl and ring_curl and thumb_curl:
        candidates.append(("I", 0.85))
    if pinch and middle_ext and ring_ext and pinky_ext:
        candidates.append(("F", 0.85))
    if pinch and middle_curl and ring_curl and pinky_curl:
        candidates.append(("O", 0.8))
    if extended_count == 0 and not thumb_ext and not pinch:
        candidates.append(("C", 0.55))
    if extended_count == 0 and thumb_ext and not thumb_over_fingers:
        candidates.append(("E", 0.5))
    if index_ext and middle_curl and ring_curl and pinky_curl and thumb_ext and state["spread_index_middle"] < 0.2:
        candidates.append(("X", 0.5))
    if extended_count == 4 and thumb_ext:
        candidates.append(("NOTHING", 0.4))

    if not candidates:
        return "UNKNOWN", 0.25
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates[0]


def extract_feature_vector(landmarks):
    landmarks = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = landmarks[0].copy()
    relative = landmarks - wrist
    middle_mcp = landmarks[9]
    palm_size = np.linalg.norm(middle_mcp - wrist) + 1e-6
    normalized = relative / palm_size
    flattened = normalized.flatten()

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    finger_mcps = [1, 5, 9, 13, 17]
    curls = []
    for tip_idx, pip_idx, mcp_idx in zip(finger_tips, finger_pips, finger_mcps):
        tip_to_mcp = np.linalg.norm(landmarks[tip_idx] - landmarks[mcp_idx])
        pip_to_mcp = np.linalg.norm(landmarks[pip_idx] - landmarks[mcp_idx]) + 1e-6
        ratio = 1.0 - min(tip_to_mcp / (pip_to_mcp * 2.2), 1.0)
        curls.append(max(0.0, min(1.0, ratio)))
    curls = np.array(curls, dtype=np.float32)

    spreads = []
    for i in range(4):
        distance = np.linalg.norm(landmarks[finger_tips[i]] - landmarks[finger_tips[i + 1]]) / palm_size
        spreads.append(distance)
    spreads = np.array(spreads, dtype=np.float32)

    feature = np.concatenate([flattened, curls, spreads]).astype(np.float32)
    return feature


FEATURE_DIM = extract_feature_vector(synthesize_hand_landmarks([0.5] * 5)).shape[0]


class GestureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def generate_synthetic_dataset(samples_per_class=400):
    features = []
    targets = []
    for label_index, label in enumerate(LABELS):
        curl_template = np.array(FINGER_CURL_TEMPLATES[label], dtype=np.float32)
        spread_template = FINGER_SPREAD_TEMPLATES.get(label, 0.2)
        for _ in range(samples_per_class):
            curl_noise = np.random.normal(0, 0.06, 5)
            curl_vector = np.clip(curl_template + curl_noise, 0.0, 1.0)
            spread = max(0.0, spread_template + np.random.normal(0, 0.05))
            rotation = np.random.uniform(-0.35, 0.35)
            scale = np.random.uniform(0.85, 1.2)
            translation = (0.5 + np.random.uniform(-0.05, 0.05), 0.6 + np.random.uniform(-0.05, 0.05))
            landmarks = synthesize_hand_landmarks(
                curl_vector, spread_factor=spread, rotation=rotation,
                scale=scale, translation=translation, noise_std=0.008,
            )
            feature = extract_feature_vector(landmarks)
            features.append(feature)
            targets.append(label_index)
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int64)


def train_gesture_classifier(device):
    print("Training gesture classifier on synthesized landmark data...")
    features, targets = generate_synthetic_dataset()
    x_train, x_val, y_train, y_val = train_test_split(
        features, targets, test_size=0.15, random_state=42, stratify=targets
    )
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std

    model = GestureClassifier(FEATURE_DIM, NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    x_val_tensor = torch.tensor(x_val_norm, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    batch_size = 128
    num_samples = x_train_tensor.shape[0]
    epochs = 40
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        for start in range(0, num_samples, batch_size):
            indices = permutation[start:start + batch_size]
            batch_x = x_train_tensor[indices]
            batch_y = y_train_tensor[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1)
            accuracy = (val_predictions == y_val_tensor).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss / num_samples:.4f} val_acc={accuracy:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "labels": LABELS,
        "feature_dim": FEATURE_DIM,
    }, MODEL_PATH)
    print(f"Model trained with best validation accuracy {best_accuracy:.4f} and saved to {MODEL_PATH}")
    return model, mean, std


def load_or_train_classifier(device):
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model = GestureClassifier(checkpoint["feature_dim"], len(checkpoint["labels"])).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint["mean"], checkpoint["std"]
    model, mean, std = train_gesture_classifier(device)
    model.eval()
    return model, mean, std


class PredictionSmoother:
    def __init__(self, buffer_size=15, confidence_threshold=0.5, stability_frames=10):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.stable_label = None
        self.stable_count = 0
        self.already_committed_this_hold = False

    def update(self, label, confidence):
        if confidence < self.confidence_threshold:
            label = "UNKNOWN"
        self.buffer.append(label)
        counts = collections.Counter(self.buffer)
        majority_label, majority_count = counts.most_common(1)[0]
        stability_ratio = majority_count / max(1, len(self.buffer))

        if majority_label == self.stable_label:
            self.stable_count += 1
        else:
            self.stable_label = majority_label
            self.stable_count = 1
            self.already_committed_this_hold = False

        committed = None
        if (
            self.stable_count >= self.stability_frames
            and stability_ratio >= 0.6
            and majority_label not in ("UNKNOWN", "NOTHING", "IDLE")
            and not self.already_committed_this_hold
        ):
            committed = majority_label
            self.already_committed_this_hold = True

        return majority_label, stability_ratio, committed

    def reset(self):
        self.buffer.clear()
        self.stable_label = None
        self.stable_count = 0
        self.already_committed_this_hold = False


class SentenceBuilder:
    def __init__(self):
        self.sentence = ""
        self.history = collections.deque(maxlen=200)
        self.letter_frequency = collections.Counter()
        self.capitalize_next = True

    def add_character(self, label):
        if label == "SPACE":
            self.sentence += " "
            self.capitalize_next = True
        elif label == "DELETE":
            self.sentence = self.sentence[:-1]
        elif len(label) == 1 and label.isalpha():
            character = label.upper() if self.capitalize_next else label.lower()
            self.sentence += character
            self.capitalize_next = False
            self.letter_frequency[label] += 1
        self.history.append((label, time.time()))

    def clear(self):
        self.sentence = ""
        self.capitalize_next = True

    def copy_to_clipboard(self):
        try:
            if sys.platform == "darwin":
                subprocess.run("pbcopy", universal_newlines=True, input=self.sentence)
            elif sys.platform.startswith("win"):
                subprocess.run("clip", universal_newlines=True, input=self.sentence, shell=True)
            else:
                subprocess.run(["xclip", "-selection", "clipboard"], input=self.sentence.encode(), check=False)
            return True
        except Exception:
            return False

    def export(self):
        export_path = os.path.join(OUTPUT_DIR, f"sentence_{int(time.time())}.txt")
        with open(export_path, "w") as file_handle:
            file_handle.write(self.sentence)
        return export_path


class AnalyticsTracker:
    def __init__(self):
        self.session_start = time.time()
        self.fps_history = collections.deque(maxlen=60)
        self.frame_count = 0
        self.recognition_events = 0
        self.letter_frequency = collections.Counter()
        self.hand_detection_count = 0

    def record_frame(self, fps):
        self.frame_count += 1
        self.fps_history.append(fps)

    def record_recognition(self, label):
        self.recognition_events += 1
        if len(label) == 1:
            self.letter_frequency[label] += 1

    def average_fps(self):
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def session_duration(self):
        return time.time() - self.session_start

    def most_recognized(self, top_n=5):
        return self.letter_frequency.most_common(top_n)


class WebcamStream:
    def __init__(self, source=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(source, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None
        self.available = self.capture.isOpened()

    def start(self):
        if not self.available:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            success, frame = self.capture.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()


class UIRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.panel_color = (24, 24, 32)
        self.accent_color = (255, 140, 0)
        self.text_color = (235, 235, 245)
        self.success_color = (80, 220, 140)
        self.warning_color = (240, 90, 90)

    def draw_glass_panel(self, frame, x, y, w, h, alpha=0.55):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.panel_color, -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.accent_color, 1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_confidence_bar(self, frame, x, y, w, h, confidence, label):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 70), -1)
        fill_width = int(w * max(0.0, min(1.0, confidence)))
        color = self.success_color if confidence > 0.7 else self.accent_color if confidence > 0.4 else self.warning_color
        cv2.rectangle(frame, (x, y), (x + fill_width, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 210), 1)
        cv2.putText(frame, f"{label} {confidence * 100:.0f}%", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1, cv2.LINE_AA)

    def draw_landmarks(self, frame, landmarks_px, connections, show_skeleton=True, show_points=True):
        if show_skeleton:
            for start_idx, end_idx in connections:
                cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx], (0, 210, 255), 2, cv2.LINE_AA)
        if show_points:
            palette = [(255, 80, 80), (255, 200, 60), (120, 255, 90), (90, 200, 255), (200, 120, 255)]
            for idx, point in enumerate(landmarks_px):
                color = palette[idx % len(palette)]
                cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, point, 6, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_bounding_box(self, frame, x1, y1, x2, y2, label, confidence):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 255, 120), 2)
        tag = f"{label} {confidence * 100:.0f}%"
        (text_width, text_height), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 12), (x1 + text_width + 10, y1), (60, 255, 120), -1)
        cv2.putText(frame, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 20, 10), 2, cv2.LINE_AA)

    def draw_hud(self, frame, state):
        self.draw_glass_panel(frame, 10, 10, 340, 190)
        cv2.putText(frame, "SIGN LANGUAGE RECOGNITION", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    self.accent_color, 2, cv2.LINE_AA)
        lines = [
            f"FPS: {state['fps']:.1f}  AVG: {state['avg_fps']:.1f}",
            f"Latency: {state['latency_ms']:.1f} ms",
            f"Frame: {state['frame_count']}",
            f"Hands: {state['hand_count']}  Tracking: {state['tracking_status']}",
            f"Detection: {state['detection_status']}",
            f"Time: {state['timestamp']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (24, 66 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        self.text_color, 1, cv2.LINE_AA)

        panel_x = self.width - 380
        self.draw_glass_panel(frame, panel_x, 10, 370, 210)
        cv2.putText(frame, "RECOGNITION", (panel_x + 14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    self.accent_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Current: {state['current_label']}", (panel_x + 14, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.success_color, 2, cv2.LINE_AA)
        self.draw_confidence_bar(frame, panel_x + 14, 92, 340, 16, state['confidence'], "Confidence")
        self.draw_confidence_bar(frame, panel_x + 14, 130, 340, 16, state['stability'], "Stability")
        cv2.putText(frame, f"Word buffer: {state['word_buffer']}", (panel_x + 14, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hand confidence: {state['hand_confidence'] * 100:.0f}%", (panel_x + 14, 194),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.text_color, 1, cv2.LINE_AA)

        bottom_y = self.height - 110
        self.draw_glass_panel(frame, 10, bottom_y, self.width - 20, 100)
        cv2.putText(frame, "SENTENCE", (24, bottom_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    self.accent_color, 2, cv2.LINE_AA)
        display_sentence = state['sentence'][-90:]
        cv2.putText(frame, display_sentence, (24, bottom_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    self.text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"History: {state['history_preview']}", (24, bottom_y + 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 180), 1, cv2.LINE_AA)

        controls = "P Pause  S Screenshot  R Record  L Landmarks  K Skeleton  H HUD  C Clear  X Copy  Q Exit"
        cv2.putText(frame, controls, (24, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (150, 150, 160), 1, cv2.LINE_AA)

        if state.get("paused"):
            cv2.putText(frame, "PAUSED", (self.width // 2 - 80, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, self.warning_color, 3, cv2.LINE_AA)


class SignLanguageRecognitionApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model, self.feature_mean, self.feature_std = load_or_train_classifier(self.device)

        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.hand_connections = list(self.mp_hands.HAND_CONNECTIONS)

        self.stream = WebcamStream().start()
        if not self.stream.available:
            print("Webcam not detected. Please connect a webcam and rerun.")
            sys.exit(1)

        self.smoother = PredictionSmoother(buffer_size=12, confidence_threshold=0.5, stability_frames=8)
        self.sentence_builder = SentenceBuilder()
        self.analytics = AnalyticsTracker()
        self.ui = UIRenderer(1280, 720)

        self.show_landmarks = True
        self.show_skeleton = True
        self.show_hud = True
        self.paused = False
        self.recording = False
        self.no_hand_since = None
        self.auto_space_inserted = True
        self.video_writer = None
        self.last_frame_time = time.time()
        self.frame_count = 0

    def classify(self, landmarks):
        label, confidence = classify_gesture_rule_based(landmarks)
        if label != "UNKNOWN":
            return label, confidence

        feature = extract_feature_vector(landmarks)
        normalized = (feature - self.feature_mean) / self.feature_std
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        fallback_label = LABELS[predicted_index]
        fallback_confidence = float(probabilities[predicted_index]) * 0.6
        return fallback_label, fallback_confidence

    def handle_key(self, key):
        if key == ord("q"):
            return False
        elif key == ord("p"):
            self.paused = not self.paused
        elif key == ord("s"):
            self.save_screenshot()
        elif key == ord("r"):
            self.toggle_recording()
        elif key == ord("l"):
            self.show_landmarks = not self.show_landmarks
        elif key == ord("k"):
            self.show_skeleton = not self.show_skeleton
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("c"):
            self.sentence_builder.clear()
        elif key == ord("x"):
            self.sentence_builder.copy_to_clipboard()
        elif key == ord("e"):
            self.sentence_builder.export()
        return True

    def save_screenshot(self, frame=None):
        if frame is None:
            return
        path = os.path.join(SCREENSHOT_DIR, f"screenshot_{int(time.time())}.png")
        cv2.imwrite(path, frame)
        logging.info(f"Screenshot saved to {path}")

    def toggle_recording(self):
        self.recording = not self.recording
        if not self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def ensure_video_writer(self, frame):
        if self.recording and self.video_writer is None:
            path = os.path.join(RECORDING_DIR, f"recording_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (width, height))

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        self.ui.width, self.ui.height = width, height
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)

        current_label = "IDLE"
        confidence = 0.0
        stability = 0.0
        hand_confidence = 0.0
        hand_count = 0
        detection_status = "No Hand"
        tracking_status = "Idle"
        committed_label = None

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            detection_status = "Tracking"
            tracking_status = "Active"
            self.no_hand_since = None
            self.auto_space_inserted = False
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0]
            hand_confidence = handedness.classification[0].score
            hand_label = handedness.classification[0].label

            landmarks_array = np.array(
                [[point.x, point.y, point.z] for point in hand_landmarks.landmark], dtype=np.float32
            )
            current_label, confidence = self.classify(landmarks_array)
            majority_label, stability, committed_label = self.smoother.update(current_label, confidence)
            current_label = majority_label

            xs = landmarks_array[:, 0] * width
            ys = landmarks_array[:, 1] * height
            landmarks_px = [(int(x), int(y)) for x, y in zip(xs, ys)]

            if self.show_landmarks or self.show_skeleton:
                self.ui.draw_landmarks(frame, landmarks_px, self.hand_connections,
                                        self.show_skeleton, self.show_landmarks)

            x1, y1 = int(xs.min()) - 20, int(ys.min()) - 20
            x2, y2 = int(xs.max()) + 20, int(ys.max()) + 20
            self.ui.draw_bounding_box(frame, max(0, x1), max(0, y1), min(width, x2), min(height, y2),
                                       f"{hand_label} | {current_label}", confidence)
        else:
            self.smoother.reset()
            if self.no_hand_since is None:
                self.no_hand_since = time.time()
            elif not self.auto_space_inserted and time.time() - self.no_hand_since > 0.7:
                if self.sentence_builder.sentence and not self.sentence_builder.sentence.endswith(" "):
                    self.sentence_builder.add_character("SPACE")
                self.auto_space_inserted = True

        if committed_label:
            self.sentence_builder.add_character(committed_label)
            self.analytics.record_recognition(committed_label)
            logging.info(f"Committed character: {committed_label}")

        now = time.time()
        frame_time = now - self.last_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.last_frame_time = now
        self.analytics.record_frame(fps)
        self.frame_count += 1

        recent_history = "".join(
            label if len(label) == 1 else "_" for label, _ in list(self.sentence_builder.history)[-15:]
        )

        state = {
            "fps": fps,
            "avg_fps": self.analytics.average_fps(),
            "latency_ms": frame_time * 1000,
            "frame_count": self.frame_count,
            "hand_count": hand_count,
            "tracking_status": tracking_status,
            "detection_status": detection_status,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "current_label": current_label,
            "confidence": confidence,
            "stability": stability,
            "hand_confidence": hand_confidence,
            "word_buffer": current_label if current_label not in ("IDLE", "UNKNOWN", "NOTHING") else "",
            "sentence": self.sentence_builder.sentence,
            "history_preview": recent_history,
            "paused": self.paused,
        }

        if self.show_hud:
            self.ui.draw_hud(frame, state)

        return frame

    def run(self):
        window_name = "Real-Time Sign Language Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        try:
            while True:
                raw_frame = self.stream.read()
                if raw_frame is None:
                    time.sleep(0.005)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                if self.paused:
                    display_frame = raw_frame.copy()
                    if self.show_hud:
                        self.ui.draw_hud(display_frame, {
                            "fps": 0, "avg_fps": self.analytics.average_fps(), "latency_ms": 0,
                            "frame_count": self.frame_count, "hand_count": 0, "tracking_status": "Paused",
                            "detection_status": "Paused", "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "current_label": "PAUSED", "confidence": 0, "stability": 0, "hand_confidence": 0,
                            "word_buffer": "", "sentence": self.sentence_builder.sentence,
                            "history_preview": "", "paused": True,
                        })
                else:
                    display_frame = self.process_frame(raw_frame)
                    self.ensure_video_writer(display_frame)
                    if self.recording and self.video_writer is not None:
                        self.video_writer.write(display_frame)

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key == ord("s"):
                        self.save_screenshot(display_frame)
                    if not self.handle_key(key):
                        break
        finally:
            self.shutdown()

    def shutdown(self):
        self.stream.stop()
        self.hands_detector.close()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

        summary = {
            "session_duration_seconds": self.analytics.session_duration(),
            "average_fps": self.analytics.average_fps(),
            "total_frames": self.analytics.frame_count,
            "recognition_events": self.analytics.recognition_events,
            "most_recognized": self.analytics.most_recognized(),
            "final_sentence": self.sentence_builder.sentence,
        }
        summary_path = os.path.join(LOG_DIR, f"session_summary_{int(time.time())}.json")
        with open(summary_path, "w") as file_handle:
            json.dump(summary, file_handle, indent=2)
        logging.info(f"Session ended. Summary saved to {summary_path}")
        print(f"Session summary saved to {summary_path}")


README_TEMPLATE = """# Real-Time Sign Language Recognition (A-Z)

## Overview
A real-time computer vision application that recognizes American Sign Language
hand alphabets using a webcam. The pipeline detects a hand with MediaPipe Hands,
extracts and normalizes 21 landmark points, classifies the gesture with a
PyTorch neural network, applies temporal smoothing and majority voting for
stable predictions, and renders a professional dark-themed HUD with live
sentence building.

## Features
- Real-time hand detection and 21-point landmark tracking
- Left/right handedness and multi-hand support
- Neural network gesture classifier with confidence scoring
- Temporal smoothing, majority voting, debounce and stability filtering
- Live sentence builder with space, delete, clear, capitalization and export
- Dark glass-style HUD with animated confidence bars and analytics dashboard
- Screenshot capture and video recording
- Automatic dependency installation and automatic model training on first run

## Technology Stack
Python, OpenCV, MediaPipe Hands, NumPy, PyTorch, Scikit-Learn, Pillow, SciPy,
Matplotlib, tqdm

## Installation
No manual installation is required. Dependencies install automatically on
first run.

```
python sign_language_recognition.py
```

## Usage
Run the script and show ASL hand shapes to the webcam. Hold a gesture steady
until it is committed to the sentence. Controls:

- P: Pause / Resume
- S: Screenshot
- R: Start / Stop recording
- L: Toggle landmark points
- K: Toggle skeleton
- H: Toggle HUD
- C: Clear sentence
- X: Copy sentence to clipboard
- E: Export sentence
- Q: Exit

## Folder Structure
```
sign_language_recognition.py
models/gesture_classifier.pt
output/screenshots/
output/recordings/
output/logs/
README.md
```

## Recognition Pipeline
1. Threaded webcam capture for low latency
2. MediaPipe Hands landmark extraction
3. Feature normalization (translation and scale invariant)
4. PyTorch MLP gesture classification with confidence scores
5. Temporal buffer, majority voting, and stability gating
6. Sentence construction with debounce to prevent duplicate characters

## Performance Highlights
- Multi-threaded capture pipeline for high FPS
- CPU compatible with optional CUDA acceleration
- Adaptive rendering with minimal per-frame overhead

## Example Output
The application displays a live annotated video feed with hand skeleton
overlays, a recognition panel showing the current character and confidence,
and a sentence bar showing the decoded text in real time.

## License
MIT License

## Developer
dev/creator=tubakhxn
"""


def write_readme():
    if not os.path.exists(README_PATH):
        with open(README_PATH, "w") as file_handle:
            file_handle.write(README_TEMPLATE)


def main():
    write_readme()
    app = SignLanguageRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()