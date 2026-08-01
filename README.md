# Real-Time Sign Language Recognition (A-Z)
# Dev/Creator=tubakhxn
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
