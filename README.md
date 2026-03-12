# Sign Language Interpreter (PyTorch, Python 3.12.7)

A fresh, capture-first project for gesture recognition with:
- OpenCV camera pipeline
- MediaPipe hand landmarks and finger/palm lines
- Square ROI on right side (covers top and bottom halves)
- Class-wise image storage under data/raw/gestures

## Dataset layout

- data/raw/gestures/A
- data/raw/gestures/B
- ...
- data/raw/gestures/0
- ...
- data/raw/gestures/Unknown
- data/raw/gestures/Backspace
- data/raw/gestures/Clear

Each class will be filled gradually by capture script.

## Setup

1. Create and activate a Python 3.12.7 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Capture images

```bash
python -m src.capture --gesture A --target 1500 --camera 0 --interval 0.04
```

For the next classes:

```bash
python -m src.capture --gesture B --target 1500 --camera 0 --interval 0.04
python -m src.capture --gesture C --target 1500 --camera 0 --interval 0.04
```

Controls:
- s: start/pause auto capture
- q: quit

Capture policy:
- Hand landmarks must be inside ROI square.
- Right hand is required by default.
- Quality gates for brightness and sharpness are applied.
- Capture interval default is 0.04 seconds (about 25 images/sec). Lower values are faster.

## View counts

```bash
python -m src.visualize_dataset
```

## Next steps

- Train classifier:

```bash
python -m src.train --epochs 12 --batch-size 32 --val-split 0.2
```

- Run live notepad inference:

```bash
python -m src.infer_live --model models/gesture_classifier.pt --camera 0 --threshold 0.65 --stability 8 --cooldown 0.55 --allow-two-hands
```

Live notepad behavior:
- Single-character labels append text
- Space inserts a blank space
- Backspace deletes one character
- Clear wipes all text (same outcome as Ctrl+A then Backspace)
