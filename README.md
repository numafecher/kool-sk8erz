# KOOL-SK8ERZ 🛹⚡
```
    __ __ ____  ____  __       _____ __ __ ___  ______  __________ ____
   / //_// __ \/ __ \/ /      / ___// //_/( _ )/ ____/ / ____/ __ \_  /
  / ,<  / / / / / / / /       \__ \/ ,<  / __ \/ __/  / /_  / /_/ // / 
 / /| |/ /_/ / /_/ / /___    ___/ / /| |/ /_/ / /___ / __/ / _, _// /_ 
/_/ |_|\____/\____/_____/   /____/_/ |_|\____/_____//_/   /_/ |_/___/ 
```

> **`[!] SYSTEM STATUS: ONLINE`** &nbsp;|&nbsp; **`DIAL-UP: 56.6 KBPS`** &nbsp;|&nbsp; **`RESOLUTION: 1024x768 32-BIT COLOR`**  
> **`PROJECT:`** HackNotts '26 &nbsp;|&nbsp; **`TARGET ARCHITECTURE:`** NEURAL-CYBER-CORE v2.0  
> **`AUDIO:`** *Now Playing: Superman - Goldfinger.mp3 via Winamp 2.8*

---

### 🌐 // 01. WHAT IS THIS CYBER-CREATION?

Welcome to **KOOL-SK8ERZ**! Forget chunky plastic controllers—it's the year 2000 and the future is *optical*. 

This is a real-time, zero-touch, camera-driven hand gesture interface designed for the **Kool Sk8erz** game engine. Powered by Google MediaPipe and a customized ultra-lean TensorFlow Lite neural net, it tracks your hand keypoints through your optical cyber-cam at lightning speed and maps your tricks straight to virtual keyboard strokes.

*No gloves. No wires. Just pure gnarly computer vision.* 🛸

---

### 🕹️ // 02. TRICK DICTIONARY & CYBER-BINDS

Throw down hand gestures in front of the lens. The neural matrix stabilizes your frames and fires the exact keystrokes:

| GESTURE NAME | RAD HAND SHAPE | VIRTUAL KEY | TRIGGER TYPE | SK8ER ACTION |
| :--- | :--- | :---: | :---: | :--- |
| **`Crouch`** | ✊ Fist down | `Key.down` (↓) | **HOLD** | Pre-load pop / duck obstacles |
| **`Jump`** | ✋ Open palm high | `Key.space` (␣) | **TAP** | Ollie into the stratosphere |
| **`LeftSpin`** | 👈 Point West | `Key.left` (←) | **TAP** | Backside 180 spin |
| **`RightSpin`** | 👉 Point East | `Key.right` (→) | **TAP** | Frontside 180 spin |
| **`Neutral`** | ✌️ Rest / Idle | *None* | **IDLE** | Cruising down the boardwalk |

> 💾 **CYBER-BUFFER STABILIZER:** The engine includes an automatic voting window (default: 5 frames, 3 votes minimum) and a `0.3s` tap cooldown so you never accidentally double-pump your ollie.

---

### 💻 // 03. SYSTEM REQUIREMENTS

* **Processor:** Intel Pentium III 800MHz or Apple Silicon G4/M-Series
* **Memory:** 128 MB RAM (512 MB recommended)
* **Optical Sensor:** USB 1.1/2.0 Webcam or built-in iSight / FaceTime camera
* **Environment:** Python 3.10+ (equipped with MediaPipe, OpenCV, and NumPy)
* **OS:** Windows 98 SE / Windows 2000 / Mac OS X Tiger / Modern macOS / Linux

---

### 🚀 // 04. BOOT SEQUENCE (HOW TO RUN)

#### STEP 1: INITIALIZE THE CYBER TERMINAL
Fire up your command prompt and step into the matrix:
```bash
# Clone the repository
git clone https://github.com/dylanlim27/kool-sk8erz.git
cd tuffsk8ers

# Mount your virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the neuro-packages
pip install -r requirements.txt
```

#### STEP 2: LAUNCH THE CONTROLLER (HEADLESS GAME ENGINE)
Run the headless driver during gameplay. It processes camera frames in the background without burning CPU cycles on a preview window:
```bash
python main.py
```
*Options:*
* `--dry-run`: Test gesture recognition in the terminal without firing actual OS keystrokes.
* `--verbose`: Print real-time detection telemetry and confidence scores.
* `--device 0`: Switch camera device index.

#### STEP 3: HEADS-UP DISPLAY & DATA COLLECTOR (DEBUG HUD)
Want to see the neural wireframe and optical tracking live on screen? Boot the HUD:
```bash
python test.py
```
* **HUD Hotkeys:**
  * `ESC` - Terminate video feed
  * `k` - Engage **Keypoint Logging Mode** (press `0`-`4` to record frames to `model/keypoint_classifier/keypoint.csv`)
  * `n` - Return to Normal Mode

---

### 🧠 // 05. RETRAINING THE NEURAL CHIP

Need to train the network on your own custom gestures? 

1. Record new hand samples in `test.py` under mode `k`.
2. Execute the re-training pipeline:
```bash
python train-keypoints.py --epochs 250
```
3. The pipeline trains with balanced class weighting, early stopping, and automatic TFLite quantization, safely generating:
   * `model/keypoint_classifier/keypoint_classifier_candidate.tflite`
4. Test the candidate model. When satisfied, promote it to active production:
```bash
cp model/keypoint_classifier/keypoint_classifier_candidate.tflite model/keypoint_classifier/keypoint_classifier.tflite
```

---

### 📁 // 06. MAINFRAME DIRECTORY TREE

```text
tuffsk8ers/
├── main.py                     # Headless live gameplay keyboard controller
├── test.py                     # Cyber HUD visualizer & keypoint recorder
├── train-keypoints.py          # Neural training & TFLite quantization script
├── requirements.txt            # Package manifest
├── README.md                   # This cyber document
│
├── model/
│   ├── tflite_interpreter.py   # Universal TFLite / NumPy fallback runtime
│   └── keypoint_classifier/
│       ├── keypoint_classifier.tflite    # Active 5-class production neural net
│       ├── keypoint_classifier_label.csv # Gesture labels (Crouch..RightSpin)
│       └── keypoint.csv                  # Raw landmark coordinate dataset
│
└── utils/
    └── cvfpscalc.py            # Optical framerate counter
```

---

### 👥 // 07. THE CREW (CONTRIBUTORS)

Constructed with pure caffeine and dial-up energy at **HackNotts '26**:

* **Numa Fecher** ([@numafecher](https://github.com/numafecher))
* **Eaint Swe** ([@freyabytes](https://github.com/freyabytes))
* **Dylan Lim** ([@dylanlim27](https://github.com/dylanlim27))

---

```
   _________________________________________________________
  /                                                         \
 |   [X] BEST VIEWED WITH NETSCAPE NAVIGATOR 4.0 OR IE 5.5   |
 |       MADE FOR HACKNOTTS '26 - KEEP ON SK8IN'             |
  \_________________________________________________________/
```