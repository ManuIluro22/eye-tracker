NOTE: This is a continuation of https://github.com/YssfDevOps/eye-tracker-vocacions, improving on it and making it usable

# Eye Tracker for Academic Vocations

A real-time eye-tracking application to analyze implicit career interests of high-school students. By comparing where students actually look (gaze trajectory) with their explicit choices, this tool delivers richer vocational guidance.

---
## 📋 Features

- **Real-time gaze tracking** (Webcam → OpenCV → MediaPipe →Custom ML model).
- **Trajectory capture**: Press **R** to start/stop recording, overlay your gaze path on a screenshot.
- **Interactive notebooks** for data exploration & model evaluation (place all notebooks under `src/notebooks/`) .
- **Modular design**: Detectors, Predictors, Utilities & Trajectory plotting all decoupled.  
- **Recording output**: Notebooks outputs (in `media/images/`) and trajectory images (in `media/trajectories/`)  

---

## 🚀 Installation

### 1. Clone the Repository
```sh
git clone https://github.com/ManuIluro22/eye-tracker.git
cd eye-tracker
```

### 2. Set Up a Virtual Environment
```sh
python -m venv venv
```
#### Windows (PowerShell):
```sh
.\venv\Scripts\Activate
```
#### macOS/Linux:
```sh
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you have CMake and Visual Studio Build Tools installed (**for `dlib`**). Then, install the required dependencies:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## 📩 Dependencies
- `pandas` – Data processing
- `torch`, `torchvision` – Machine learning models
- `PIL` – Image processing
- `opencv-python` – Computer vision tasks
- `MediaPipe` – Face and eye tracking
- `numpy`, `matplotlib`, `scipy` – Data analysis

## ▶️ Running the Tracker
Run the main script to start eye tracking:
```sh
python main.py
```

## ⚙️ Project Structure
```
eye-tracker-vocacions/
│
├─ src/                    # All source code + assets
│   ├─ data/               # Raw & preprocessed data
│   │   ├─ face/
│   │   ├─ face_aligned/
│   │   ├─ head_pos/
│   │   ├─ l_eye/
│   │   ├─ r_eye/
│   │   ├─ positions.csv
│   │   ├─ region_map.npy
│   │   └─ region_map.png
│   │
│   ├─ media/              # Outputs: trajectory images, notebook plots
│   │   ├─ images/         # Images for notebooks
│   │   └─ trajectories/   # Saved gaze trajectory
│   │
│   ├─ menu/               # Button images for the UI
│   │
│   ├─ notebooks/          # Jupyter notebooks
│   │
│   ├─ trained_models/     # Pretrained model weights & configs
│   │   ├─ face/
│   │   ├─ face_aligned/
│   │   ├─ eyes/
│   │   └─ full/
│   │
│   ├─ data_collection.py  # Calibration & data-gathering scripts
│   ├─ realtime_tracker.py # Main eye-tracking & trajectory logic
│   ├─ trajectory.py       # Gaze-path plotting utilities
│   ├─ utils.py            # Configuration, clamping, plotting helpers
│   ├─ models.py           # Model definitions
│   ├─ gaze.py             # Detector & Predictor classes
│   ├─ config.ini          # Configuration file
│   └─ landmarks.json      # Face landmarks for facial detection
│
├─ main.py                 # Entry point (runs `realtime_tracker.py`)
├─ requirements.txt        # `pip install` list
├─ README.md               # ← You are here
└─ .gitignore
```
## 📓 Working with Notebooks

All Jupyter notebooks live in `src/notebooks/` for organization, but **to run them successfully** (so that they can `import` the project modules), you must place or copy the `.ipynb` files into the `src/` folder root before launching. For example:

```bash
cd src
cp src/notebooks/*.ipynb src/
```

## Contributing
1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Commit your changes**: `git commit -m "Add new feature"`
4. **Push to GitHub**: `git push origin feature-name`
5. **Open a Pull Request**

## License
This project is licensed under the **MIT License**.

## Contact
For questions or suggestions, reach out via GitHub Issues.

---
