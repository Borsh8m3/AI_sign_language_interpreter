# Sign Language Translator (AI)

This repository is a small demo project for recognizing sign-language letters using MediaPipe for hand landmark detection and a scikit-learn classifier. The project includes a Streamlit UI (`main.py`) for live camera inference, a data collection script (`detection.py`), and a training script (`train_model.py`) that produces a model saved as `model.joblib`.

## Main features
- Live single-character recognition from webcam input.
- Data collection into `gestures.csv` for building training datasets.
- Training an SVM classifier and saving label mapping to `labels.json`.

**Note:** A pre-trained `model.joblib` may be included — you can use it directly without retraining.

## Requirements
- Python 3.9+ (3.10/3.11 recommended)
- Install dependencies with:

```bash
pip install -r requirements.txt
# and install Streamlit if you plan to use the UI
pip install streamlit
```

Recommended libraries: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `pyttsx3`, `streamlit`.

## Quick start
1. Clone the repository or download the files.
2. (Optional) Create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
```

## Running the app (inference / UI)
Run the Streamlit interface:

```bash
streamlit run main.py
```

Open the page in your browser, enable "Run Camera" in the sidebar and select the camera index. Adjust confidence threshold and hold duration as needed.

## Collecting data
Use `detection.py` to collect samples for a specific character:

```bash
python detection.py
# Enter the character when prompted; the script will collect samples and append them to gestures.csv
```

Each row in `gestures.csv` is saved as: label, x1, y1, z1, x2, y2, z2, ...

## Training the model
After collecting data, run:

```bash
python train_model.py
```

The training script performs a cleaning step (`gestures.csv` → `gestures_clean.csv`), trains an SVM pipeline, and writes `model.joblib` and `labels.json`.

## Files overview
- `main.py` — Streamlit app (UI and camera loop).
- `detection.py` — Data collection script.
- `train_model.py` — Data cleaning, training, and artifact saving.
- `gestures.csv` — Raw collected samples.
- `gestures_clean.csv` — Cleaned dataset used for training.
- `model.joblib` — Saved classifier model (if present).
- `labels.json` — Label list used by the model.
- `utils.py` — Helper functions (feature conversion, constants).

## Troubleshooting
- If Streamlit cannot access the camera, check OS permissions and try different `camera_index` values.
- If training reports incorrect column counts, re-run `train_model.py` — it performs a cleaning step and reports bad rows.
- Ensure `mediapipe` and `opencv-python` versions are compatible with your Python and OS.

## Next steps / improvements
- Extend recognition to phrases or continuous signing.
- Use deep learning models (RNNs/CNNs/Transformers) for higher accuracy.
- Add multi-hand support and contextual understanding.

If you want, I can also update `requirements.txt`, add a sample `.env`, or prepare a short Windows-specific setup guide.
