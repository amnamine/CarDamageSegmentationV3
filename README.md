# Car Damage Segmentation (CODEV3)

Semantic **instance segmentation** for vehicle damage using a compact **YOLO segmentation** model ([Ultralytics](https://github.com/ultralytics/ultralytics)), trained on a multi-class car-damage dataset from [Roboflow](https://roboflow.com/). This repository bundles training artifacts, evaluation plots, and a small **desktop GUI** to run inference on photos.

---

## What this project does

- **Detects and segments** damaged regions on car images with class labels (16 damage-related classes in the trained dataset).
- Ships **`cardmgv3.pt`**: trained **YOLO26n-seg** weights used by the GUI for prediction.
- Includes **`CarDamageSegV3.ipynb`**: Colab-oriented workflow—install deps, download the dataset export, and launch `yolo train`.
- Provides **`tkinter_guiv3.py`**: load an image, run the model, and view overlaid boxes and masks.

---

## Repository layout

| Item | Description |
|------|-------------|
| `tkinter_guiv3.py` | Tkinter app; loads `cardmgv3.pt` and displays predictions |
| `cardmgv3.pt` | Fine-tuned YOLO segmentation weights |
| `CarDamageSegV3.ipynb` | Dataset download + training command (Google Colab / GPU) |
| `results.csv` | Per-epoch training and validation metrics from the last run |
| `results.png` | Ultralytics training summary plot |
| `labels.jpg` | Dataset label distribution visualization from training |
| `confusion_matrix*.png` | Validation confusion matrices (counts and normalized) |
| `Box*.png`, `Mask*.png` | Precision–recall and related curves for boxes and masks |

---

## Model and training summary

Training was performed with **Ultralytics** (example environment: Colab, **Tesla T4**, CUDA), using:

| Setting | Value |
|---------|--------|
| Base weights | `yolo26n-seg.pt` |
| Epochs | 20 |
| Image size | 640 |
| Batch size | 64 |
| AMP / half precision | Enabled (`half=true`) |
| Dataset format | YOLO26 export (Roboflow **car damage segmentation**, version 3) |

Approximate scale from training logs: **~2.7k** training images and **491** validation images (**938** instances on the val split).

**Final validation metrics** (epoch 20, from `results.csv`):

| Task | mAP50 | mAP50–95 |
|------|-------|----------|
| Bounding boxes | **0.258** | **0.139** |
| Instance masks | **0.217** | **0.100** |

These numbers reflect this specific run and split; your mileage may vary if you change data, epochs, or hyperparameters.

---

## Requirements

- **Python** 3.10+ recommended  
- **PyTorch** (CPU or CUDA; GPU speeds up training and inference)

Python packages used by the GUI and inference:

```bash
pip install ultralytics opencv-python pillow
```

For the notebook workflow you also need:

```bash
pip install roboflow
```

---

## Run the desktop GUI

1. Place **`cardmgv3.pt`** in the same directory as the script (as shipped in this repo).
2. From the project folder:

```bash
python tkinter_guiv3.py
```

3. Use **Load Image** → choose JPG/PNG/BMP → **Predict** to see boxes and masks. **Reset** clears the view.

If the weights file is missing or corrupt, the app shows an error and disables meaningful prediction until `cardmgv3.pt` is restored.

---

## Retrain or adapt (notebook)

Open `CarDamageSegV3.ipynb` in Jupyter, **Google Colab**, or VS Code. Typical steps:

1. Install `ultralytics` and `roboflow`.
2. Authenticate Roboflow and download your dataset export (the notebook targets a YOLO26-style layout and a `data.yaml`).
3. Run training, for example:

```bash
yolo train model=yolo26n-seg.pt data=/path/to/your/data.yaml epochs=20 imgsz=640 batch=64 half=true
```

Adjust paths for your machine (`/content/...` in the notebook is Colab-specific). Use your own Roboflow API key via **environment variables** or Roboflow’s documented auth—**do not commit secrets** to the notebook or git history.

After training, copy the best weights from the Ultralytics run directory (e.g. `runs/segment/train/weights/best.pt`) and rename or symlink to `cardmgv3.pt` if you want the GUI to load them without code changes.

---

## Credits

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — training and inference stack  
- **Roboflow** — dataset hosting and export tooling (see your project page for dataset license and citation)  

---

## License

This README does not set a license for third-party datasets or pretrained weights. Confirm terms for the Roboflow dataset and Ultralytics/YOLO usage in their respective documentation before redistribution or commercial use.
