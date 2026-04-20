# Car Damage Segmentation (CODEV3)

Car damage **instance segmentation** project built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).  
This repository includes:

- a notebook training workflow (`CarDamageSegV3.ipynb`)
- a desktop inference GUI (`tkinter_guiv3.py`)
- two trained checkpoints:
  - `cardmgv3_20.pt` (20 epochs)
  - `cardmgv3_100.pt` (100 epochs)
- full metrics/plots for each run in:
  - `metrics20/`
  - `metrics100/`

---

## What this project does

- Detects and segments damaged car parts/areas from input images.
- Outputs box + mask predictions using a YOLO segmentation model.
- Provides an easy Tkinter app to run inference on local images.
- Preserves training artifacts so you can compare shorter vs longer training runs.

---

## Repository structure

| Path | Description |
|------|-------------|
| `tkinter_guiv3.py` | Desktop GUI for loading an image and visualizing segmentation predictions |
| `CarDamageSegV3.ipynb` | Notebook used for dataset download and model training |
| `cardmgv3_20.pt` | Trained checkpoint after 20 epochs |
| `cardmgv3_100.pt` | Trained checkpoint after 100 epochs |
| `metrics20/results.csv` | Per-epoch train/val metrics for the 20-epoch run |
| `metrics100/results.csv` | Per-epoch train/val metrics for the 100-epoch run |
| `metrics20/*.png` | Curves, confusion matrices, and visual summaries for 20 epochs |
| `metrics100/*.png` | Curves, confusion matrices, and visual summaries for 100 epochs |

`metrics20/` and `metrics100/` each include:
- `results.png`
- `BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
- `MaskF1_curve.png`, `MaskPR_curve.png`, `MaskP_curve.png`, `MaskR_curve.png`
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `labels.jpg`
- `results.csv`

---

## Training setup (from notebook artifacts)

The notebook (`CarDamageSegV3.ipynb`) shows a Colab-style workflow:

1. install dependencies (`ultralytics`, `roboflow`)
2. download Roboflow dataset export (YOLO format)
3. train with a command similar to:

```bash
yolo train model=yolo26n-seg.pt data=/content/car-damage-segmentation-3/data.yaml epochs=20 imgsz=640 batch=64 half=true
```

Key observed training configuration:

| Setting | Value |
|---------|-------|
| Base model | `yolo26n-seg.pt` |
| Image size | 640 |
| Batch size | 64 |
| Precision | mixed precision / `half=true` |
| GPU (example in logs) | Tesla T4 |
| Classes in dataset | 16 |
| Validation split (from logs) | 491 images, 938 instances |

---

## 20 vs 100 Epoch Results

The two runs in this repo are separated cleanly by folder and checkpoint name:

- **20 epochs**: `cardmgv3_20.pt` + `metrics20/`
- **100 epochs**: `cardmgv3_100.pt` + `metrics100/`

### Final epoch comparison

| Run | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|-----|-----------|--------------|------------|---------------|
| 20 epochs (`metrics20/results.csv`, epoch 20) | 0.25816 | 0.13879 | 0.21738 | 0.10038 |
| 100 epochs (`metrics100/results.csv`, epoch 100) | 0.26871 | 0.15796 | 0.26473 | 0.13287 |

### Best achieved during training (not only last epoch)

| Run | Best Box mAP50-95 (epoch) | Best Mask mAP50-95 (epoch) |
|-----|---------------------------|----------------------------|
| 20 epochs | 0.13990 (epoch 18) | 0.10285 (epoch 18) |
| 100 epochs | 0.16549 (epoch 84) | 0.14253 (epoch 80) |

### Practical interpretation

- The **100-epoch model** is clearly stronger on segmentation quality metrics, especially masks (`mAP50-95(M)` improvement from `0.10038` to `0.13287` at final epoch).
- The 100-epoch run also has higher best peaks, suggesting better convergence with longer training.
- If you want better accuracy, prefer `cardmgv3_100.pt`.
- If you need faster experimentation and smaller training budget, `cardmgv3_20.pt` remains a valid lightweight baseline.

---

## Running inference (GUI)

### 1) Install requirements

```bash
pip install ultralytics opencv-python pillow
```

### 2) Choose which checkpoint the GUI loads

The current `tkinter_guiv3.py` loads this fixed filename:

```python
self.model = YOLO("cardmgv3.pt")
```

Since this repo now stores `cardmgv3_20.pt` and `cardmgv3_100.pt`, you have two options:

- **Option A (no code change):** duplicate/rename your chosen file to `cardmgv3.pt`
- **Option B (recommended):** edit `tkinter_guiv3.py` and replace the filename with:
  - `cardmgv3_20.pt`, or
  - `cardmgv3_100.pt`

### 3) Launch app

```bash
python tkinter_guiv3.py
```

### 4) Use app

- `Load Image` to select a JPG/JPEG/PNG/BMP
- `Predict` to run segmentation and display overlays
- `Reset` to clear the canvas

If model loading fails, the app shows a message box and prediction is disabled until a valid checkpoint is provided.

---

## Re-training notes

- Use `CarDamageSegV3.ipynb` as the reference training pipeline.
- For local training, adapt Colab paths (`/content/...`) to local filesystem paths.
- Keep each experiment in separate output folders (as done with `metrics20/` and `metrics100/`) for clean comparison.
- Track both final and best metrics from `results.csv`, not only final epoch.

---

## Important security note

The notebook currently contains a hard-coded Roboflow API key in source cells.  
Best practice is to rotate that key and use environment variables instead of committing secrets to git.

---

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for training/inference framework
- Roboflow for dataset management and export tooling

---

## License and usage

This repository README does not itself define dataset/model redistribution rights.  
Please verify:

- dataset license and terms from the Roboflow project
- Ultralytics/YOLO licensing terms

before commercial usage or redistribution.
