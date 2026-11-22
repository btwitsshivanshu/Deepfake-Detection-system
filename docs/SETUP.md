# Setup & Run Guide

Complete instructions to run AuthentiScan (Deep Fake Detection System) locally.

---

## Prerequisites
- Python 3.10+ (tested on 3.13)
- Node.js 18+ (tested on 22)
- MySQL 8
- *(Optional)* NVIDIA GPU + recent driver for CUDA-accelerated training (tested on an
  RTX 4050 Laptop, driver 591.44). CPU works too, just slower.

---

## 1. Clone & environment
```powershell
cd d:\projects\Deepfake_finalyear
Copy-Item .env.example .env      # then edit DB creds + SECRET_KEY
```

Generate a strong secret key:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```
Paste the output as `SECRET_KEY` in `.env`.

---

## 2. Database
```powershell
mysql -u root -p < database/schema.sql
```
This creates `deepfake_db`, all six tables, a default admin, and seeds the model registry.

> Default admin login: `admin@deepfake.local`. The seeded bcrypt hash is a placeholder —
> register a fresh admin or update the hash before real use.

---

## 3. Python dependencies (ML + backend)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 3b. (Optional) Enable GPU / CUDA
`requirements.txt` installs a CPU build of PyTorch. For GPU training, replace it with a
CUDA build (cu128, tested with an RTX 4050), then set `DEVICE=cuda` in `.env`:
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
> A plain `pip install torch` will **not** replace an existing `+cpu` build — uninstall first.

---

## 4. Model weights (required for detection)
The default model is the **ensemble** and needs these files in `ml/checkpoints/`:

| File | Detector |
|---|---|
| `xception_best.pth` | GAN faces |
| `efficientnet_b4_sbi_best.pth` | Face swaps (Self-Blended Images) |
| `clip_vitl14_head.pt` | AI-generated images (head; the CLIP backbone downloads automatically, ~1.7 GB) |
| `combiner.pkl` | Merges the three scores |
| `resnet50_best.pth`, `efficientnet_b0_best.pth` | Optional comparison models |

`ml/weights/face_detection_yunet_2023mar.onnx` (YuNet face detector) is in the repo.
Without weights the API returns a clear 503.

### Training them yourself
Full commands and timings: [phase13_generalization.md §13.10](phase13_generalization.md).
```powershell
python -m ml.dataset.fetch_hf_full --out datasets/hemg        # ~1.8 GB download
python -m ml.training.train_gpu --arch xception --data datasets/hemg --epochs 15
python -m ml.training.sbi_train --per-source 30000 --epochs 3  # needs D:\archive (140k faces) too
python -m ml.training.clip_train                              # needs datasets/openfake
python -m ml.evaluation.benchmark && python -m ml.evaluation.fit_combiner
```
**Restart the backend** after training (models are cached in-process). Stop the backend
*before* GPU training or evaluation — two processes sharing a 6 GB GPU can crash CUDA.

---

## 5. Run the backend
```powershell
cd backend
uvicorn main:app --reload
```
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

---

## 6. Run the frontend
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

---

## 7. First run flow
1. Register a user in the UI.
2. Go to **Image Detection**, upload an image (the default model is the ensemble), click **Detect**.
3. View the verdict (REAL / UNCERTAIN / FAKE), likely fake type, per-detector scores and Grad-CAM
   heatmap. Phone screenshots are cropped to the photo automatically.
4. Check **History** and **Analytics**.

---

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `Model weights not found` (503) | Train a model or place a `.pth` in `ml/checkpoints/` |
| DB connection error | Verify `.env` credentials and that MySQL is running |
| CORS error in browser | Ensure `FRONTEND_ORIGIN` matches the Vite URL |
| `ModuleNotFoundError: ml` | Run backend from the `backend/` folder (main.py adds project root to path) |
| Frontend 401 loops | Clear localStorage; log in again |
| 500 on register/login (`password cannot be longer than 72 bytes`) | bcrypt 5.x breaks passlib 1.7.4 — pin `bcrypt==4.0.1` |
| `module 'cv2' has no attribute 'CascadeClassifier'` | Bad OpenCV wheel — `pip install opencv-python==4.10.0.84` |
| `UnicodeEncodeError` during training (Windows) | Fixed in `ml/utils/logger.py` (UTF-8 handlers); update if on an old copy |
| `Data truncated for column 'verdict'` | Database created before the UNCERTAIN verdict: `ALTER TABLE predictions MODIFY verdict ENUM('real','fake','uncertain') NOT NULL;` |
| `No active model for architecture 'ensemble'` | Older database: `INSERT INTO models (name, architecture, weights_path, is_active) VALUES ('Ensemble-DF','ensemble','ml/checkpoints/combiner.pkl',1);` |
| `CUDA error: unknown error` / `bad allocation` | Another process is using the GPU or RAM is low — stop the backend during training, use `--workers 0` |
| First detection takes ~20 s | Models load on startup in the background; wait for `[startup] ensemble models loaded` |

---

## Docs map
| Phase | Document |
|-------|----------|
| 1 | [phase1_planning.md](phase1_planning.md) |
| 2 | [phase2_architecture.md](phase2_architecture.md) |
| 3 | [phase3_database.md](phase3_database.md) |
| 4–7 | [phase4-7_ml_pipeline.md](phase4-7_ml_pipeline.md) |
| 10 | [phase10_results.md](phase10_results.md) |
| 11 | [phase11_final_report.md](phase11_final_report.md) |
| 12 | [phase12_viva.md](phase12_viva.md) |
| 13 | [phase13_generalization.md](phase13_generalization.md) |
