# Setup & Run Guide

Complete instructions to run the Deep Fake Detection System locally.

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
CUDA build (cu124 matches recent drivers), then set `DEVICE=cuda` in `.env`:
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
> A plain `pip install torch` will **not** replace an existing `+cpu` build — uninstall first.

---

## 4. (Optional) Train models
Detection endpoints need trained weights in `ml/checkpoints/`. Without them the API returns
a clear 503 telling you to train first.

```powershell
# 1. Fetch a small balanced dataset (public HuggingFace, no login) — see phase4-7 docs
python -m ml.dataset.fetch_hf_sample --per-class 1500 --only-label 0 --start-offset 0
python -m ml.dataset.fetch_hf_sample --per-class 1500 --only-label 1 --start-offset 95000

# 2. Train (GPU) — one architecture at a time
python -m ml.training.train_gpu --arch xception        --epochs 15
python -m ml.training.train_gpu --arch resnet50        --epochs 15
python -m ml.training.train_gpu --arch efficientnet_b0 --epochs 15
#    …or CPU fallback:
#    python -m ml.training.train_cpu --arch xception --epochs 8
```
Checkpoints save to `ml/checkpoints/<arch>_best.pth`, the paths the seeded model registry
already expects. **Restart the backend** after training (models are cached in-process).

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
2. Go to **Image Detection**, upload an image, choose a model, click **Detect**.
3. View the verdict, confidence, and Grad-CAM heatmap.
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
