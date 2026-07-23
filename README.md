# Deep Fake Detection System Using Deep Learning for Image and Video Authentication

A production-quality, full-stack deep learning system that detects manipulated (deepfake) images and
videos, produces confidence scores, visualizes tampered regions with Grad-CAM, and exposes everything
through a secured REST API and a modern React dashboard.

> **Final-Year B.Tech (CSE) Major Project** — suitable for university evaluation and GitHub portfolio.

---

## ✨ Key Features

| Category | Capability |
|----------|------------|
| Detection | Deepfake **image** & **video** detection |
| Models | ResNet50, EfficientNet-B0, XceptionNet (comparable) |
| Explainability | Grad-CAM heatmaps of manipulated regions |
| Confidence | Calibrated probability / confidence score |
| History | Persisted prediction history per user |
| Auth | JWT-based registration & login, role-based access |
| Analytics | Dashboard with model & usage analytics |
| Comparison | Side-by-side multi-model comparison |

---

## 🏗️ Tech Stack

- **Frontend:** React.js, Tailwind CSS, React Router, Axios, Recharts
- **Backend:** FastAPI, SQLAlchemy, Pydantic, JWT (python-jose), Passlib
- **Database:** MySQL 8
- **ML:** Python, PyTorch, torchvision, timm, OpenCV, NumPy, Pandas, scikit-learn
- **XAI:** Grad-CAM (pytorch-grad-cam / custom hooks)

---

## 📁 Repository Structure

```
Deepfake_finalyear/
├── README.md
├── requirements.txt              # Python (ML + backend) dependencies
├── .env.example                  # Environment variable template
├── docs/                         # All project documentation & diagrams
│   ├── phase1_planning.md
│   ├── phase2_architecture.md
│   ├── phase3_database.md
│   └── ...
├── database/                     # SQL schema, seeds, migrations
│   └── schema.sql
├── ml/                           # Machine-learning pipeline (training/eval)
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explain/                  # Grad-CAM
│   └── utils/
├── backend/                      # FastAPI application
│   ├── app/
│   │   ├── routers/
│   │   ├── services/
│   │   ├── models/               # SQLAlchemy ORM
│   │   ├── schemas/              # Pydantic
│   │   ├── auth/
│   │   ├── database/
│   │   └── ml/                    # Inference wrappers
│   └── main.py
└── frontend/                     # React application
    └── src/
```

---

## 🚀 Quick Start

### 1. Clone & configure
```bash
git clone <repo-url>
cd Deepfake_finalyear
cp .env.example .env      # edit DB credentials & secret key
```

### 2. Database (MySQL)
```bash
mysql -u root -p < database/schema.sql
```

### 3. Python environment (ML + backend)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the backend
```bash
cd backend
uvicorn main:app --reload
# API docs: http://localhost:8000/docs
```

### 5. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🗺️ Build Roadmap (Phases)

1. Project Planning · 2. System Architecture · 3. Database Design · 4. Dataset Module ·
5. ML Pipeline · 6. Model Implementation · 7. Explainable AI · 8. FastAPI Backend ·
9. React Frontend · 10. Results Module · 11. Documentation · 12. Viva Prep

See [docs/](docs/) for the full documentation set.

---

## 📜 License

MIT — for academic and portfolio use.
