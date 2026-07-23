# Deep Fake Detection System Using Deep Learning for Image and Video Authentication
## Final Year Project Report

> This is the master report document (Phase 11). It is structured to expand to **80–100 pages**
> when formatted (Times New Roman 12pt, 1.5 spacing) with the diagrams from
> [`phase2_architecture.md`](phase2_architecture.md), screenshots, and result plots inserted.

---

## Table of Contents
1. Abstract
2. Synopsis
3. Introduction
4. Literature Survey
5. Software Requirements Specification (SRS)
6. System Design
7. Methodology
8. Implementation
9. Results & Discussion
10. Conclusion
11. Future Scope
12. References

---

## 1. Abstract

The proliferation of generative deep-learning techniques — Generative Adversarial Networks
(GANs), variational autoencoders, and diffusion models — has enabled the creation of
hyper-realistic synthetic media popularly known as *deepfakes*. While these technologies have
legitimate creative applications, their misuse for misinformation, financial fraud, identity
theft, and defamation poses a serious threat to digital trust. Human observers can no longer
reliably distinguish authentic media from sophisticated forgeries.

This project presents an end-to-end **Deep Fake Detection System** that authenticates both
images and videos using convolutional neural networks. Three state-of-the-art architectures —
**ResNet50, EfficientNet-B0, and XceptionNet** — are fine-tuned via transfer learning
(ImageNet-pretrained backbones) and compared. In this prototype they are trained on a balanced
**3,000-image** real/fake face dataset (public HuggingFace `Hemg/deepfake-and-real-images`);
the pipeline also supports the standard video benchmarks (FaceForensics++, Celeb-DF, DFDC) for
scale-up. The system extracts faces from media, performs frame-level inference for video, and
aggregates results into a final verdict with a confidence score. To make predictions
interpretable, **Grad-CAM** is used to visualize the facial regions responsible for each decision.

The models are served through a secure **FastAPI** backend with **JWT authentication** and a
**MySQL** database that persists users, uploads, predictions, and audit logs. A responsive
**React + Tailwind** dashboard lets users upload media, view explainable results, browse their
prediction history, and explore analytics. On the held-out test split, **XceptionNet** achieves
**96.4% accuracy / 0.991 ROC-AUC**, **ResNet50** 96.2% / 0.993, and **EfficientNet-B0**
89.1% / 0.972. The system follows clean architecture and SOLID principles, making it suitable
for both academic evaluation and real-world extension.

---

## 2. Synopsis

**Title:** Deep Fake Detection System Using Deep Learning for Image and Video Authentication

**Problem:** Automated, explainable detection of manipulated facial media.

**Objective:** Build a full-stack system that classifies images/videos as real or fake,
explains its reasoning with Grad-CAM, and exposes the capability via a secure web application.

**Approach:**
- Preprocess media → detect and crop faces → sample frames for video.
- Train and compare three CNN detectors.
- Aggregate per-frame predictions to a video-level verdict.
- Generate Grad-CAM heatmaps for explainability.
- Serve via FastAPI + JWT; persist in MySQL; present via React.

**Outcome:** A working, documented, production-quality prototype in which the best model
(XceptionNet) reaches **~96% accuracy / 0.99 ROC-AUC** on the held-out test split, with an
interpretable, user-facing interface.

**Tools:** Python, PyTorch, timm, OpenCV, scikit-learn, FastAPI, SQLAlchemy, MySQL, React,
Tailwind CSS.

---

## 3. Introduction

### 3.1 Background
Deepfakes are synthetic media in which a person's likeness is replaced or reenacted using deep
learning. The most common facial manipulation categories are **face-swap** (identity
replacement), **face-reenactment** (expression transfer), **attribute editing**, and **entire
face synthesis**. Detection methods exploit the subtle artifacts these techniques leave behind:
blending boundaries, inconsistent lighting, unnatural textures, temporal flicker, and frequency
anomalies.

### 3.2 Motivation
As generation quality improves, manual detection becomes infeasible. Society needs scalable,
trustworthy, and **explainable** detection tools — not black boxes. Explainability (via
Grad-CAM) is essential for forensic credibility and user trust.

### 3.3 Problem Statement
*(See [phase1_planning.md §1.1](phase1_planning.md).)* Build an automated, explainable system
that classifies images/videos as real or fake with confidence scoring and manipulated-region
visualization, backed by authentication, history, and analytics.

### 3.4 Objectives
*(See [phase1_planning.md §1.2](phase1_planning.md).)*

### 3.5 Scope
*(See [phase1_planning.md §1.3](phase1_planning.md).)*

### 3.6 Organization of the Report
Chapters proceed from literature and requirements through design, methodology, implementation,
results, and conclusions.

---

## 4. Literature Survey

| # | Work / Method | Key Idea | Relevance |
|---|---------------|----------|-----------|
| 1 | Rossler et al., **FaceForensics++** (ICCV 2019) | Large benchmark + XceptionNet baseline | Dataset & baseline model |
| 2 | Li et al., **Celeb-DF** (CVPR 2020) | High-quality deepfake benchmark | Harder evaluation set |
| 3 | Dolhansky et al., **DFDC** (2020) | Large-scale challenge dataset | Diversity & robustness |
| 4 | Chollet, **Xception** (CVPR 2017) | Depthwise-separable convolutions | Strong detector backbone |
| 5 | Tan & Le, **EfficientNet** (ICML 2019) | Compound scaling | Efficient accurate backbone |
| 6 | He et al., **ResNet** (CVPR 2016) | Residual learning | Baseline backbone |
| 7 | Selvaraju et al., **Grad-CAM** (ICCV 2017) | Gradient-based localization | Explainability |
| 8 | Afchar et al., **MesoNet** (WIFS 2018) | Compact mesoscopic detector | Early deepfake detector |
| 9 | Nguyen et al., **Capsule-Forensics** (2019) | Capsule networks for forgery | Alternative approach |
| 10 | Frank et al., **Frequency analysis** (ICML 2020) | GAN spectral artifacts | Complementary cue |

### 4.1 Gap Analysis
Most academic detectors stop at a model and metrics. Few provide an **integrated, explainable,
authenticated, full-stack application** with history and analytics. This project fills that gap
by combining strong detection with usability, security, and interpretability.

---

## 5. Software Requirements Specification (SRS)

### 5.1 Functional Requirements
*(Full list in [phase1_planning.md §1.6](phase1_planning.md).)* FR1–FR13 cover registration,
authentication, upload, validation, preprocessing, inference, Grad-CAM, persistence, history,
comparison, analytics, admin, and audit logging.

### 5.2 Non-Functional Requirements
*(Full table in [phase1_planning.md §1.7](phase1_planning.md).)* Performance, accuracy,
security, scalability, usability, reliability, maintainability, portability, explainability.

### 5.3 Hardware Requirements
- Training: NVIDIA GPU (≥ 6 GB VRAM), 16 GB RAM, 50 GB disk.
- Serving: CPU sufficient for inference; GPU optional for speed.

### 5.4 Software Requirements
- Python 3.10+, Node.js 18+, MySQL 8.
- Libraries per [`requirements.txt`](../requirements.txt) and
  [`frontend/package.json`](../frontend/package.json).

### 5.5 Use Cases
*(Use-case diagram in [phase2_architecture.md §2.8](phase2_architecture.md).)*

---

## 6. System Design

All diagrams are in [`phase2_architecture.md`](phase2_architecture.md):
- High-Level & Low-Level Architecture (§2.1–2.2)
- Component, Class, Sequence, Activity, Use-Case diagrams (§2.3–2.8)
- DFD Levels 0/1/2 (§2.9)

Database design (ER diagram, schema, constraints) is in
[`phase3_database.md`](phase3_database.md) with the runnable script
[`database/schema.sql`](../database/schema.sql).

### 6.1 Architectural Style
The system uses a **layered / clean architecture**: presentation (React) → API (FastAPI
routers) → services (business logic) → domain (ORM models) → persistence (MySQL), with the ML
inference engine as a separable dependency injected into the detection service. This enforces
separation of concerns and the Dependency Inversion Principle.

### 6.2 Design Principles
- **SOLID** — `BaseDetector` abstraction + factory (OCP, DIP, LSP); single-responsibility
  services; thin routers.
- **DRY** — shared transforms, metrics, and the reusable `UploadDetect` React component.
- **Security by design** — hashed passwords, JWT, input/upload validation, CORS.

---

## 7. Methodology

### 7.1 Overall Pipeline
```mermaid
flowchart LR
    A[Data Collection] --> B[Preprocessing<br/>face crop, frame sample]
    B --> C[Augmentation]
    C --> D[Train/Val/Test split]
    D --> E[Model Training<br/>ResNet50/EffNet/Xception]
    E --> F[Evaluation & Comparison]
    F --> G[Best Model → Inference Engine]
    G --> H[Grad-CAM Explainability]
    H --> I[API + Web App]
```

### 7.2 Data Preprocessing
Faces are detected and cropped with an OpenCV Haar cascade (swappable for MTCNN/RetinaFace).
Videos are uniformly sampled to N frames; each frame's face is cropped. Details in
[`phase4-7_ml_pipeline.md`](phase4-7_ml_pipeline.md).

### 7.3 Model Training
AdamW optimizer, cosine LR schedule, label smoothing, mixed-precision (AMP), and early stopping
on validation loss. The best checkpoint (lowest val loss) is retained.

### 7.4 Inference & Aggregation
Image → single face → verdict. Video → per-frame fake probabilities → aggregated by mean
(default), max, or majority vote → verdict + confidence.

### 7.5 Explainability
Grad-CAM highlights the regions driving the *fake* class score, saved as an overlay and
surfaced in the UI.

---

## 8. Implementation

### 8.1 Project Structure
*(See root [README.md](../README.md).)* Four pillars: `ml/`, `backend/`, `frontend/`,
`database/`, with `docs/` for documentation.

### 8.2 Machine Learning
- Models & factory: [`ml/models/`](../ml/models/)
- Training: [`ml/training/train.py`](../ml/training/train.py)
- Evaluation: [`ml/evaluation/`](../ml/evaluation/)
- Grad-CAM: [`ml/explain/gradcam.py`](../ml/explain/gradcam.py)
- Inference engine: [`ml/inference.py`](../ml/inference.py)

### 8.3 Backend (FastAPI)
- Auth (JWT, bcrypt): [`backend/app/auth/`](../backend/app/auth/)
- Routers: [`backend/app/routers/`](../backend/app/routers/)
- Services: [`backend/app/services/`](../backend/app/services/)
- ORM models: [`backend/app/models/`](../backend/app/models/)
- Entry point: [`backend/main.py`](../backend/main.py)

### 8.4 Frontend (React)
- Routing & pages: [`frontend/src/pages/`](../frontend/src/pages/)
- Reusable components & hooks: [`frontend/src/components/`](../frontend/src/components/),
  [`frontend/src/context/AuthContext.jsx`](../frontend/src/context/AuthContext.jsx)
- API layer: [`frontend/src/services/api.js`](../frontend/src/services/api.js)

### 8.5 Database
MySQL schema with six tables and full referential integrity —
[`database/schema.sql`](../database/schema.sql).

### 8.6 Key API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Obtain JWT |
| POST | `/api/detect/image` | Detect image |
| POST | `/api/detect/video` | Detect video |
| POST | `/api/detect/compare` | Run all models |
| GET | `/api/history` | List predictions |
| GET | `/api/analytics/summary` | Analytics |

---

## 9. Results & Discussion

Full metrics, plots, and interpretation are in [`phase10_results.md`](phase10_results.md).
Summary (held-out test split): **XceptionNet 96.4% acc / 0.991 AUC** and **ResNet50 96.2% /
0.993** are essentially tied for best; **EfficientNet-B0 (89.1% / 0.972)** is the smallest and
fastest model, illustrating the accuracy-vs-compute trade-off; Grad-CAM confirms the models
attend to facial regions rather than background.

---

## 10. Conclusion

This project delivers a complete, explainable, full-stack deepfake detection system. It meets
its objectives: accurate image/video classification, confidence scoring, Grad-CAM
explainability, multi-model comparison, secure authentication, persistent history, and
analytics. The clean, modular architecture makes it maintainable and extensible, and the
results validate the effectiveness of CNN-based facial forgery detection.

---

## 11. Future Scope

- **Temporal modelling** (LSTM/3D-CNN/transformer) to exploit inter-frame inconsistencies.
- **Frequency-domain features** (FFT/DCT) to catch GAN spectral fingerprints.
- **Audio-visual detection** for lip-sync and voice-clone forgeries.
- **Robustness** to compression, adversarial perturbations, and unseen generators.
- **Real-time / streaming** detection and browser-extension deployment.
- **Ensemble** of the three detectors for higher reliability.
- **Dockerization & cloud deployment** with autoscaling and model versioning.

---

## 12. References

1. A. Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV, 2019.
2. Y. Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics," CVPR, 2020.
3. B. Dolhansky et al., "The DeepFake Detection Challenge (DFDC) Dataset," arXiv:2006.07397, 2020.
4. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR, 2017.
5. M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML, 2019.
6. K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
7. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV, 2017.
8. D. Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection Network," WIFS, 2018.
9. H. Nguyen et al., "Capsule-Forensics," ICASSP, 2019.
10. J. Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition," ICML, 2020.
11. FastAPI Documentation — https://fastapi.tiangolo.com
12. PyTorch Documentation — https://pytorch.org
13. React Documentation — https://react.dev
14. timm (PyTorch Image Models) — https://github.com/huggingface/pytorch-image-models
