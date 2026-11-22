# AuthentiScan: An Explainable AI System for Synthetic Media Detection
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

This project presents **AuthentiScan**, an end-to-end deepfake detection system that authenticates both
images and videos. It was built in two stages. First, three CNN architectures — **ResNet50,
EfficientNet-B0 and XceptionNet** — were fine-tuned via transfer learning and compared on a
GAN-face dataset; scaling the data from 3,000 to 200,000 images raised XceptionNet to **98%**
accuracy. Testing on other kinds of fakes, however, showed that this in-distribution accuracy
did not generalize: the same model was a coin-flip on face swaps and *inverted* (ROC-AUC 0.215)
on images from modern diffusion generators such as Gemini, Midjourney and Flux.

The final system therefore uses an **ensemble of three specialised detectors** — XceptionNet
for GAN-generated faces, an **EfficientNet-B4 trained with Self-Blended Images** for face swaps,
and a **frozen CLIP ViT-L/14 with a linear classifier** for AI-generated images — whose outputs
are merged by a gradient-boosted **combiner** fitted only on validation data. Training used four
public datasets covering more than 90 generators. The system also locates the photo inside phone
**screenshots**, reports an explicit **UNCERTAIN** verdict for borderline cases, names the likely
type of manipulation, and explains decisions with **Grad-CAM**.

The models are served through a secure **FastAPI** backend with **JWT authentication** and a
**MySQL** database that persists users, uploads, predictions, and audit logs. A responsive
**React + Tailwind** dashboard lets users upload media, view explainable results, browse their
prediction history, and explore analytics. On a held-out benchmark of 8,000 decisions spanning
GAN faces, StyleGAN faces, diffusion images and face swaps (clean and WhatsApp-compressed), the
ensemble reaches **84.5% accuracy / 0.925 ROC-AUC**, up from 70.0% for the best single model,
with accuracy on diffusion images and face swaps rising from about 49% to about 80%. The system
follows clean architecture and SOLID principles, making it suitable for both academic evaluation
and real-world extension.

---

## 2. Synopsis

**Title:** AuthentiScan: An Explainable AI System for Synthetic Media Detection

**Problem:** Automated, explainable detection of manipulated facial media.

**Objective:** Build a full-stack system that classifies images/videos as real or fake,
explains its reasoning with Grad-CAM, and exposes the capability via a secure web application.

**Approach:**
- Preprocess media → detect and crop faces → sample frames for video.
- Train and compare three CNN detectors; scale the training data.
- Evaluate on *other* kinds of fakes, then build specialised detectors (GAN faces, face swaps,
  AI-generated images) and a learned combiner.
- Crop the photo out of phone screenshots before analysis.
- Aggregate per-frame predictions to a video-level verdict.
- Generate Grad-CAM heatmaps for explainability.
- Serve via FastAPI + JWT; persist in MySQL; present via React.

**Outcome:** A working, documented, production-quality system. Single-dataset accuracy reached
98%; on a four-domain benchmark the ensemble reaches **84.5% accuracy / 0.925 ROC-AUC**
(70.0% for the best single model), with an interpretable, user-facing interface.

**Tools:** Python, PyTorch, timm, OpenCLIP, OpenCV (YuNet), scikit-learn, Albumentations,
FastAPI, SQLAlchemy, MySQL, React, Tailwind CSS.

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
| 11 | Shiohara & Yamasaki, **Self-Blended Images** (CVPR 2022) | Train swap detectors from real faces blended with themselves | Our face-swap detector |
| 12 | Radford et al., **CLIP** (ICML 2021) | Image-text pre-training on 400M pairs | Backbone of our AI-image detector |
| 13 | Ojha et al., **UnivFD** (CVPR 2023) | Linear probe on frozen CLIP generalizes across generators | Our AI-image detector design |
| 14 | Livernoche et al., **OpenFake** (2025) | Real vs. many modern generators | Training / test data for AI images |
| 15 | Song et al., **DeepFakeFace** (2023) | Diffusion and InsightFace swaps of celebrities | Unseen face-swap test set |
| 16 | Wu et al., **YuNet** (2023) | Lightweight CNN face detector | Face detection & landmarks |

### 4.1 Gap Analysis
Most academic detectors stop at a model and metrics, and are usually evaluated on the same
dataset they were trained on. Few provide an **integrated, explainable, authenticated,
full-stack application**, and fewer still are evaluated across *different kinds* of fakes and
real-world conditions (compression, screenshots). This project addresses both: a usable,
secure, explainable application, and a detector evaluated — and redesigned — for
cross-domain generalization.

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
    A[Data Collection<br/>4 public datasets] --> B[Preprocessing<br/>YuNet face crop, frame sample,<br/>screenshot crop]
    B --> C[Augmentation<br/>incl. JPEG / downscale / blur]
    C --> D[Fixed Train/Val/Test splits]
    D --> E1[Stage 1: compare<br/>ResNet50 / EffNet / Xception]
    E1 --> E2[Stage 2: specialised detectors<br/>Xception · SBI EffNet-B4 · CLIP]
    E2 --> F[Combiner fitted on validation]
    F --> G[Cross-domain benchmark<br/>clean + WhatsApp + screenshots]
    G --> H[Inference engine + Grad-CAM]
    H --> I[API + Web App]
```

### 7.2 Data Preprocessing
Faces are detected with **OpenCV YuNet** (a small CNN that also returns five landmarks; it
replaced the Haar cascade) and cropped at the same framing as the training images. Phone
screenshots are detected and reduced to the embedded photo. Videos are uniformly sampled to N
frames. Details in [`phase4-7_ml_pipeline.md`](phase4-7_ml_pipeline.md) and
[`phase13_generalization.md`](phase13_generalization.md).

### 7.3 Model Training
AdamW optimizer, cosine LR schedule, label smoothing, mixed-precision (AMP), and early stopping
on validation loss. The best checkpoint (lowest val loss) is retained.

### 7.4 Specialised Detectors and Combiner
- **GAN faces:** XceptionNet fine-tuned on 200k Hemg + 140k-Faces images.
- **Face swaps:** EfficientNet-B4 trained with **Self-Blended Images** — each real face is
  blended with a slightly altered copy of itself, so the model learns blending artefacts
  without ever seeing a real swap.
- **AI-generated images:** frozen **CLIP ViT-L/14** features + a logistic-regression head,
  trained on OpenFake (89 generators) with clean and degraded copies.
- **Combiner:** gradient-boosted trees over the three probabilities, fitted only on
  validation data.

### 7.5 Inference & Aggregation
Image → (screenshot crop) → three detector scores → combiner → **REAL / UNCERTAIN / FAKE**
(uncertain band 0.35–0.65) + likely fake type. Video → per-frame combined probabilities →
mean → verdict + confidence.

### 7.6 Explainability
Grad-CAM (from whichever face detector is more suspicious) highlights the regions driving the
*fake* score; the UI also shows each detector's score, so the user sees *which* detector
fired.

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
- Inference engine: [`ml/inference.py`](../ml/inference.py) → ensemble in
  [`ml/ensemble.py`](../ml/ensemble.py)
- Specialised training: [`ml/training/sbi_train.py`](../ml/training/sbi_train.py),
  [`ml/training/clip_train.py`](../ml/training/clip_train.py),
  [`ml/training/mixed_train.py`](../ml/training/mixed_train.py)
- Self-Blended Images: [`ml/dataset/sbi.py`](../ml/dataset/sbi.py); screenshot cropping:
  [`ml/dataset/screenshot.py`](../ml/dataset/screenshot.py)
- Benchmark & combiner: [`ml/evaluation/benchmark.py`](../ml/evaluation/benchmark.py),
  [`ml/evaluation/fit_combiner.py`](../ml/evaluation/fit_combiner.py),
  [`ml/evaluation/screenshot_eval.py`](../ml/evaluation/screenshot_eval.py)

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

Full metrics, plots, and interpretation are in [`phase10_results.md`](phase10_results.md) and
[`phase13_generalization.md`](phase13_generalization.md).

**Stage 1 (architecture comparison, 3k images):** XceptionNet 96.4% / 0.991 AUC and ResNet50
96.2% / 0.993 are essentially tied; EfficientNet-B0 (89.1% / 0.972) is smallest and fastest.

**Scaling (XceptionNet):** 30k images → 97.5%; 181k → 98.4% (AUC 0.9986); adding the 140k-Faces
dataset lifted unseen StyleGAN faces from about 51% to 97.7%.

**Final ensemble (8,000 held-out decisions):**

| Kind of fake | Best single model | Ensemble |
|---|:---:|:---:|
| GAN faces (Hemg) | 98.3% | 93.8% |
| StyleGAN faces | 97.9% | 95.4% |
| Diffusion / AI images | 48.8% | 81.0% |
| Face swaps | 49.2% | 79.1% |
| WhatsApp-compressed (4 sets) | 66.4% | 81.8% |
| **Overall** | **70.0%** | **84.5% (AUC 0.925)** |
| Phone screenshots | 72.9% | 84.4% |

The key finding is that in-distribution accuracy overstates real-world performance; dedicated
detectors for each kind of manipulation, combined by a learned model, generalize far better.

---

## 10. Conclusion

This project delivers a complete, explainable, full-stack deepfake detection system. It meets
its objectives: accurate image/video classification, confidence scoring, Grad-CAM
explainability, multi-model comparison, secure authentication, persistent history, and
analytics. The clean, modular architecture makes it maintainable and extensible, and the
results validate the effectiveness of CNN-based facial forgery detection. Beyond the original
objectives, the project shows — with measurements — that a detector must be evaluated across
kinds of manipulation, and that a specialised ensemble raises cross-domain accuracy from 70% to
84.5% while staying explainable and usable.

---

## 11. Future Scope

- **Temporal modelling** (LSTM/3D-CNN/transformer) to exploit inter-frame inconsistencies.
- **Frequency-domain features** (FFT/DCT) to catch GAN spectral fingerprints.
- **Audio-visual detection** for lip-sync and voice-clone forgeries.
- **Real face-swap video data** (Celeb-DF, FaceForensics++) to strengthen the weakest category
  (face swaps, 72–79%) and to evaluate video detection properly.
- **More diffusion data** (more of OpenFake's 2.3M images) and fine-tuning CLIP's last layers,
  especially for Midjourney 7 (57%).
- **Finer SBI masks** (468-point face mesh) and longer SBI training.
- **Robustness** to adversarial perturbations and generators released after training.
- **Real-time / streaming** detection and browser-extension deployment.
- **Knowledge distillation** of the ensemble into one compact model for mobile use.
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
11. K. Shiohara and T. Yamasaki, "Detecting Deepfakes with Self-Blended Images," CVPR, 2022.
12. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML, 2021.
13. U. Ojha, Y. Li and Y. J. Lee, "Towards Universal Fake Image Detectors that Generalize Across Generative Models," CVPR, 2023.
14. V. Livernoche et al., "OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection," arXiv:2509.09495, 2025.
15. H. Song et al., "Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models" (DeepFakeFace), arXiv:2309.02218, 2023.
16. W. Wu, H. Peng and S. Yu, "YuNet: A Tiny Millisecond-level Face Detector," Machine Intelligence Research, 2023.
17. Q. Xie et al., "Self-training with Noisy Student improves ImageNet classification," CVPR, 2020.
18. FastAPI Documentation — https://fastapi.tiangolo.com
19. PyTorch Documentation — https://pytorch.org
20. React Documentation — https://react.dev
21. timm (PyTorch Image Models) — https://github.com/huggingface/pytorch-image-models
22. OpenCLIP — https://github.com/mlfoundations/open_clip
23. scikit-learn (HistGradientBoostingClassifier) — https://scikit-learn.org
