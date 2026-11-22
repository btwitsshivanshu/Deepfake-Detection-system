# Phase 1 — Project Planning

**Project:** AuthentiScan — Deep Fake Detection System Using Deep Learning for Image and Video Authentication
**Domain:** Deep Learning · Computer Vision · Media Forensics · Full-Stack Web Engineering

---

## 1.1 Problem Statement

The rapid democratization of generative AI (GANs, autoencoders, diffusion models) has made it
trivially easy to synthesize hyper-realistic fake images and videos — "deepfakes". These are
increasingly weaponized for misinformation, financial fraud, identity theft, non-consensual media,
and political manipulation. Human perception alone can no longer reliably distinguish authentic media
from AI-generated forgeries.

There is a critical need for an **automated, explainable, and accessible** system that can:
1. Accept an image or video from any user,
2. Classify it as **REAL** or **FAKE** with a calibrated confidence score,
3. **Explain** its decision by highlighting the manipulated facial regions, and
4. Maintain an auditable history for verification and analytics.

This project delivers such a system by combining state-of-the-art CNN architectures
(ResNet50, EfficientNet-B0, XceptionNet) with Grad-CAM explainability, wrapped in a secure
FastAPI backend and a React dashboard.

---

## 1.2 Objectives

1. **Primary:** Build a deep-learning model that classifies images/videos as real or fake with
   high accuracy (target ≥ 90% on benchmark datasets).
2. Extract faces and frames from video and run frame-level + aggregated video-level inference.
3. Provide **confidence scoring** and probability calibration.
4. Implement **Grad-CAM** to visualize which regions influenced the prediction (explainable AI).
5. Train and **compare three architectures** (ResNet50, EfficientNet-B0, XceptionNet).
6. Build a **secure REST API** (JWT auth) exposing detection, history and analytics endpoints.
7. Build a **responsive React dashboard** for upload, results, history and analytics.
8. Persist predictions, files and audit logs in a **MySQL** database.
9. Produce complete **documentation** (SRS, design, results) and viva material.

---

## 1.3 Scope

### In Scope
- Face-based deepfake detection for images (JPG/PNG) and videos (MP4/AVI/MOV).
- Three CNN detectors + model comparison.
- Grad-CAM explanations and heatmap overlays.
- User authentication, prediction history, analytics dashboard.
- Training/evaluation pipelines on FaceForensics++, Celeb-DF, DFDC.

### Out of Scope
- Audio deepfake / voice-cloning detection.
- Real-time streaming / live-camera detection (batch upload only).
- Full-body or scene manipulation (focus is on **faces**).
- Mobile native apps (web-responsive only).
- Detection of diffusion-based whole-image synthesis beyond face-swap/reenactment datasets.

---

## 1.4 Features

| # | Feature | Description |
|---|---------|-------------|
| F1 | Image Detection | Upload image → REAL/FAKE + confidence |
| F2 | Video Detection | Upload video → frame sampling → aggregated verdict |
| F3 | Confidence Score | Probability output per prediction |
| F4 | Grad-CAM Heatmap | Visual overlay of manipulated regions |
| F5 | Model Selection | Choose ResNet50 / EfficientNet / Xception |
| F6 | Model Comparison | Compare outputs of all models on one file |
| F7 | Prediction History | Persisted, filterable per-user history |
| F8 | Authentication | Register, login, JWT sessions, roles |
| F9 | Analytics Dashboard | Charts: real vs fake, model usage, accuracy |
| F10 | Audit Logging | Track key user & system actions |

---

## 1.5 User Roles

| Role | Permissions |
|------|-------------|
| **Guest** | View landing page, register/login |
| **User** | Upload media, view own predictions/history, run comparisons, view own analytics |
| **Admin** | All user rights + manage users, view all predictions, manage/activate models, view audit logs and global analytics |

---

## 1.6 Functional Requirements

- **FR1** The system shall allow a user to register with a unique email and password.
- **FR2** The system shall authenticate users and issue JWT access/refresh tokens.
- **FR3** The system shall allow authenticated users to upload an image or video.
- **FR4** The system shall validate file type and size before processing.
- **FR5** The system shall extract faces (image) or sample & extract face frames (video).
- **FR6** The system shall run inference using the selected model and return REAL/FAKE + confidence.
- **FR7** The system shall generate a Grad-CAM heatmap for each prediction.
- **FR8** The system shall persist each prediction, its result and the uploaded file metadata.
- **FR9** The system shall allow users to view and filter their prediction history.
- **FR10** The system shall let a user run all models on one input and compare results.
- **FR11** The system shall present analytics (counts, distributions, model usage) via charts.
- **FR12** The system shall let admins manage users, models and view audit logs.
- **FR13** The system shall record audit log entries for auth and critical actions.

---

## 1.7 Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Performance** | Single image inference < 2 s; video (≤ 30 s clip) < 20 s on GPU |
| **Accuracy** | ≥ 90% test accuracy, ROC-AUC ≥ 0.92 on benchmark sets |
| **Security** | Bcrypt-hashed passwords, JWT auth, input validation, OWASP-safe uploads |
| **Scalability** | Stateless API, model loaded once, horizontally scalable behind a load balancer |
| **Usability** | Responsive UI, clear result visualization, ≤ 3 clicks to a result |
| **Reliability** | Graceful error handling, validation, transactional DB writes |
| **Maintainability** | Modular clean architecture, SOLID, typed code, documented |
| **Portability** | Dockerizable; runs on CPU or GPU |
| **Explainability** | Every verdict accompanied by a Grad-CAM visualization |

---

## 1.8 Assumptions & Constraints

- A CUDA-capable GPU is available for training (inference can fall back to CPU).
- Benchmark datasets (FaceForensics++, Celeb-DF, DFDC) are accessible under their licenses.
- Uploaded media contains at least one detectable frontal face.
- The project timeline follows the 12-phase incremental plan.
