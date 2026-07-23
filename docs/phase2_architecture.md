# Phase 2 — System Architecture

All diagrams use **Mermaid** and render directly on GitHub / VS Code.

---

## 2.1 High-Level Architecture

```mermaid
flowchart TB
    subgraph Client["🖥️ Client Layer"]
        UI["React + Tailwind SPA<br/>(Login, Dashboard, Detect, History, Analytics)"]
    end

    subgraph API["⚙️ Application Layer — FastAPI"]
        AUTH["Auth Service<br/>JWT / Passlib"]
        ROUT["Routers<br/>/auth /detect /history /analytics /admin"]
        SVC["Services<br/>Detection · History · Analytics"]
        MLW["ML Inference Wrapper<br/>model registry + Grad-CAM"]
    end

    subgraph ML["🧠 ML Layer"]
        PRE["Preprocessing<br/>face crop · frame sampling"]
        MODELS["Models<br/>ResNet50 · EfficientNet-B0 · Xception"]
        CAM["Grad-CAM Explainer"]
    end

    subgraph Data["🗄️ Data Layer"]
        DB[("MySQL<br/>users, predictions, results,<br/>models, files, audit_logs")]
        FS["File Storage<br/>uploads / heatmaps"]
    end

    UI -->|HTTPS REST + JWT| ROUT
    ROUT --> AUTH
    ROUT --> SVC
    SVC --> MLW
    MLW --> PRE --> MODELS --> CAM
    SVC --> DB
    MLW --> FS
    SVC --> FS
```

---

## 2.2 Low-Level Architecture (Backend Modules)

```mermaid
flowchart LR
    subgraph backend["backend/app"]
        main["main.py<br/>app factory · CORS · router mount"]
        subgraph routers
            r1["auth.py"]
            r2["detection.py"]
            r3["history.py"]
            r4["analytics.py"]
            r5["admin.py"]
        end
        subgraph services
            s1["auth_service.py"]
            s2["detection_service.py"]
            s3["history_service.py"]
            s4["analytics_service.py"]
        end
        subgraph models_orm["models (ORM)"]
            m1["user.py"]
            m2["prediction.py"]
            m3["detection_result.py"]
            m4["ml_model.py"]
            m5["uploaded_file.py"]
            m6["audit_log.py"]
        end
        subgraph schemas
            sc["Pydantic DTOs"]
        end
        subgraph mlw["ml wrappers"]
            w1["model_registry.py"]
            w2["inference.py"]
            w3["gradcam.py"]
            w4["preprocess.py"]
        end
        db["database/session.py"]
    end
    main --> routers --> services --> models_orm --> db
    services --> mlw
    routers --> schemas
```

---

## 2.3 Component Diagram

```mermaid
flowchart TB
    FE["React Frontend"] -->|Axios| GW["FastAPI Gateway"]
    GW --> AUTHC["Authentication Component"]
    GW --> DETC["Detection Component"]
    GW --> HISTC["History Component"]
    GW --> ANLC["Analytics Component"]
    DETC --> INFC["Inference Engine"]
    INFC --> PREC["Preprocessing"]
    INFC --> MODC["Model Registry"]
    INFC --> CAMC["Grad-CAM Explainer"]
    AUTHC --> DBC["Persistence (SQLAlchemy)"]
    DETC --> DBC
    HISTC --> DBC
    ANLC --> DBC
    DETC --> STO["File Storage"]
    DBC --> MYSQL[("MySQL")]
```

---

## 2.4 Class Diagram (Domain / ORM)

```mermaid
classDiagram
    class User {
        +int id
        +str username
        +str email
        +str hashed_password
        +str role
        +bool is_active
        +datetime created_at
        +register()
        +authenticate()
    }
    class Prediction {
        +int id
        +int user_id
        +int file_id
        +int model_id
        +str media_type
        +str verdict
        +float confidence
        +datetime created_at
    }
    class DetectionResult {
        +int id
        +int prediction_id
        +int frame_index
        +float fake_probability
        +str heatmap_path
    }
    class MLModel {
        +int id
        +str name
        +str architecture
        +str weights_path
        +float accuracy
        +bool is_active
    }
    class UploadedFile {
        +int id
        +int user_id
        +str filename
        +str storage_path
        +str media_type
        +int size_bytes
    }
    class AuditLog {
        +int id
        +int user_id
        +str action
        +str ip_address
        +datetime created_at
    }

    User "1" --> "*" Prediction : makes
    User "1" --> "*" UploadedFile : uploads
    User "1" --> "*" AuditLog : generates
    UploadedFile "1" --> "*" Prediction : source
    MLModel "1" --> "*" Prediction : used_by
    Prediction "1" --> "*" DetectionResult : contains
```

---

## 2.5 Sequence Diagram — Image Detection

```mermaid
sequenceDiagram
    actor U as User
    participant FE as React UI
    participant API as FastAPI /detect/image
    participant SVC as DetectionService
    participant ML as InferenceEngine
    participant DB as MySQL
    participant FS as FileStorage

    U->>FE: Select image + model, click Detect
    FE->>API: POST /detect/image (JWT, file)
    API->>API: Validate JWT + file type/size
    API->>FS: Save uploaded file
    API->>SVC: detect(file, model)
    SVC->>ML: preprocess + face crop
    ML->>ML: forward pass (CNN)
    ML->>ML: Grad-CAM heatmap
    ML-->>SVC: verdict, confidence, heatmap
    SVC->>FS: Save heatmap
    SVC->>DB: INSERT prediction + result + file
    SVC-->>API: result DTO
    API-->>FE: 200 {verdict, confidence, heatmap_url}
    FE-->>U: Show verdict + heatmap overlay
```

---

## 2.6 Sequence Diagram — Video Detection

```mermaid
sequenceDiagram
    actor U as User
    participant FE as React UI
    participant API as FastAPI /detect/video
    participant SVC as DetectionService
    participant ML as InferenceEngine
    participant DB as MySQL

    U->>FE: Upload video, click Detect
    FE->>API: POST /detect/video (JWT, file)
    API->>SVC: detect_video(file, model)
    SVC->>ML: sample N frames + detect faces
    loop each sampled frame
        ML->>ML: forward pass → fake_prob
    end
    ML->>ML: aggregate (mean / max voting)
    ML-->>SVC: verdict, confidence, per-frame results
    SVC->>DB: INSERT prediction + N detection_results
    SVC-->>API: aggregated result
    API-->>FE: 200 {verdict, confidence, frames[]}
    FE-->>U: Show verdict + timeline of fake frames
```

---

## 2.7 Activity Diagram — Detection Flow

```mermaid
flowchart TD
    A([Start]) --> B[User uploads media]
    B --> C{Authenticated?}
    C -- No --> Z[Return 401] --> E([End])
    C -- Yes --> D{Valid type & size?}
    D -- No --> Y[Return 400] --> E
    D -- Yes --> F[Store file]
    F --> G{Image or Video?}
    G -- Image --> H[Detect & crop face]
    G -- Video --> I[Sample frames + detect faces]
    H --> J[Run CNN inference]
    I --> J
    J --> K[Compute confidence]
    K --> L[Generate Grad-CAM]
    L --> M[Persist prediction + result]
    M --> N[Return verdict + heatmap]
    N --> E
```

---

## 2.8 Use Case Diagram

```mermaid
flowchart LR
    guest([Guest])
    user([User])
    admin([Admin])

    guest --- UC1((Register))
    guest --- UC2((Login))
    user --- UC3((Detect Image))
    user --- UC4((Detect Video))
    user --- UC5((Compare Models))
    user --- UC6((View History))
    user --- UC7((View Analytics))
    user --- UC8((View Grad-CAM))
    admin --- UC9((Manage Users))
    admin --- UC10((Manage Models))
    admin --- UC11((View Audit Logs))
    admin --- UC7
    admin --- UC6
    user --- UC2
```

---

## 2.9 Data Flow Diagrams

### DFD Level 0 (Context)

```mermaid
flowchart LR
    U[User] -->|media, credentials| S((Deepfake Detection System))
    S -->|verdict, confidence, heatmap, history| U
    A[Admin] -->|manage users/models| S
    S -->|reports, logs| A
    S <-->|read/write| DB[(MySQL)]
```

### DFD Level 1

```mermaid
flowchart TB
    U[User]
    U -->|register/login| P1[1.0 Authentication]
    U -->|upload media| P2[2.0 Detection]
    U -->|request| P3[3.0 History]
    U -->|request| P4[4.0 Analytics]

    P1 <--> D1[(Users)]
    P2 --> D2[(UploadedFiles)]
    P2 --> D3[(Predictions)]
    P2 --> D4[(DetectionResults)]
    P2 -->|use| D5[(Models)]
    P3 <--> D3
    P4 <--> D3
    P1 --> D6[(AuditLogs)]
    P2 --> D6

    P2 -->|verdict + heatmap| U
    P3 -->|history| U
    P4 -->|charts| U
```

### DFD Level 2 (Detection Process — expand 2.0)

```mermaid
flowchart TB
    IN[/Uploaded media/]
    IN --> P21[2.1 Validate File]
    P21 --> P22[2.2 Preprocess<br/>face crop / frame sample]
    P22 --> P23[2.3 Model Inference]
    P23 --> P24[2.4 Confidence Scoring]
    P24 --> P25[2.5 Grad-CAM Generation]
    P25 --> P26[2.6 Persist Results]
    P26 --> D3[(Predictions)]
    P26 --> D4[(DetectionResults)]
    P22 --> D2[(UploadedFiles)]
    P23 --> D5[(Models)]
    P26 --> OUT[/Verdict + Heatmap/]
```
