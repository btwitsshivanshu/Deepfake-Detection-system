# Phase 12 — Viva Preparation (100 Questions & Answers)

Grouped into: Project/General · Deep Learning · Research · System Design ·
Backend/Frontend/Database. Concise, exam-ready answers.

---

## A. Project & General (Q1–Q20)

1. **What is your project about?**
   An explainable full-stack system that detects deepfake images and videos using CNNs
   (ResNet50, EfficientNet-B0, Xception), returns a confidence score, and highlights
   manipulated regions with Grad-CAM.

2. **What is a deepfake?** Synthetic media where a person's face is swapped or reenacted using
   deep learning (GANs/autoencoders), producing realistic but fake images/videos.

3. **Why is deepfake detection important?** To combat misinformation, fraud, identity theft,
   and defamation, and to preserve digital trust.

4. **What problem does your project solve?** Automated, explainable authentication of facial
   media that humans can no longer reliably judge.

5. **What are the main features?** Image/video detection, confidence scores, Grad-CAM heatmaps,
   model comparison, history, analytics, JWT auth.

6. **Which dataset did you use?** A balanced 3,000-image real/fake face dataset (public
   HuggingFace `Hemg/deepfake-and-real-images`, 1,500 real + 1,500 fake, split 70/15/15).
   The pipeline also supports the standard video benchmarks (FaceForensics++, Celeb-DF, DFDC)
   for scale-up.

7. **What is your tech stack?** PyTorch, OpenCV, FastAPI, SQLAlchemy, MySQL, React, Tailwind.

8. **Why three models?** To compare architectures and select the best (Xception ≈ ResNet50)
   while showing trade-offs (EfficientNet-B0 is the smallest/fastest).

9. **What accuracy did you achieve?** On the held-out test split: XceptionNet **96.4% acc /
   0.991 AUC**, ResNet50 96.2% / 0.993, EfficientNet-B0 89.1% / 0.972. (Trained via transfer
   learning on the HuggingFace face dataset; benchmark-grade video numbers require retraining
   on FaceForensics++-style video frames.)

10. **Who are the users?** Guests, registered users, and admins (role-based access).

11. **What is the novelty?** Integration of strong detection + Grad-CAM explainability +
    secure full-stack app with history and analytics.

12. **What are the limitations?** Face-only, batch (not live), no audio, and sensitivity to
    heavy compression / unseen generators.

13. **How is confidence computed?** Softmax probability of the predicted class; for video, the
    aggregated fake probability.

14. **How do you handle video?** Sample N frames, crop faces, run per-frame inference, aggregate
    (mean/max/vote).

15. **What is Grad-CAM used for?** To visualize which facial regions drove the "fake" decision.

16. **How is the project structured?** `ml/`, `backend/`, `frontend/`, `database/`, `docs/`.

17. **What software principles did you follow?** Clean/layered architecture, SOLID, DRY,
    security-by-design.

18. **How is data stored?** MySQL with six tables (users, models, uploaded_files, predictions,
    detection_results, audit_logs).

19. **Is it production-ready?** It's a production-quality prototype; future work adds Docker,
    cloud deploy, and ensembles.

20. **What did you learn?** End-to-end ML system building: data, training, explainability, API,
    frontend, and security.

---

## B. Deep Learning (Q21–Q50)

21. **What is a CNN?** A neural network using convolutional filters to learn spatial feature
    hierarchies from images.

22. **What is convolution?** A sliding dot-product between a learnable kernel and input regions
    producing feature maps.

23. **What is pooling?** Downsampling (e.g., max/average) that reduces spatial size and adds
    translation invariance.

24. **What is ReLU?** $f(x)=\max(0,x)$ — a non-linear activation that mitigates vanishing
    gradients and adds sparsity.

25. **What is a fully connected layer?** A dense layer connecting every input to every output,
    used in the classifier head.

26. **What is ResNet's key idea?** Residual (skip) connections $y=F(x)+x$ that enable very deep
    networks by easing gradient flow.

27. **What is EfficientNet's key idea?** Compound scaling of depth, width, and resolution for
    optimal accuracy/efficiency.

28. **What is Xception's key idea?** Depthwise-separable convolutions that decouple spatial and
    channel correlations, improving efficiency and accuracy.

29. **What is a depthwise-separable convolution?** A depthwise conv (per-channel) followed by a
    pointwise 1×1 conv — far fewer parameters than standard conv.

30. **What is transfer learning?** Reusing ImageNet-pretrained weights and fine-tuning for the
    deepfake task to save data and time.

31. **What is fine-tuning?** Continuing training of a pretrained model on the target dataset,
    often with a lower learning rate.

32. **What is the loss function used?** Cross-entropy loss with label smoothing.

33. **What is label smoothing?** Softening one-hot targets to reduce overconfidence and improve
    calibration/generalization.

34. **What optimizer did you use?** AdamW (Adam with decoupled weight decay).

35. **What is the difference between Adam and AdamW?** AdamW decouples weight decay from the
    gradient update, giving better regularization.

36. **What is a learning-rate scheduler?** A policy that changes LR over training; we use cosine
    annealing.

37. **What is mixed-precision training?** Using FP16/FP32 together (AMP) to speed up training and
    save memory with minimal accuracy loss.

38. **What is overfitting?** When a model memorizes training data and generalizes poorly; we use
    augmentation, dropout, and early stopping.

39. **What is dropout?** Randomly zeroing activations during training to prevent co-adaptation
    and overfitting.

40. **What is early stopping?** Halting training when validation loss stops improving to avoid
    overfitting.

41. **What is data augmentation?** Applying label-preserving transforms (flips, noise, JPEG,
    color) to expand and diversify data.

42. **Why JPEG-compression augmentation?** Deepfakes often appear compressed online; it improves
    robustness to real-world artifacts.

43. **What is batch normalization?** Normalizing layer inputs per mini-batch to stabilize and
    speed up training.

44. **What is the vanishing gradient problem?** Gradients shrinking through deep layers; solved
    by ReLU, residuals, and normalization.

45. **What is softmax?** Converts logits into a probability distribution over classes.

46. **What is the difference between sigmoid and softmax?** Sigmoid is per-class independent
    (binary/multi-label); softmax is mutually exclusive multi-class.

47. **How does Grad-CAM work?** It back-props the target class score, GAP-pools gradients of the
    last conv layer into weights, and forms a weighted, ReLU'd activation map.

48. **Why the last convolutional layer for Grad-CAM?** It retains high-level semantics while
    preserving spatial location.

49. **What is the difference between CAM and Grad-CAM?** CAM needs a GAP architecture; Grad-CAM
    generalizes to any CNN using gradients.

50. **What are common deepfake artifacts?** Blending boundaries, warping, inconsistent lighting,
    unnatural texture, temporal flicker, and frequency anomalies.

---

## C. Research & Datasets (Q51–Q65)

51. **What is FaceForensics++?** A benchmark of 1000 real videos with four manipulation methods
    (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

52. **What is Celeb-DF?** A high-quality deepfake dataset of celebrities with reduced visual
    artifacts, making detection harder.

53. **What is DFDC?** The large-scale Deepfake Detection Challenge dataset by Meta/Kaggle.

54. **How did you split data?** Stratified 70/15/15 train/val/test.

55. **Why stratified split?** To preserve class balance across all splits.

56. **What is a GAN?** A generator vs. discriminator adversarial framework that learns to
    synthesize realistic data.

57. **How are deepfakes generated?** Via autoencoders/GANs that encode a source face and decode
    it onto a target identity.

58. **What is cross-dataset generalization?** Performance on datasets unseen during training —
    a key robustness measure.

59. **What metrics matter for detection?** Accuracy, precision, recall, F1, and especially
    ROC-AUC.

60. **Why is ROC-AUC useful?** It is threshold-independent and measures class separability.

61. **What is precision vs recall trade-off?** Higher precision reduces false alarms; higher
    recall catches more fakes; F1 balances them.

62. **What is a confusion matrix?** A table of TP/FP/FN/TN showing error types.

63. **What baseline does FaceForensics++ recommend?** XceptionNet — hence our top model choice.

64. **What are alternative approaches?** MesoNet, Capsule-Forensics, frequency analysis, and
    temporal/transformer models.

65. **How could you improve generalization?** Diverse training data, frequency features,
    augmentation, and ensembles.

---

## D. System Design & Architecture (Q66–Q85)

66. **What architecture pattern did you use?** Layered/clean architecture with separated
    presentation, API, service, domain, and persistence layers.

67. **What is FastAPI?** A modern, async Python web framework with automatic OpenAPI docs and
    Pydantic validation.

68. **Why FastAPI over Flask?** Async support, built-in validation, type hints, and auto docs.

69. **What is JWT?** A signed JSON token (header.payload.signature) used for stateless
    authentication.

70. **How does your auth flow work?** Login verifies bcrypt password, issues access + refresh
    JWTs; protected routes validate the access token.

71. **Why hash passwords with bcrypt?** It is slow and salted, resisting brute-force and rainbow
    tables.

72. **What is the difference between access and refresh tokens?** Access is short-lived for
    requests; refresh is long-lived to obtain new access tokens.

73. **What is CORS and how did you handle it?** Cross-Origin Resource Sharing; configured via
    FastAPI middleware to allow the frontend origin.

74. **How do you validate uploads?** Check extension/MIME and enforce a max size to prevent
    abuse.

75. **What is SQLAlchemy?** A Python ORM mapping classes to database tables.

76. **What is an ORM?** Object-Relational Mapping — work with objects instead of raw SQL.

77. **How is the ML model served efficiently?** Loaded once and cached in-process to avoid
    reloading weights per request.

78. **What is dependency injection here?** FastAPI `Depends` injects DB sessions and the current
    user into routes/services.

79. **How do you separate concerns?** Thin routers → services (business logic) → models
    (persistence); ML engine is a separate module.

80. **How does the frontend talk to the backend?** Axios REST calls with a JWT interceptor;
    Vite proxies `/api` and `/storage`.

81. **How is state managed in React?** Auth via Context; component-local state via hooks.

82. **How is routing protected?** A `Protected` wrapper redirects unauthenticated users to
    login.

83. **How are heatmaps delivered?** Saved to a static dir and served via a mounted static route,
    referenced by `heatmap_url`.

84. **How would you scale this?** Stateless API behind a load balancer, shared object storage,
    and a model-serving layer/GPU workers.

85. **How do you ensure data integrity?** Foreign keys, constraints (CHECK on probabilities),
    and transactional writes.

---

## E. Implementation, DB & Misc (Q86–Q100)

86. **Describe your database schema.** Six tables: users, models, uploaded_files, predictions,
    detection_results, audit_logs, linked by foreign keys.

87. **Why store detection_results separately?** To keep per-frame/per-face breakdowns for videos
    (1-to-many with predictions).

88. **What are audit logs for?** Security traceability of logins, uploads, and admin actions.

89. **What indexing did you add?** Indexes on FK columns and on `predictions.created_at/verdict`
    for fast history/analytics.

90. **How do you prevent SQL injection?** ORM parameterization — no string-concatenated SQL.

91. **What is Pydantic used for?** Request/response validation and serialization (DTOs).

92. **How is the model chosen at inference?** By architecture name from the active model registry
    (defaults to Xception).

93. **How is multi-model comparison implemented?** The file is read once and each active model is
    run on a fresh in-memory copy.

94. **What happens if no face is detected?** The system falls back to the whole resized frame.

95. **How is video verdict aggregated?** Mean (default), max, or majority vote of per-frame fake
    probabilities.

96. **How do you make results reproducible?** Global seeding of Python/NumPy/PyTorch and
    deterministic cuDNN.

97. **How do you handle large files?** Enforced max upload size and streamed disk writes.

98. **What testing/validation exists for data?** A dataset validator checks class balance and
    corrupt images before training.

99. **What are the security measures overall?** Hashed passwords, JWT, input/upload validation,
    CORS, RBAC, and audit logging (OWASP-aware).

100. **What is the biggest challenge and how did you solve it?** Balancing accuracy,
     explainability, and usability — solved by reusing one inference engine across research and
     serving, adding Grad-CAM, and a clean layered architecture.

---

### Rapid-fire one-liners
- **Epochs:** up to 15 (GPU) with early stopping. · **Batch size:** 32. · **Image size:** 224×224.
- **Training:** transfer learning (ImageNet-pretrained), AdamW + cosine LR, AMP mixed precision, on an NVIDIA RTX 4050 GPU (~18 s/epoch).
- **LR:** 1e-4, cosine schedule. · **Positive class:** fake (index 1).
- **Best model:** XceptionNet. · **Explainability:** Grad-CAM. · **Auth:** JWT + bcrypt.
