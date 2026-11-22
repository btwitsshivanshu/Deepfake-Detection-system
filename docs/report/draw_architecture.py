"""
Draw the system architecture flow chart.

    python docs/report/draw_architecture.py

Output: docs/report/architecture_flowchart.png (and .pdf for print)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture_flowchart"
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

# palette: (border, fill)
BLUE = ("#2F6FC0", "#EAF2FC")
GREEN = ("#2E8B3C", "#EDF7EE")
AMBER = ("#B8860B", "#FFF7DD")
PURPLE = ("#5B33B0", "#F4F0FC")
TEAL = ("#16808F", "#E6F5F7")
RED = ("#C8334A", "#FDEEF0")
INDIGO = ("#3F51B5", "#ECEEFA")
GREY = ("#5F6B7A", "#F3F5F8")
INK = "#1B1F24"

fig, ax = plt.subplots(figsize=(20, 12.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 63)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, color, title=None, lines=(), title_size=12.5, size=10.2, pad_top=1.15,
        bullet=True, align="left", lw=1.6, radius=0.8, line_gap=1.62):
    edge, fill = color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ty = y + h - pad_top
    tx = x + 0.9 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    if title:
        ax.text(tx, ty, title, fontsize=title_size, fontweight="bold", color=INK, ha=ha, va="top", zorder=3)
        ty -= 2.0
    for ln in lines:
        cont = ln.startswith("   ")
        txt = ("•  " + ln) if (bullet and align == "left" and not cont) else ("     " + ln.strip() if cont else ln)
        ax.text(tx, ty, txt, fontsize=size, color="#2B2F36", ha=ha, va="top", zorder=3)
        ty -= line_gap


def group(x, y, w, h, color, label):
    edge, fill = color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=2.2, edgecolor=edge, facecolor=fill, zorder=1))
    ax.text(x + 1.0, y + h - 0.9, label, fontsize=14, fontweight="bold", color=edge, ha="left", va="top", zorder=3)


def arrow(x1, y1, x2, y2, label=None, num=None, color=INK, ls="-", lx=None, ly=None, la="center", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17,
                                 linewidth=1.7, color=color, linestyle=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        mx = lx if lx is not None else (x1 + x2) / 2
        my = ly if ly is not None else (y1 + y2) / 2 + 0.9
        t = (f"{num}  " if num else "") + label
        ax.text(mx, my, t, fontsize=9.6, color=color, ha=la, va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


def badge(x, y, n, color="#1B1F24"):
    ax.add_patch(plt.Circle((x, y), 0.95, color=color, zorder=6))
    ax.text(x, y - 0.05, str(n), fontsize=9.5, color="white", fontweight="bold", ha="center", va="center", zorder=7)


# ------------------------------------------------------------------ title
ax.text(50, 61.8, "AuthentiScan — System Architecture & Flow", fontsize=22, fontweight="bold",
        color="#1B2A4A", ha="center", va="top")
ax.text(50, 59.3, "Deepfake detection for images and videos: three specialised detectors + a learned combiner, "
                  "served through a secure web application", fontsize=12, color="#4A5361", ha="center", va="top")

# ------------------------------------------------------------------ live request flow
group(0.6, 20.2, 98.8, 37.6, ("#9AA5B4", "#FBFCFD"), "A.  WHAT HAPPENS WHEN A USER UPLOADS A FILE")

box(1.8, 41.5, 10.2, 9.5, GREY, "User", ["Browser", "Uploads an image", "or a video"], bullet=False, align="center",
    title_size=13)

box(14.6, 23.0, 16.4, 31.0, BLUE, "Frontend", [
    "React + Vite + Tailwind",
    "Login / Register",
    "Dashboard",
    "Image Detection",
    "Video Detection",
    "History",
    "Analytics (Recharts)",
    "Axios: sends the JWT",
    "   with every request",
    "Result card: verdict,",
    "   detector scores, heatmap",
], title_size=13)

box(33.6, 37.5, 17.4, 16.5, GREEN, "Backend (FastAPI)", [
    "Auth: JWT login,",
    "   bcrypt password hashes",
    "Detection service",
    "History & Analytics",
    "Stores uploads + heatmaps",
], title_size=13)

box(33.6, 23.0, 17.4, 12.3, AMBER, "MySQL database", [
    "users · models",
    "uploaded_files · predictions",
    "detection_results",
    "audit_logs",
], title_size=13)

# ML inference group
group(53.6, 21.4, 45.2, 33.4, PURPLE, "ML Inference Engine")
ax.text(97.9, 53.9, "ml/ensemble.py", fontsize=9.5, color=PURPLE[0], ha="right", va="top", style="italic")

box(55.0, 46.6, 42.4, 5.8, ("#7B8794", "#FFFFFF"), "1.  Prepare the input", [
    "Image  →  if it is a phone screenshot, crop out the photo",
    "Video  →  sample 20 frames; each frame follows the same path",
], bullet=False, size=9.8, title_size=11.2, pad_top=0.9, line_gap=1.45)

box(55.0, 41.3, 27.4, 4.2, ("#7B8794", "#FFFFFF"), "2.  Find the face — OpenCV YuNet", [
    "Face cropped with the same framing as the training images"], bullet=False, size=9.6, title_size=11.2,
    pad_top=0.8)

det_y, det_h, det_w = 32.6, 7.6, 13.4
box(55.0, det_y, det_w, det_h, TEAL, "3a  XceptionNet", ["GAN-generated faces", "face crop · 224 px"],
    bullet=False, align="center", title_size=11.5, size=9.6, pad_top=0.9)
box(69.0, det_y, det_w, det_h, RED, "3b  EfficientNet-B4 + SBI", ["Face swaps / edits", "face crop · 256 px"],
    bullet=False, align="center", title_size=11.5, size=9.6, pad_top=0.9)
box(83.9, det_y, det_w, det_h, INDIGO, "3c  CLIP ViT-L/14", ["AI-generated images", "whole image · 224 px"],
    bullet=False, align="center", title_size=11.5, size=9.6, pad_top=0.9)

box(55.0, 27.2, 42.4, 4.0, ("#7B8794", "#FFFFFF"), "4.  Combiner — gradient-boosted trees (fitted on validation data)",
    ["merges the three fake probabilities into one"], bullet=False, size=9.6, title_size=11.2, pad_top=0.75,
    line_gap=1.4)

box(55.0, 22.0, 42.4, 4.4, GREEN, None, [], lw=1.8)
ax.text(76.2, 25.2, "5.  Verdict:  REAL (≤ 0.35)  ·  UNCERTAIN  ·  FAKE (≥ 0.65)", fontsize=10.6,
        fontweight="bold", color="#1E5E29", ha="center", va="center", zorder=5)
ax.text(76.2, 23.15, "+ confidence  ·  likely fake type  ·  each detector's score  ·  Grad-CAM heatmap",
        fontsize=9.8, color="#1E5E29", ha="center", va="center", zorder=5)

# inner arrows
arrow(68.7, 46.6, 68.7, 45.5)                       # prepare -> face
arrow(61.7, 41.3, 61.7, det_y + det_h)              # face -> xception
arrow(75.7, 41.3, 75.7, det_y + det_h)              # face -> sbi
arrow(90.6, 46.6, 90.6, det_y + det_h)              # prepare -> clip (whole image)
ax.text(91.2, 43.6, "whole\nimage", fontsize=9, color=INDIGO[0], ha="left", va="center", style="italic")
for xx in (61.7, 75.7, 90.6):
    arrow(xx, det_y, xx, 31.2)                      # detectors -> combiner
arrow(76.2, 27.2, 76.2, 26.4)                       # combiner -> verdict

# request / response arrows
arrow(12.0, 48.5, 14.6, 48.5)
badge(13.3, 50.4, 1)
arrow(14.6, 44.0, 12.0, 44.0, color="#2E8B3C")
badge(13.3, 42.1, 7, "#2E8B3C")

arrow(31.0, 50.5, 33.6, 50.5)
badge(32.3, 52.4, 2)
arrow(33.6, 42.0, 31.0, 42.0, color="#2E8B3C")
badge(32.3, 40.1, 6, "#2E8B3C")

arrow(51.0, 49.5, 55.0, 49.5)
badge(53.0, 51.4, 3)
arrow(53.6, 43.0, 51.0, 43.0, color="#2E8B3C")
badge(52.3, 41.1, 4, "#2E8B3C")

arrow(42.3, 37.5, 42.3, 35.3)
badge(44.3, 36.4, 5)

# legend for numbered steps
legend = ["1  Upload file", "2  POST /api/detect (with JWT)", "3  Run the ML pipeline", "4  Result returned",
          "5  Save prediction & history", "6  JSON response", "7  Show verdict, scores, heatmap"]
ax.text(1.8, 38.6, "Request steps", fontsize=10.5, fontweight="bold", color=INK, ha="left", va="top")
for i, t in enumerate(legend):
    ax.text(1.8, 36.6 - i * 1.75, t, fontsize=9.2, color="#2B2F36", ha="left", va="top")

# ------------------------------------------------------------------ offline training
group(0.6, 0.6, 98.8, 18.2, ("#9AA5B4", "#FBFCFD"), "B.  OFFLINE — HOW THE MODELS WERE BUILT (done once, on an RTX 4050 laptop GPU)")

ty, th = 1.8, 13.2
box(2.0, ty, 22.0, th, AMBER, "Datasets (4 public)", [
    "Hemg — 190k real & GAN-style faces",
    "140k Faces — real vs StyleGAN",
    "OpenFake — real vs 89 AI generators",
    "   (Gemini, Imagen, Midjourney, Flux…)",
    "DeepFakeFace — face swaps (test only)",
], size=9.6, title_size=12)
box(27.0, ty, 22.0, th, TEAL, "Training", [
    "XceptionNet on mixed GAN data",
    "EfficientNet-B4 with Self-Blended",
    "   Images (made from 60k real faces)",
    "CLIP ViT-L/14 frozen + linear head",
    "Augmentation: JPEG, blur, noise",
], size=9.6, title_size=12)
box(52.0, ty, 22.0, th, INDIGO, "Evaluation & combiner", [
    "Combiner fitted on validation sets",
    "Benchmark: 8,000 test decisions",
    "   (clean + WhatsApp-compressed)",
    "Accuracy 70.0% → 84.5%",
    "AUC 0.925 · screenshots 84.4%",
], size=9.6, title_size=12)
box(77.0, ty, 21.0, th, GREY, "Model files", [
    "xception_best.pth",
    "efficientnet_b4_sbi_best.pth",
    "clip_vitl14_head.pt",
    "combiner.pkl",
    "+ YuNet face detector (.onnx)",
], size=9.6, title_size=12)

arrow(24.0, ty + th / 2, 27.0, ty + th / 2)
arrow(49.0, ty + th / 2, 52.0, ty + th / 2)
arrow(74.0, ty + th / 2, 77.0, ty + th / 2)
arrow(87.5, ty + th, 87.5, 21.4, label="loaded when the server starts", color=PURPLE[0], ls="--",
      lx=86.7, ly=17.3, la="right")

fig.savefig(OUT.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("saved", OUT.with_suffix(".png"))
