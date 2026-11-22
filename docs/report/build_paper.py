"""
Build the IEEE-style research paper PDF.

    python docs/report/build_paper.py

Output: docs/report/AuthentiScan_Research_Paper.pdf
Layout: A4, two columns; page 2 carries the full-width architecture figure,
and the page where Section VI starts carries the full-width comparison table.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
OUT = HERE / "AuthentiScan_Research_Paper.pdf"
FIG = HERE / "architecture_flowchart.png"

F = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("TNR", str(F / "times.ttf")))
pdfmetrics.registerFont(TTFont("TNR-B", str(F / "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("TNR-I", str(F / "timesi.ttf")))
pdfmetrics.registerFont(TTFont("TNR-BI", str(F / "timesbi.ttf")))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

W, H = A4
M = 54                    # page margin
GAP = 14                  # column gap
CW = (W - 2 * M - GAP) / 2  # column width
FW = W - 2 * M            # full text width

# ---------------------------------------------------------------- styles
BODY = ParagraphStyle("body", fontName="TNR", fontSize=9.8, leading=11.9, alignment=TA_JUSTIFY,
                      firstLineIndent=10, spaceAfter=2)
BODY0 = ParagraphStyle("body0", parent=BODY, firstLineIndent=0)
ABS = ParagraphStyle("abs", fontName="TNR-B", fontSize=9, leading=10.9, alignment=TA_JUSTIFY, spaceAfter=4)
SEC = ParagraphStyle("sec", fontName="TNR", fontSize=9.8, leading=12, alignment=TA_CENTER, spaceBefore=8,
                     spaceAfter=4, keepWithNext=1)
SUB = ParagraphStyle("sub", fontName="TNR-I", fontSize=9.8, leading=12, alignment=TA_LEFT, spaceBefore=5,
                     spaceAfter=2, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, firstLineIndent=0, leftIndent=11, bulletIndent=2, spaceAfter=1.5)
EQ = ParagraphStyle("eq", fontName="TNR-I", fontSize=9.8, leading=13, alignment=TA_CENTER, spaceBefore=2, spaceAfter=3)
CAP = ParagraphStyle("cap", fontName="TNR", fontSize=8, leading=9.6, alignment=TA_CENTER, spaceBefore=4, spaceAfter=3)
TCAP = ParagraphStyle("tcap", fontName="TNR", fontSize=8, leading=9.6, alignment=TA_CENTER, spaceBefore=6, spaceAfter=3)
CELL = ParagraphStyle("cell", fontName="TNR", fontSize=7.6, leading=8.9)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="TNR-B")
CELL_C = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)
REF = ParagraphStyle("ref", fontName="TNR", fontSize=7.9, leading=9.4, alignment=TA_JUSTIFY, leftIndent=14,
                     firstLineIndent=-14, spaceAfter=1.6)
TITLE = ParagraphStyle("title", fontName="TNR", fontSize=21, leading=25, alignment=TA_CENTER, spaceAfter=10)
AUTH = ParagraphStyle("auth", fontName="TNR", fontSize=10.5, leading=13, alignment=TA_CENTER)
AFF = ParagraphStyle("aff", fontName="TNR-I", fontSize=9, leading=11, alignment=TA_CENTER)


def md(text):
    t = escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", t)
    return t


def P(text, style=BODY):
    return Paragraph(md(text), style)


def sec(roman, title):
    return Paragraph(f"{roman}. {escape(title.upper())}", SEC)


def sub(letter, title):
    return Paragraph(f"{letter}. {escape(title)}", SUB)


def bullets(items):
    return [Paragraph(md(i), BUL, bulletText="•") for i in items]


def table(rows, widths, center_cols=(), bold_last=False, font=None):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            st = CELL_B if r == 0 else (CELL_C if c in center_cols else CELL)
            if bold_last and r == len(rows) - 1:
                st = ParagraphStyle("bl", parent=st, fontName="TNR-B")
            if font:
                st = ParagraphStyle("f", parent=st, fontSize=font, leading=font + 1.3)
            cells.append(Paragraph(md(v), st))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def ttable(caption, rows, widths, **kw):
    return KeepTogether([Paragraph(caption, TCAP), table(rows, widths, **kw), Spacer(1, 4)])


# ---------------------------------------------------------------- full-width page items
def comparison_table():
    rows = [
        ["Aspect", "FaceForensics++ / XceptionNet [1]", "Self-Blended Images [6]", "UnivFD (CLIP) [8]", "**AuthentiScan (this work)**"],
        ["Kinds of fake covered", "Face manipulations", "Face swaps", "AI-generated images", "**All three in one system**"],
        ["How decisions are combined", "—", "—", "—", "**Learned combiner** (gradient-boosted trees, fitted on validation data); names the likely fake type"],
        ["Evaluation across kinds of fake", "Mostly within one benchmark family", "Other face-swap datasets", "Other image generators",
         "**4 kinds of fake in one benchmark** (GAN faces, StyleGAN, diffusion/AR images, face swaps)"],
        ["Compressed (shared) media", "Compressed video", "Not the focus", "Not the focus", "**Every test image also WhatsApp-compressed** (8,000 decisions)"],
        ["Phone screenshots", "No", "No", "No", "**Automatic photo-region crop**: 72.9% → 84.4%, 0/400 false crops"],
        ["Abstains when unsure", "No", "No", "No", "**UNCERTAIN verdict** (13.4% of inputs; 89.1% accuracy on the rest)"],
        ["User-facing explanation", "No", "No", "No", "**Grad-CAM + per-detector scores + fake type**"],
        ["Deployment", "Research code", "Research code", "Research code", "**Web app**: JWT auth, image & video, history, analytics"],
        ["Hardware constraint", "Not a design constraint", "Not a design constraint", "Not a design constraint", "**Trained on one 6 GB laptop GPU**"],
    ]
    t = table(rows, [86, 92, 82, 82, 145], font=7.4)
    return t, "TABLE I. COMPARISON WITH THE BASE METHODS. AuthentiScan does not try to out-score each specialist on its own benchmark; it combines them and evaluates them under realistic conditions."


class Doc(BaseDocTemplate):
    def __init__(self, path, wide_pages, marks):
        super().__init__(str(path), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                         title="AuthentiScan: An Explainable Ensemble of Specialised Detectors for Deepfake Detection",
                         author="P. Satnani, S. Agarwal, U. Vishnoi, V. Gupta")
        self.wide_pages, self.marks, self.idx = wide_pages, marks, {}
        top_h = 168
        first = PageTemplate("first", [
            Frame(M, H - M - top_h, FW, top_h, id="title", leftPadding=0, rightPadding=0),
            Frame(M, M, CW, H - 2 * M - top_h - 6, id="c1", leftPadding=0, rightPadding=0),
            Frame(M + CW + GAP, M, CW, H - 2 * M - top_h - 6, id="c2", leftPadding=0, rightPadding=0)],
            onPage=self._footer)
        two = PageTemplate("two", self._cols(0), onPage=self._footer)
        templates = [first, two]
        for pno, (kind, height) in wide_pages.items():
            name = f"wide{pno}"
            templates.append(PageTemplate(name, self._cols(height + 14), onPage=self._wide(kind, height)))
        self.addPageTemplates(templates)
        self.idx = {t.id: i for i, t in enumerate(self.pageTemplates)}

    @staticmethod
    def _cols(reserved):
        h = H - 2 * M - reserved
        return [Frame(M, M, CW, h, id="c1", leftPadding=0, rightPadding=0),
                Frame(M + CW + GAP, M, CW, h, id="c2", leftPadding=0, rightPadding=0)]

    def _footer(self, c, doc):
        c.setFont("TNR", 8.5)
        c.drawCentredString(W / 2, 30, str(doc.page))

    def _wide(self, kind, height):
        def draw(c, doc):
            self._footer(c, doc)
            top = H - M
            if kind == "fig":
                iw, ih = PILImage.open(FIG).size
                h_img = FW * ih / iw
                c.drawImage(str(FIG), M, top - h_img, width=FW, height=h_img, mask="auto")
                cap = Paragraph("Fig. 1.  Architecture and data flow of AuthentiScan. (A) A request travels from the React "
                                "frontend through the FastAPI backend to the inference engine, where the input is prepared "
                                "(screenshot crop or video frame sampling), the face is located with YuNet, three specialised "
                                "detectors score it, and a learned combiner produces the verdict. (B) The offline pipeline "
                                "that produced the model files.", CAP)
                _, ch = cap.wrap(FW, 100)
                cap.drawOn(c, M, top - h_img - ch - 2)
            else:
                t, caption = comparison_table()
                cp = Paragraph(caption, TCAP)
                _, chh = cp.wrap(FW, 100)
                cp.drawOn(c, M, top - chh)
                _, th = t.wrap(FW, 600)
                t.drawOn(c, M, top - chh - 3 - th)
        return draw

    def handle_pageEnd(self):
        nxt = self.page + 1
        name = f"wide{nxt}" if nxt in self.wide_pages else "two"
        self._nextPageTemplateIndex = self.idx[name]
        super().handle_pageEnd()

    def afterFlowable(self, f):
        key = getattr(f, "mark", None)
        if key:
            self.marks.setdefault(key, self.page)


def wide_height(kind):
    if kind == "fig":
        iw, ih = PILImage.open(FIG).size
        cap = Paragraph("x" * 600, CAP)
        _, ch = cap.wrap(FW, 100)
        return FW * ih / iw + ch + 4
    t, caption = comparison_table()
    _, th = t.wrap(FW, 600)
    _, chh = Paragraph(caption, TCAP).wrap(FW, 100)
    return th + chh + 6


def marked(flowable, key):
    flowable.mark = key
    return flowable


# ---------------------------------------------------------------- content
def story():
    s = [P("AuthentiScan: An Explainable Ensemble of Specialised Detectors for Deepfake Image and Video Authentication", TITLE),
         P("Priyansh Satnani, Shivanshu Agarwal, Unnati Vishnoi, Vaibhav Gupta", AUTH),
         Spacer(1, 3),
         P("Department of Computer Science and Engineering, Shri Ram Murti Smarak College of Engineering & Technology, "
           "Bareilly, India", AFF),
         P("Under the guidance of Ms. Meenakshi Pathak", AFF),
         FrameBreak()]

    s += [Paragraph("<i>Abstract</i>—" + escape("Deepfake detectors are usually trained and evaluated on a single dataset, where they report "
            "near-perfect accuracy. We show that this hides a severe generalization gap: an XceptionNet detector "
            "fine-tuned on 181,000 face images reached 98.4% accuracy on its own test split, yet scored 48.8% on images "
            "from modern diffusion generators and 49.2% on face swaps — no better than chance. We present AuthentiScan, "
            "a deployable system that closes much of this gap with three specialised detectors built on established base "
            "methods: XceptionNet for GAN-generated faces, an EfficientNet-B4 trained with Self-Blended Images for face "
            "swaps, and a frozen CLIP ViT-L/14 with a linear head for AI-generated images, merged by a gradient-boosted "
            "combiner fitted only on validation data. The system also extracts the photo from phone screenshots, reports "
            "an explicit Uncertain verdict, names the likely manipulation type and explains decisions with Grad-CAM, inside "
            "a secure full-stack web application. On a held-out benchmark of 8,000 decisions spanning four kinds of fake "
            "media, each clean and WhatsApp-compressed, the ensemble raises accuracy from 70.0% to 84.5% (ROC-AUC 0.925); "
            "on 2,000 phone screenshots it raises accuracy from 72.9% to 84.4%. All models were trained on a single 6 GB "
            "laptop GPU."), ABS),
          Paragraph("<i>Index Terms</i>—deepfake detection, generalization, ensemble learning, self-blended "
                    "images, CLIP, explainable AI, Grad-CAM.", ABS)]

    # I. Introduction
    s += [sec("I", "Introduction"),
          P("Generative models can now produce faces, face swaps and complete photographs that people cannot reliably "
            "tell apart from real ones. Such media are used for misinformation, fraud and harassment, which makes "
            "automated detection important. The standard approach, popularised by FaceForensics++ [1], fine-tunes an "
            "ImageNet-pretrained CNN such as XceptionNet [2] on face crops from one benchmark and reports accuracy on a "
            "test split of the same benchmark."),
          P("In-domain accuracy, however, says little about the fakes users actually encounter, which come from many "
            "generators and are usually shared as compressed copies or screenshots. Recent surveys identify "
            "cross-dataset generalization and robustness as the main open problems [15], [16]. Methods that target "
            "generalization exist — Self-Blended Images (SBI) for face swaps [6] and CLIP-based universal detectors (UnivFD) "
            "for generated images [8] — but each addresses one kind of manipulation and is evaluated on clean benchmark "
            "data."),
          P("This paper describes AuthentiScan, which builds on these base methods. Its contributions are:"),
          *bullets([
              "**A measured generalization gap.** A detector with 98.4% in-domain accuracy falls to chance on face swaps "
              "and scores an *inverted* ROC-AUC of 0.215 on diffusion images.",
              "**A specialised-detector ensemble** that combines three base methods through a learned combiner, covers GAN "
              "faces, face swaps and diffusion/autoregressive images, and names the likely manipulation type.",
              "**A realistic evaluation protocol**: four kinds of fake, clean and compressed, with strict separation of "
              "training, validation and test data, plus a phone-screenshot benchmark.",
              "**Screenshot handling** that locates the embedded photo, restoring accuracy on screenshots to the level of "
              "original images with no false crops on ordinary photos.",
              "**An explainable, deployable system** with an Uncertain verdict, per-detector scores and Grad-CAM [5], "
              "trained entirely on consumer hardware.",
          ])]

    # II. Related work
    s += [sec("II", "Base Methods and Related Work"),
          sub("A", "XceptionNet baseline"),
          P("FaceForensics++ [1] established fine-tuned XceptionNet [2], which uses depthwise-separable convolutions, as "
            "the standard baseline for face-manipulation detection. ResNet [3] and EfficientNet [4] are common "
            "alternatives. Such detectors learn generator-specific artefacts and are known to degrade on unseen "
            "manipulation methods and datasets such as Celeb-DF [13]."),
          sub("B", "Self-Blended Images"),
          P("Shiohara and Yamasaki [6] train a face-swap detector on real faces only: each face is blended with a slightly "
            "transformed copy of itself, producing pseudo-fakes whose only cue is the blending boundary common to all swap "
            "tools. With an EfficientNet-B4 backbone [14], this gave strong cross-dataset results, approximately 0.93 "
            "ROC-AUC on Celeb-DF."),
          sub("C", "CLIP-based universal detection"),
          P("Ojha *et al.* [8] showed that a linear classifier on frozen CLIP ViT-L/14 features [7] generalizes to unseen "
            "generators far better than a CNN trained end-to-end, because CLIP's representation is not specialised to any "
            "generator's artefacts."),
          sub("D", "Gap addressed"),
          P("Each base method covers one kind of manipulation and is evaluated on clean benchmark images. None combines "
            "detectors for different kinds of fake, evaluates on compressed media and screenshots, abstains when "
            "uncertain, or is delivered as a usable application. Table I summarises the differences.")]

    # III. System
    s += [sec("III", "Proposed System"),
          sub("A", "Overview"),
          P("Fig. 1 shows the architecture. A user uploads an image or a video in a React frontend; a FastAPI backend "
            "authenticates the request with a JSON Web Token, stores the file and calls the inference engine; the "
            "prediction is stored in MySQL and returned with the verdict, confidence, likely fake type, per-detector "
            "scores and a Grad-CAM heatmap."),
          sub("B", "Input preparation and screenshot handling"),
          P("Videos are reduced to 20 uniformly sampled frames, each processed like an image. For images, AuthentiScan "
            "checks whether the input is a phone screenshot and, if so, extracts the embedded photo:"),
          *bullets([
              "Accept only screen-shaped inputs: aspect ratio ≥ 1.7, or ≥ 1.2 with a long side ≥ 1200 px.",
              "Estimate the app background as the most frequent colour.",
              "Mark rows that are mostly non-background and contain at least six distinct coarse colours; keep the "
              "tallest block of such rows, then repeat for columns.",
              "Accept the region only if it covers 8–85% of the screen, contains ≥ 300 distinct colours, and its "
              "surroundings are flat UI background.",
          ]),
          sub("C", "Face localisation"),
          P("Faces are detected with YuNet [11], a lightweight CNN detector in OpenCV that also returns five landmarks. "
            "Crops are taken with the same face-to-image ratio as the training images (0.36 of the crop width); "
            "Section VI-C shows why this matters."),
          sub("D", "Specialised detectors"),
          *bullets([
              "**GAN faces — XceptionNet** [2] (224 px face crop), fine-tuned on mixed GAN-face data.",
              "**Face swaps — EfficientNet-B4 + SBI** [6] (256 px face crop). Each pseudo-fake is",
          ]),
          P("x′ = M · T(x) + (1 − M) · x", EQ),
          P("where *x* is a real face, *T* applies colour, sharpness, ±5% scale and ±3% shift changes, and *M* is a soft, "
            "randomly deformed face mask built from the YuNet landmarks.", BODY0),
          *bullets([
              "**AI-generated images — CLIP ViT-L/14** [7], frozen, with a logistic-regression head on L2-normalised "
              "768-d features (whole image, 224 px), following [8].",
          ]),
          sub("E", "Combiner and verdict"),
          P("A gradient-boosted tree model (scikit-learn [17], depth 3, 200 iterations) maps the three fake probabilities "
            "to one probability *p*. Trees can express rules such as \"trust XceptionNet when it is confident\", which a "
            "weighted average cannot. The verdict is"),
          P("REAL if p ≤ 0.35;  FAKE if p ≥ 0.65;  otherwise UNCERTAIN,", EQ),
          P("and the likely fake type is that of the detector with the highest score. Grad-CAM [5] is computed from the "
            "more suspicious face detector.", BODY0)]

    # IV. Experimental setup
    s += [sec("IV", "Experimental Setup"),
          sub("A", "Datasets"),
          P("Four public datasets are used (Table II). OpenFake [9] is 3.4 TB in full, so individual parquet row groups "
            "were streamed from many shards. All OpenFake images, real and fake, were resized to a 256 px short side and "
            "re-encoded as JPEG to remove a format shortcut (Section VI-C). DeepFakeFace [10] is never used for training."),
          ttable("TABLE II. DATASETS", [
              ["Dataset", "Content", "Use"],
              ["Hemg [19]", "190,335 real and GAN-style face images", "Train / val / test"],
              ["140k Faces [20]", "Real FFHQ vs StyleGAN [12] faces", "Train / val / test"],
              ["OpenFake [9]", "Real photos vs 89 generators (e.g. Imagen, Gemini, GPT-image, Midjourney, Flux)",
               "6,462 / 1,018 / 1,776"],
              ["DeepFakeFace [10]", "30k real celebrities + 30k InsightFace swaps", "Val / test only"],
          ], [56, 116, 65]),
          sub("B", "Training"),
          Paragraph(md("All training used one NVIDIA RTX 4050 laptop GPU (6 GB) with mixed precision. XceptionNet: AdamW, "
            "learning rate ").replace("&amp;", "&") + "10<super>−4</super> (3×10<super>−5</super> " + md("for the final mixed-data pass), batch size 32, cosine schedule, label smoothing "
            "0.05. SBI: EfficientNet-B4 (Noisy Student weights) [14], AdamW with a one-cycle schedule, 3 epochs over "
            "174,564 real/pseudo-fake pairs built from 60,000 real faces. CLIP head: 10,462 images, each embedded clean "
            "and after random degradation (20,924 vectors); regularisation chosen on validation data. Augmentations "
            "include JPEG compression, downscaling, blur and noise (Albumentations [18]). The combiner was fitted on "
            "7,962 validation samples (four validation sets, clean and compressed)."), BODY),
          sub("C", "Evaluation protocol"),
          P("Each of the four test sets contains 500 real and 500 fake images that are excluded from all training and "
            "combiner fitting. Every image is scored as-is and after WhatsApp-style degradation (downscaling and JPEG "
            "quality 50), giving 8,000 decisions. A separate benchmark renders 2,000 test images inside synthetic phone "
            "screenshots (status bar, header, captions, buttons; half additionally JPEG-compressed). We report accuracy "
            "at a 0.5 threshold and ROC-AUC.")]

    # V. Results
    s += [sec("V", "Results"),
          sub("A", "Architecture comparison and scaling"),
          P("On a 3,000-image Hemg sample (450 test images), XceptionNet and ResNet50 were essentially tied and "
            "EfficientNet-B0 was weaker but smallest (Table III). Scaling XceptionNet's training data increased in-domain "
            "accuracy to 98.4% (Table IV), but unseen StyleGAN faces were detected at only ~51% until that data family was "
            "added."),
          ttable("TABLE III. STAGE-1 ARCHITECTURE COMPARISON (450 TEST IMAGES)", [
              ["Model", "Acc.", "Prec.", "Rec.", "F1", "AUC"],
              ["XceptionNet", "96.4", "96.5", "96.4", "96.5", "0.991"],
              ["ResNet50", "96.2", "95.6", "96.9", "96.2", "0.993"],
              ["EfficientNet-B0", "89.1", "90.7", "87.1", "88.9", "0.972"],
          ], [72, 33, 33, 33, 33, 33], center_cols=(1, 2, 3, 4, 5)),
          ttable("TABLE IV. EFFECT OF TRAINING DATA ON XCEPTIONNET", [
              ["Training data", "Held-out result"],
              ["3,000 Hemg", "96.4% acc., AUC 0.991"],
              ["30,000 Hemg", "97.5% acc., AUC 0.996"],
              ["181,000 Hemg", "98.4% acc., AUC 0.9986; ~51% on StyleGAN"],
              ["100k Hemg + 100k 140k-Faces", "98.3% Hemg, 97.7% StyleGAN"],
          ], [92, 145]),
          sub("B", "The generalization gap and the individual detectors"),
          P("Table V scores every detector on every test set. Each detector is strong exactly where it was designed to be "
            "and weak elsewhere. The 98% XceptionNet is at chance on face swaps, and its ROC-AUC of 0.215 on OpenFake "
            "means it ranks real photos as *more* fake than generated ones: cues learned from GAN faces point the wrong "
            "way for diffusion output. SBI reaches 0.888 ROC-AUC on InsightFace swaps it never saw, and CLIP 0.960 on 89 "
            "generators. Per generator, CLIP detects 95% of real photos, 94% of GPT-image-1.5, 80% of Gemini "
            "(nano-banana-pro) and only 57% of Midjourney 7 images."),
          ttable("TABLE V. ACCURACY / ROC-AUC OF EACH DETECTOR (CLEAN TEST SETS)", [
              ["Detector", "Hemg", "140k", "OpenFake", "Swaps"],
              ["XceptionNet", "**98.3 / .999**", "**97.9 / .998**", "48.8 / .215", "49.2 / .522"],
              ["SBI EffNet-B4", "65.1 / .678", "48.7 / .445", "47.7 / .354", "**77.6 / .888**"],
              ["CLIP ViT-L/14", "78.8 / .870", "87.1 / .947", "**86.9 / .960**", "65.8 / .732"],
          ], [61, 44, 44, 44, 44], center_cols=(1, 2, 3, 4)),
          sub("C", "Ensemble"),
          P("Table VI compares the final system with the best single model on all 8,000 decisions. Overall accuracy rises "
            "from 70.0% to 84.5% (ROC-AUC 0.925); accuracy on diffusion images and face swaps rises by about 30 percentage "
            "points; false alarms on real images fall from 10.7% to 8.6%. 13.4% of inputs are reported as Uncertain, and "
            "accuracy on the remainder is 89.1%. The cost is 3–5 points on Hemg faces, where the new detectors "
            "occasionally flag real photos."),
          ttable("TABLE VI. BEST SINGLE MODEL VS. ENSEMBLE (ACCURACY %, CLEAN / COMPRESSED)", [
              ["Test set", "XceptionNet", "Ensemble", "AUC"],
              ["GAN faces (Hemg)", "98.3 / 97.4", "93.8 / 92.3", ".989 / .976"],
              ["StyleGAN (140k)", "97.9 / 76.2", "95.4 / 89.1", ".993 / .962"],
              ["Diffusion/AR (OpenFake)", "48.8 / 43.8", "81.0 / 73.3", ".934 / .952"],
              ["Face swaps (DFF)", "49.2 / 48.0", "79.1 / 72.4", ".873 / .800"],
              ["All 8,000", "70.0", "84.5", "0.925"],
          ], [84, 50, 50, 53], center_cols=(1, 2, 3), bold_last=True),
          sub("D", "Screenshots"),
          P("The photo region was found in 94.3% of screenshots (mean IoU ≈ 0.90), and none of 400 ordinary photos was "
            "cropped. Table VII shows that cropping restores screenshot accuracy to the clean-image level. The gain is "
            "small for face swaps, whose blending traces are erased by screenshot compression."),
          ttable("TABLE VII. ENSEMBLE ON 2,000 PHONE SCREENSHOTS", [
              ["Test set", "Whole screenshot", "Photo cropped"],
              ["Hemg", "78.2% (AUC .87)", "93.6% (AUC .99)"],
              ["140k", "73.4% (.84)", "93.8% (.98)"],
              ["OpenFake", "65.0% (.68)", "73.8% (.91)"],
              ["Face swaps", "75.0% (.83)", "76.2% (.84)"],
              ["All", "72.9% (.81)", "84.4% (.93)"],
          ], [67, 85, 85], center_cols=(1, 2), bold_last=True)]

    # VI. Comparison
    s += [marked(sec("VI", "Comparison with the Base Methods"), "sec6"),
          sub("A", "What is different"),
          P("Table I contrasts AuthentiScan with its base methods. The individual detectors are adaptations of [1], [6] "
            "and [8]; the differences lie in combining them, in how they are evaluated, and in how the result is "
            "delivered: one system for three kinds of manipulation with a learned combiner and a named fake type; "
            "evaluation on four domains with compression and screenshots; an abstaining verdict; per-detector "
            "explanations; and a secured application trained on a laptop GPU."),
          sub("B", "Where the base methods remain stronger"),
          *bullets([
              "**SBI** [6] used 81 dlib landmarks, about 100 epochs and video frames from FaceForensics++; our variant "
              "uses 5 landmarks, 3 epochs and still images. Our 0.888 ROC-AUC on DeepFakeFace is not directly comparable "
              "to their ~0.93 on Celeb-DF, and we do not claim to exceed it.",
              "**UnivFD** [8] was trained on far more images; our head used 10,462 because of a mobile-data limit, which "
              "shows in the weak Midjourney 7 result (57%).",
              "**FaceForensics++** [1] evaluated on real manipulated videos; our video path has not yet been tested on a "
              "face-swap video benchmark.",
          ]),
          sub("C", "Lessons from building the system"),
          *bullets([
              "**Crop framing.** Inference crops were initially tighter than the training faces; the detector then "
              "called 47% of real faces fake (75.8% vs. 97.8% accuracy on the same images). Matching the training "
              "framing fixed it.",
              "**Format shortcut.** In OpenFake, real images are mostly JPEG and fakes mostly large PNGs. Without "
              "re-encoding every image identically, a detector can learn the file format instead of the content.",
              "**Leakage.** Fixed test splits, a dataset reserved for testing, and a combiner fitted on validation data "
              "only keep the reported numbers honest.",
              "**Face detection.** YuNet found faces in about 95% of Hemg images, versus about 81% for a Haar cascade.",
          ])]

    # VII. Limitations, VIII. Conclusion
    s += [sec("VII", "Limitations and Future Work"),
          P("Face swaps remain the weakest category (72–79%) and are often reported as Uncertain. Generators released "
            "after training, such as future versions of Midjourney, will reduce accuracy. Video is analysed frame by "
            "frame without temporal modelling. The screenshot benchmark uses synthetic screenshots. Planned work: "
            "fine-tuning the face-swap detector on Celeb-DF [13] and FaceForensics++ [1], training the CLIP head on more of "
            "OpenFake, fine-tuning CLIP's last layers, and distilling the ensemble into a single compact model."),
          sec("VIII", "Conclusion"),
          P("A deepfake detector's in-domain accuracy can be badly misleading: our 98.4% XceptionNet was at chance on face "
            "swaps and inverted on diffusion images. By combining three specialised base methods with a learned combiner, "
            "evaluating across four kinds of fake under compression and screenshots, and adding an Uncertain verdict and "
            "explanations, AuthentiScan raises cross-domain accuracy from 70.0% to 84.5% in a deployable application "
            "trained on consumer hardware."),
          Paragraph("ACKNOWLEDGMENT", SEC),
          P("The authors thank Ms. Meenakshi Pathak for her guidance, Dr. Jyoti Agarwal, Dr. Shahjahan Ali and "
            "Dr. Shailesh Saxena of the Department of Computer Science and Engineering, SRMS CET, Bareilly, for their "
            "support, and the creators of the public datasets used in this work.")]

    refs = [
        "A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, \"FaceForensics++: Learning to detect manipulated facial images,\" in *Proc. IEEE/CVF ICCV*, 2019, pp. 1–11.",
        "F. Chollet, \"Xception: Deep learning with depthwise separable convolutions,\" in *Proc. IEEE CVPR*, 2017, pp. 1251–1258.",
        "K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in *Proc. IEEE CVPR*, 2016, pp. 770–778.",
        "M. Tan and Q. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks,\" in *Proc. ICML*, 2019, pp. 6105–6114.",
        "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in *Proc. IEEE ICCV*, 2017, pp. 618–626.",
        "K. Shiohara and T. Yamasaki, \"Detecting deepfakes with self-blended images,\" in *Proc. IEEE/CVF CVPR*, 2022, pp. 18720–18729.",
        "A. Radford *et al.*, \"Learning transferable visual models from natural language supervision,\" in *Proc. ICML*, 2021, pp. 8748–8763.",
        "U. Ojha, Y. Li, and Y. J. Lee, \"Towards universal fake image detectors that generalize across generative models,\" in *Proc. IEEE/CVF CVPR*, 2023, pp. 24480–24489.",
        "V. Livernoche *et al.*, \"OpenFake: An open dataset and platform toward real-world deepfake detection,\" arXiv:2509.09495, 2025.",
        "H. Song *et al.*, \"Robustness and generalizability of deepfake detection: A study with diffusion models,\" arXiv:2309.02218, 2023.",
        "W. Wu, H. Peng, and S. Yu, \"YuNet: A tiny millisecond-level face detector,\" *Machine Intelligence Research*, vol. 20, no. 5, pp. 656–665, 2023.",
        "T. Karras, S. Laine, and T. Aila, \"A style-based generator architecture for generative adversarial networks,\" in *Proc. IEEE/CVF CVPR*, 2019, pp. 4401–4410.",
        "Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, \"Celeb-DF: A large-scale challenging dataset for DeepFake forensics,\" in *Proc. IEEE/CVF CVPR*, 2020, pp. 3207–3216.",
        "Q. Xie, M.-T. Luong, E. Hovy, and Q. V. Le, \"Self-training with Noisy Student improves ImageNet classification,\" in *Proc. IEEE/CVF CVPR*, 2020, pp. 10687–10698.",
        "M. S. Rana, M. N. Nobi, B. Murali, and A. H. Sung, \"Deepfake detection: A systematic literature review,\" *IEEE Access*, vol. 10, pp. 25494–25513, 2022.",
        "T. Wang, X. Liao, K. P. Chow, X. Lin, and Y. Wang, \"Deepfake detection: A comprehensive survey from the reliability perspective,\" *ACM Computing Surveys*, vol. 57, no. 3, 2024.",
        "F. Pedregosa *et al.*, \"Scikit-learn: Machine learning in Python,\" *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.",
        "A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin, \"Albumentations: Fast and flexible image augmentations,\" *Information*, vol. 11, no. 2, p. 125, 2020.",
        "Hemg, \"Deepfake and real images,\" Hugging Face Datasets. [Online]. Available: https://huggingface.co/datasets/Hemg/deepfake-and-real-images",
        "xhlulu, \"140k real and fake faces,\" Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
    ]
    s += [Paragraph("REFERENCES", SEC)]
    s += [Paragraph(f"[{i}]&nbsp;&nbsp;{md(r)}", REF) for i, r in enumerate(refs, 1)]
    return s


def build():
    wide = {2: ("fig", wide_height("fig"))}
    marks: dict = {}
    Doc(OUT, wide, marks).build(story())                     # pass 1: find Section VI page
    p6 = marks.get("sec6", 3)
    p6 = p6 if p6 != 2 else 3
    wide[p6] = ("cmp", wide_height("cmp"))
    marks2: dict = {}
    Doc(OUT, wide, marks2).build(story())                    # pass 2: with Table I on that page
    print(f"wrote {OUT}  (Fig. 1 on p.2, Table I on p.{p6})")


if __name__ == "__main__":
    build()
