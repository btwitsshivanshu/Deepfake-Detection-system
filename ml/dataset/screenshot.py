"""
Screenshot handling.

People rarely upload the original image: they upload a phone SCREENSHOT of it
(status bar, app header, buttons, captions around the photo). The detectors
then judge the whole screen — UI included — and the real evidence is diluted.

find_photo_region() locates the embedded photo so only that region is analysed.

Idea: app UI is made of perfectly flat colour areas and thin text, while a
photo is textured and has thousands of distinct colours. We:
  1. build a "textured" mask (local std-dev > small threshold),
  2. close it morphologically so a photo becomes one solid blob,
  3. score each blob's bounding box by colour diversity + texture density,
  4. accept the best box only if the area AROUND it is UI-like (mostly
     perfectly flat) — so ordinary photos are left untouched.

make_fake_screenshot() renders a synthetic phone screenshot around an image
(used to measure the cropper with known ground-truth boxes).
"""
from __future__ import annotations

import cv2
import numpy as np

_WORK = 640                    # analysis resolution (long side)


def _local_std(gray: np.ndarray, k: int = 5) -> np.ndarray:
    f = gray.astype(np.float32)
    mu = cv2.blur(f, (k, k))
    mu2 = cv2.blur(f * f, (k, k))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0))


def _unique_colors(rgb: np.ndarray) -> int:
    q = (rgb >> 3).astype(np.int32)                       # 32 levels / channel
    codes = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    return int(np.unique(codes).size)


def _bg_color(small: np.ndarray) -> np.ndarray:
    """Most common colour of the screen = the app's background colour."""
    q = (small >> 4).astype(np.int32)
    codes = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    vals, counts = np.unique(codes, return_counts=True)
    m = codes == vals[np.argmax(counts)]
    return np.median(small[m], axis=0)


def _runs(mask: np.ndarray, max_gap: int) -> tuple[int, int]:
    """Longest run of True after bridging gaps of <= max_gap False values."""
    m = mask.copy()
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            m[a:b] = True
    best, start, n = (0, 0), 0, 0
    for i, v in enumerate(m):
        if v:
            if n == 0:
                start = i
            n += 1
            if n > best[1]:
                best = (start, n)
        else:
            n = 0
    return best


def _photo_lines(seg: np.ndarray, is_bg: np.ndarray, axis: int) -> np.ndarray:
    """Lines (rows if axis=1, cols if axis=0) that look like photo content."""
    bg_frac = is_bg.mean(axis)
    q = (seg >> 5).astype(np.int32)
    codes = (q[..., 0] << 6) | (q[..., 1] << 3) | q[..., 2]
    lines = codes if axis == 1 else codes.transpose(1, 0)
    n_col = np.array([np.unique(l).size for l in lines])
    return (bg_frac < 0.5) & (n_col >= 6)


def looks_like_screen(W: int, H: int) -> bool:
    """Phone screenshots are tall (or wide) and high-res; desktop ones are large."""
    long_side, ratio = max(W, H), max(W, H) / max(1, min(W, H))
    if long_side < 600:
        return False
    return ratio >= 1.7 or (ratio >= 1.2 and long_side >= 1200)


def find_photo_region(img: np.ndarray, min_area: float = 0.08, max_area: float = 0.85):
    """
    Return (x, y, w, h) of the photo inside a phone screenshot (original pixel
    coordinates), or None if the image does not look like a screenshot.
    """
    H, W = img.shape[:2]
    if not looks_like_screen(W, H):
        return None
    s = min(1.0, _WORK / max(H, W))
    small = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    bg = _bg_color(small)
    is_bg = (np.abs(small.astype(np.float32) - bg).max(-1) < 10)
    if is_bg.mean() < 0.2:                       # no dominant flat background -> ordinary photo
        return None

    rows = _photo_lines(small, is_bg, 1)
    r0, rn = _runs(rows, max_gap=max(2, h // 60))
    if rn < 48:
        return None
    seg, seg_bg = small[r0:r0 + rn], is_bg[r0:r0 + rn]
    cols = _photo_lines(seg, seg_bg, 0)
    c0, cn = _runs(cols, max_gap=max(2, w // 60))
    if cn < 48:
        return None

    area = rn * cn / (h * w)
    if not (min_area <= area <= max_area):
        return None
    roi = small[r0:r0 + rn, c0:c0 + cn]
    if _unique_colors(roi) < 300:                # a big flat banner / card, not a photo
        return None
    outside = np.ones((h, w), bool); outside[r0:r0 + rn, c0:c0 + cn] = False
    if is_bg[outside].mean() < 0.35:             # surroundings are not app UI
        return None
    # app backgrounds are perfectly flat; a real wall / studio backdrop is not
    flat = _local_std(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)) < 1.0
    if flat[outside & is_bg].mean() < 0.7:
        return None

    X0, Y0 = int(c0 / s), int(r0 / s)
    X1, Y1 = min(W, int((c0 + cn) / s)), min(H, int((r0 + rn) / s))
    return X0, Y0, X1 - X0, Y1 - Y0


def crop_screenshot(img: np.ndarray):
    """Return (image_to_analyse, box or None)."""
    box = find_photo_region(img)
    if box is None:
        return img, None
    x, y, w, h = box
    return img[y:y + h, x:x + w], box


# ---------------------------------------------------------------------------
# Synthetic screenshots for evaluation
# ---------------------------------------------------------------------------
_WORDS = "the a photo shared viral news today look this amazing see more comments reply like share " \
         "breaking video official image trending follow save download".split()


def _text_line(canvas, x, y, width, color, rng, scale=0.5):
    words, cx = [], x
    while True:
        wd = str(rng.choice(_WORDS))
        tw = cv2.getTextSize(wd + " ", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if cx + tw > x + width:
            break
        words.append(wd); cx += tw
    cv2.putText(canvas, " ".join(words), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_fake_screenshot(img: np.ndarray, rng: np.random.Generator):
    """Render a phone-style screenshot containing `img`. Returns (screenshot, (x, y, w, h))."""
    Wc, Hc = 540, int(540 * rng.uniform(1.9, 2.2))
    dark = rng.random() < 0.35
    bg = (18, 18, 18) if dark else tuple(int(v) for v in rng.choice([[255, 255, 255], [245, 246, 248], [236, 229, 221]]))
    fg = (230, 230, 230) if dark else (30, 30, 30)
    sub = (150, 150, 150) if dark else (110, 110, 110)
    c = np.full((Hc, Wc, 3), bg, np.uint8)

    # status bar
    cv2.putText(c, f"{rng.integers(1, 13)}:{rng.integers(10, 60)}", (18, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)
    for i in range(4):
        cv2.rectangle(c, (Wc - 120 + i * 22, 12), (Wc - 108 + i * 22, 26), fg, -1)
    # app header
    hy = 44
    if rng.random() < 0.6:
        head = tuple(int(v) for v in rng.choice([[7, 94, 84], [24, 119, 242], [40, 40, 40], [255, 255, 255]]))
        cv2.rectangle(c, (0, hy), (Wc, hy + 56), head, -1)
        cv2.putText(c, str(rng.choice(["Chats", "Search", "Photos", "Home"])), (60, hy + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255) if sum(head) < 400 else fg, 2, cv2.LINE_AA)
    y = hy + 70
    for _ in range(int(rng.integers(0, 3))):
        _text_line(c, 20, y, Wc - 40, sub, rng); y += 24

    # the photo
    pw = int(Wc * rng.uniform(0.7, 1.0))
    ph = int(pw * img.shape[0] / img.shape[1])
    ph = min(ph, int(Hc * 0.55))
    pw = int(ph * img.shape[1] / img.shape[0])
    px = (Wc - pw) // 2 if rng.random() < 0.7 else int(rng.integers(0, Wc - pw + 1))
    py = y + int(rng.integers(5, 30))
    c[py:py + ph, px:px + pw] = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    y = py + ph + 30

    # caption + buttons
    for _ in range(int(rng.integers(1, 4))):
        if y > Hc - 140:
            break
        _text_line(c, 20, y, Wc - 40, fg, rng, 0.6); y += 28
    if y < Hc - 120:
        btn = tuple(int(v) for v in rng.choice([[11, 87, 208], [0, 128, 105], [60, 60, 60]]))
        cv2.rectangle(c, (Wc // 2 + 10, y), (Wc - 20, y + 48), btn, -1)
        cv2.rectangle(c, (20, y), (Wc // 2 - 10, y + 48), (210, 225, 250) if not dark else (60, 60, 70), -1)
    # nav bar
    cv2.rectangle(c, (0, Hc - 56), (Wc, Hc), tuple(max(0, v - 8) for v in bg), -1)
    for i, cx in enumerate((Wc // 4, Wc // 2, 3 * Wc // 4)):
        cv2.circle(c, (cx, Hc - 28), 10, sub, 2)
    return c, (px, py, pw, ph)


def iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)
