import hashlib
import os
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image

# Define formats
SBS_FULL = "SBS_FULL"
SBS_HALF = "SBS_HALF"
SBS_FULL_CROSS = "SBS_FULL_CROSS"
TB_FULL = "TB_FULL"
TB_HALF = "TB_HALF"
SBS_180 = "SBS_180"
FLAT = "FLAT"

# Map tags to formats.
# Note: Based on spec, _TBF is in both, so we'll prioritize it as Full.
TAG_MAP = {
    "_Full_SBS": SBS_FULL,
    "_fullsbs": SBS_FULL,
    "_LRF": SBS_FULL,
    "_SBS": SBS_FULL,
    "_3DHF": SBS_FULL,
    "_3DPHF": SBS_FULL,
    "_RLF": SBS_FULL_CROSS,
    "_LR_180": SBS_180,
    "_180x180_LR": SBS_180,
    "VR180": SBS_180,
    "_LR": SBS_HALF,
    "_3DH": SBS_HALF,
    "_3DPH": SBS_HALF,
    "_Full_TB": TB_FULL,
    "_fulltb": TB_FULL,
    "_TBF": TB_FULL,
    "_3DVF": TB_FULL,
    "_3DPVF": TB_FULL,
    "_TB": TB_HALF,
    "_3DV": TB_HALF,
    "_3DPV": TB_HALF,
}

# Sort tags by length (longest first) to ensure precise matching
SORTED_TAGS = sorted(TAG_MAP.keys(), key=len, reverse=True)

# MLP weights and biases for format classification
_W1 = np.array(
    [
        [5.3580708503723145, -3.846473455429077, -3.1798648834228516, 3.3081345558166504],
        [8.403752326965332, -2.244128942489624, -3.960606575012207, 1.3691649436950684],
        [-3.6235997676849365, 7.777614593505859, 3.1487159729003906, -3.4909191131591797],
        [-3.336681604385376, -6.300732612609863, 3.242278575897217, 4.757264614105225],
    ],
    dtype=np.float32,
)
_B1 = np.array([0.25426536798477173, 0.4942944347858429, 1.4740190505981445, 1.2720032930374146], dtype=np.float32)
_W2 = np.array(
    [
        [0.1533418595790863, 2.6059467792510986, 1.5183905363082886, -6.48093318939209],
        [-6.113797187805176, -5.228984355926514, 3.082887887954712, 0.6357945799827576],
        [5.335931301116943, 2.043487071990967, -6.655498504638672, 7.333863735198975],
    ],
    dtype=np.float32,
)
_B2 = np.array([1.5812703371047974, -0.1364462822675705, -1.9182605743408203], dtype=np.float32)
_LABELS = ["FLAT", "SBS", "TB"]


def compute_media_etag(path: str, mtime: Optional[int] = None) -> str:
    """Computes a SHA-256 ETag hash from normalized path and optional modification time."""
    norm_path = path.lstrip("/")
    raw_key = f"{norm_path}{int(mtime)}" if mtime is not None else norm_path
    return hashlib.sha256(raw_key.encode()).hexdigest()


def compute_format_cache_key(path: str, mtime: Optional[int] = None) -> str:
    """Computes a cache key matching the thumbnail ETag hash scheme."""
    return f"fmt_{compute_media_etag(path, mtime)}"


def extract_candidate_names(path_or_candidates: Union[str, Sequence[str], None]) -> list[str]:
    """Extracts candidate names (filename and parent directory) for tag and keyword matching."""
    if not path_or_candidates:
        return []
    items = [path_or_candidates] if isinstance(path_or_candidates, str) else path_or_candidates

    res: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        if "/" in item or "\\" in item:
            base = os.path.basename(item)
            if base and base not in seen:
                seen.add(base)
                res.append(base)
            parent = os.path.basename(os.path.dirname(item))
            if parent and parent not in seen:
                seen.add(parent)
                res.append(parent)
        else:
            if item not in seen:
                seen.add(item)
                res.append(item)
    return res


def detect_format_from_tags(path_or_candidates: Union[str, Sequence[str], None]) -> Optional[str]:
    """Detects stereo format purely from filename tags or directory names. Returns None if not found."""
    candidates = extract_candidate_names(path_or_candidates)
    for name in candidates:
        name_lower = name.lower()
        for tag in SORTED_TAGS:
            if tag.lower() in name_lower:
                return TAG_MAP[tag]
    return None


def extract_features(im: Image.Image) -> np.ndarray:
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return np.zeros(4, dtype=np.float32)

    mid_w = w // 2
    mid_h = h // 2

    d_lr = float(np.mean(np.abs(arr[:, :mid_w] - arr[:, mid_w : mid_w * 2])))
    d_tb = float(np.mean(np.abs(arr[:mid_h, :] - arr[mid_h : mid_h * 2, :])))
    d_hflip = float(np.mean(np.abs(arr - arr[:, ::-1])))
    d_vflip = float(np.mean(np.abs(arr - arr[::-1, :])))

    return np.array([d_lr, d_tb, d_hflip, d_vflip], dtype=np.float32)


def predict(
    im: Image.Image,
    w1: np.ndarray | None = None,
    b1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
) -> str:
    x = extract_features(im)
    # Uniform or near-uniform blank/black/white images have zero/near-zero diffs
    if float(np.max(x)) < 0.01:
        return "FLAT"

    _w1 = w1 if w1 is not None else _W1
    _b1 = b1 if b1 is not None else _B1
    _w2 = w2 if w2 is not None else _W2
    _b2 = b2 if b2 is not None else _B2

    # 2-layer MLP: ReLU(x @ W1.T + B1) @ W2.T + B2
    h_out = np.maximum(0.0, x @ _w1.T + _b1)
    out = h_out @ _w2.T + _b2
    return _LABELS[int(np.argmax(out))]


def detect_format_from_image(
    image: Image.Image,
    path_or_candidates: Union[str, Sequence[str], None] = None,
) -> str:
    """Predicts stereo format from image content with VR180 filename keyword heuristics."""
    im = image
    if max(im.size) > 128:
        scale = 128 / max(im.size)
        im = im.resize(
            (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
            Image.Resampling.BILINEAR,
        )

    pred = predict(im)
    aspect_ratio = image.width / max(1, image.height)
    if pred == "SBS":
        candidates = extract_candidate_names(path_or_candidates)
        has_vr_or_180 = any("vr" in c.lower() or "180" in c.lower() for c in candidates)
        if has_vr_or_180:
            return SBS_180
        # Check aspect ratio: images wider than 16:9 are likely Full SBS (e.g. 2:1, 8:3, 32:9).
        # We use a slight margin (0.01) to avoid rounding inaccuracies (e.g. 854x480) for 16:9 Half SBS.
        if aspect_ratio > (16.0 / 9.0 + 0.01):
            return SBS_FULL
        return SBS_HALF
    elif pred == "TB":
        # Check aspect ratio: Full TB is stacked vertically with 1:1 (VR), 8:9 (16:9 Full), or 2:3 (4:3 Full).
        # Half TB maintains standard horizontal frames (16:9 ≈ 1.78, 4:3 ≈ 1.33).
        # Using 1.1 allows 1:1 VR and smaller ratios to be Full TB, while 4:3 and 16:9 are Half TB.
        if aspect_ratio < 1.1:
            return TB_FULL
        return TB_HALF
    return FLAT


def detect_stereo_format(
    path_or_candidates: Union[str, Sequence[str], None] = None,
    image: Optional[Image.Image] = None,
    cache_db: Optional[Any] = None,
    cache_key: Optional[str] = None,
    fallback_flat: bool = False,
) -> Optional[str]:
    """
    Detects stereo format by checking filename/directory tags first,
    then cache_db, and finally image MLP prediction if an image is provided.
    Falls back to FLAT if fallback_flat is True, otherwise None.
    """
    tag_fmt = detect_format_from_tags(path_or_candidates)
    if tag_fmt is not None:
        return tag_fmt

    if cache_db is not None and cache_key is not None:
        cached = cache_db.get(cache_key)
        if cached is not None:
            return cached

    if image is not None:
        fmt = detect_format_from_image(image, path_or_candidates)
        if cache_db is not None and cache_key is not None:
            if hasattr(cache_db, "set"):
                cache_db.set(cache_key, fmt)
            else:
                cache_db[cache_key] = fmt
        return fmt

    return FLAT if fallback_flat else None
