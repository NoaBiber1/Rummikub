"""Two-stage Rummikub tile recognition for project_rummikub_backend.

Stage 1: detect tiles   -> Roboflow hosted detector OR a local Ultralytics weights file
Stage 2: classify tiles -> local YOLO11s-cls weights (best.pt from Colab)

Environment variables
---------------------
CLS_WEIGHTS         path to classify/weights/best.pt              (required)
DETECTOR_WEIGHTS    path to detect/weights/best.pt                (optional; if set, the
                    detector runs locally and Roboflow is not used at all)
ROBOFLOW_API_KEY    only needed when DETECTOR_WEIGHTS is not set
DETECTOR_MODEL_ID   default "rummikub-tiles-qu08l/1"
MIN_CLS_CONF        default 0.35; below this a tile is labelled "Unknown"
"""
from __future__ import annotations

import base64
import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------
# The crop geometry is defined once, in tools/tile_crops.py, and shared with
# training, evaluation and auto-labeling. Inside the repo it is loaded from
# that file by path (no sys.path changes). The Docker image is built from this
# folder only, so the same math is inlined in _workflow_view_inline() and
# tests/test_local_pipeline_crops.py checks both give identical pixels.
_TILE_CROPS_PATH = Path(__file__).resolve().parent.parent / "tools" / "tile_crops.py"


def _load_tile_crops() -> ModuleType | None:
    if not _TILE_CROPS_PATH.is_file():
        logger.info("%s not found; using the inlined crop geometry.", _TILE_CROPS_PATH)
        return None
    spec = importlib.util.spec_from_file_location("tile_crops", _TILE_CROPS_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TILE_CROPS = _load_tile_crops()
_HAVE_TILE_CROPS = _TILE_CROPS is not None
CLS_SIZE = _TILE_CROPS.CLS_SIZE if _TILE_CROPS else 224
# 5 % per side, the Roboflow Detection Offset step (10 % in total).
_MARGIN_PER_SIDE = _TILE_CROPS.MARGIN_PER_SIDE if _TILE_CROPS else 0.05

DET_SIZE = 1024
DET_CONF = 0.35
# Total enlargement added by the Roboflow "Detection Offset" step (10 % total,
# 5 % per side).  Kept here only for the output box scaling so that the
# coordinates returned to vision_grouping.py match the old workflow's output.
CROP_OFFSET = 0.10


# ---------------------------------------------------------------------------
# Models (loaded lazily and cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _classifier():
    from ultralytics import YOLO  # type: ignore[import]
    return YOLO(os.environ["CLS_WEIGHTS"])


@lru_cache(maxsize=1)
def _local_detector():
    from ultralytics import YOLO  # type: ignore[import]
    return YOLO(os.environ["DETECTOR_WEIGHTS"])


# ---------------------------------------------------------------------------
# Stage 1 – detection
# ---------------------------------------------------------------------------

def _detect_local(img_bgr: np.ndarray) -> list[dict]:
    """Run the YOLO detector locally; returns xywh centre-format boxes."""
    r = _local_detector().predict(img_bgr, imgsz=DET_SIZE, conf=DET_CONF, verbose=False)[0]
    out = []
    for (x, y, w, h), c in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist()):
        out.append({"x": x, "y": y, "width": w, "height": h, "confidence": c, "class": "tile"})
    return out


def _detect_hosted(img_bgr: np.ndarray) -> list[dict]:
    """Send the image to the Roboflow hosted detector."""
    import cv2
    import requests  # type: ignore[import]
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("could not encode image for Roboflow detector")
    model_id = os.environ.get("DETECTOR_MODEL_ID", "rummikub-tiles-qu08l/1")
    resp = requests.post(
        f"https://detect.roboflow.com/{model_id}",
        params={"api_key": os.environ["ROBOFLOW_API_KEY"], "confidence": int(DET_CONF * 100)},
        data=base64.b64encode(buf.tobytes()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("predictions", [])


def detect_tiles(img_bgr: np.ndarray) -> list[dict]:
    """Dispatch to local or hosted detector based on environment."""
    return _detect_local(img_bgr) if os.environ.get("DETECTOR_WEIGHTS") else _detect_hosted(img_bgr)


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def _crop_tile_pil(img_bgr: np.ndarray, det: dict):
    """Crop using the exact same preprocessing as tools/tile_crops.workflow_view().

    Converts BGR to RGB PIL image, expands the box by 5 % per side (matching
    the Roboflow Detection Offset step), clips to image boundaries, then
    stretches non-aspect-preserving to 224x224.
    """
    from PIL import Image  # type: ignore[import]
    import cv2

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # detector boxes are centre-format (x, y, w, h)
    x, y, w, h = det["x"], det["y"], det["width"], det["height"]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    if _TILE_CROPS is not None:
        return _TILE_CROPS.workflow_view(pil_img, (x1, y1, x2, y2), CLS_SIZE)
    return _workflow_view_inline(pil_img, (x1, y1, x2, y2), CLS_SIZE)


def _workflow_view_inline(pil_img, box_xyxy: tuple[float, float, float, float], size: int):
    """Same math as tools/tile_crops.workflow_view, for builds without tools/."""
    from PIL import Image  # type: ignore[import]

    x1, y1, x2, y2 = box_xyxy
    dx, dy = (x2 - x1) * _MARGIN_PER_SIDE, (y2 - y1) * _MARGIN_PER_SIDE
    left = max(0, round(x1 - dx))
    top = max(0, round(y1 - dy))
    right = min(pil_img.width, round(x2 + dx))
    bottom = min(pil_img.height, round(y2 + dy))
    right, bottom = max(right, left + 1), max(bottom, top + 1)
    crop = pil_img.crop((left, top, right, bottom))
    return crop.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def _pil_to_bgr(pil_img) -> np.ndarray:
    """Convert a PIL RGB image to a BGR numpy array for Ultralytics."""
    import cv2
    arr = np.array(pil_img)          # H x W x 3, RGB uint8
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Stage 2 – classification
# ---------------------------------------------------------------------------

def classify_crops(crops_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
    """Run the YOLO classifier on a batch of BGR crop arrays."""
    if not crops_bgr:
        return []
    results = _classifier().predict(crops_bgr, imgsz=CLS_SIZE, verbose=False)
    return [(r.names[int(r.probs.top1)], float(r.probs.top1conf)) for r in results]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def recognize(image_bytes: bytes) -> dict:
    """Detect and classify all tiles in *image_bytes*.

    Returns a Roboflow-style payload::

        {
            "image": {"width": W, "height": H},
            "predictions": [
                {
                    "x": cx, "y": cy,
                    "width": w, "height": h,   # enlarged by 10 % (CROP_OFFSET)
                    "confidence": <detector conf>,
                    "class": "Blue7",           # or "Unknown" below MIN_CLS_CONF
                    "class_confidence": <cls conf>,
                },
                ...
            ]
        }

    The boxes are enlarged by ``CROP_OFFSET`` so that the coordinates match
    what the Roboflow workflow's "Detection Offset" step produced, keeping all
    thresholds in ``vision_grouping.py`` valid.
    """
    import cv2

    img_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("invalid or unreadable image")

    dets = detect_tiles(img_bgr)

    # Build crops using the PIL-based helper so preprocessing exactly matches
    # what the classifier was trained on (via tools/tile_crops.workflow_view).
    crops_bgr = [_pil_to_bgr(_crop_tile_pil(img_bgr, d)) for d in dets]
    labels = classify_crops(crops_bgr)

    min_conf = float(os.environ.get("MIN_CLS_CONF", "0.35"))
    preds = []
    for d, (name, p) in zip(dets, labels):
        preds.append({
            # Boxes enlarged to match the old Roboflow Detection Offset output
            # so vision_grouping.py spatial thresholds remain valid.
            "x":      d["x"],
            "y":      d["y"],
            "width":  d["width"]  * (1 + CROP_OFFSET),
            "height": d["height"] * (1 + CROP_OFFSET),
            "confidence":       d.get("confidence", 0.0),
            "class":            name if p >= min_conf else "Unknown",
            "class_confidence": p,
        })

    return {
        "image":       {"width": img_bgr.shape[1], "height": img_bgr.shape[0]},
        "predictions": preds,
    }
