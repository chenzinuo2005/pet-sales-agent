"""CNN inference for single-image pet breed prediction.

Provides the public API `predict_breed(image_path)` used by pet_agent.py
to preprocess pet images before agent dispatch.

Architecture:
    - Lazy-loads the EfficientNet-B0 model on first call (~2s), reuses thereafter.
    - Preprocessing matches training (Resize→CenterCrop→ToTensor→Normalize)
      WITHOUT data augmentation.
    - Applies 4-tier confidence threshold strategy (see spec section 6).
    - Supports dependency injection via `predict_with_model()` and
      parameterized `_get_model(weights_path)`.
    - Provides TTA (Test-Time Augmentation) via `predict_with_tta()` and
      `predict_breed_tta()` for improved accuracy.
"""

import os
import threading

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.common.logger import logger
from app.models.schemas import CNNPredictResult

# ---------------------------------------------------------------------------
# Oxford-IIIT Pet breed class names in torchvision order (indices 0..36)
# Must be 1:1 with the class_to_idx ordering used during training.
# Verified against torchvision.datasets.OxfordIIITPet.classes.
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
]

# ---------------------------------------------------------------------------
# English → Chinese breed name mapping (hardcoded per spec section 6).
# Keys are lowercase_underscore versions of CLASS_NAMES.
# ---------------------------------------------------------------------------
BREED_MAPPING: dict[str, str] = {
    # Dogs (25)
    "american_bulldog": "美国斗牛犬",
    "american_pit_bull_terrier": "美国比特犬",
    "basset_hound": "巴吉度猎犬",
    "beagle": "比格犬",
    "boxer": "拳师犬",
    "chihuahua": "吉娃娃",
    "english_cocker_spaniel": "英国可卡犬",
    "english_setter": "英国塞特犬",
    "german_shorthaired": "德国短毛指示犬",
    "great_pyrenees": "大白熊犬",
    "havanese": "哈瓦那犬",
    "japanese_chin": "日本狆",
    "keeshond": "荷兰毛狮犬",
    "leonberger": "莱昂贝格犬",
    "miniature_pinscher": "迷你品犬",
    "newfoundland": "纽芬兰犬",
    "pomeranian": "博美犬",
    "pug": "巴哥犬",
    "saint_bernard": "圣伯纳犬",
    "samoyed": "萨摩耶犬",
    "scottish_terrier": "苏格兰梗",
    "shiba_inu": "柴犬",
    "staffordshire_bull_terrier": "斯塔福德斗牛梗",
    "wheaten_terrier": "软毛麦色梗",
    "yorkshire_terrier": "约克夏梗",
    # Cats (12)
    "abyssinian": "阿比西尼亚猫",
    "bengal": "孟加拉猫",
    "birman": "伯曼猫",
    "bombay": "孟买猫",
    "british_shorthair": "英国短毛猫",
    "egyptian_mau": "埃及猫",
    "maine_coon": "缅因猫",
    "persian": "波斯猫",
    "ragdoll": "布偶猫",
    "russian_blue": "俄罗斯蓝猫",
    "siamese": "暹罗猫",
    "sphynx": "斯芬克斯猫",
}

assert len(BREED_MAPPING) == 37, (
    f"BREED_MAPPING must have 37 entries, got {len(BREED_MAPPING)}"
)

# ---------------------------------------------------------------------------
# Image preprocessing pipeline
# Matches training preprocessing WITHOUT augmentation.
# Spec section 12.1: Resize(256) → CenterCrop(224) → ToTensor → Normalize
# ---------------------------------------------------------------------------
# Temperature for confidence calibration.
# Computed by minimizing NLL on the validation set after CutMix training.
# T < 1 sharpens the overly-smooth softmax distribution caused by soft label training.
TEMPERATURE: float = 0.25

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------
_model: "nn.Module | None" = None
_model_lock = threading.Lock()


def _get_model(weights_path: str | None = None) -> "nn.Module":
    """Lazy-load CNN model on first call (~2s), reuse thereafter.

    Thread-safe: uses double-checked locking so concurrent requests
    never load the model twice.

    Args:
        weights_path: Optional path to a .pth weights file. If None,
            defaults to settings.model_weights_path.

    Raises:
        RuntimeError: If the model weights file is missing.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from app.cnn.model import create_model
                from app.common.config import settings

                if weights_path is None:
                    weights_path = settings.model_weights_path

                _model = create_model()

                if not os.path.exists(weights_path):
                    raise RuntimeError(
                        f"模型文件未找到: {weights_path}，请先运行 train-cnn"
                    )

                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                _model.load_state_dict(state_dict)
                _model.eval()

                logger.info("cnn_model_loaded", extra={"path": weights_path})

    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _en_to_key(en_name: str) -> str:
    """Convert a CLASS_NAMES entry to a BREED_MAPPING key."""
    return en_name.lower().replace(" ", "_")


def _build_top3_entry(idx: int, conf: float) -> dict:
    """Build a single top-3 result entry."""
    en_name = CLASS_NAMES[idx]
    cn_name = BREED_MAPPING.get(_en_to_key(en_name), en_name)
    return {"breed_en": en_name, "breed_cn": cn_name, "conf": round(conf, 4)}


# ---------------------------------------------------------------------------
# Core inference logic (device-agnostic helper)
# ---------------------------------------------------------------------------

def _run_inference(
    model: nn.Module, device: torch.device, input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single forward pass and return (top3_probs, top3_indices)."""
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)                      # (1, 37)  raw logits
        calibrated_logits = logits / TEMPERATURE          # temperature scaling
        probs = torch.softmax(calibrated_logits, dim=1).squeeze(0)  # (37,)

    top3_probs, top3_indices = torch.topk(probs, 3)
    return top3_probs, top3_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_with_model(model: nn.Module, image_path: str) -> CNNPredictResult:
    """Predict pet breed using an externally-provided model (DI variant).

    This function accepts the model as a parameter instead of relying
    on the module-level singleton, making it suitable for testing,
    custom model instances, or scenarios where the caller manages
    the model lifecycle.

    Confidence threshold strategy (spec section 6):
            >= 85%  → status="success"
            60-85%  → status="low_confidence"
            40-60%  → status="low_confidence"
            < 40%   → status="failed"

    Args:
        model: A pre-loaded, eval-mode nn.Module (e.g. from _get_model()).
        image_path: Absolute or relative path to a JPG/PNG pet image.

    Returns:
        CNNPredictResult with breed_en, breed_cn, confidence, top3, and status.
    """
    # --- 1. Validate image path ---
    if not os.path.exists(image_path):
        logger.warning("image_not_found", extra={"path": image_path})
        return CNNPredictResult(
            breed_en="",
            breed_cn="无法读取图片文件",
            confidence=0.0,
            top3=[],
            status="failed",
        )

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        logger.warning("unsupported_format", extra={"extension": ext})
        return CNNPredictResult(
            breed_en="",
            breed_cn="请使用 JPG/PNG/WebP 格式",
            confidence=0.0,
            top3=[],
            status="failed",
        )

    # --- 2. Load & convert image ---
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.warning("image_open_failed", extra={"path": image_path, "error": str(exc)})
        return CNNPredictResult(
            breed_en="",
            breed_cn="无法读取图片文件",
            confidence=0.0,
            top3=[],
            status="failed",
        )

    # --- 3. Preprocess ---
    input_tensor = _preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)

    # --- 4. Device selection (auto CPU fallback) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.info("device_selected", extra={"device": "cpu", "reason": "cuda_unavailable"})

    # --- 5. Inference ---
    top3_probs, top3_indices = _run_inference(model, device, input_tensor)

    # --- 6. Top-3 ---
    top1_conf = top3_probs[0].item()
    top3 = [
        _build_top3_entry(top3_indices[i].item(), top3_probs[i].item())
        for i in range(3)
    ]

    top1_en = CLASS_NAMES[top3_indices[0].item()]
    top1_cn = BREED_MAPPING.get(_en_to_key(top1_en), top1_en)

    # --- 7. Confidence threshold strategy ---
    if top1_conf >= 0.85:
        status = "success"
    elif top1_conf >= 0.40:
        status = "low_confidence"
    else:
        status = "failed"

    logger.debug(
        "prediction_result",
        extra={
            "file": os.path.basename(image_path),
            "breed_cn": top1_cn,
            "confidence_pct": round(top1_conf * 100, 1),
            "status": status,
        },
    )

    return CNNPredictResult(
        breed_en=top1_en,
        breed_cn=top1_cn,
        confidence=round(top1_conf, 4),
        top3=top3,
        status=status,
    )


def predict_breed(image_path: str) -> CNNPredictResult:
    """Predict pet breed from a single image with confidence thresholding.

    This is THE public entry point called by pet_agent.py for image
    preprocessing before agent dispatch.

    Uses the module-level lazy-loaded model singleton. For dependency
    injection (testing or custom model instances), use predict_with_model().

    Confidence threshold strategy (spec section 6):
        >= 85%  → status="success"
        60-85%  → status="low_confidence"
        40-60%  → status="low_confidence"
        < 40%   → status="failed"

    Args:
        image_path: Absolute or relative path to a JPG/PNG pet image.

    Returns:
        CNNPredictResult with breed_en, breed_cn, confidence, top3, and status.

    Raises:
        RuntimeError: If the CNN model weights file is missing (not trained).
    """
    model = _get_model()
    return predict_with_model(model, image_path)


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

_tta_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_with_tta(model, image_path: str, num_crops: int = 10) -> CNNPredictResult:
    """Predict pet breed using Test-Time Augmentation (multi-crop ensemble).

    Runs inference on *num_crops* augmented versions of the image and averages
    the softmax probabilities. This typically improves top-1 accuracy by 1-2%
    at the cost of num_crops × inference time.

    Args:
        model: Pre-loaded CNN model (from _get_model or container).
        image_path: Absolute or relative path to a JPG/PNG pet image.
        num_crops: Number of augmented crops to average (default 10).

    Returns:
        CNNPredictResult with ensembled confidence scores.
    """
    if not os.path.exists(image_path):
        return CNNPredictResult(
            breed_en="", breed_cn="无法读取图片文件",
            confidence=0.0, top3=[], status="failed",
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return CNNPredictResult(
            breed_en="", breed_cn="无法读取图片文件",
            confidence=0.0, top3=[], status="failed",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Collect probabilities from multiple augmented crops
    all_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(num_crops):
            crop = _tta_transform(image).unsqueeze(0).to(device)
            logits = model(crop)
            calibrated_logits = logits / TEMPERATURE
            probs = torch.softmax(calibrated_logits, dim=1)
            all_probs.append(probs)

    # Average probabilities across all crops
    avg_probs = torch.stack(all_probs).mean(dim=0).squeeze(0)  # (37,)

    # Extract top-3 from averaged probabilities
    top3_probs, top3_indices = torch.topk(avg_probs, 3)
    top1_conf = top3_probs[0].item()
    top3 = [
        _build_top3_entry(top3_indices[i].item(), top3_probs[i].item())
        for i in range(3)
    ]

    top1_en = CLASS_NAMES[top3_indices[0].item()]
    top1_cn = BREED_MAPPING.get(_en_to_key(top1_en), top1_en)

    if top1_conf >= 0.85:
        status = "success"
    elif top1_conf >= 0.40:
        status = "low_confidence"
    else:
        status = "failed"

    return CNNPredictResult(
        breed_en=top1_en,
        breed_cn=top1_cn,
        confidence=round(top1_conf, 4),
        top3=top3,
        status=status,
    )


def predict_breed_tta(image_path: str, num_crops: int = 10) -> CNNPredictResult:
    """Convenience wrapper: predict_breed with TTA using lazy-loaded model."""
    model = _get_model()
    return predict_with_tta(model, image_path, num_crops=num_crops)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from app.common.logger import setup_logging

    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m app.cnn.inference <image_path>")
        sys.exit(1)

    result = predict_breed(sys.argv[1])
    print(result.model_dump_json(indent=2))
