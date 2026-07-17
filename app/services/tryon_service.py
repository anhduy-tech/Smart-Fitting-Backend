"""
app/services/tryon_service.py
"""
import logging
import os
import uuid
from functools import lru_cache

from django.conf import settings
from PIL import Image

from app.services.catvton_engine import TryOnEngine

logger = logging.getLogger(__name__)

# =============================================================================
# Cau hinh (co the ghi de qua settings.py / .env)
# =============================================================================
TRYON_MASK_FREE = getattr(settings, "TRYON_MASK_FREE", False)
TRYON_MIXED_PRECISION = getattr(settings, "TRYON_MIXED_PRECISION", "fp16")
TRYON_DEVICE = getattr(settings, "TRYON_DEVICE", "cuda")
TRYON_STEPS = getattr(settings, "TRYON_STEPS", 40)
TRYON_GUIDANCE_SCALE = getattr(settings, "TRYON_GUIDANCE_SCALE", 2.5)
TRYON_WIDTH = getattr(settings, "TRYON_WIDTH", 576)
TRYON_HEIGHT = getattr(settings, "TRYON_HEIGHT", 768)
TRYON_LOW_VRAM = getattr(settings, "TRYON_LOW_VRAM_MODE", True)
TRYON_DEBUG_DIR = getattr(settings, "TRYON_DEBUG_DIR", "tryon_debug")

# category_type (models.py) -> cloth_type CatVTON dung
_CATEGORY_TO_CLOTH_TYPE = {
    "shirt": "upper",
    "pants": "lower",
    "dress": "overall",
}


@lru_cache(maxsize=1)
def _get_engine() -> TryOnEngine:
    logger.info(f"Khoi tao TryOnEngine (mask_free={TRYON_MASK_FREE})...")
    return TryOnEngine(
        mask_free=TRYON_MASK_FREE,
        mixed_precision=TRYON_MIXED_PRECISION,
        device=TRYON_DEVICE,
        low_vram=TRYON_LOW_VRAM,
    )


def _flatten_to_rgb(image: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    """Chuyen RGBA/LA/P-transparency sang RGB an toan (composite len nen mau
    dong nhat, tranh vien den quanh chu the)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, bg_color)
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def generate_tryon_image(portrait: Image.Image, garment: Image.Image, category_type: str) -> Image.Image:
    """GIU NGUYEN SIGNATURE de tasks.py khong doi gi ca."""
    portrait = _flatten_to_rgb(portrait)
    garment = _flatten_to_rgb(garment)
    cloth_type = _CATEGORY_TO_CLOTH_TYPE.get(category_type, "upper")

    engine = _get_engine()

    logger.info(f"generate_tryon_image: category={category_type} -> cloth_type={cloth_type}, "
                f"size={TRYON_WIDTH}x{TRYON_HEIGHT}, steps={TRYON_STEPS}, mask_free={TRYON_MASK_FREE}")

    result = engine.run(
        person_image=portrait,
        cloth_image=garment,
        cloth_type=cloth_type,
        num_inference_steps=TRYON_STEPS,
        guidance_scale=TRYON_GUIDANCE_SCALE,
        width=TRYON_WIDTH,
        height=TRYON_HEIGHT,
    )

    if TRYON_DEBUG_DIR:
        try:
            debug_dir = os.path.join(settings.MEDIA_ROOT, TRYON_DEBUG_DIR)
            os.makedirs(debug_dir, exist_ok=True)
            ts = uuid.uuid4().hex[:8]
            result.save(os.path.join(debug_dir, f"{ts}_1_catvton_output.png"))
            logger.info(f"generate_tryon_image: da luu anh debug tai {debug_dir}/{ts}_*.png")
        except Exception as e:
            logger.warning(f"generate_tryon_image: khong luu duoc anh debug: {e}")

    return result


def compose_with_frame(tryon_image: Image.Image, frame_image: Image.Image) -> Image.Image:

    import cv2
    import numpy as np
    from app.services.background_inpaint import get_person_mask

    # dilate=0/blur=0: bo qua buoc no rong + lam mo mac dinh (danh cho use
    # case khac), tu xu ly erode+blur rieng ben duoi cho phu hop cat-dan.
    person_mask = get_person_mask(tryon_image, dilate=0, blur=0)

    # Erode mask vao trong ~2-3px de cat bo han vung bien pha mau nen cu,
    # roi blur NHE lai de co canh mem tu nhien ma khong keo mau nen cu vao.
    mask_np = np.array(person_mask)
    erode_kernel = np.ones((10, 10), np.uint8)
    mask_np = cv2.erode(mask_np, erode_kernel, iterations=1)
    mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)
    person_mask = Image.fromarray(mask_np)

    frame = frame_image.convert("RGB")

    scale = frame.height / tryon_image.height
    new_w = int(tryon_image.width * scale)
    person_resized = tryon_image.resize((new_w, frame.height), Image.LANCZOS)
    mask_resized = person_mask.resize((new_w, frame.height), Image.LANCZOS)

    paste_x = (frame.width - new_w) // 2
    result = frame.copy()
    result.paste(person_resized, (paste_x, 0), mask_resized)
    return result