"""
app/services/tryon_service.py

Sinh anh thu do ao (yeu cau 1.1.6: chon Quan/Ao/Vay + Khung nen -> sinh anh
nguoi mac do trong khung nen).

## Huong tiep can (MVP - xem README/ghi chu trong PR de biet ly do)
Pipeline HR-VITON goc (repo Fashion-U-Want) can 4 model rieng biet chay tuan
tu (OpenPose build C++, Graphonomy human-parsing, Detectron2 DensePose, GAN
generator voi checkpoint rieng) - qua phuc tap/de vo khi tu host tren server
Django thuong. Ban nay dung huong nhe hon, tu host duoc ngay:

  1. mediapipe Pose (thuan Python, KHONG can build C++ nhu OpenPose) de lay
     toa do khop xuong (vai, hong, dau goi, mat ca chan...).
  2. Tu toa do khop xuong, dung 1 vung da giac (polygon) lam mask cho vung
     ao/quan/vay tuong ung voi category_type cua san pham.
  3. Stable Diffusion Inpainting + IP-Adapter (dieu kien hoa theo anh san
     pham qua ip_adapter_image) de "ve" trang phuc vao dung vung mask do.
  4. Composite lai (giong het nguyen tac trong background_inpaint.py): giu
     nguyen 100% pixel ngoai mask, chi lay ket qua model trong vung mask.

Chat luong se KHONG bang cac model try-on chuyen dung (CatVTON, IDM-VTON)
nhung khong can toi 4 model rieng + checkpoint GAN rieng, va co the nang cap
sau: chi can sua ham `_run_diffusion_inpaint()` ben duoi, phan con lai
(mask theo pose, composite, ghep frame, Celery task, view) khong doi.
"""
import io
import logging
from functools import lru_cache

import numpy as np
import cv2
from django.conf import settings
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

# =============================================================================
# Cau hinh (co the ghi de qua settings.py / bien .env tuong ung neu can)
# =============================================================================
TRYON_BASE_MODEL = getattr(settings, "TRYON_BASE_MODEL", "runwayml/stable-diffusion-inpainting")
TRYON_IP_ADAPTER_REPO = getattr(settings, "TRYON_IP_ADAPTER_REPO", "h94/IP-Adapter")
TRYON_IP_ADAPTER_WEIGHT = getattr(settings, "TRYON_IP_ADAPTER_WEIGHT", "ip-adapter_sd15.bin")
TRYON_IP_ADAPTER_SCALE = getattr(settings, "TRYON_IP_ADAPTER_SCALE", 0.75)
TRYON_STRENGTH = getattr(settings, "TRYON_STRENGTH", 0.85)  # 0-1, cang cao cang "ve lai" nhieu
TRYON_GUIDANCE_SCALE = getattr(settings, "TRYON_GUIDANCE_SCALE", 6.0)
TRYON_STEPS = getattr(settings, "TRYON_STEPS", 30)
TRYON_DEVICE = getattr(settings, "TRYON_DEVICE", "cuda")  # fallback tu dong ve "cpu" neu khong co GPU
TRYON_IMAGE_SIZE = getattr(settings, "TRYON_IMAGE_SIZE", 512)  # SD 1.5 lam viec tot nhat o 512x512

# Padding (ti le % so voi kich thuoc vung) khi ve mask tu toa do khop xuong,
# de mask khong bam sat khung xuong qua (ao/vay co do rong hon co the).
_REGION_PADDING_RATIO = 0.12
_MASK_FEATHER_BLUR = 15


# =============================================================================
# 1. Uoc luong vung ao/quan/vay tu mediapipe Pose
# =============================================================================
@lru_cache(maxsize=1)
def _get_pose_detector():
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )


# Chi so landmark cua mediapipe Pose (33 diem) can dung.
_LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_elbow": 13, "right_elbow": 14,
}


def _get_landmark_points(image: Image.Image) -> dict | None:
    """Tra ve dict {ten_khop: (x, y) pixel} hoac None neu khong phat hien duoc nguoi."""
    detector = _get_pose_detector()
    image_rgb = np.array(image.convert("RGB"))
    result = detector.process(image_rgb)
    if not result.pose_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    points = {}
    for name, idx in _LM.items():
        lm = result.pose_landmarks.landmark[idx]
        points[name] = (lm.x * w, lm.y * h)
    return points


def _polygon_mask(size: tuple, points: list, padding_ratio: float = _REGION_PADDING_RATIO) -> Image.Image:
    """Ve mask (anh 'L') tu 1 danh sach diem da giac, co nong ra (padding) theo % kich thuoc vung."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    region_w = max(xs) - min(xs)
    region_h = max(ys) - min(ys)
    pad_x = region_w * padding_ratio
    pad_y = region_h * padding_ratio

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    padded_points = []
    for x, y in points:
        # Day diem ra xa tam mot chut de "nong" da giac ra ngoai, khong chi
        # dich chuyen theo truc x/y rieng le (tranh mask bi meo o goc).
        dx = (x - cx)
        dy = (y - cy)
        padded_points.append((x + (pad_x if dx >= 0 else -pad_x) * 0.5,
                               y + (pad_y if dy >= 0 else -pad_y) * 0.5))

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(padded_points, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(_MASK_FEATHER_BLUR))
    return mask


def get_garment_region_mask(portrait: Image.Image, category_type: str) -> Image.Image:
    """
    Tra ve mask (anh 'L', cung kich thuoc voi portrait) danh dau vung can
    "ve lai" trang phuc, dua theo category_type cua san pham:
      - shirt : tu vai toi hong
      - pants : tu hong toi mat ca chan
      - dress : tu vai toi mat ca chan
    Neu khong phat hien duoc pose (anh khong ro nguoi/goc chup xau), fallback
    ve 1 vung hinh chu nhat co dinh o giua anh (tot hon la khong xu ly gi).
    """
    points = _get_landmark_points(portrait)
    w, h = portrait.size

    if points is None:
        logger.warning("Khong phat hien duoc pose - dung mask fallback hinh chu nhat")
        top = int(h * (0.25 if category_type == "pants" else 0.15))
        bottom = int(h * (0.60 if category_type == "shirt" else 0.90))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle([w * 0.2, top, w * 0.8, bottom], fill=255)
        return mask.filter(ImageFilter.GaussianBlur(_MASK_FEATHER_BLUR))

    ls, rs = points["left_shoulder"], points["right_shoulder"]
    lh, rh = points["left_hip"], points["right_hip"]
    lk, rk = points["left_knee"], points["right_knee"]
    la, ra = points["left_ankle"], points["right_ankle"]

    if category_type == "shirt":
        polygon = [ls, rs, rh, lh]
    elif category_type == "pants":
        polygon = [lh, rh, ra, la]
    else:  # 'dress' hoac gia tri khac -> ca vai toi mat ca chan
        polygon = [ls, rs, ra, la]

    return _polygon_mask((w, h), polygon)


# =============================================================================
# 2. Model sinh anh (Stable Diffusion Inpainting + IP-Adapter)
# =============================================================================
@lru_cache(maxsize=1)
def _load_pipeline():
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = TRYON_DEVICE if torch.cuda.is_available() else "cpu"
    if TRYON_DEVICE == "cuda" and device == "cpu":
        logger.warning("Khong tim thay GPU - chay TRYON tren CPU (se rat cham, khuyen nghi dung Celery + gioi han so worker song song).")

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(TRYON_BASE_MODEL, torch_dtype=dtype, safety_checker=None)
    pipe.load_ip_adapter(TRYON_IP_ADAPTER_REPO, subfolder="models", weight_name=TRYON_IP_ADAPTER_WEIGHT)
    pipe.set_ip_adapter_scale(TRYON_IP_ADAPTER_SCALE)
    pipe = pipe.to(device)
    return pipe, device


_CATEGORY_PROMPT = {
    "shirt": "a person wearing this exact shirt, same garment, photo realistic, well fitted, high quality",
    "pants": "a person wearing this exact pants, same garment, photo realistic, well fitted, high quality",
    "dress": "a person wearing this exact dress, same garment, photo realistic, well fitted, high quality",
}
_NEGATIVE_PROMPT = "blurry, deformed, extra limbs, bad anatomy, low quality, watermark, text, cropped"


def _resize_for_model(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.LANCZOS)


def _run_diffusion_inpaint(portrait: Image.Image, mask: Image.Image, garment: Image.Image, category_type: str) -> Image.Image:
    """Goi model, tra ve NGUYEN output (cung kich thuoc TRYON_IMAGE_SIZE, chua composite/resize lai)."""
    pipe, device = _load_pipeline()
    size = TRYON_IMAGE_SIZE

    portrait_sq = _resize_for_model(portrait, size)
    mask_sq = mask.resize((size, size), Image.LANCZOS)
    garment_sq = _resize_for_model(garment, size)

    prompt = _CATEGORY_PROMPT.get(category_type, _CATEGORY_PROMPT["shirt"])

    result = pipe(
        prompt=prompt,
        negative_prompt=_NEGATIVE_PROMPT,
        image=portrait_sq,
        mask_image=mask_sq,
        ip_adapter_image=garment_sq,
        strength=TRYON_STRENGTH,
        guidance_scale=TRYON_GUIDANCE_SCALE,
        num_inference_steps=TRYON_STEPS,
    ).images[0]
    return result


def _composite(before: Image.Image, after: Image.Image, mask: Image.Image) -> Image.Image:
    """Giong het nguyen tac trong background_inpaint.py: giu nguyen pixel goc
    ngoai mask, chi lay ket qua model trong vung mask (gradient o vien mask)."""
    if after.size != before.size:
        after = after.resize(before.size, Image.LANCZOS)
    mask_l = mask.convert("L")
    if mask_l.size != before.size:
        mask_l = mask_l.resize(before.size, Image.LANCZOS)
    return Image.composite(after, before, mask_l)


def generate_tryon_image(portrait: Image.Image, garment: Image.Image, category_type: str) -> Image.Image:
    """
    Ham chinh: sinh anh nguoi (portrait) mac trang phuc (garment) theo dung
    vung co the (category_type: 'shirt' | 'pants' | 'dress').
    Tra ve anh PIL cung kich thuoc voi portrait goc.
    """
    original_size = portrait.size
    portrait = portrait.convert("RGB")

    mask = get_garment_region_mask(portrait, category_type)
    raw_result = _run_diffusion_inpaint(portrait, mask, garment, category_type)

    # raw_result dang o kich thuoc TRYON_IMAGE_SIZE vuong -> resize ve dung
    # kich thuoc portrait goc truoc khi composite, tranh meo ti le.
    raw_result_resized = raw_result.resize(original_size, Image.LANCZOS)
    final = _composite(portrait, raw_result_resized, mask)
    return final


# =============================================================================
# 3. Ghep ket qua vao Khung nen (Frame) da chon - yeu cau 1.1.6
# =============================================================================
def compose_with_frame(tryon_image: Image.Image, frame_image: Image.Image) -> Image.Image:
    """
    Tach nguoi ra khoi tryon_image (dung lai ham get_person_mask da co san
    trong background_inpaint.py de khong trung lap logic tach nen), roi dan
    len tren frame_image (resize + can giua theo chieu ngang, neo day anh).
    """
    from app.services.background_inpaint import get_person_mask

    person_mask = get_person_mask(tryon_image)
    frame = frame_image.convert("RGB")

    # Scale nguoi cho vua chieu cao khung nen (giu ti le), can giua theo
    # chieu ngang, neo o day (gia dinh nguoi dung tren "san" cua khung nen).
    scale = frame.height / tryon_image.height
    new_w = int(tryon_image.width * scale)
    person_resized = tryon_image.resize((new_w, frame.height), Image.LANCZOS)
    mask_resized = person_mask.resize((new_w, frame.height), Image.LANCZOS)

    paste_x = (frame.width - new_w) // 2
    result = frame.copy()
    result.paste(person_resized, (paste_x, 0), mask_resized)
    return result
