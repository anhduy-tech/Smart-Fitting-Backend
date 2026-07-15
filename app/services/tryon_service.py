"""
app/services/tryon_service.py

Sinh anh thu do (yeu cau 1.1.6). BAN NAY THAY THE hoan toan model cu
(mediapipe Pose + SD Inpainting + IP-Adapter) bang CatVTON - model
chuyen dung cho virtual try-on, giu dung hoa tiet/mau sac trang phuc
(khac voi IP-Adapter chi "goi y phong cach chung", khong truyen chinh
xac pixel/hoa tiet - day la nguyen nhan ket qua ban cu bi sai mau/khong
doi ao ro rang nhu da quan sat thuc te).

## GIAY PHEP - DOC TRUOC KHI DUNG PRODUCTION THUONG MAI
CatVTON phat hanh theo Creative Commons BY-NC-SA 4.0 (NonCommercial).
Neu Smart-Fitting-Backend la san pham/dich vu THUONG MAI, can tu kiem
tra ky dieu khoan giay phep truoc khi dung trong production (tham khao
y kien phap ly neu can). Neu chi la do an hoc tap/nghien cuu, NC
thuong khong van de.

## Interface KHONG DOI so voi ban cu (tasks.py khong can sua gi):
    generate_tryon_image(portrait: PIL.Image, garment: PIL.Image,
                          category_type: str) -> PIL.Image
    compose_with_frame(tryon_image: PIL.Image, frame_image: PIL.Image) -> PIL.Image

## Vendor code
Thu muc con `catvton_lib/` chua nguyen ban `model/pipeline.py`,
`model/attn_processor.py`, `model/utils.py`, `utils.py` tu repo
https://github.com/Zheng-Chong/CatVTON (KHONG copy cloth_masker.py/
DensePose/SCHP vi ban mask-free khong can AutoMasker). Code nay dung
IMPORT TUYET DOI dang `model.xxx` / `utils` (giong repo goc, khong
phai package Django binh thuong) nen PHAI them thu muc catvton_lib/
vao sys.path truoc khi import - xem _ensure_catvton_on_path() ben duoi.
"""
import io
import logging
import os
import sys
import uuid
from functools import lru_cache

from django.conf import settings
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Cau hinh (co the ghi de qua settings.py / .env)
# =============================================================================
# mask_free=True (mac dinh): dung checkpoint CatVTON-MaskFree, KHONG can
# AutoMasker (DensePose+SCHP) -> nhe hon nhieu ve VRAM/RAM, phu hop GPU
# 6-8GB. Doi False chi khi co GPU >=8GB VRAM rieng va muon do chinh xac
# cao hon (AutoMasker tu dinh vi vung ao chinh xac hon crop theo pose).
TRYON_MASK_FREE = getattr(settings, "TRYON_MASK_FREE", True)
TRYON_CATVTON_REPO = getattr(
    settings, "TRYON_CATVTON_REPO",
    "zhengchong/CatVTON-MaskFree" if TRYON_MASK_FREE else "zhengchong/CatVTON",
)
# QUAN TRONG: 2 kien truc UNet khac nhau ve so kenh input - dung sai se
# loi "conv_in channel mismatch" (da gap thuc te khi test):
#   - mask_free=True  -> UNet pix2pix 8 kenh  -> base "timbrooks/instruct-pix2pix"
#   - mask_free=False -> UNet inpainting 9 kenh -> base "booksforcharlie/stable-diffusion-inpainting"
TRYON_BASE_MODEL = getattr(
    settings, "TRYON_BASE_MODEL",
    "timbrooks/instruct-pix2pix" if TRYON_MASK_FREE else "booksforcharlie/stable-diffusion-inpainting",
)
TRYON_MIXED_PRECISION = getattr(settings, "TRYON_MIXED_PRECISION", "fp16")  # "no"|"fp16"|"bf16"
TRYON_DEVICE = getattr(settings, "TRYON_DEVICE", "cuda")
TRYON_STEPS = getattr(settings, "TRYON_STEPS", 40)
TRYON_GUIDANCE_SCALE = getattr(settings, "TRYON_GUIDANCE_SCALE", 2.5)
TRYON_WIDTH = getattr(settings, "TRYON_WIDTH", 576)   # giam so voi 768 mac dinh cua CatVTON de vua GPU VRAM nho
TRYON_HEIGHT = getattr(settings, "TRYON_HEIGHT", 768)  # giam so voi 1024 mac dinh
TRYON_LOW_VRAM_MODE = getattr(settings, "TRYON_LOW_VRAM_MODE", True)
TRYON_DEBUG_DIR = getattr(settings, "TRYON_DEBUG_DIR", "tryon_debug")

# category_type cua san pham (models.py) -> cloth_type CatVTON dung
_CATEGORY_TO_CLOTH_TYPE = {
    "shirt": "upper",
    "pants": "lower",
    "dress": "overall",
}


def _ensure_catvton_on_path():
    """Them thu muc catvton_lib/ vao sys.path (1 lan) de import duoc
    `model.pipeline` / `utils` dung nguyen ban goc repo CatVTON."""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catvton_lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)


# =============================================================================
# 1. Load pipeline (nang - chi load 1 lan, dung chung cho moi request)
# =============================================================================
@lru_cache(maxsize=1)
def _load_engine():
    _ensure_catvton_on_path()
    import torch
    from huggingface_hub import snapshot_download
    from model.pipeline import CatVTONPipeline, CatVTONPix2PixPipeline  # noqa: E402 (path da them o tren)

    device = TRYON_DEVICE if torch.cuda.is_available() else "cpu"
    if TRYON_DEVICE == "cuda" and device == "cpu":
        logger.warning("Khong tim thay GPU - CatVTON se chay tren CPU (rat cham).")

    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map.get(TRYON_MIXED_PRECISION, torch.float16)

    repo_path = snapshot_download(repo_id=TRYON_CATVTON_REPO)

    # LUU Y (bug nho trong CatVTONPix2PixPipeline.auto_attn_ckpt_load): no
    # dung thang `version` lam TEN THU MUC thay vi map qua ten day du nhu
    # CatVTONPipeline goc -> phai truyen dung ten thu muc that "mix-48k-1024".
    attn_version = "mix-48k-1024" if TRYON_MASK_FREE else "mix"

    pipeline_cls = CatVTONPix2PixPipeline if TRYON_MASK_FREE else CatVTONPipeline
    pipeline = pipeline_cls(
        base_ckpt=TRYON_BASE_MODEL,
        attn_ckpt=repo_path,
        attn_ckpt_version=attn_version,
        weight_dtype=weight_dtype,
        use_tf32=True,
        device=device,
    )

    if TRYON_LOW_VRAM_MODE:
        try:
            pipeline.unet.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        except Exception:
            pass

    automasker = None
    if not TRYON_MASK_FREE:
        from model.cloth_masker import AutoMasker  # chi can khi mask_free=False

        automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device=device,
        )

    logger.info(f"CatVTON engine da san sang (mask_free={TRYON_MASK_FREE}, device={device})")
    return pipeline, automasker, device


def _clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# =============================================================================
# 2. Tien xu ly anh (tai su dung nguyen tac tu ban cu)
# =============================================================================
def _flatten_to_rgb(image: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    """Chuyen anh RGBA/LA/P-transparency sang RGB AN TOAN (composite len nen
    mau dong nhat truoc, khong .convert("RGB") truc tiep - tranh vien/quang
    den quanh chu the). Giong het logic ban cu."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, bg_color)
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


# =============================================================================
# 3. Ham chinh - GIU NGUYEN SIGNATURE de tasks.py khong doi
# =============================================================================
def generate_tryon_image(portrait: Image.Image, garment: Image.Image, category_type: str) -> Image.Image:
    """
    Sinh anh nguoi (portrait) mac trang phuc (garment) bang CatVTON.
    Tra ve anh PIL (kich thuoc TRYON_WIDTH x TRYON_HEIGHT - KHAC ban cu
    la tra ve dung kich thuoc portrait goc, vi CatVTON lam viec o resolution
    co dinh; neu can giu dung kich thuoc goc, resize lai o tasks.py hoac
    them logic resize tai day).
    """
    _ensure_catvton_on_path()
    from model.utils import resize_and_crop, resize_and_padding  # noqa: E402

    portrait = _flatten_to_rgb(portrait)
    garment = _flatten_to_rgb(garment)
    cloth_type = _CATEGORY_TO_CLOTH_TYPE.get(category_type, "upper")

    pipeline, automasker, device = _load_engine()

    portrait_resized = resize_and_crop(portrait, (TRYON_WIDTH, TRYON_HEIGHT))
    garment_resized = resize_and_padding(garment, (TRYON_WIDTH, TRYON_HEIGHT))

    if device == "cuda":
        _clear_gpu_cache()

    call_kwargs = dict(
        image=portrait_resized,
        condition_image=garment_resized,
        num_inference_steps=TRYON_STEPS,
        guidance_scale=TRYON_GUIDANCE_SCALE,
        width=TRYON_WIDTH,
        height=TRYON_HEIGHT,
    )

    if automasker is not None:
        mask = automasker(portrait_resized, cloth_type)["mask"]
        call_kwargs["mask"] = mask

    logger.info(f"generate_tryon_image: category={category_type} -> cloth_type={cloth_type}, "
                f"size={TRYON_WIDTH}x{TRYON_HEIGHT}, steps={TRYON_STEPS}")

    result = pipeline(**call_kwargs)[0]

    if device == "cuda":
        _clear_gpu_cache()

    if TRYON_DEBUG_DIR:
        try:
            debug_dir = os.path.join(settings.MEDIA_ROOT, TRYON_DEBUG_DIR)
            os.makedirs(debug_dir, exist_ok=True)
            ts = uuid.uuid4().hex[:8]
            result.save(os.path.join(debug_dir, f"{ts}_1_catvton_output.png"))
            portrait_resized.save(os.path.join(debug_dir, f"{ts}_2_portrait_input.png"))
            garment_resized.save(os.path.join(debug_dir, f"{ts}_3_garment_input.png"))
            logger.info(f"generate_tryon_image: da luu anh debug tai {debug_dir}/{ts}_*.png")
        except Exception as e:
            logger.warning(f"generate_tryon_image: khong luu duoc anh debug: {e}")

    return result


# =============================================================================
# 4. Ghep ket qua vao Khung nen (Frame) - GIU NGUYEN 100% tu ban cu, khong
#    lien quan gi den viec doi model try-on ben tren.
# =============================================================================
def compose_with_frame(tryon_image: Image.Image, frame_image: Image.Image) -> Image.Image:
    """Tach nguoi khoi tryon_image (dung lai get_person_mask co san trong
    background_inpaint.py), roi dan len tren frame_image."""
    from app.services.background_inpaint import get_person_mask

    person_mask = get_person_mask(tryon_image)
    frame = frame_image.convert("RGB")

    scale = frame.height / tryon_image.height
    new_w = int(tryon_image.width * scale)
    person_resized = tryon_image.resize((new_w, frame.height), Image.LANCZOS)
    mask_resized = person_mask.resize((new_w, frame.height), Image.LANCZOS)

    paste_x = (frame.width - new_w) // 2
    result = frame.copy()
    result.paste(person_resized, (paste_x, 0), mask_resized)
    return result