"""
app/services/background_inpaint.py

Tach nguoi khoi anh + AI ve lai (inpaint) phan nen bi khuyet, dung cho
yeu cau 1.2.4: "Chuc nang tach nen: khi tai len hinh anh co chua nguoi,
he thong cho phep tach nguoi ra khoi nen muc dich la chi giu lai khung
nen (Frame)".

## NGUYEN TAC QUAN TRONG NHAT (da sua trong ban nay)
Model inpaint (LaMa) luon tra ve TOAN BO canvas da qua xu ly, ke ca vung
NGOAI mask. Neu dung thang output do lam anh cuoi, ca buc anh se bi "ve
lai" qua mang no-ron -> mo/nhoe/mat chi tiet o CA NHUNG VUNG KHONG CAN
SUA (day chinh la nguyen nhan gay ra loi "ca tam anh bi mo ngay ca cho
khong co nguoi" da gap truoc do).

Cach sua: sau MOI lan goi model, luon GHEP (composite) lai:
    ket_qua_cuoi = anh_truoc_do NGOAI vung mask (giu 100% pixel goc)
                 + output_model  TRONG vung mask (phan da duoc ve lai)
Ham `_composite()` ben duoi lam viec nay bang PIL.Image.composite(),
dam bao vung nen that su khong bi dong den mot chut nao.

Ngoai ra, ban nay cung:
- Giam kich thuoc mask nguoi (dilate) tu ~18px*2 lan xuong con so it hon,
  tranh bat model phai "bia" ra vung nen lon hon can thiet.
- Bo lan inpaint pass 2 chay lai gan nhu toan bo mask cu (gay chong lap
  loi), thay bang 1 pass chinh + (tuy chon) 1 pass "vien" rat mong chi
  quanh duong bien mask de xu ly seam, khong dung lai ca vung lon.
- Sharpen/Contrast/Color CHI ap dung trong vung mask (qua composite),
  khong con anh huong den vung nen von da dung tu dau.
"""
import logging
from functools import lru_cache
import numpy as np
import cv2
import os
from django.conf import settings
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEFAULT_INPAINT_MODEL = getattr(settings, "INPAINT_MODEL_NAME", "lama")
INPAINT_HD_RESIZE_LIMIT = getattr(settings, "INPAINT_HD_RESIZE_LIMIT", 1400)

# Chien luoc xu ly anh lon cua iopaint khi inpaint. CROP (mac dinh cua
# chinh thu vien) = chi cat vung quanh mask ra xu ly o DO PHAN GIAI GOC,
# giu chi tiet/net hon nhieu so voi RESIZE (thu nho ca anh). Xem
# _run_inpaint_raw() de biet cach dung.
HD_STRATEGY_CROP_MARGIN = getattr(settings, "HD_STRATEGY_CROP_MARGIN", 160)
HD_STRATEGY_CROP_TRIGGER_SIZE = getattr(settings, "HD_STRATEGY_CROP_TRIGGER_SIZE", 640)

# Cuong do "cay" lai texture/chi tiet be mat vao vung vua inpaint (0.0 =
# tat, 1.0 = giu nguyen 100% bien do chi tiet vay muon tu nen that). Gia
# tri qua cao co the lam vung inpaint bi "nhieu hat" khong tu nhien neu
# nen goc it texture (vd tuong son phang) - 0.5-0.7 thuong hop ly cho
# tuong/be mat co van (gach, be tong, go).
TEXTURE_DETAIL_STRENGTH = getattr(settings, "TEXTURE_DETAIL_STRENGTH", 0.6)

# Kich thuoc dilate mask nguoi (px). TRUOC DAY la 18 (x2 lan = ~30-40px
# moi canh) - qua lon, khien model phai "bia" ra mot vung nen rong hon
# nhieu so voi nguoi that, de gay hallucinate/mo. Gia tri moi nho hon,
# chi du de xoa vien/halo quanh nguoi (toc bay, bong do, vien anti-alias)
# ma khong an qua nhieu nen that vao mask.
PERSON_MASK_DILATE = getattr(settings, "PERSON_MASK_DILATE", 8)
PERSON_MASK_DILATE_ITER = getattr(settings, "PERSON_MASK_DILATE_ITER", 1)

# Do mo (Gaussian blur, px, so le) cua vien mask -> tao gradient alpha
# muot ma khi ghep (composite), tranh duong ranh gioi anh gay cung.
PERSON_MASK_FEATHER_BLUR = getattr(settings, "PERSON_MASK_FEATHER_BLUR", 9)


@lru_cache(maxsize=1)
def _get_rembg_session():
    from rembg import new_session
    return new_session("u2net_human_seg")


@lru_cache(maxsize=4)
def _get_inpaint_model_manager(model_name: str):
    import torch
    from iopaint.model import models as iopaint_models
    from iopaint.model_manager import ModelManager
    if model_name in iopaint_models and iopaint_models[model_name].is_erase_model:
        model_cls = iopaint_models[model_name]
        if not model_cls.is_downloaded():
            logger.info(f"Đang tải model '{model_name}'...")
            model_cls.download()
        device = torch.device("cpu")
        return ModelManager(name=model_name, device=device)
    raise ValueError(f"Model {model_name} không hỗ trợ")


def _get_inpaint_model_manager_with_fallback(model_name: str):
    try:
        return _get_inpaint_model_manager(model_name), model_name
    except Exception as e:
        if model_name == "lama":
            raise
        logger.warning(f"Fallback sang lama: {e}")
        return _get_inpaint_model_manager("lama"), "lama"


def preprocess_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if max(w, h) > INPAINT_HD_RESIZE_LIMIT:
        scale = INPAINT_HD_RESIZE_LIMIT / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


# Kich thuoc "noi lien" (morphological close) cac mieng mask bi dut gay
# gan nhau - vi du: ban tay dat rieng tren lan can, khong dinh lien voi
# than nguoi trong mask cua rembg, se bi BO SOT (khong duoc inpaint) neu
# khong noi lai. Gia tri lon hon PERSON_MASK_DILATE nhieu vi muc dich la
# "bac cau" khoang trong, khong phai "nong rong vien".
PERSON_MASK_CLOSE_KERNEL = getattr(settings, "PERSON_MASK_CLOSE_KERNEL", 25)

# Nguong nhi phan hoa mask (0-255) truoc khi dua vao model inpaint. Gia
# tri thap hon se giu lai ca nhung vung rembg "khong chac chan lam" (vi
# du ria ban tay, toc mong) - giam nguy co bo sot bo phan nguoi, doi lai
# co the an lan mot chut nen that o vien. TRUOC DAY la 100 (kha cao).
PERSON_MASK_BINARY_THRESHOLD = getattr(settings, "PERSON_MASK_BINARY_THRESHOLD", 60)

# So lan toi da tu dong "va lai" neu sau khi xoa nguoi van con sot mot
# phan nguoi trong ket qua (vd ban tay/chan bi bo sot o lan dau). Gioi
# han lai de tranh treo/tốn tai nguyen neu anh qua kho.
MAX_RESIDUAL_FIX_PASSES = getattr(settings, "MAX_RESIDUAL_FIX_PASSES", 2)

# Dien tich toi thieu (ti le % so voi tong anh) de coi la "con sot nguoi"
# va can va them - tranh va lai vi nhung noise/false-positive rat nho
# (vai chuc pixel) khong dang ke.
RESIDUAL_PERSON_MIN_AREA_RATIO = getattr(settings, "RESIDUAL_PERSON_MIN_AREA_RATIO", 0.003)


def get_person_mask(image: Image.Image, dilate: int = None, blur: int = None) -> Image.Image:
    """
    Tao mask (anh grayscale 'L') danh dau vung co nguoi trong anh, dung
    rembg (u2net_human_seg). Cac buoc theo thu tu:
      1. Morphological CLOSE: noi lien cac mieng mask bi dut gay gan nhau
         (vd ban tay dat tren lan can tach roi khoi than nguoi) thanh 1
         vung lien mach - neu khong lam buoc nay, phan bi tach roi de bi
         BO SOT (khong duoc inpaint), de lai "vet den" trong ket qua.
      2. Nhi phan hoa o nguong thap hon truoc (PERSON_MASK_BINARY_THRESHOLD)
         de khong bo sot vien mo/khong chac chan (toc, ngon tay...).
      3. Dilate NHE: chi du xoa vien/halo quanh nguoi (KHONG dilate qua
         lon nhu ban dau, tranh bat qua nhieu nen that vao vung inpaint).
      4. Lam mo bien (Gaussian blur) de co gradient alpha muot khi ghep
         anh (composite) sau nay.
    """
    dilate = PERSON_MASK_DILATE if dilate is None else dilate
    blur = PERSON_MASK_FEATHER_BLUR if blur is None else blur

    from rembg import remove
    session = _get_rembg_session()
    mask = remove(image, session=session, only_mask=True).convert("L")

    mask_np = np.array(mask)

    # Nhi phan hoa o nguong thap truoc khi close/dilate, de khong bo sot
    # cac vung rembg cho diem thap (vd ban tay, toc mong).
    mask_np = (mask_np > PERSON_MASK_BINARY_THRESHOLD).astype(np.uint8) * 255

    if PERSON_MASK_CLOSE_KERNEL > 0:
        close_kernel = np.ones((PERSON_MASK_CLOSE_KERNEL, PERSON_MASK_CLOSE_KERNEL), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, close_kernel)

    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=PERSON_MASK_DILATE_ITER)
    if blur > 0:
        mask_np = cv2.GaussianBlur(mask_np, (blur, blur), 0)
    return Image.fromarray(mask_np)


def _detect_residual_person_mask(image: Image.Image):
    """
    Chay lai rembg tren anh KET QUA (sau khi da xoa nguoi) de kiem tra
    con sot phan nguoi nao khong. Tra ve (mask, area_ratio):
      - mask: anh 'L' danh dau vung nghi la con sot nguoi (da close+dilate+blur)
      - area_ratio: ti le dien tich vung do so voi tong anh (0.0 - 1.0)
    Neu area_ratio < RESIDUAL_PERSON_MIN_AREA_RATIO thi coi nhu khong con
    sot gi dang ke (tranh va lai vi vai pixel nhieu/false-positive).
    """
    residual_mask = get_person_mask(image)
    mask_np = np.array(residual_mask)
    area_ratio = float(np.count_nonzero(mask_np > 30)) / mask_np.size
    return residual_mask, area_ratio


def create_manual_mask(image: Image.Image, mask_points=None):
    mask = Image.new("L", image.size, 0)
    if mask_points:
        ImageDraw.Draw(mask).polygon(mask_points, fill=255)
    return mask


def _edge_ring_mask(mask: Image.Image, ring_px: int = 6) -> Image.Image:
    """
    Tao 1 mask MONG chi bao quanh duong bien cua mask goc (vi du de refine
    seam sau pass inpaint dau, KHONG dung lai toan bo mask cu - tranh
    inpaint chong inpaint gay mo tich luy).
    """
    mask_np = np.array(mask.convert("L"))
    binary = (mask_np > 127).astype(np.uint8) * 255
    dilated = cv2.dilate(binary, np.ones((ring_px, ring_px), np.uint8), iterations=1)
    eroded = cv2.erode(binary, np.ones((ring_px, ring_px), np.uint8), iterations=1)
    ring = cv2.subtract(dilated, eroded)
    ring = cv2.GaussianBlur(ring, (5, 5), 0)
    return Image.fromarray(ring)


def _composite(before: Image.Image, after: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Ghep lai: giu NGUYEN pixel goc cua `before` o vung mask=0, chi lay
    ket qua tu `after` (output model) o vung mask=255 (gradient o giua
    nho mask da duoc lam mo bien). Day la buoc BAT BUOC sau moi lan goi
    model inpaint, neu khong ca buc anh se bi "ve lai" toan bo (xem
    giai thich o dau file).
    """
    if after.size != before.size:
        after = after.resize(before.size, Image.LANCZOS)
    mask_l = mask.convert("L")
    if mask_l.size != before.size:
        mask_l = mask_l.resize(before.size, Image.LANCZOS)
    return Image.composite(after, before, mask_l)


def _run_inpaint_raw(image: Image.Image, mask: Image.Image, model_name: str) -> Image.Image:
    """Goi model inpaint, tra ve NGUYEN output cua model (chua ghep lai)."""
    model_manager, used_model = _get_inpaint_model_manager_with_fallback(model_name)
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))
    mask_np = (mask_np > 100).astype(np.uint8) * 255

    from iopaint.schema import InpaintRequest, HDStrategy
    config = InpaintRequest(
        # HDStrategy.CROP (mac dinh cua chinh thu vien iopaint, KHONG phai
        # RESIZE nhu ban truoc): chi CAT RIENG vung quanh mask ra xu ly o
        # DO PHAN GIAI GOC (khong bi thu nho ca buc anh xuong 1400px roi
        # phong to lai nhu RESIZE) -> giu duoc nhieu chi tiet/van/hat nhieu
        # cua nen that hon han, do la nguyen nhan chinh gay ket qua mo/nhoe
        # truoc day.
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=HD_STRATEGY_CROP_MARGIN,
        hd_strategy_crop_trigger_size=HD_STRATEGY_CROP_TRIGGER_SIZE,
        hd_strategy_resize_limit=INPAINT_HD_RESIZE_LIMIT,
        use_croper=False,
        inpaint_mode="full",
    )

    result_bgr = model_manager(image_np, mask_np, config)
    return Image.fromarray(result_bgr[:, :, ::-1])


def _extract_high_freq_detail(image: Image.Image, blur_radius: float = 3.0) -> np.ndarray:
    """
    Trich xuat lop "chi tiet tan so cao" (texture/hat nhieu/van be mat)
    cua 1 anh: detail = anh_goc - anh_lam_mo. Day la phan LaMa thuong
    KHONG ve lai duoc (model co xu huong "an toan" bang mau trung binh
    muot ma), can "cay" lai thu cong tu vung nen that xung quanh.
    """
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    blurred = cv2.GaussianBlur(img_np, (0, 0), blur_radius)
    return img_np - blurred


def _synthesize_detail_for_mask(reference_image: Image.Image, mask: Image.Image, blur_radius: float = 3.0) -> np.ndarray:
    """
    Dung texture synthesis co dien (cv2.inpaint kieu Telea) de "keo dai"
    pattern chi tiet/hat nhieu tu vung nen THAT (ngoai mask) vao trong
    vung mask. Khac voi LaMa (hoc sau, hieu cau truc/mau nhung de ra
    ket qua muot), Telea chi lan truyen pattern tan so cao mot cach co
    hoc - phu hop de "vay muon" texture (khong hoc duoc cau truc phuc
    tap, nhung dung o day chi de lay lop chi tiet nen an toan).
    """
    detail = _extract_high_freq_detail(reference_image, blur_radius)
    mask_np = np.array(mask.convert("L"))
    hard_mask = (mask_np > 100).astype(np.uint8) * 255

    # cv2.inpaint chi ho tro anh 8-bit, can chuan hoa detail (co the am)
    # ve khoang [0, 255] truoc, roi giai chuan hoa lai sau khi inpaint.
    detail_u8 = np.clip(detail + 128, 0, 255).astype(np.uint8)
    synthesized = cv2.inpaint(detail_u8, hard_mask, 5, cv2.INPAINT_TELEA)
    return synthesized.astype(np.float32) - 128


def _apply_texture_detail(base_result: Image.Image, reference_image: Image.Image, mask: Image.Image, strength: float = None) -> Image.Image:
    """
    "Cay" lai chi tiet/texture (tu _synthesize_detail_for_mask) vao ket
    qua LaMa, CHI trong vung mask (qua composite) - giup vung vua duoc
    ve lai co do "gai"/van be mat gan giong nen that xung quanh hon,
    thay vi mo phang mot mau nhu truoc.
    """
    strength = TEXTURE_DETAIL_STRENGTH if strength is None else strength
    if strength <= 0:
        return base_result

    detail = _synthesize_detail_for_mask(reference_image, mask)
    base_np = np.array(base_result.convert("RGB")).astype(np.float32)
    if detail.shape[:2] != base_np.shape[:2]:
        detail = cv2.resize(detail, (base_np.shape[1], base_np.shape[0]))
    textured_np = np.clip(base_np + detail * strength, 0, 255).astype(np.uint8)
    textured_img = Image.fromarray(textured_np)
    return _composite(base_result, textured_img, mask)


def _run_inpaint(image: Image.Image, mask: Image.Image, model_name: str) -> Image.Image:
    """
    Chay inpaint VA ghep lai ngay (composite) - luon tra ve anh da ghep,
    dam bao vung ngoai mask giu nguyen 100% pixel goc cua `image`.
    """
    raw = _run_inpaint_raw(image, mask, model_name)
    return _composite(image, raw, mask)


def remove_person_and_reconstruct(
    image: Image.Image,
    model_name: str = None,
    use_manual_mask: bool = False,
    manual_mask_points: list = None,
    refine: bool = True,
) -> Image.Image:
    model_name = model_name or DEFAULT_INPAINT_MODEL
    original_size = image.size
    image = image.convert("RGB")
    image = preprocess_image(image)

    logger.info(f"Xử lý | Model: {model_name} | Size: {image.size}")

    if use_manual_mask and manual_mask_points:
        mask = create_manual_mask(image, manual_mask_points)
    else:
        mask = get_person_mask(image)

    # Pass 1: inpaint toan bo vung mask, ket qua da duoc ghep (composite)
    # nen vung ngoai mask giu nguyen pixel goc.
    result1 = _run_inpaint(image, mask, model_name)

    if refine:
        # Pass 2 (tuy chon): chi refine 1 VIEN MONG quanh duong bien mask
        # (khong phai toan bo mask cu) de xu ly seam/rang gioi con sot,
        # tranh inpaint chong inpaint gay mo tich luy nhu ban truoc.
        ring_mask = _edge_ring_mask(mask, ring_px=6)
        result_final = _run_inpaint(result1, ring_mask, model_name)
    else:
        result_final = result1

    # Cay lai chi tiet/texture that (van tuong, hat nhieu...) tu vung nen
    # xung quanh vao vung vua duoc AI ve lai - thay the cho sharpen tho
    # truoc day (chi khuech dai canh/tuong phan chu KHONG tao ra chi tiet
    # that). Van chi ap dung trong vung mask (qua composite), khong dung
    # den vung nen goc.
    textured = _apply_texture_detail(result_final, image, mask)
    enhanced = ImageEnhance.Contrast(textured).enhance(1.03)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.02)
    final = _composite(result_final, enhanced, mask)

    # Tu kiem tra + tu dong va lai neu con sot mot phan nguoi (vd ban tay
    # dat rieng tren lan can, khong dinh lien than nguoi nen bi bo sot o
    # lan xu ly dau). Chay lai rembg tren CHINH ket qua vua xu ly, neu
    # van con phat hien vung nguoi dang ke thi inpaint them 1 lan nua CHI
    # tren vung con sot do. Gioi han so lan de tranh treo neu anh qua kho.
    for retry_i in range(MAX_RESIDUAL_FIX_PASSES):
        residual_mask, area_ratio = _detect_residual_person_mask(final)
        if area_ratio < RESIDUAL_PERSON_MIN_AREA_RATIO:
            break
        logger.warning(
            f"Van con sot ~{area_ratio*100:.2f}% dien tich nghi la nguoi sau "
            f"lan xu ly {retry_i + 1} - tu dong va them 1 lan nua."
        )
        final = _run_inpaint(final, residual_mask, model_name)
    else:
        # Het so lan cho phep ma van con sot -> ghi log canh bao ro rang
        # de admin biet ma kiem tra thu cong, thay vi am tham tra ve anh
        # con loi.
        logger.warning(
            "Da va toi da %d lan nhung van con sot vung nghi la nguoi trong "
            "ket qua tach nen. Nen kiem tra thu cong anh nay truoc khi dung "
            "lam Frame.", MAX_RESIDUAL_FIX_PASSES
        )

    if final.size != original_size:
        final = final.resize(original_size, Image.LANCZOS)

    return final