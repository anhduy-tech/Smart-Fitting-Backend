"""
app/services/portrait_processing.py

Tach NGUOI ra khoi NEN cho anh chan dung nguoi dung (yeu cau 1.1.5: "Neu
anh co nen, he thong cho phep tach nguoi ra khoi nen de lay chan dung cua
nguoi su dung").

QUAN TRONG - de phan biet voi background_inpaint.py:
  - background_inpaint.remove_person_and_reconstruct(): XOA NGUOI, GIU/VE
    LAI NEN (dung cho admin tao Frame, yeu cau 1.2.4) -> can inpainting
    (LaMa) de "bia" lai phan nen bi khuyet sau khi xoa nguoi.
  - portrait_processing.extract_person() (file nay): GIU NGUOI, XOA NEN
    (dung cho nguoi dung tach chan dung cua chinh minh, yeu cau 1.1.5) ->
    chi can segmentation (rembg), KHONG can inpainting vi khong "bia" gi
    ca - phan nen bi xoa thi de trong suot (RGBA) hoac thay mau nen dong
    nhat, khong can AI "doan" lai noi dung nen.
"""
import logging
from functools import lru_cache

from PIL import Image

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_rembg_session():
    from rembg import new_session
    # Dung chung model "u2net_human_seg" (toi uu cho nguoi) giong
    # background_inpaint.py, nhung la 1 session/cache RIENG - rembg cho
    # phep nhieu session cung model song song khong xung dot, giu tach
    # biet ro rang ve muc dich su dung giua 2 file.
    return new_session("u2net_human_seg")


def extract_person(image: Image.Image, background_color: tuple = None) -> Image.Image:
    """
    Tach nguoi khoi nen cho 1 anh chan dung.

    Args:
        image: anh goc (PIL Image, se tu convert sang RGB).
        background_color: neu None (mac dinh) -> tra ve anh RGBA voi nen
            TRONG SUOT (kenh alpha=0 o vung nen). Neu truyen vao 1 tuple
            RGB (vd (255, 255, 255) cho nen trang) -> tra ve anh RGB voi
            nen duoc thay bang mau dong nhat do (phu hop neu client/app
            khong xu ly duoc anh nen trong suot).

    Returns:
        PIL Image (RGBA neu background_color=None, nguoc lai RGB).
    """
    from rembg import remove

    session = _get_rembg_session()
    cutout = remove(image.convert("RGB"), session=session)  # -> RGBA, nen alpha=0

    if background_color is not None:
        bg = Image.new("RGB", cutout.size, background_color)
        bg.paste(cutout, mask=cutout.split()[3])  # dan theo kenh alpha lam mask
        return bg

    return cutout