"""
app/services/celebrity_detection.py

Nhan dien anh nguoi noi tieng Viet Nam (yeu cau 1.3.1):
"Khi nguoi su dung tai anh len. He thong tu dong phat hien anh nguoi
noi tieng o Viet Nam, neu phat hien se khong hien thi anh va tu dong
xoa anh".

## Cach hoat dong
1. Dung `insightface` (model `buffalo_l`, kien truc ArcFace) de phat hien
   khuon mat trong anh nguoi dung tai len va trich xuat vector dac trung
   (embedding) 512 chieu cho khuon mat lon nhat trong anh.
2. So sanh (cosine similarity) vector nay voi vector dai dien cua tung
   nguoi noi tieng trong danh sach ~224 nguoi noi tieng Viet Nam (ca si,
   dien vien, hoa hau) da duoc tinh san tu truoc.
3. Neu do tuong dong cao nhat >= CELEBRITY_THRESHOLD -> coi la trung khop
   voi nguoi noi tieng do.

## Nguon du lieu (dataset) nguoi noi tieng
Dataset: fptudsc/face-celeb-vietnamese (HuggingFace Datasets, giay phep
Apache-2.0, ~8.557 anh cua 224 nguoi noi tieng Viet Nam - ca si/dien
vien/hoa hau, thu thap tu nguoinoitieng.tv).
https://huggingface.co/datasets/fptudsc/face-celeb-vietnamese

Dataset nay KHONG duoc tai/xu ly truc tiep tren server production. Thay
vao do, chay `scripts/build_celebrity_index.py` MOT LAN, OFFLINE (tren
may co Internet) de:
  - Tai dataset ve
  - Phat hien khuon mat + trich xuat embedding cho tung anh
  - Gom trung binh embedding theo tung nguoi
  - Luu ket qua vao file gon nhe `app/data/celebrity_embeddings.npz`
    (chi vai MB, khong chua anh goc) roi commit file nay vao source code.
Server production chi can load file .npz nay len, khong can tai dataset
hay ket noi HuggingFace khi chay.

## Gioi han / luu y quan trong
- Day la buoc loc TU DONG dua tren du lieu cong khai, co the co sai sot
  (nham lan giua nhung nguoi co khuon mat giong nhau, khong nhan dien
  duoc nguoi noi tieng ngoai danh sach 224 nguoi, hoac bo sot anh chat
  luong thap/goc nghieng/bi che). Nen:
    + Cho phep nguoi dung gui khieu nai (`support/create`) neu bi tu choi
      nham, de admin xem xet thu cong.
    + Dinh ky mo rong dataset / danh sach nguoi noi tieng can chan.
- Neu file `celebrity_embeddings.npz` CHUA duoc tao (chua chay script
  build), ham `check_celebrity()` se luon tra ve is_celebrity=False
  (fail-open) va ghi log canh bao, thay vi lam loi ca API upload anh.

## Cach dung
    from app.services.celebrity_detection import check_celebrity
    result = check_celebrity(request.FILES['image'])
    # result = {'is_celebrity': bool, 'celebrity_name': str | None, 'score': float}
"""
import os
import logging
from functools import lru_cache

import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Nguong do tuong dong (cosine similarity, tu 0.0 - 1.0) de coi la "trung
# khop" voi mot nguoi noi tieng. Vector ArcFace da L2-normalize nen cosine
# similarity = dot product. Co the override trong settings.py / .env bang
# CELEBRITY_THRESHOLD. Nguong khuyen nghi ban dau: 0.42 (uu tien giam sai
# sot "tu choi nham" hon la bo sot - co the tang len 0.5+ neu qua nhieu
# false positive sau khi test thuc te).
CELEBRITY_THRESHOLD = getattr(settings, 'CELEBRITY_THRESHOLD', 0.42)

# Duong dan file du lieu embedding nguoi noi tieng (tao boi
# scripts/build_celebrity_index.py). Mac dinh: app/data/celebrity_embeddings.npz
CELEBRITY_EMBEDDINGS_PATH = getattr(
    settings, 'CELEBRITY_EMBEDDINGS_PATH',
    os.path.join(settings.BASE_DIR, 'app', 'data', 'celebrity_embeddings.npz')
)

# Model insightface dung de phat hien khuon mat + trich embedding.
# 'buffalo_l' (mac dinh, chinh xac hon, ~326MB) hoac 'buffalo_s' (nhe hon,
# nhanh hon, phu hop may yeu / CPU cham).
CELEBRITY_MODEL_NAME = getattr(settings, 'CELEBRITY_MODEL_NAME', 'buffalo_l')

# Bo qua khuon mat qua nho (pixel) trong anh - thuong la nguoi o xa/phia
# sau, khong phai chu the chinh cua anh chan dung, de tranh nham lan.
CELEBRITY_MIN_FACE_SIZE = getattr(settings, 'CELEBRITY_MIN_FACE_SIZE', 40)


@lru_cache(maxsize=1)
def _get_face_app():
    """
    Khoi tao model phat hien khuon mat + trich embedding (insightface).
    Chi load 1 lan duy nhat cho ca process (worker), tranh load lai model
    tren moi request (rat cham neu load lai moi lan). Lan goi dau tien se
    tu dong tai model tu insightface model zoo va cache lai (~/.insightface).
    """
    from insightface.app import FaceAnalysis
    # allowed_modules=['detection', 'recognition']: chi can 2 model nay de
    # phat hien khuon mat + trich embedding so sanh. Bo qua landmark_3d_68/
    # landmark_2d_106/genderage (buffalo_l mac dinh load ca 5 model) giup
    # giam dang ke thoi gian xu ly moi anh (~2-3 lan) ma khong doi ket qua.
    face_app = FaceAnalysis(
        name=CELEBRITY_MODEL_NAME,
        providers=['CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition'],
    )
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return face_app


@lru_cache(maxsize=1)
def _get_celebrity_index():
    """
    Load danh sach vector dac trung cua nguoi noi tieng tu file .npz.

    Returns:
        (names, embeddings): names la np.ndarray kieu str shape (N,),
        embeddings la np.ndarray float32 shape (N, 512) - da L2-normalize.
        Tra ve (None, None) neu file chua ton tai.
    """
    if not os.path.exists(CELEBRITY_EMBEDDINGS_PATH):
        logger.warning(
            "Khong tim thay file du lieu nguoi noi tieng tai '%s'. Chuc nang "
            "nhan dien nguoi noi tieng (yeu cau 1.3.1) dang bi TAT (luon tra "
            "ve is_celebrity=False). Chay `python scripts/build_celebrity_index.py` "
            "de tao file nay truoc.",
            CELEBRITY_EMBEDDINGS_PATH,
        )
        return None, None
    data = np.load(CELEBRITY_EMBEDDINGS_PATH, allow_pickle=False)
    return data['names'], data['embeddings']


def _extract_largest_face_embedding(image_bgr: np.ndarray):
    """
    Phat hien tat ca khuon mat trong anh (mang numpy BGR), tra ve embedding
    (512-d, da L2-normalize) cua khuon mat co dien tich lon nhat, hoac None
    neu khong phat hien duoc khuon mat nao du lon.
    """
    face_app = _get_face_app()
    faces = face_app.get(image_bgr)
    if not faces:
        return None

    def _area(f):
        return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])

    face = max(faces, key=_area)
    w = face.bbox[2] - face.bbox[0]
    h = face.bbox[3] - face.bbox[1]
    if w < CELEBRITY_MIN_FACE_SIZE or h < CELEBRITY_MIN_FACE_SIZE:
        return None
    return face.normed_embedding


def check_celebrity(image_file) -> dict:
    """
    Kiem tra 1 anh co chua khuon mat trung voi nguoi noi tieng Viet Nam
    trong danh sach da biet hay khong.

    Args:
        image_file: Django UploadedFile (request.FILES['image']),
                    duong dan file (str), hoac PIL.Image co san.

    Returns:
        dict: {
            'is_celebrity': bool,      # True neu do tuong dong >= CELEBRITY_THRESHOLD
            'celebrity_name': str|None,# ten nguoi noi tieng trung khop cao nhat (neu match)
            'score': float,            # do tuong dong cao nhat tim duoc, 0.0 - 1.0
        }
    """
    from PIL import Image

    names, embeddings = _get_celebrity_index()
    if names is None:
        # Chua co du lieu nguoi noi tieng -> khong the kiem tra, khong chan
        # nham nguoi dung (fail-open).
        return {'is_celebrity': False, 'celebrity_name': None, 'score': 0.0}

    if hasattr(image_file, 'read'):
        image_file.seek(0)
        img = Image.open(image_file).convert('RGB')
        image_file.seek(0)  # tra con tro ve dau de code sau con doc lai duoc file
    elif isinstance(image_file, Image.Image):
        img = image_file.convert('RGB')
    else:
        img = Image.open(image_file).convert('RGB')

    # insightface/opencv lam viec voi anh dang BGR
    image_bgr = np.array(img)[:, :, ::-1].copy()
    query_embedding = _extract_largest_face_embedding(image_bgr)

    if query_embedding is None:
        # Khong phat hien duoc khuon mat ro rang trong anh -> khong co co
        # so de so sanh, coi nhu khong trung khop.
        return {'is_celebrity': False, 'celebrity_name': None, 'score': 0.0}

    # Ca 2 phia deu da L2-normalize -> cosine similarity = dot product
    similarities = embeddings @ query_embedding
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    is_match = best_score >= CELEBRITY_THRESHOLD

    return {
        'is_celebrity': is_match,
        'celebrity_name': str(names[best_idx]) if is_match else None,
        'score': round(best_score, 4),
    }