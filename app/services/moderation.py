"""
app/services/moderation.py

Kiem duyet noi dung anh - phat hien anh khoa than / sex (NSFW).
Dung cho yeu cau 1.3.2: "Khi nguoi su dung tai anh len. He thong tu dong
phat hien anh Sex, neu phat hien se khong hien thi anh va tu dong xoa anh".

Model: Falconsai/nsfw_image_detection (Vision Transformer, HuggingFace).
- Nhe, chay tot tren CPU, khong can GPU.
- Chi phan loai 2 nhan: 'normal' / 'nsfw'.
- Lan goi dau tien se tai model (~350MB) tu HuggingFace Hub va cache lai,
  cac lan sau load tu cache local rat nhanh.

Cach dung:
    from app.services.moderation import check_nsfw
    result = check_nsfw(request.FILES['image'])
    # result = {'is_nsfw': bool, 'score': float, 'label': str}
"""
from functools import lru_cache
from django.conf import settings

# Nguong tin cay de coi la NSFW. Anh co score >= nguong nay se bi tu dong xoa.
# Co the override trong settings.py bang NSFW_THRESHOLD.
NSFW_THRESHOLD = getattr(settings, 'NSFW_THRESHOLD', 0.85)
NSFW_MODEL_NAME = getattr(settings, 'NSFW_MODEL_NAME', 'Falconsai/nsfw_image_detection')


@lru_cache(maxsize=1)
def _get_classifier():
    """
    Load model 1 lan duy nhat cho ca process (worker), tranh load lai
    tren moi request (rat cham neu load lai moi lan).
    """
    from transformers import pipeline
    return pipeline('image-classification', model=NSFW_MODEL_NAME)


def check_nsfw(image_file) -> dict:
    """
    Kiem tra 1 anh co phai noi dung nhay cam (sex/nudity) hay khong.

    Args:
        image_file: co the la Django UploadedFile (request.FILES['image']),
                    duong dan file (str), hoac PIL.Image co san.

    Returns:
        dict: {
            'is_nsfw': bool,   # True neu vuot nguong NSFW_THRESHOLD
            'score': float,    # xac suat NSFW, tu 0.0 - 1.0
            'label': str,      # 'nsfw' hoac 'normal'
        }
    """
    from PIL import Image

    if hasattr(image_file, 'read'):
        # Django UploadedFile / file-like object
        image_file.seek(0)
        img = Image.open(image_file).convert('RGB')
        image_file.seek(0)  # tra con tro ve dau de code sau con doc lai duoc file
    elif isinstance(image_file, Image.Image):
        img = image_file.convert('RGB')
    else:
        img = Image.open(image_file).convert('RGB')

    classifier = _get_classifier()
    results = classifier(img)  # vd: [{'label': 'nsfw', 'score': 0.97}, {'label': 'normal', 'score': 0.03}]

    scores = {r['label'].lower(): r['score'] for r in results}
    nsfw_score = scores.get('nsfw', 0.0)
    is_nsfw = nsfw_score >= NSFW_THRESHOLD

    return {
        'is_nsfw': is_nsfw,
        'score': round(float(nsfw_score), 4),
        'label': 'nsfw' if is_nsfw else 'normal',
    }