"""
scripts/test_celebrity_detection.py

Script test nhanh 2 tinh nang kiem duyet anh (nhan dien nguoi noi tieng +
NSFW) TRUC TIEP tren service layer - KHONG can chay server Django, KHONG
can dang ky/dang nhap/token. Dung de kiem tra nhanh xem model co hoat
dong dung khong va tinh chinh nguong (threshold) truoc khi dung API that.

Cach dung (chay tu thu muc goc project, noi co file manage.py):
    python scripts/test_celebrity_detection.py duong/dan/anh1.jpg
    python scripts/test_celebrity_detection.py anh1.jpg anh2.png anh3.jpg

Vi du ket qua in ra:
    === anh1.jpg ===
      Nguoi noi tieng: True (ten: ca sĩ Bigdaddy, score: 0.5123)
      NSFW: False (score: 0.0021)
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fitting_app.settings')

import django  # noqa: E402
django.setup()

from app.services.celebrity_detection import check_celebrity  # noqa: E402
from app.services.moderation import check_nsfw  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print('Cach dung: python scripts/test_celebrity_detection.py anh1.jpg [anh2.jpg ...]')
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f'\n=== {path} ===')
        if not os.path.exists(path):
            print('  [!] Khong tim thay file nay.')
            continue

        try:
            with open(path, 'rb') as f:
                celeb_result = check_celebrity(f)
            with open(path, 'rb') as f:
                nsfw_result = check_nsfw(f)
        except Exception as e:
            print(f'  [!] Loi khi xu ly anh: {e}')
            continue

        mark_celeb = '⚠️ ' if celeb_result['is_celebrity'] else '✓ '
        mark_nsfw = '⚠️ ' if nsfw_result['is_nsfw'] else '✓ '

        print(f"  {mark_celeb}Nguoi noi tieng: {celeb_result['is_celebrity']} "
              f"(ten: {celeb_result['celebrity_name']}, score: {celeb_result['score']})")
        print(f"  {mark_nsfw}NSFW: {nsfw_result['is_nsfw']} (score: {nsfw_result['score']})")


if __name__ == '__main__':
    main()