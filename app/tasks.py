"""
app/tasks.py

Cac tac vu Celery chay bat dong bo (nen), tranh timeout request HTTP khi
model AI xu ly lau. Chay worker bang:
    celery -A fitting_app worker --loglevel=info
"""
import io
import logging
import time
import uuid

from celery import shared_task
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def generate_tryon_task(self, generated_image_id):
    """
    Sinh anh thu do cho 1 ban ghi Generated_Image (yeu cau 1.1.6):
      1. Doc anh chan dung (uu tien ban da tach nen neu co) + anh san pham.
      2. Goi tryon_service.generate_tryon_image() de "mac" trang phuc vao.
      3. Neu co chon Frame -> ghep them vao khung nen bang compose_with_frame().
      4. Luu ket qua vao result_image, cap nhat status/processing_time.
    """
    from PIL import Image
    from app.models import Generated_Image

    generated = Generated_Image.objects.filter(id=generated_image_id).first()
    if not generated:
        logger.error(f"generate_tryon_task: Generated_Image id={generated_image_id} khong ton tai")
        return

    generated.status = 'processing'
    generated.save(update_fields=['status'])

    start_time = time.time()
    try:
        from app.services.tryon_service import generate_tryon_image, compose_with_frame

        if not generated.portrait or not generated.product:
            raise ValueError('Thieu portrait hoac product de sinh anh thu do')

        # LUON dung original_image (KHONG dung processed_image da tach nen)
        # cho buoc try-on:
        #   1. processed_image la anh RGBA nen trong suot - qua
        #      generate_tryon_image() se bi convert("RGB") lam lo vien/
        #      quang DEN quanh nguoi (da gap thuc te, xem ghi chu trong
        #      tryon_service._flatten_to_rgb()).
        #   2. Nen den bat thuong lam mediapipe Pose kho dinh vi khop
        #      xuong chinh xac -> mask vung ao/quan co the sai/rong ->
        #      ket qua "khong doi ao" nhu da gap.
        #   3. Khong can thiet: compose_with_frame() o cuoi PIPELINE nay
        #      da tu tach nen lai tu dau roi, tach nen som o day chi thua
        #      va gay hai.
        portrait_img = Image.open(generated.portrait.original_image.path)
        garment_img = Image.open(generated.product.image.path)
        category_type = generated.product.category.category_type if generated.product.category else 'shirt'

        result = generate_tryon_image(portrait_img, garment_img, category_type)

        if generated.frame and generated.frame.image:
            frame_img = Image.open(generated.frame.image.path)
            result = compose_with_frame(result, frame_img)

        buffer = io.BytesIO()
        result.convert('RGB').save(buffer, format='JPEG', quality=92)
        buffer.seek(0)
        filename = f"tryon_{uuid.uuid4().hex[:12]}.jpg"

        generated.result_image.save(filename, ContentFile(buffer.read()), save=False)
        generated.status = 'completed'
        generated.processing_time = round(time.time() - start_time, 2)
        generated.save()
        logger.info(f"generate_tryon_task: hoan thanh Generated_Image id={generated_image_id} trong {generated.processing_time}s")

    except Exception as e:
        logger.exception(f"generate_tryon_task: loi khi xu ly Generated_Image id={generated_image_id}: {e}")
        generated.status = 'failed'
        generated.processing_time = round(time.time() - start_time, 2)
        generated.save()
        # Khong retry vo han - tac vu AI hay loi do anh dau vao (khong phat
        # hien duoc pose, anh loi...), retry lai se ra cung 1 loi.
        raise