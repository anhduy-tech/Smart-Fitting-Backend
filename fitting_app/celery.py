"""
fitting_app/celery.py

Khoi tao Celery app - dung Redis lam broker (xem CELERY_BROKER_URL trong settings.py).
Dung de chay bat dong bo cac tac vu AI nang (kiem duyet NSFW, tach nguoi + inpainting)
thay vi xu ly truc tiep trong request HTTP - tranh timeout khi model AI chay lau.

Cach chay worker (sau khi da cai `celery` va co Redis server dang chay):
    celery -A fitting_app worker --loglevel=info

Windows can them --pool=solo:
    celery -A fitting_app worker --loglevel=info --pool=solo
"""
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fitting_app.settings')

app = Celery('fitting_app')

# Doc cac cau hinh CELERY_* tu settings.py cua Django (da khai bao trong settings.py)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Tu dong tim file tasks.py trong tung app (vi du: app/tasks.py)
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')