import os
import django
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fitting_app.settings')  # ← thay tên project cho đúng

django.setup()

from app.models import Product

print("=== Tìm có is_active=True ===")
print(Product.objects.filter(id=2, is_active=True).first())

print("\n=== Tìm tất cả id=2 ===")
qs = Product.objects.filter(id=2).values('id', 'is_active', 'name')
print(list(qs))