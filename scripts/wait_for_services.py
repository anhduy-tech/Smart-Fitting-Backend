"""
scripts/wait_for_services.py

Kiem tra Database va Redis da san sang va ket noi duoc chua truoc khi
chay backend. Dung boi run.sh.

Cach dung:
    python scripts/wait_for_services.py --timeout 30
"""
import os
import sys
import time
import argparse

# Script nam trong thu muc scripts/, can them thu muc goc project vao
# sys.path thi moi `import fitting_app.settings` (qua django.setup()) duoc,
# vi khi chay `python scripts/wait_for_services.py` (khong phai `-m`),
# Python chi tu dong them thu muc CHUA file script (scripts/) vao sys.path,
# khong phai thu muc goc project.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fitting_app.settings')


def _setup_django():
    """
    Khoi tao Django app registry - BAT BUOC phai goi truoc khi dung
    bat cu thu gi tu django.db, django.conf.settings, v.v. Neu khong,
    se bi loi django.core.exceptions.AppRegistryNotReady.
    """
    import django
    django.setup()


def wait_for_db(timeout=30):
    from django.db import connections
    from django.db.utils import OperationalError

    start = time.time()
    while time.time() - start < timeout:
        try:
            for conn in connections.all():
                conn.cursor()
            print("Database is ready")
            return True
        except OperationalError:
            print("Waiting for database...")
            time.sleep(2)
        except Exception as e:
            print(f"Database error: {e}")
            time.sleep(2)
    print("Database timeout")
    return False


def wait_for_redis(timeout=30):
    from django.conf import settings

    start = time.time()
    redis_url = getattr(settings, 'REDIS_URL', None) or os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')

    while time.time() - start < timeout:
        try:
            import redis
            r = redis.from_url(redis_url)
            if r.ping():
                print("Redis is ready")
                return True
        except Exception as e:
            print(f"Waiting for Redis... ({e})")
            time.sleep(2)
    print("Redis timeout")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()

    _setup_django()

    if wait_for_db(args.timeout) and wait_for_redis(args.timeout):
        sys.exit(0)
    else:
        sys.exit(1)