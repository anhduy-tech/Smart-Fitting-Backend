"""
Django settings for fitting_app project.

Toan bo cau hinh nhay cam (SECRET_KEY, DATABASE, REDIS, ...) duoc doc tu
bien moi truong (file .env o thu muc goc). Xem file .env.example de biet
day du danh sach bien can khai bao.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/
"""
import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Doc file .env o thu muc goc project (khong commit file .env len git)
load_dotenv(BASE_DIR / '.env')


# =============================================================================
# Helper - parse gia tri tu bien moi truong
# =============================================================================
def env(key, default=None):
    return os.environ.get(key, default)


def env_bool(key, default=False):
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


def env_list(key, default=''):
    val = os.environ.get(key, default)
    return [item.strip() for item in val.split(',') if item.strip()]


def env_int(key, default=0):
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def env_float(key, default=0.0):
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# =============================================================================
# Bao mat co ban
# =============================================================================
# SECURITY WARNING: bat buoc phai dat SECRET_KEY rieng trong .env khi len production!
SECRET_KEY = env('SECRET_KEY', 'django-insecure-CHANGE-ME-IN-PRODUCTION')

# SECURITY WARNING: khong duoc bat DEBUG=True khi len production!
DEBUG = env_bool('DEBUG', default=True)

# Vi du .env: ALLOWED_HOSTS=localhost,127.0.0.1,api.smartfitting.vn
ALLOWED_HOSTS = env_list('ALLOWED_HOSTS', 'localhost,127.0.0.1')


# =============================================================================
# Application definition
# =============================================================================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third party
    'rest_framework',
    'rest_framework_simplejwt.token_blacklist',  # can thiet de RefreshToken.blacklist() hoat dong (BLACKLIST_AFTER_ROTATION=True ben duoi). Nho chay migrate sau khi them app nay.
    'corsheaders',
    'django_filters',
    'cacheops',
    # Local apps
    'app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'fitting_app.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'fitting_app.wsgi.application'


# =============================================================================
# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases
#
# Mac dinh dung SQLite cho dev (khong can cau hinh gi them).
# De dung PostgreSQL cho production, khai bao trong .env:
#   DB_ENGINE=django.db.backends.postgresql
#   DB_NAME=smart_fitting
#   DB_USER=postgres
#   DB_PASSWORD=postgres
#   DB_HOST=localhost
#   DB_PORT=5432
# =============================================================================
DB_ENGINE = env('DB_ENGINE', 'django.db.backends.sqlite3')

if DB_ENGINE == 'django.db.backends.sqlite3':
    DATABASES = {
        'default': {
            'ENGINE': DB_ENGINE,
            'NAME': BASE_DIR / env('DB_NAME', 'db.sqlite3'),
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': DB_ENGINE,
            'NAME': env('DB_NAME', 'smart_fitting'),
            'USER': env('DB_USER', 'postgres'),
            'PASSWORD': env('DB_PASSWORD', ''),
            'HOST': env('DB_HOST', 'localhost'),
            'PORT': env('DB_PORT', '5432'),
        }
    }


# =============================================================================
# Password validation
# =============================================================================
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# =============================================================================
# Internationalization
# =============================================================================
LANGUAGE_CODE = env('LANGUAGE_CODE', 'vi-vn')
TIME_ZONE = env('TIME_ZONE', 'Asia/Ho_Chi_Minh')
USE_I18N = True
USE_TZ = True


# =============================================================================
# Static / Media files
# =============================================================================
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'app/static'),
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# =============================================================================
# REST Framework
# =============================================================================
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
    ],
    'DATETIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'DATE_FORMAT': '%Y-%m-%d',
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=env_int('JWT_ACCESS_TOKEN_LIFETIME_MINUTES', 60)),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=env_int('JWT_REFRESH_TOKEN_LIFETIME_DAYS', 1)),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
}


# =============================================================================
# CORS settings
# =============================================================================
CORS_ALLOW_ALL_ORIGINS = env_bool('CORS_ALLOW_ALL_ORIGINS', default=True)
CORS_ALLOWED_ORIGINS = env_list('CORS_ALLOWED_ORIGINS', '')
CORS_ALLOW_CREDENTIALS = True


# =============================================================================
# Custom user model
# =============================================================================
AUTH_USER_MODEL = 'app.User'


# =============================================================================
# OTP settings
# =============================================================================
OTP_MAX_PER_DAY = env_int('OTP_MAX_PER_DAY', 5)
OTP_EXPIRY_MINUTES = env_int('OTP_EXPIRY_MINUTES', 5)


# =============================================================================
# Upload settings
# =============================================================================
MAX_UPLOAD_SIZE = env_int('MAX_UPLOAD_SIZE', 10 * 1024 * 1024)  # 10MB mac dinh
ALLOWED_IMAGE_EXTENSIONS = env_list('ALLOWED_IMAGE_EXTENSIONS', 'jpg,jpeg,png,webp')


# =============================================================================
# NSFW moderation settings (app/services/moderation.py)
# =============================================================================
NSFW_MODEL_NAME = env('NSFW_MODEL_NAME', 'Falconsai/nsfw_image_detection')
NSFW_THRESHOLD = env_float('NSFW_THRESHOLD', 0.85)


# =============================================================================
# Celebrity detection settings (app/services/celebrity_detection.py)
#
# Nhan dien anh chua khuon mat nguoi noi tieng Viet Nam (yeu cau 1.3.1).
# Du lieu tham chieu (vector dac trung cua ~224 nguoi noi tieng) duoc tao
# san boi scripts/build_celebrity_index.py, luu tai
# CELEBRITY_EMBEDDINGS_PATH - KHONG can Internet luc chay server.
# =============================================================================
CELEBRITY_MODEL_NAME = env('CELEBRITY_MODEL_NAME', 'buffalo_l')
CELEBRITY_THRESHOLD = env_float('CELEBRITY_THRESHOLD', 0.42)
CELEBRITY_MIN_FACE_SIZE = env_int('CELEBRITY_MIN_FACE_SIZE', 40)
CELEBRITY_EMBEDDINGS_PATH = env(
    'CELEBRITY_EMBEDDINGS_PATH',
    str(BASE_DIR / 'app' / 'data' / 'celebrity_embeddings.npz')
)


# =============================================================================
# Redis
#
# Du an CO dung Redis, cho 2 muc dich:
#   1. Cache query (django-cacheops) - da khai bao trong requirements.txt
#      nhung truoc day CHUA duoc cau hinh, gio duoc noi day o day.
#   2. Hang doi tac vu bat dong bo (Celery) - dung cho cac tac vu AI nang
#      (kiem duyet NSFW, tach nguoi + inpainting o app/services/) de khong
#      lam block request HTTP.
#
# Neu chua co Redis server, cai nhanh bang Docker:
#   docker run -d --name redis -p 6379:6379 redis:7-alpine
# =============================================================================
REDIS_URL = env('REDIS_URL', 'redis://127.0.0.1:6379/0')
CACHEOPS_REDIS = env('CACHEOPS_REDIS_URL', 'redis://127.0.0.1:6379/1')

CACHEOPS_ENABLED = env_bool('CACHEOPS_ENABLED', default=True)
CACHEOPS_DEGRADE_ON_FAILURE = True  # Redis chet -> app van chay binh thuong, chi mat cache

CACHEOPS = {
    # Danh muc / san pham / frame it thay doi, doc nhieu -> cache lau
    'app.Product': {'ops': 'get', 'timeout': 60 * 15},
    'app.Product_Category': {'ops': 'get', 'timeout': 60 * 60},
    'app.Frame': {'ops': 'get', 'timeout': 60 * 30},
    'app.Frame_Category': {'ops': 'get', 'timeout': 60 * 60},
    'app.Slide': {'ops': 'get', 'timeout': 60 * 30},
    'app.Setting': {'ops': 'get', 'timeout': 60 * 5},
} if CACHEOPS_ENABLED else {}


# =============================================================================
# Celery (hang doi tac vu bat dong bo, dung Redis lam broker)
# Dinh nghia app Celery tai fitting_app/celery.py
# =============================================================================
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TIME_LIMIT = 5 * 60  # tac vu AI toi da 5 phut truoc khi bi kill

# =============================================================================
# Celery - phan tach QUEUE cho task AI nang (GPU) va task nhe (I/O, thong bao)
#
# Task AI nang (SD Inpainting/try-on...) PHAI chay tren worker rieng voi
# --pool=solo --concurrency=1 (xem run.sh), vi nhieu process cung luc se
# tranh nhau VRAM gay CUDA out of memory (da gap thuc te). Task nhe (gui
# OTP, notification...) khong dung GPU nen chay duoc tren worker prefork
# binh thuong voi concurrency cao de xu ly duoc nhieu viec cung luc.
#
# Neu sau nay them task AI nang moi (vd background-remove chay bat dong bo),
# nho khai bao them vao CELERY_TASK_ROUTES ben duoi de vao dung queue
# 'gpu_tasks', neu khong se roi vao 'default' va co the bi chay chung voi
# task nhe tren worker prefork concurrency cao -> lai gap OOM y het cu.
# =============================================================================
CELERY_TASK_DEFAULT_QUEUE = 'default'
CELERY_TASK_ROUTES = {
    'app.tasks.generate_tryon_task': {'queue': 'gpu_tasks'},
}