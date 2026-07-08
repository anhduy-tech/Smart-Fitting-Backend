# Smart Fitting App - Backend (Django REST API)

Backend cho ứng dụng thử đồ ảo Smart Fitting (đăng ký/đăng nhập bằng SĐT, quản lý sản phẩm, giỏ hàng, đơn hàng, thử đồ AI, kiểm duyệt ảnh NSFW, tách nền tạo Frame bằng AI...).

---

## 1. Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| **Python** | **3.10.x** (khuyến nghị 3.10 hoặc 3.11 — đã kiểm tra tương thích với toàn bộ thư viện trong `requirements.txt`, bao gồm `torch`, `transformers`, `rembg`). Không dùng Python 3.13 trở lên vì một số thư viện AI chưa có bản build sẵn. |
| **Hệ điều hành** | Linux / macOS / Windows đều chạy được. Khuyến nghị Linux (Ubuntu 22.04+) cho môi trường production. |
| **Database** | PostgreSQL (khuyến nghị cho production) hoặc SQLite (mặc định, dùng cho dev/test). |
| **RAM** | Tối thiểu 4GB. Khuyến nghị 8GB+ vì các model AI (NSFW detection, tách nền, inpainting) load vào RAM khi chạy trên CPU. |
| **Dung lượng ổ đĩa** | Tối thiểu ~3GB trống để tải các model AI lần đầu (PyTorch + các model NSFW/rembg/LaMa). |
| **GPU** | Không bắt buộc. Có GPU (NVIDIA + CUDA) sẽ giúp chức năng tách nền/inpainting chạy nhanh hơn đáng kể, nhưng code vẫn chạy được trên CPU. |

---

## 2. Cài đặt

### Cách nhanh — dùng script tự động

```bash
# Linux / macOS
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

Script sẽ tự tạo `venv`, cài `requirements.txt`, tạo `.env` từ `.env.example`, và chạy `migrate`. Sau đó chỉ cần kích hoạt venv, tạo superuser và chạy server (xem bước cuối bên dưới).

### Cách thủ công — làm từng bước

### Bước 1 — Clone / giải nén source code

```bash
cd smart-fitting-app
```

### Bước 2 — Tạo và kích hoạt môi trường ảo (virtual environment)

```bash
# Kiểm tra phiên bản Python trước khi tạo venv
python3 --version   # phải là 3.10.x hoặc 3.11.x

# Tạo venv
python3 -m venv venv

# Kích hoạt venv
# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
```

> **Lưu ý:** mọi lệnh `pip`/`python` bên dưới đều phải chạy **sau khi đã kích hoạt venv** — dấu hiệu nhận biết là dòng lệnh có tiền tố `(venv)`. Mỗi khi mở terminal mới để làm việc với project, phải `source venv/bin/activate` (hoặc `venv\Scripts\activate.bat` trên Windows) lại trước.

### Bước 3 — Cài đặt thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý:** Lần cài đặt đầu tiên sẽ khá lâu (vài phút) vì `torch` khá nặng (~700MB). Nếu chỉ chạy trên CPU (không có GPU), có thể cài bản `torch` CPU-only nhẹ hơn:
> ```bash
> pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Bước 4 — Cấu hình biến môi trường

Copy file mẫu và chỉnh sửa (nếu chạy `setup.sh`/`setup.bat` thì bước này đã tự làm rồi):

```bash
cp .env.example .env
```

Mở `.env` và chỉnh các giá trị cho phù hợp — xem đầy đủ danh sách biến và giải thích trong `.env.example`. Quan trọng nhất:

```env
SECRET_KEY=doi-thanh-chuoi-bi-mat-that-su-khi-len-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Bo trong de dung SQLite (dev). Dien vao neu dung PostgreSQL (production).
# DB_ENGINE=django.db.backends.postgresql
# DB_NAME=smart_fitting
# DB_USER=postgres
# DB_PASSWORD=postgres
# DB_HOST=localhost
# DB_PORT=5432

REDIS_URL=redis://127.0.0.1:6379/0
CACHEOPS_REDIS_URL=redis://127.0.0.1:6379/1

NSFW_THRESHOLD=0.85
CELEBRITY_THRESHOLD=0.42
```

`settings.py` đọc toàn bộ cấu hình từ `.env` này qua `python-dotenv` — **không cần sửa trực tiếp `settings.py`** cho việc đổi môi trường dev/production nữa.

### Bước 5 — Khởi tạo database

```bash
python manage.py makemigrations
python manage.py migrate
```

### Bước 6 — Tạo tài khoản quản trị (admin)

```bash
python manage.py createsuperuser
# Nhập số điện thoại (thay cho username) và mật khẩu khi được hỏi
```

### Bước 7 — Thu thập static files (chỉ cần khi chạy production với whitenoise)

```bash
python manage.py collectstatic --noinput
```

---

## 3. Chạy dự án

### Cách khuyến nghị — dùng `run.sh` (tự kiểm tra Database + Redis trước khi chạy)

```bash
chmod +x run.sh   # chỉ cần làm 1 lần

./run.sh                  # chạy dev server (runserver) ở 0.0.0.0:8000
./run.sh --prod           # chạy bằng gunicorn (production)
./run.sh --port 8080      # đổi port (mặc định 8000)
./run.sh --skip-migrate   # bỏ qua bước migrate tự động
./run.sh --timeout 60     # đợi tối đa 60s cho DB/Redis sẵn sàng (mặc định 30s)
```

`run.sh` sẽ tự động: kích hoạt venv → kiểm tra đã cài dependencies chưa → chờ Database và Redis sẵn sàng (qua `scripts/wait_for_services.py`, dùng đúng config thật trong `settings.py`) → chạy `migrate` → chạy server. Nếu Database/Redis chưa sẵn sàng trong thời gian `--timeout`, script sẽ dừng lại thay vì chạy backend với dữ liệu/cache không kết nối được.

### Cách thủ công (không qua run.sh)

```bash
python manage.py runserver 0.0.0.0:8000
```

Truy cập:
- API: `http://localhost:8000/api/`
- Trang quản trị Django: `http://localhost:8000/admin/`

### Môi trường production (gợi ý, dùng gunicorn)

```bash
./run.sh --prod --port 8000
# tương đương chạy tay:
gunicorn fitting_app.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120
```

> `--timeout 120` (của gunicorn) được khuyến nghị vì các API xử lý ảnh AI (kiểm duyệt NSFW, tách nền + inpainting) có thể mất vài giây đến vài chục giây trên CPU, dễ bị timeout mặc định (30s) của gunicorn.

---

## 4. Ghi chú riêng cho các chức năng AI (quan trọng)

Dự án dùng các model AI mã nguồn mở, chạy **local trên server**, không gọi API trả phí bên ngoài:

| Chức năng | Thư viện / Model | Ghi chú |
|---|---|---|
| Kiểm duyệt ảnh NSFW (`app/services/moderation.py`) | `transformers` + model `Falconsai/nsfw_image_detection` | Model ~350MB, tự tải về từ HuggingFace Hub trong lần gọi API đầu tiên. |
| Nhận diện người nổi tiếng VN (`app/services/celebrity_detection.py`) | `insightface` (ArcFace, model `buffalo_l`) + file dữ liệu `app/data/celebrity_embeddings.npz` | Model nhận diện khuôn mặt ~326MB, tự tải lần đầu chạy. File `.npz` (vài MB, ~224 người nổi tiếng) được build **offline, một lần** — xem hướng dẫn ngay dưới đây. |
| Tách người khỏi ảnh (`app/services/background_inpaint.py`) | `rembg` (model `u2net_human_seg`) | Model ~176MB, tự tải về lần đầu chạy. |
| AI vẽ lại nền sau khi xoá người | `simple-lama-inpainting` (LaMa) | Model ~200MB, tự tải về lần đầu chạy. |

**Yêu cầu bắt buộc:** server phải có kết nối Internet ra ngoài (ít nhất trong lần đầu chạy các API liên quan) để tải model. Sau khi tải xong, model được cache lại tại thư mục cache mặc định của HuggingFace/`rembg`/`insightface` (thường là `~/.cache/huggingface`, `~/.u2net`, `~/.insightface`), các lần sau không cần tải lại.

Nếu server chạy trong môi trường không có Internet (air-gapped), cần tải sẵn các model này từ máy có mạng rồi copy thư mục cache tương ứng vào server.

**Khuyến nghị hiệu năng:** vì các tác vụ AI (đặc biệt tách nền + inpainting) tốn thời gian, nếu traffic upload ảnh lớn nên tách các tác vụ này ra chạy nền bằng **Celery + Redis** thay vì xử lý đồng bộ ngay trong request HTTP (xem mục 5 bên dưới — dự án đã cấu hình sẵn Celery + Redis).

### 4.1. Thiết lập tính năng nhận diện người nổi tiếng (yêu cầu 1.3.1)

Khác với NSFW (model tự tải về khi gọi API lần đầu), tính năng này cần một **bước chuẩn bị dữ liệu offline một lần** trước khi deploy, vì cần một danh sách "vector đặc trưng khuôn mặt" của người nổi tiếng để so sánh.

**Nguồn dữ liệu:** dataset [`fptudsc/face-celeb-vietnamese`](https://huggingface.co/datasets/fptudsc/face-celeb-vietnamese) trên HuggingFace (giấy phép Apache-2.0, ~8.557 ảnh của 224 người nổi tiếng Việt Nam — ca sĩ, diễn viên, hoa hậu).

**Cách build (chạy trên máy có Internet, KHÔNG cần chạy trên server production):**

```bash
# Cài thêm các thư viện chỉ dùng cho bước build này (không cần trong requirements.txt của server)
pip install datasets huggingface_hub insightface onnxruntime pillow numpy

python scripts/build_celebrity_index.py
```

Script sẽ tải dataset, phát hiện khuôn mặt + trích vector đặc trưng cho từng ảnh, gộp trung bình theo từng người, rồi lưu kết quả (chỉ vài MB) vào:

```
app/data/celebrity_embeddings.npz
```

Chạy xong, **commit file `.npz` này vào source code** — server production chỉ cần đọc file có sẵn, không cần tải dataset hay gọi HuggingFace lúc chạy thật. Nếu chưa build file này, tính năng sẽ tự tắt an toàn (luôn coi ảnh là "không phải người nổi tiếng", không chặn nhầm người dùng, có log cảnh báo).

**Tinh chỉnh độ nhạy:** biến `CELEBRITY_THRESHOLD` trong `.env` (mặc định `0.42`, thang 0.0–1.0). Tăng lên nếu bị từ chối nhầm quá nhiều (false positive), giảm xuống nếu bỏ sót quá nhiều. Nên test trên tập ảnh thật của app trước khi chốt ngưỡng cho production.

**Lưu ý:** đây là bộ lọc tự động dựa trên dữ liệu công khai, không tuyệt đối chính xác — nên giữ chức năng gửi khiếu nại (`support/create`) để người dùng phản ánh nếu bị từ chối nhầm.

---

## 5. Redis & Celery

Dự án **có dùng Redis**, cho 2 mục đích:

1. **Cache dữ liệu** (`django-cacheops`) — cache các bảng ít thay đổi, đọc nhiều như `Product`, `Product_Category`, `Frame`, `Slide`, `Setting` để giảm tải database. Cấu hình tại `CACHEOPS` trong `settings.py`. Nếu Redis chết, app vẫn chạy bình thường (`CACHEOPS_DEGRADE_ON_FAILURE = True`), chỉ mất tác dụng cache.
2. **Hàng đợi tác vụ bất đồng bộ** (`celery`) — dùng để chạy các API xử lý ảnh AI nặng (kiểm duyệt NSFW, tách người + AI vẽ lại nền) ở nền, tránh block/timeout request HTTP.

### Cài Redis

```bash
# Cách nhanh nhất — dùng Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Hoặc cài trực tiếp trên Ubuntu/Debian
sudo apt install redis-server
sudo systemctl enable --now redis-server
```

Kiểm tra Redis đã chạy:

```bash
redis-cli ping
# Kết quả mong đợi: PONG
```

### Chạy Celery worker (xử lý tác vụ nền)

```bash
# đảm bảo đã kích hoạt venv trước
celery -A fitting_app worker --loglevel=info

# Windows cần thêm --pool=solo
celery -A fitting_app worker --loglevel=info --pool=solo
```

> File cấu hình Celery app: `fitting_app/celery.py`. Các biến `CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND` lấy từ `REDIS_URL` trong `.env`.

### Đóng gói tác vụ AI thành Celery task (bước tiếp theo, tuỳ chọn)

Hiện tại `app/services/moderation.py` và `app/services/background_inpaint.py` được gọi **đồng bộ** trực tiếp trong view. Để chuyển sang chạy nền qua Celery, tạo file `app/tasks.py`:

```python
from celery import shared_task

@shared_task
def check_nsfw_task(uploaded_image_id):
    from app.models import Uploaded_Image
    from app.services.moderation import check_nsfw
    obj = Uploaded_Image.objects.get(id=uploaded_image_id)
    result = check_nsfw(obj.image.path)
    obj.is_nsfw = result['is_nsfw']
    obj.nsfw_score = result['score']
    obj.status = 'rejected' if result['is_nsfw'] else 'approved'
    obj.save()
```

Rồi gọi `check_nsfw_task.delay(uploaded.id)` thay vì gọi `check_nsfw()` trực tiếp trong view — client sẽ nhận phản hồi ngay, kết quả kiểm duyệt cập nhật sau (cần thêm cơ chế poll trạng thái hoặc push notification cho client).

---

## 6. Xác thực bằng JWT

Dự án dùng **JWT** (`djangorestframework-simplejwt`) thay cho DRF Token truyền thống. Cấu hình tại `SIMPLE_JWT` trong `settings.py`:

| Cấu hình | Mặc định | Biến `.env` tương ứng |
|---|---|---|
| Access token sống bao lâu | 60 phút | `JWT_ACCESS_TOKEN_LIFETIME_MINUTES` |
| Refresh token sống bao lâu | 1 ngày | `JWT_REFRESH_TOKEN_LIFETIME_DAYS` |
| Xoay refresh token mỗi lần dùng | Bật (`ROTATE_REFRESH_TOKENS`) | — |
| Vô hiệu hoá refresh token cũ sau khi xoay | Bật (`BLACKLIST_AFTER_ROTATION`) | — |

**Quan trọng:** tính năng blacklist refresh token cần app `rest_framework_simplejwt.token_blacklist` (đã có sẵn trong `INSTALLED_APPS`) và **phải chạy `migrate`** để tạo bảng lưu token bị blacklist — bước này đã nằm trong `python manage.py migrate` / `./run.sh` ở trên, không cần làm gì thêm, nhưng nếu nâng cấp từ bản cũ (trước khi có JWT) thì nhớ chạy lại `migrate` một lần.

Client gọi API xác thực bằng header:
```
Authorization: Bearer <access_token>
```

> **Lưu ý nếu bạn nâng cấp từ bản dùng `TokenAuthentication` (DRF Token) cũ:** `settings.py` đã đổi `DEFAULT_AUTHENTICATION_CLASSES` sang `JWTAuthentication` và **gỡ `rest_framework.authtoken` khỏi `INSTALLED_APPS`**. Nếu `views.py` của bạn vẫn đang cấp/kiểm tra DRF `Token` (ví dụ trong view `login`/`register` cũ có `Token.objects.create(user=...)`), các view đó cần được cập nhật để cấp JWT (`RefreshToken.for_user(user)`) thay thế, nếu không client sẽ không đăng nhập được sau khi đổi sang cấu hình này.

---

## 7. Cấu trúc thư mục chính

```
smart-fitting-app/
├── manage.py
├── requirements.txt
├── .env.example                 # mau bien moi truong - commit len git
├── .env                          # tu tao tu .env.example, KHONG commit len git
├── setup.sh                       # script tu dong cai dat (Linux/macOS)
├── setup.bat                      # script tu dong cai dat (Windows)
├── run.sh                         # kiem tra DB/Redis roi chay backend (dev/prod)
├── scripts/
│   ├── wait_for_services.py       # kiem tra Database + Redis san sang (dung boi run.sh)
│   └── build_celebrity_index.py   # (chay 1 lan, offline) build app/data/celebrity_embeddings.npz
├── fitting_app/                 # cấu hình project Django
│   ├── settings.py               # doc config tu .env
│   ├── celery.py                 # khoi tao Celery app (dung Redis)
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── app/                         # app chính
    ├── models.py
    ├── views.py
    ├── serializers.py
    ├── urls.py
    ├── admin.py
    ├── apps.py
    ├── tasks.py                 # (tuy chon) Celery tasks bat dong bo
    ├── data/
    │   └── celebrity_embeddings.npz  # vector dac trung nguoi noi tieng (build san, xem muc 4.1)
    └── services/                # các module xử lý AI
        ├── moderation.py            # kiểm duyệt NSFW (1.3.2)
        ├── celebrity_detection.py   # nhận diện người nổi tiếng VN (1.3.1)
        └── background_inpaint.py    # tách người + AI vẽ lại nền
```

---

## 8. Trước khi deploy production — checklist bảo mật

File `.env.example` mặc định phù hợp cho **development**. Trước khi deploy production, tạo `.env` riêng cho production và đảm bảo:

- [ ] `SECRET_KEY` — đổi sang chuỗi bí mật ngẫu nhiên đủ dài, không dùng lại giá trị mẫu.
- [ ] `DEBUG=False`
- [ ] `ALLOWED_HOSTS` — chỉ liệt kê domain thật, không để trống/`*`.
- [ ] `CORS_ALLOW_ALL_ORIGINS=False` và khai báo `CORS_ALLOWED_ORIGINS` cụ thể.
- [ ] `DB_ENGINE` chuyển sang PostgreSQL thay vì SQLite mặc định.
- [ ] Redis production nên bật auth (`requirepass`) và không để mở public port.
- [ ] Bật HTTPS (`SECURE_SSL_REDIRECT`, `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`) — các setting này chưa có sẵn trong `settings.py`, cần bổ sung thêm khi deploy.

---

## 9. Kiểm tra nhanh sau khi cài đặt (smoke test)

```bash
# Kiểm tra server chạy được
curl http://localhost:8000/api/products/

# Đăng ký thử tài khoản
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"phone": "0900000000", "password": "123456", "full_name": "Test User"}'
```

Nếu nhận được response JSON hợp lệ (không phải lỗi 500) là cài đặt thành công.

---

## 10. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Chưa cài `opencv-python-headless` | `pip install -r requirements.txt` lại, đảm bảo dòng `opencv-python-headless` có trong file |
| `ModuleNotFoundError: No module named 'dotenv'` | Chưa cài `python-dotenv`, hoặc chưa kích hoạt venv | Kích hoạt venv, `pip install -r requirements.txt` lại |
| Model AI tải rất chậm / bị treo ở request đầu tiên | Đang tải model lần đầu (300-700MB) | Chờ hoàn tất, hoặc kiểm tra kết nối Internet của server |
| `psycopg2` cài lỗi trên macOS/Windows | Thiếu thư viện hệ thống cho PostgreSQL | Cài `postgresql` client libs trước, hoặc dùng SQLite cho dev |
| Timeout khi gọi API tách nền / kiểm duyệt ảnh | Xử lý AI trên CPU chậm hơn timeout mặc định | Tăng timeout của gunicorn/nginx, hoặc chuyển tác vụ đó sang Celery (xem mục 5) |
| `ConnectionError` / app chậm bất thường liên quan Redis | Redis server chưa chạy | Kiểm tra `redis-cli ping`, khởi động lại Redis; app vẫn chạy được nhờ `CACHEOPS_DEGRADE_ON_FAILURE=True`, chỉ mất cache |
| `celery: command not found` | Chưa kích hoạt venv hoặc chưa cài `celery` | `source venv/bin/activate` rồi `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'rest_framework_simplejwt'` | Chưa cài `djangorestframework-simplejwt` | `pip install -r requirements.txt` lại |
| Đăng nhập được nhưng gọi API khác báo `401 Unauthorized` dù có gửi token | Client đang gửi sai kiểu header, hoặc `views.py` vẫn đang cấp DRF Token cũ thay vì JWT | Header phải là `Authorization: Bearer <access_token>`; kiểm tra `views.py` đã cấp JWT chưa (xem mục 6) |
| Lỗi liên quan `token_blacklist` / bảng `outstandingtoken` không tồn tại khi logout hoặc refresh token | Chưa chạy `migrate` sau khi thêm app `rest_framework_simplejwt.token_blacklist` | `python manage.py migrate` (hoặc `./run.sh`, tự động migrate) |
| `python scripts/wait_for_services.py` báo `ModuleNotFoundError: No module named 'fitting_app'` | Chạy script không đúng từ thư mục gốc project | Luôn chạy qua `./run.sh` (đã tự `cd` đúng thư mục), hoặc `cd` vào thư mục gốc project trước khi chạy tay |
| Cài `insightface` báo lỗi build wheel: `fatal error: Python.h: No such file or directory` | `requirements.txt` đang ghim bản `insightface` cũ (`0.7.x`) — bản này không có sẵn wheel, pip phải tự biên dịch một extension C++ và cần header `Python.h` (gói `python3-dev`/`python3.11-dev` của hệ điều hành) | Đảm bảo `requirements.txt` đang dùng `insightface==1.0.1` trở lên (bản này là wheel Python thuần, không cần biên dịch gì) — nếu vẫn dùng bản cũ, cập nhật lại dòng đó trong `requirements.txt` rồi `pip install -r requirements.txt` lại |