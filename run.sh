#!/usr/bin/env bash
#
# run.sh - Kiem tra Database + Redis da san sang va ket noi duoc chua,
# roi moi chay backend Django (Smart Fitting App).
#
# Cach dung:
#   ./run.sh                  # chay dev server (runserver) o 0.0.0.0:8000
#   ./run.sh --prod           # chay bang gunicorn (production)
#   ./run.sh --port 8080      # doi port (mac dinh 8000)
#   ./run.sh --skip-migrate   # bo qua buoc migrate tu dong
#   ./run.sh --timeout 60     # doi toi da 60s cho DB/Redis (mac dinh 30s)
#
#   ./run.sh --worker-gpu       # chay Celery worker CHO QUEUE 'gpu_tasks'
#                                # (task AI nang: try-on...). LUON chay
#                                # --pool=solo --concurrency=1 - BAT BUOC,
#                                # nhieu process cung luc se tranh nhau
#                                # VRAM gay "CUDA out of memory".
#   ./run.sh --worker-default   # chay Celery worker CHO QUEUE 'default'
#                                # (task nhe: OTP, notification...) - prefork,
#                                # concurrency cao hon binh thuong duoc.
#   ./run.sh --all              # chay CUNG LUC: Django dev server +
#                                # worker-gpu + worker-default trong 1
#                                # terminal (tien cho dev). Ctrl+C se tu
#                                # dong dung ca 3. KHONG dung --all cho
#                                # production - production nen chay 3
#                                # process nay o 3 service/container rieng
#                                # (vd 3 systemd unit hoac 3 Docker service)
#                                # de scale/restart doc lap nhau.
#
set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Cau hinh mac dinh + doc tham so dong lenh
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PORT=8000
MODE="dev"
TIMEOUT=30
SKIP_MIGRATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prod)
            MODE="prod"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --skip-migrate)
            SKIP_MIGRATE=true
            shift
            ;;
        --worker-gpu)
            MODE="worker-gpu"
            shift
            ;;
        --worker-default)
            MODE="worker-default"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Tham so khong hop le: $1"
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_info() { echo -e "${YELLOW}[..]${NC} $1"; }
log_err()  { echo -e "${RED}[FAIL]${NC} $1"; }

# ---------------------------------------------------------------------------
# 1. Kich hoat virtualenv (neu chua active)
# ---------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        log_info "Kich hoat virtualenv (venv/)..."
        # shellcheck disable=SC1091
        source venv/bin/activate
    else
        log_err "Khong tim thay venv/bin/activate. Hay tao venv truoc:"
        echo "    python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi
log_ok "Dang dung Python: $(command -v python) ($(python --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Kiem tra da cai dependencies chua (django + redis-py la 2 cai bat buoc
#    de cac buoc kiem tra ben duoi chay duoc)
# ---------------------------------------------------------------------------
if ! python -c "import django" >/dev/null 2>&1; then
    log_err "Chua cai dependencies. Hay chay: pip install -r requirements.txt"
    exit 1
fi
if [[ "$MODE" == "worker-gpu" || "$MODE" == "worker-default" || "$MODE" == "all" ]]; then
    if ! python -c "import celery" >/dev/null 2>&1; then
        log_err "Chua cai celery. Hay chay: pip install -r requirements.txt"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Kiem tra Database + Redis da san sang va ket noi duoc chua
#    (dung dung config that trong settings.py, xem scripts/wait_for_services.py)
# ---------------------------------------------------------------------------
log_info "Kiem tra Database va Redis (timeout ${TIMEOUT}s)..."
if ! python scripts/wait_for_services.py --timeout "$TIMEOUT"; then
    log_err "Database va/hoac Redis chua san sang. Dung lai, khong chay backend."
    echo ""
    echo "Goi y kiem tra thu cong:"
    echo "  - Postgres: pg_isready -h \$DB_HOST -p \$DB_PORT   (hoac kiem tra file SQLite neu dung SQLite)"
    echo "  - Redis:    redis-cli -u \"\${REDIS_URL:-redis://127.0.0.1:6379/0}\" ping"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Chay migrate (tu dong ap dung migration con thieu) - co the tat bang --skip-migrate
#    Bo qua hoan toan neu chi chay worker (worker khong can migrate, Django
#    server da lo migrate roi - tranh 2 process cung migrate 1 luc).
# ---------------------------------------------------------------------------
if [[ "$SKIP_MIGRATE" == false && "$MODE" != "worker-gpu" && "$MODE" != "worker-default" ]]; then
    log_info "Ap dung migrations..."
    if ! python manage.py migrate --noinput; then
        log_err "Migrate that bai. Dung lai."
        exit 1
    fi
    log_ok "Migrate xong."
else
    log_info "Bo qua migrate."
fi

# ---------------------------------------------------------------------------
# 5. Cac ham chay tung loai worker (dung chung cho MODE=worker-* va MODE=all)
# ---------------------------------------------------------------------------
run_worker_gpu() {
    # --pool=solo --concurrency=1 la BAT BUOC cho queue nay: cac task AI
    # nang (try-on, Stable Diffusion...) dung chung 1 GPU, chay >1 process
    # cung luc se tranh nhau VRAM va gay loi "CUDA out of memory" (da gap
    # thuc te). KHONG tang concurrency cho worker nay du server co GPU
    # manh hay VRAM lon co nao - GPU khong chia se VRAM tot giua nhieu
    # process nhu CPU/RAM.
    log_ok "Chay Celery worker [gpu_tasks] (pool=solo, concurrency=1)..."
    exec celery -A fitting_app worker --loglevel=info --pool=solo --concurrency=1 \
        -Q gpu_tasks -n gpu_worker@%h
}

run_worker_default() {
    # Task nhe (gui OTP, notification, cac tac vu I/O khac sau nay) khong
    # dung GPU nen chay prefork binh thuong duoc, concurrency cao hon de
    # xu ly nhieu viec song song.
    log_ok "Chay Celery worker [default] (pool=prefork, concurrency=4)..."
    exec celery -A fitting_app worker --loglevel=info --concurrency=4 \
        -Q default -n default_worker@%h
}

if [[ "$MODE" == "worker-gpu" ]]; then
    run_worker_gpu
    exit 0
fi

if [[ "$MODE" == "worker-default" ]]; then
    run_worker_default
    exit 0
fi

# ---------------------------------------------------------------------------
# 6. Che do --all: chay Django + ca 2 Celery worker cung luc (chi cho dev).
#    Dung trap de dam bao Ctrl+C se giet HET cac process con, khong de
#    worker chay "mo coi" ngam sau khi tuong da tat het.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "all" ]]; then
    log_info "Che do --all: chay Django + worker-gpu + worker-default cung luc (chi danh cho dev)."

    PIDS=()
    cleanup() {
        log_info "Dang dung tat ca process con..."
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null
        log_ok "Da dung xong."
    }
    trap cleanup EXIT INT TERM

    celery -A fitting_app worker --loglevel=info --pool=solo --concurrency=1 \
        -Q gpu_tasks -n gpu_worker@%h &
    PIDS+=("$!")

    celery -A fitting_app worker --loglevel=info --concurrency=4 \
        -Q default -n default_worker@%h &
    PIDS+=("$!")

    log_ok "Chay dev server tren 0.0.0.0:${PORT} ..."
    python manage.py runserver "0.0.0.0:${PORT}" &
    PIDS+=("$!")

    wait
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. Chay backend (mac dinh: dev hoac --prod)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "prod" ]]; then
    log_ok "Chay backend bang gunicorn tren 0.0.0.0:${PORT} ..."
    exec gunicorn fitting_app.wsgi:application \
        --bind "0.0.0.0:${PORT}" \
        --workers 3 \
        --timeout 120
else
    log_ok "Chay dev server tren 0.0.0.0:${PORT} ..."
    exec python manage.py runserver "0.0.0.0:${PORT}"
fi