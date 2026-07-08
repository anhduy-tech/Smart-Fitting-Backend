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
# ---------------------------------------------------------------------------
if [[ "$SKIP_MIGRATE" == false ]]; then
    log_info "Ap dung migrations..."
    if ! python manage.py migrate --noinput; then
        log_err "Migrate that bai. Dung lai."
        exit 1
    fi
    log_ok "Migrate xong."
else
    log_info "Bo qua migrate (--skip-migrate)."
fi

# ---------------------------------------------------------------------------
# 5. Chay backend
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