import os
import csv
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# Supabase client — khởi tạo 1 lần duy nhất
# ══════════════════════════════════════════════════════════════
_supabase = None

def _get_supabase():
    """Lazy init — chỉ import và kết nối khi cần."""
    global _supabase
    if _supabase is not None:
        return _supabase
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
        if url and key:
            _supabase = create_client(url, key)
    except Exception as e:
        print(f"⚠️ Không kết nối được Supabase: {e}")
    return _supabase


# ══════════════════════════════════════════════════════════════
# Fallback: CSV local (dùng khi chạy local, không có Supabase)
# ══════════════════════════════════════════════════════════════
LOG_FILE = "chat_logs.csv"

def _init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Thời_Gian", "Route", "Category",
                "Câu_Hỏi", "Độ_Dài_Trả_Lời"
            ])

def _log_csv(route: str, category: str, user_query: str, response_length: int):
    _init_csv()
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, route, category, user_query, response_length])
    except Exception as e:
        print(f"⚠️ Lỗi ghi CSV: {e}")


# ══════════════════════════════════════════════════════════════
# PUBLIC API — dùng trong app.py
# ══════════════════════════════════════════════════════════════
def init_logger():
    """Gọi 1 lần khi app khởi động — kiểm tra kết nối Supabase."""
    client = _get_supabase()
    if client:
        print("✅ Logger: kết nối Supabase thành công")
    else:
        _init_csv()
        print(f"⚠️ Logger: dùng CSV local ({LOG_FILE})")


def log_interaction(route: str, category: str, user_query: str, ai_response: str):
    """
    Ghi log mỗi lần user hỏi.
    Ưu tiên Supabase, fallback về CSV local nếu không có kết nối.
    """
    response_length = len(ai_response)
    timestamp       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Thử ghi vào Supabase ─────────────────────────────────
    client = _get_supabase()
    if client:
        try:
            client.table("chat_logs").insert({
                "thoi_gian":      timestamp,
                "route":          route,
                "category":       category,
                "cau_hoi":        user_query,
                "do_dai_tra_loi": response_length,
            }).execute()
            return  # ghi Supabase thành công → xong
        except Exception as e:
            print(f"⚠️ Lỗi ghi Supabase, fallback CSV: {e}")

    # ── Fallback: ghi CSV local ───────────────────────────────
    _log_csv(route, category, user_query, response_length)


# ══════════════════════════════════════════════════════════════
# ĐỌC LOG — dùng cho analytics trong sidebar Streamlit
# ══════════════════════════════════════════════════════════════
def get_logs(limit: int = 200) -> list[dict]:
    """
    Lấy log gần nhất để vẽ biểu đồ.
    Trả về list[dict] với các key: thoi_gian, route, category, cau_hoi, do_dai_tra_loi
    """
    client = _get_supabase()
    if client:
        try:
            res = (
                client.table("chat_logs")
                .select("*")
                .order("thoi_gian", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception as e:
            print(f"⚠️ Lỗi đọc Supabase: {e}")

    # Fallback: đọc CSV
    rows = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            rows = rows[-limit:]
        except Exception:
            pass
    return rows