import csv
import os
from datetime import datetime

# Tên file log (Sẽ tự động được tạo ngang hàng với app.py)
LOG_FILE = "chat_logs.csv"


def init_logger():
    """Kiểm tra và tạo file CSV với tiêu đề cột nếu chưa tồn tại."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # Tạo các cột dữ liệu cần thiết cho việc vẽ biểu đồ sau này
            writer.writerow(
                ["Thời_Gian", "Luồng_Xử_Lý (Route)", "Chủ_Đề (Category)", "Câu_Hỏi_User", "Độ_Dài_Câu_Trả_Lời"])
        print(f"✅ Đã tạo file log mới: {LOG_FILE}")


def log_interaction(route: str, category: str, user_query: str, ai_response: str):
    """Ghi lại thông tin mỗi lần người dùng hỏi."""
    # Lấy thời gian thực
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tính độ dài câu trả lời (thay vì lưu cả đoạn text dài gây nặng file,
    # Khoa thường chỉ quan tâm user hỏi gì và AI có trả lời dài/đầy đủ không)
    response_length = len(ai_response)

    try:
        with open(LOG_FILE, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, route, category, user_query, response_length])
    except Exception as e:
        print(f"⚠️ Lỗi khi ghi log: {e}")