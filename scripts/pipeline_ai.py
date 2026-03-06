import pandas as pd
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from src.ai.translator import featuring_data, generate_edge_case_report
from src.ai.slm import get_slm_analysis, prime_group_context

def run_ai_pipeline(X_test_df, output_path="report/ai_assistance_report.csv"):
    """
    Quy trình tích hợp: Feature Engineering -> Filter Edge Cases -> SLM Analysis
    """
    print("Bước 1: Trích xuất đặc trưng hành vi ")
    # Sử dụng hàm featuring_data từ translator.py
    df_featured = featuring_data(X_test_df)
    
    print("Lọc các trường hợp bất thường nghiêm trọng (rb3 >= 2 & rb4 >= 1) ")
    # Sử dụng hàm generate_edge_case_report từ translator.py
    report_df = generate_edge_case_report(df_featured).copy()
    
    if len(report_df)==0:
        print("Không tìm thấy trường hợp nào thỏa mãn điều kiện bất thường nghiêm trọng.")
        return None

    print(f"Tìm thấy {len(report_df)} trường hợp cần xử lý. Đang chạy AI chẩn đoán...")
    # Prime context theo workflow supervisor (group signature + ngưỡng entropy động).
    prime_group_context(report_df)

    # --- Bước 3: Gọi SLM để phân tích từng dòng ---
    # Chúng ta sử dụng cột 'fact' và 'action_sequence' làm đầu vào cho mô hình
    def apply_ai(row):
        try:
            return get_slm_analysis(row['fact'], row['action_sequence'])
        except Exception as e:
            return f"Lỗi phân tích AI: {str(e)}"

    report_df['ai_assistance'] = report_df.apply(apply_ai, axis=1)

    priority_order = {
        'HIGH': 0,
        'MEDIUM': 1,
        'LOW': 2
    }

    def get_priority_rank(ai_string):
        """Hàm phụ để lấy mức độ ưu tiên từ chuỗi kết quả của SLM"""
        if 'P=HIGH' in ai_string: return priority_order['HIGH']
        if 'P=MEDIUM' in ai_string: return priority_order['MEDIUM']
        if 'P=LOW' in ai_string: return priority_order['LOW']
        return 3 # Mức mặc định nếu không tìm thấy

    # 2. Tạo cột tạm để sắp xếp
    report_df['temp_priority_rank'] = report_df['ai_assistance'].apply(get_priority_rank)

    # 3. Sắp xếp: Ưu tiên (Priority) trước, sau đó đến ID hoặc Entropy (tùy chọn)
    report_df = report_df.sort_values(by=['temp_priority_rank', 'id'], ascending=[True, True])

    # --- Bước 4: Định dạng lại và xuất file CSV ---
    # Giữ đúng các cột yêu cầu: id, action_sequence, fact, ai_assistance
    final_output = report_df[['id', 'action_sequence','first_action_rb', 'ai_assistance']]
    
    # Xuất file với encoding utf-8-sig để đọc được tiếng Việt trong Excel
    final_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f" Hoàn thành! Báo cáo đã được lưu tại: {output_path} ")
    return final_output

if __name__ == "__main__":
    # Giả sử bạn đã load dữ liệu X_test từ trước
    # X_test = pd.read_csv("data/X_test.csv") 
    
    # Demo với dữ liệu giả lập nếu chạy độc lập
    X_train=pd.read_csv(r"data/layer1_raw/X_test.csv")
    # Chạy pipeline
    result = run_ai_pipeline(X_train)
    if result is not None:
        print(result.head(5))
