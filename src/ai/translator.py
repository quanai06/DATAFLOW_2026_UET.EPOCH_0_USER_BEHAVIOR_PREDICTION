import numpy as np 
import pandas as pd 
def featuring_data(df):
    df = df.copy()
    seq_cols = [f"feature_{i}" for i in range(1, 38)]
    # Length 
    df["length"] = df[seq_cols].notna().sum(axis=1)

    # Entropy 
    from collections import Counter
    def calculate_entropy(row, cols):
        seq = row[cols].dropna().values
        if len(seq) == 0: 
            return 0
        
        counts = Counter(seq)
        probs = np.array(list(counts.values())) / len(seq)
        return -np.sum(probs * np.log2(probs + 1e-9))
    
    df["entropy"] = df.apply(lambda x: calculate_entropy(x, seq_cols), axis=1)

    def count_rollbacks(sequence):
        seq = [x for x in sequence if pd.notnull(x) and x != 0]
        n = len(seq)
        
        rollback_3 = 0
        rollback_4 = 0
        rb3_actions=set()
        rb4_actions=set()
        
        if n < 3:
            return 0, 0
        
        # 1. Quét rollback 3 bước: A -> B -> A
        for i in range(n - 2):
            # Bước đầu và bước cuối của cụm 3 giống nhau, nhưng bước giữa phải khác
            if seq[i] == seq[i+2]:
                rollback_3 += 1
                rb3_actions.add(str(int(seq[i])))
                
        # 2. Quét rollback 4 bước: A -> B -> C -> A
        if n >= 4:
            for i in range(n - 3):
                # Bước đầu và bước cuối của cụm 4 giống nhau
                if seq[i] == seq[i+3]:
                    # Đảm bảo đây không phải là một chuỗi lặp đơn điệu (như A -> A -> A -> A)
                    if seq[i] != seq[i+1] :
                        rollback_4 += 1
                        rb4_actions.add(str(int(seq[i])))
                        
        return rollback_3, rollback_4,list(rb3_actions),list(rb4_actions)

    # Ví dụ áp dụng cho DataFrame
        
    # Tính toán rollback cho từng dòng
    results = df[seq_cols].apply(lambda row: count_rollbacks(row.values), axis=1)

# Bước 2: Bung kết quả ra thành các cột (Cách này nhanh và sạch nhất)
    res_df = pd.DataFrame(results.tolist(), index=df.index)
    df['rb_3_steps'] = res_df[0]
    df['rb_4_steps'] = res_df[1]

    # Bước 3: Gộp chi tiết mã lỗi và loại bỏ trùng lặp bằng set
    df['first_action_rb'] = [list(set(a + b)) for a, b in zip(res_df[2], res_df[3])]
    def row_to_string(row):
        actions = row[seq_cols].dropna().astype(int).astype(str).tolist()
        return "-".join(actions)
    # 'action_sequence'
    df['action_sequence'] = df.apply(row_to_string, axis=1)
    return df 


def generate_edge_case_report(df):
    """
    df: DataFrame đã có các cột rb_3_steps, rb_4_steps, length, entropy, action_sequence
    """
    # 1. Lọc các ID theo điều kiện bất thường nghiêm trọng
    # rb_3_steps >= 2 VÀ rb_4_steps >= 1
    edge_cases = df[
        (df['rb_3_steps'] >= 2) |
        (df['rb_4_steps'] >= 1)
    ].copy()
    
    # Hàm con để tạo fact cho từng dòng (tối ưu từ hàm bạn đã viết)
    def create_fact(row):
        rb3 = row['rb_3_steps']
        rb4 = row['rb_4_steps']
        length = row['length']
        ent = round(row['entropy'], 2)
        
        fact = f"Phát hiện {rb3} lần lặp 3 bước (A-B-A), "
        fact += f"{rb4} lần lặp 4 bước (A-B-C-A). "
        fact += f"Độ dài chuỗi: {length} thao tác. "
        fact += f"Chỉ số hỗn loạn (Entropy): {ent}."

        return fact

    # Xây dựng DataFrame kết quả
    report_df = pd.DataFrame()
    report_df['id'] = edge_cases['id']
    report_df['action_sequence'] = edge_cases['action_sequence']
    report_df['first_action_rb'] =edge_cases['first_action_rb']
    
    # Tạo cột fact chứa câu diễn giải văn bản
    report_df['fact'] = edge_cases.apply(create_fact, axis=1)

    return report_df.reset_index(drop=True)