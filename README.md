##README
# USER BEHAVIOR PREDICTION – DATAFLOW 2026

Giải pháp dự đoán hành vi người dùng từ chuỗi Action ID ẩn danh, kết hợp **EDA**, **Feature Engineering**, **Edge-case Analysis** và các mô hình học sâu để phục vụ dự báo nhu cầu, tối ưu vận hành và hỗ trợ giám sát hành vi bất thường.

---

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Dữ liệu](#2-dữ-liệu)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Mô hình](#4-mô-hình)
5. [Xử lý edge-case](#5-xử-lý-edge-case)
6. [Đánh giá](#6-đánh-giá)
7. [Cấu trúc thư mục](#7-cấu-trúc-thư-mục)
8. [Cài đặt và chạy dự án](#8-cài-đặt-và-chạy-dự-án)
9. [Giới hạn và hướng phát triển](#9-giới-hạn-và-hướng-phát-triển)

---

## 1. Tóm tắt

Dự án được thực hiện trong khuôn khổ cuộc thi **DATAFLOW 2026: THE ALCHEMY OF MINDS**.

### Bối cảnh bài toán

Trong các hệ thống bán lẻ, kho bãi hoặc nền tảng giao dịch, hành vi người dùng thường diễn ra theo chuỗi thao tác liên tiếp. Nếu chỉ dựa trên thống kê tĩnh, doanh nghiệp khó phát hiện:

- người dùng có khả năng hoàn tất giao dịch,
- người dùng có dấu hiệu rời bỏ sớm,
- các luồng thao tác bất thường như rollback, loop hoặc spam-like behavior.

### Mục tiêu

Dự án hướng tới ba mục tiêu chính:

1. **Dự đoán hành vi người dùng** từ chuỗi Action ID đã ẩn danh.
2. **Phân tích edge-case** để hỗ trợ giám sát các mẫu hành vi bất thường.
3. **Tăng khả năng triển khai thực tế** thông qua pipeline có thể mở rộng và giảm chi phí review thủ công.

### Cách tiếp cận

Hệ thống được xây dựng theo 3 lớp dữ liệu:

- **Layer 1 – Raw Layer**: lưu dữ liệu gốc.
- **Layer 2 – Feature Engineering Layer**: sinh các đặc trưng thủ công từ chuỗi hành động.
- **Layer 3 – Model-Special Layer**: định dạng dữ liệu theo yêu cầu từng mô hình.

Các mô hình chính gồm:

- **BiLSTM + manual features**
- **Transformer Encoder + Multi-head Classification**
- **Ensemble LSTM + GRU + CNN**
- **SLM / Rule-based fallback** cho bài toán phân nhóm edge-case

---

## 2. Dữ liệu

Dữ liệu đầu vào là chuỗi **Action ID** đã được ẩn danh, mô tả hành vi của người dùng theo thời gian.

### Đặc điểm dữ liệu

- Chuỗi có độ dài biến thiên, tối đa khoảng **37 bước**.
- Hành vi được mã hóa thành các **Action ID** số nguyên.
- Dữ liệu có tính **tuần tự**, **mất cân bằng nhãn** và tồn tại **long-tail distribution**.
- Có xuất hiện các mẫu bất thường như:
  - rollback 3 bước: `A → B → A`
  - rollback 4 bước: `A → B → C → A`
  - loop cục bộ
  - chuỗi dài bất thường

### Feature Engineering

Từ chuỗi Action ID, hệ thống trích xuất các đặc trưng như:

- **Độ dài chuỗi**: `length`
- **Độ đa dạng**: `nunique`
- **Hành động đầu / cuối**: `first_item`, `last_item`
- **Hành động thống trị**: `mode_val`, `action_dominance`
- **Độ phức tạp chuỗi**: `entropy`, `n_transitions`, `transition_ratio`
- **Đặc trưng hành động hiếm**: `n_rare_actions`, `rare_action_ratio`
- **Rollback features**: `rb_3_steps`, `rb_4_steps`, `first_action_rb`

---

## 3. Kiến trúc hệ thống

Hệ thống được thiết kế theo kiến trúc dữ liệu 3 tầng để cô lập tiền xử lý và tối ưu đầu vào cho từng nhóm mô hình.

### Layer 1 – Raw Layer
Lưu trữ dữ liệu gốc do ban tổ chức cung cấp, đảm bảo tính toàn vẹn.

### Layer 2 – Feature Engineering Layer
Sinh đặc trưng từ chuỗi hành vi phục vụ EDA, ML và edge-case analysis.

### Layer 3 – Model-Special Layer
Chuẩn hóa dữ liệu theo từng loại mô hình:

- Tensor / padded sequence cho Transformer
- `input_ids + input_stats` cho LSTM
- dữ liệu fusion cho ensemble
- signature nhóm cho pipeline edge-case

---

## 4. Mô hình

### 4.1 LSTM

Mô hình LSTM được xây dựng để khai thác phụ thuộc tuần tự trong chuỗi hành vi.

**Đầu vào**
- `input_ids`: chuỗi hành động đã padding
- `input_stats`: đặc trưng thủ công

**Kiến trúc**
- Embedding (`dim = 128`)
- Bidirectional LSTM (`hidden = 128`, `dropout = 0.3`)
- Dense branch cho manual features
- Concatenate
- Dense + Dropout
- Multi-head output

**Đặc điểm**
- Kết hợp thông tin chuỗi và tín hiệu thống kê
- Phù hợp với dữ liệu tuần tự có độ dài biến thiên
- Hỗ trợ phân loại đa nhãn

---

### 4.2 Transformer

Transformer được dùng để học các quan hệ phi tuyến trong chuỗi hành vi mà không phụ thuộc mạnh vào khoảng cách vị trí.

**Đầu vào**
- `seq`: chuỗi hành động đã padding
- `mask`: mặt nạ bỏ qua padding

**Kiến trúc**
- Embedding (`vocab_size = 25000`, `d_model = 128`)
- Positional Encoding
- Transformer Encoder (`num_layers = 3`, `nheads = 4`)
- Masked Mean Pooling
- 6 nhánh phân loại độc lập

**Ưu điểm**
- Khả năng học quan hệ xa tốt hơn RNN
- Tận dụng cơ chế self-attention
- Hữu ích với chuỗi có mẫu hành vi phức tạp

---

### 4.3 Ensemble LSTM + GRU + CNN

Để tăng khả năng tổng quát hóa, nhóm triển khai một hệ thống **heterogeneous ensemble** kết hợp:

- **BiLSTM**: học phụ thuộc dài hạn
- **BiGRU**: hội tụ nhanh, xử lý tín hiệu biến thiên
- **1D-CNN**: bắt các mẫu cục bộ trong chuỗi

Kết quả cuối cùng được tổng hợp bằng **soft voting** trên xác suất đầu ra.

---

## 5. Xử lý edge-case

Ngoài mô hình dự đoán chính, hệ thống còn có một pipeline riêng để xử lý các trường hợp bất thường.

### Mục tiêu
- phát hiện rollback / loop / chuỗi hiếm,
- gom nhóm các chuỗi tương tự,
- định tuyến ưu tiên cho người giám sát.

### Signature nhóm hành vi

Mỗi chuỗi được ánh xạ sang một signature gồm:

- `anchor`
- `aba_count`
- `b_unique`
- `rb4`
- bucket độ dài
- bucket entropy

### Nhãn nhóm (TAG)

Hệ thống gán các nhãn hình thái:

- `RB4`
- `ANCHOR_LOOP`
- `LONG`
- `HIGH_VAR`
- `MIXED`

### Mức ưu tiên (P)

Mỗi nhóm được gán ưu tiên:

- `HIGH`
- `MED`
- `LOW`

### Output phục vụ giám sát

Dạng output compact:

```text
G=<gid>|TAG=<TAG>|P=<P>|COUNT=<n>|REP=<id>|A=<anchor>|ABA=<k>|Buniq=<m>
```
## 6. Đánh giá


### 1. Accuracy theo từng thuộc tính

| Model            | Attr_1 | Attr_2 | Attr_3 | Attr_4 | Attr_5 | Attr_6 | Overall |
|------------------|-------:|-------:|-------:|-------:|-------:|-------:|--------:|
| LSTM             | 0.9922 | 0.9982 | 0.9972 | 0.9982 | 0.9971 | 0.9992 | 0.99139 |
| LSTM+GRU+CNN     | 0.9875 | 0.9985 | 0.9973 | 0.9849 | 0.9991 | 0.9992 | 0.99389 |
| Transformer      | **0.9983** | **0.9990** | **0.9979** | **0.9986** | **0.9985** | **0.9981** | **0.9926** |

### 2. F1-score theo từng thuộc tính

| Model            | Attr_1 | Attr_2 | Attr_3 | Attr_4 | Attr_5 | Attr_6 | Macro F1 |
|------------------|-------:|-------:|-------:|-------:|-------:|-------:|---------:|
| LSTM             | 0.9768 | 0.9975 | 0.9972 | 0.9719 | 0.9971 | 0.9991 | 0.98994 |
| LSTM+GRU+CNN     | 0.9875 | 0.9985 | 0.9973 | 0.9849 | 0.9991 | 0.9992 | 0.99444 |
| Transformer      | **0.9750** | **0.9986** | **0.9978** | **0.9777** | **0.9976** | **0.9980** | **0.9908** |

### 3. Nhận xét

- **Transformer** đạt accuracy cao nhất ở hầu hết các thuộc tính.
- **LSTM+GRU+CNN** cho kết quả **Macro F1 cao nhất**, cho thấy khả năng cân bằng hiệu quả giữa các nhãn tốt hơn.
- **LSTM** có kết quả ổn định nhưng nhìn chung thấp hơn hai mô hình còn lại.
- Nhìn tổng thể, **Transformer** mạnh về độ chính xác theo từng thuộc tính, trong khi **LSTM+GRU+CNN** có lợi thế hơn về hiệu quả phân loại tổng quát theo F1.
## 7. Cấu trúc thư mục


### Cấu trúc thư mục

  ```text
  DATAFLOW_2026_UET.EPOCH_0_USER_BEHAVIOR_PREDICTION/
  ├── .gitignore                                  # Khai báo file/thư mục bỏ qua khi commit
  ├── README.md                                   # Tài liệu mô tả dự án
  ├── requirements.txt                            # Danh sách thư viện Python
  ├── submission.csv                              # Kết quả dự đoán để nộp
  ├── submission_combine_seed42_final.csv         # Kết quả từ pipeline combine/ensemble
  │
  ├── data/                                       # Dữ liệu theo từng tầng xửlý
  │   ├── layer1_raw/                             # Dữ liệu gốc đã tách train/val/test
  │   │   ├── X_train.csv                         # Feature train
  │   │   ├── X_val.csv                           # Feature validation
  │   │   ├── X_test.csv                          # Feature test
  │   │   ├── Y_train.csv                         # Label train
  │   │   └── Y_val.csv                           # Label validation
  │   ├── layer2/                                 # Dữ liệu sau tiền xử lý +manual features
  │   │   ├── X_train.csv
  │   │   ├── X_val.csv
  │   │   ├── X_test.csv
  │   │   ├── Y_train.csv
  │   │   ├── Y_val.csv
  │   │   └── full_with_manual_features.csv       # Bảng dữ liệu đầy đủ kèm đặc trưng thủ công
  │   └── layer3_features/
  │       └── transformer/                        # Feature chuyên biệt cho Transformer
  │
  ├── models/                                     # Trọng số model và object đã huấn luyện
  │   ├── combine/
  │   │   ├── model_*_{lstm,gru,cnn}.keras        # Các model con dùng choensemble
  │   │   ├── encoder_attr_*.pkl                  # Encoder cho biến phân loại
  │   │   └── scaler_*.pkl                        # Scaler chuẩn hóa đặc trưng
  │   └── transformer/
  │       ├── transformer_fold_0.pt               # Checkpoint fold 0
  │       ├── transformer_fold_1.pt               # Checkpoint fold 1
  │       ├── transformer_fold_2.pt               # Checkpoint fold 2
  │       ├── transformer_fold_3.pt               # Checkpoint fold 3
  │       ├── transformer_fold_4.pt               # Checkpoint fold 4
  │       └── transformer_full.pt                 # Model huấn luyện trên toànbộ dữ liệu
  │
  ├── src/                                        # Mã nguồn chính
  │   ├── ai/
  │   │   ├── slm.py                              # Thành phần AI/LLM hỗ trợ
  │   │   └── translator.py                       # Dịch/chuẩn hóa dữ liệu text
  │   ├── data/
  │   │   ├── loaders.py                          # Hàm đọc và chuẩn bị dữ liệu
  │   │   ├── create_feature.py                   # Tạo đặc trưng thủ công
  │   │   └── build_transformer_features.py       # Tạo feature cho Transformer
  │   ├── metrics/
  │   │   └── metrics.py                          # Metric đánh giá mô hình
  │   ├── models/
  │   │   ├── transformer_model.py                # Định nghĩa kiến trúc
  Transformer
  │   │   └── losses.py                           # Custom loss functions
  │   └── training/
  │       ├── train_lstm.py                       # Huấn luyện LSTM
  │       ├── train_transformer.py                # Huấn luyện Transformer
  │       └── train_combine.py                    # Huấn luyện/kết hợp mô hìnhensemble
  │
  ├── scripts/                                    # Script chạy pipeline
  │   ├── pipeline_training.py                    # Pipeline huấn luyện end-to-end
  │   ├── pipeline_ai.py                          # Pipeline có hỗ trợ AI
  │   └── predict_test.py                         # Sinh dự đoán trên tập test
  │
  ├── notebooks/
  │   └── processing_data.ipynb                   # Notebook EDA + tiền xử lý thử nghiệm
  │
  ├── figures/                                    # Hình minh họa/biểu đồ phân tích
  └── report/
      └── ai_assistance_report.csv                # Báo cáo hỗ trợ AI trong quá trình làm
```
## 8. Cài đặt và chạy dự án 
**Yêu cầu hệ thống:**

Python 3.10 trở lên, tối thiểu 8gb RAM 

Các thư viện phụ thuộc trong file requirements.txt

### 1. Clone dự án
```bash
git clone https://github.com/quanai06/DATAFLOW_2026_UET.EPOCH_0_USER_BEHAVIOR_PREDICTION.git
cd DATAFLOW_2026_UET.EPOCH_0_USER_BEHAVIOR_PREDICTION
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv .venv
# Trên Linux/Macos
source .venv/bin/activate  
# Trên Windows: 
source .venv\Scripts\activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

##  Cách sử dụng

### Chạy toàn bộ pipeline 
Chạy lần lượt các file theo thứ tự sau:
Chú ý: Nếu huấn luyện lại tất cả model, thời gian xấp xỉ 3-4 tiếng.
```bash

# scripts train model (local)
python scripts/pipeline_training.py
# scripts test (load model)
python scripts/pipeline_test.py
# scripts predict submit kaggle
python scripts/make_v3_submission.py
# scripts ai
python scripts/pipeline_ai.py
```
## 9. Giới hạn và hướng phát triển
### Hạn chế

- Dữ liệu đầu vào chỉ gồm các chuỗi **Action ID ẩn danh**, không mang ngữ nghĩa nghiệp vụ trực tiếp, nên khả năng diễn giải sâu từng hành vi vẫn còn hạn chế.
- Một số **Action ID** có mức tương quan rất cao với nhãn, khiến mô hình có nguy cơ học theo các tín hiệu nổi bật thay vì học đầy đủ cấu trúc hành vi của toàn chuỗi.
- Các trường hợp **hiếm, bất thường hoặc phức tạp** vẫn khó xử lý triệt để và cần một lớp phân tích riêng để nhận diện tốt hơn.
- Hệ thống hiện chủ yếu được đánh giá trong bối cảnh huấn luyện và kiểm thử offline, nên chưa phản ánh đầy đủ các biến động khi triển khai thực tế theo thời gian thực.
- Giải pháp **ensemble đa mô hình** giúp cải thiện hiệu năng nhưng đồng thời làm tăng chi phí huấn luyện, suy luận và độ phức tạp khi triển khai.

### Hướng phát triển

- Xây dựng pipeline theo hướng **streaming-friendly** để cập nhật online các đặc trưng hành vi và hỗ trợ cơ chế cảnh báo sớm cho các phiên có rủi ro cao.
- Theo dõi sự thay đổi của dữ liệu theo thời gian nhằm tăng độ bền vững của mô hình khi hành vi người dùng hoặc quy trình nghiệp vụ thay đổi.
- Hoàn thiện lớp xử lý **edge-case** để nhận diện tốt hơn các chuỗi bất thường, hiếm gặp hoặc có dấu hiệu nhiễu.
- Nghiên cứu các cách giảm sự phụ thuộc của mô hình vào một số Action ID nổi trội, từ đó nâng cao khả năng tổng quát hóa.
- Tối ưu kiến trúc theo hướng nhẹ hơn, cân bằng giữa hiệu năng dự báo và khả năng triển khai thực tế.
## 👥 8. Tác giả và giấy phép

Dự án này được thực hiện bởi nhóm sinh viên từ trường [UET - VNU](https://uet.vnu.edu.vn) gồm 4 thành viên:

* **Lê Hoàng Quân** - Trưởng Nhóm
* **Vũ Hoàng Diệu Linh** 
* **Nguyễn Thị Hiền** 
* **Dương Trọng Nguyên** 

<div align="center">
  <p>Được phát triển bởi nhóm UET_EPOCH0</p>
  <p>Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội</p>
</div>
