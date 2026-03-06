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
9. [Ứng dụng thực tiễn](#9-ứng-dụng-thực-tiễn)
10. [Giới hạn và hướng phát triển](#10-giới-hạn-và-hướng-phát-triển)
11. [Tác giả](#11-tác-giả)

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
