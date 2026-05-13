# Dự đoán điểm cuối kỳ — CS523.Q21

> Dự đoán điểm cuối kỳ từ điểm giữa kỳ bằng mô hình Hồi quy tuyến tính
> được cài đặt thủ công qua Đồ thị tính toán (Computational Graph).

## Mô hình toán học

```math
\hat{y} = w \cdot x + b
```

```math
L = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2
```

## Chạy trên máy cục bộ

```bash
git clone https://github.com/Zadkiel-qvux/EndTermPrediction
cd EndTermPrediction
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc dự án

```text
model.py          — Hồi quy tuyến tính từ đầu (NumPy)
graph.py          — Vẽ Đồ thị tính toán (matplotlib)
app.py            — Ứng dụng web Streamlit
requirements.txt  — Danh sách thư viện
data/             — Bộ dữ liệu (TRAIN2.xlsx)
```

## Bộ dữ liệu

- 515 mẫu: điểm giữa kỳ → điểm cuối kỳ (thang điểm 0–10)
- Tỉ lệ chia: 80% huấn luyện / 20% kiểm tra

## Tài liệu tham khảo

- [GeeksforGeeks — Computational Graphs in Deep Learning](https://www.geeksforgeeks.org/computational-graphs-in-deep-learning/)
- CS523.Q21 slides
