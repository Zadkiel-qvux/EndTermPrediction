"""
app.py — Ứng dụng web Streamlit
Mục tiêu: Hiển thị kết quả huấn luyện Linear Regression với Computational Graph, cho phép người dùng điều chỉnh hyperparameters và trực quan hóa dữ liệu, đường hồi quy, loss curve, và graph tính toán.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import LinearRegressionCG, load_data
from graph import draw_graph

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán điểm cuối kỳ | CS523",
    layout="wide"
)

st.title("Dự đoán điểm cuối kỳ")
st.caption("CS523.Q11 — Linear Regression via Computational Graph")

# ── Sidebar controls ──────────────────────────────────────────────────
st.sidebar.header("Siêu tham số (Hyperparameters)")
lr     = st.sidebar.slider("Tốc độ học — Learning rate (η)", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
epochs = st.sidebar.slider("Số vòng lặp — Epochs", 100, 5000, 1000, step=100)
st.sidebar.markdown("---")
st.sidebar.markdown("**Bộ dữ liệu:** TRAIN2.xlsx  \n515 mẫu · thang điểm 0–10")

# ── Train ─────────────────────────────────────────────────────────────
@st.cache_data
def train(lr, epochs):
    (x_tr, y_tr), (x_te, y_te) = load_data("data/TRAIN2.xlsx")
    model = LinearRegressionCG(lr=lr)
    model.fit(x_tr, y_tr, epochs=epochs)
    metrics = model.evaluate(x_te, y_te)
    return model, x_tr, y_tr, x_te, y_te, metrics

model, x_tr, y_tr, x_te, y_te, metrics = train(lr, epochs)

# ── Metric cards ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trọng số w (slope)",       f"{model.w:.4f}")
c2.metric("Hệ số chặn b (intercept)", f"{model.b:.4f}")
c3.metric("MSE tập kiểm tra",         f"{metrics['mse']}")
c4.metric("R² tập kiểm tra",          f"{metrics['r2']}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Dự đoán", "Dữ liệu & Fit", "Loss Curve", "Computational Graph"
])

# Tab 1 — Prediction
with tab1:
    st.subheader("Dự đoán điểm cuối kỳ")
    midterm_input = st.slider("Nhập điểm giữa kỳ", 0.0, 10.0, 5.0, step=0.01)
    predicted = model.predict(np.array([midterm_input]))[0]
    predicted = float(np.clip(predicted, 0, 10))
    st.markdown(f"### Điểm cuối kỳ dự đoán: **{predicted:.2f} / 10**")
    st.progress(predicted / 10)

    st.markdown("**Công thức áp dụng:**")
    st.latex(r"\hat{y} = w \cdot x + b = " +
             f"{model.w:.4f} \\times {midterm_input:.2f} + {model.b:.4f} = {predicted:.4f}")

# Tab 2 — Scatter plot
with tab2:
    st.subheader("Phân bố dữ liệu & đường hồi quy")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(x_tr, y_tr, alpha=0.4, s=20, color="#1D9E75", label="Tập huấn luyện")
    ax2.scatter(x_te, y_te, alpha=0.6, s=20, color="#D85A30", label="Tập kiểm tra")
    x_line = np.linspace(0, 10, 200)
    ax2.plot(x_line, model.predict(x_line), color="#534AB7", lw=2,
             label=f"ŷ = {model.w:.3f}x + {model.b:.3f}")
    ax2.set_xlabel("Điểm giữa kỳ", fontsize=12)
    ax2.set_ylabel("Điểm cuối kỳ", fontsize=12)
    ax2.legend()
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

# Tab 3 — Loss curve
with tab3:
    st.subheader("Đường loss trong quá trình huấn luyện")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(model.history, color="#534AB7", lw=1.5)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("MSE Loss", fontsize=12)
    ax3.set_title("Gradient Descent Convergence")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("**Quy tắc cập nhật Gradient Descent:**")
    st.latex(r"w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}")
    st.latex(r"b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}")

# Tab 4 — Computational Graph
with tab4:
    st.subheader("Computational Graph — Forward & Backward Pass")
    st.markdown("""
    Chọn một mẫu để quan sát cách giá trị và gradient lan truyền qua đồ thị.
    - **Node xanh lá:** giá trị đầu vào
    - **Node tím:** node phép tính (×, +, −, ½e²)
    - **Node cam:** loss đầu ra L
    - **Nhãn trên:** giá trị Forward Pass
    - **Nhãn dưới:** gradient Backward Pass
    """)

    sample_idx = st.slider("Chỉ số mẫu (tập kiểm tra)", 0, len(x_te) - 1, 0)
    x_s = float(x_te[sample_idx])
    y_s = float(y_te[sample_idx])

    # Compute gradients for this single sample
    y_hat_s = model.w * x_s + model.b
    error_s  = y_hat_s - y_s
    dw_s = error_s * x_s
    db_s = error_s

    fig4 = draw_graph(model.w, model.b, x_s, y_s, dw_s, db_s)
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("**Công thức đạo hàm (Chain Rule):**")
    st.latex(r"\frac{\partial L}{\partial w} = \underbrace{(\hat{y} - y)}_{\text{error}} \cdot x")
    st.latex(r"\frac{\partial L}{\partial b} = (\hat{y} - y)")
