"""
model.py — Hồi quy tuyến tính qua Đồ thị tính toán (Computational Graph)
"""

import numpy as np
import pandas as pd


def load_data(path: str = "data/TRAIN2.xlsx"):
    """Tải dữ liệu từ file Excel và chia tập train/test (80/20)."""
    df = pd.read_excel(path)
    df = df.dropna()

    x = df["midterm"].values.astype(float)
    y = df["final"].values.astype(float)

    # shuffle
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    split = int(0.8 * len(x))
    return (x[:split], y[:split]), (x[split:], y[split:])


class LinearRegressionCG:
    """
    Hồi quy tuyến tính được cài đặt dưới dạng Đồ thị tính toán.

    Forward Pass:  y_hat = w * x + b
    Hàm mất mát:   L = mean( 0.5 * (y_hat - y)^2 )
    Backward Pass: dL/dw, dL/db theo quy tắc chain rule
    Cập nhật:      gradient descent
    """

    def __init__(self, lr: float = 0.01):
        self.w = 0.0          # weight — trọng số (slope / hệ số góc)
        self.b = 0.0          # bias — hệ số chặn (intercept)
        self.lr = lr          # learning rate — tốc độ học (η)
        self.history = []     # loss history — lịch sử loss qua từng epoch

    # ------------------------------------------------------------------
    # FORWARD PASS — traverse graph input → output
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Node nhân và cộng: y_hat = w * x + b"""
        return self.w * x + self.b

    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Node hàm mất mát: L = mean(0.5 * (y_hat - y)^2)"""
        return float(np.mean(0.5 * (y_hat - y) ** 2))

    # ------------------------------------------------------------------
    # BACKWARD PASS — chain rule, traverse graph output → input
    # ------------------------------------------------------------------
    def backward(self, x: np.ndarray, y_hat: np.ndarray, y: np.ndarray):
        """
        Tính gradient theo quy tắc chain rule:
            dL/dy_hat = (y_hat - y)             [gradient cục bộ tại node loss]
            dL/dw     = mean(dL/dy_hat * x)     [lan truyền qua node nhân]
            dL/db     = mean(dL/dy_hat * 1)     [lan truyền qua node cộng]
        """
        error = y_hat - y                        # error — sai số dự đoán (∂L/∂ŷ)
        dw = np.mean(error * x)                  # dw — gradient của loss theo weight (∂L/∂w)
        db = np.mean(error)                      # db — gradient của loss theo bias (∂L/∂b)
        return dw, db

    # ------------------------------------------------------------------
    # GRADIENT DESCENT UPDATE
    # ------------------------------------------------------------------
    def update(self, dw: float, db: float):
        self.w -= self.lr * dw
        self.b -= self.lr * db

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000):
        """Vòng lặp huấn luyện đầy đủ: forward → loss → backward → update."""
        for epoch in range(epochs):
            y_hat = self.forward(x)
            L = self.loss(y_hat, y)
            dw, db = self.backward(x, y_hat, y)
            self.update(dw, db)
            self.history.append(L)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {L:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        y_hat  = self.predict(x)                         # y_hat — điểm cuối kỳ dự đoán
        mse    = np.mean((y_hat - y) ** 2)               # mse — Mean Squared Error (sai số bình phương trung bình)
        ss_res = np.sum((y - y_hat) ** 2)                # ss_res — Sum of Squared Residuals (tổng bình phương phần dư)
        ss_tot = np.sum((y - np.mean(y)) ** 2)           # ss_tot — Sum of Squares Total (tổng phương sai so với đường trung bình)
        r2     = 1 - ss_res / ss_tot                     # r2 — R² coefficient of determination (hệ số xác định)
        return {"mse": round(mse, 4), "rmse": round(mse ** 0.5, 4), "r2": round(r2, 4)}
