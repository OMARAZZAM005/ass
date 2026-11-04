import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


df = fetch_california_housing(as_frame=True).frame.sample(300, random_state=42)
X_raw = df[['MedInc']].values
y = df[['MedHouseVal']].values

# 2. عمل Polynomial وتجهيز المدخلات
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X_raw)
X = StandardScaler().fit_transform(X)
X[:, 0] = 1.0  # Bias column

m, n = X.shape
lr = 0.01
epochs = 50
batch_size = 32

# دالة بسيطة لطباعة النتيجة
def print_metrics(name, w):
  preds = X @ w
  loss = np.mean((preds - y) ** 2)
  acc = r2_score(y, preds) * 100
  print(f"[{name}] -> Loss: {loss:.4f} | Accuracy: {acc:.2f}%")


np.random.seed(42)
w = np.zeros((n, 1))

for epoch in range(epochs):
  # خلط البيانات في بداية كل Epoch
  indices = np.random.permutation(m)
  X_shuffled = X[indices]
  y_shuffled = y[indices]

  # تقسيم البيانات إلى دفعات صغيرة (Mini-Batches)
  for start in range(0, m, batch_size):
    X_batch = X_shuffled[start:start + batch_size]
    y_batch = y_shuffled[start:start + batch_size]
    b_size = len(X_batch)

    grad = (2 / b_size) * X_batch.T @ (X_batch @ w - y_batch)
    w -= lr * grad

print_metrics("Mini-Batch SGD", w)
