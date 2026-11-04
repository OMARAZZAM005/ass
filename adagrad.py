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
lr = 0.5  # معدل تعلم ابتدائي أكبر لأن Adagrad يقلله تلقائياً مع الوقت
epochs = 50

# دالة بسيطة لطباعة النتيجة
def print_metrics(name, w):
  preds = X @ w
  loss = np.mean((preds - y) ** 2)
  acc = r2_score(y, preds) * 100
  print(f"[{name}] -> Loss: {loss:.4f} | Accuracy: {acc:.2f}%")


w = np.zeros((n, 1))
G = np.zeros((n, 1))  # مجموع مربعات الجرادينت السابقة
eps = 1e-8

for epoch in range(epochs):
  grad = (2 / m) * X.T @ (X @ w - y)

  # مراكمة مربع الجرادينت وتحديث الأوزان
  G += grad ** 2
  w -= (lr / (np.sqrt(G) + eps)) * grad

print_metrics("Adagrad", w)
