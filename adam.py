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
lr = 0.1
epochs = 50

# دالة بسيطة لطباعة النتيجة
def print_metrics(name, w):
  preds = X @ w
  loss = np.mean((preds - y) ** 2)
  acc = r2_score(y, preds) * 100
  print(f"[{name}] -> Loss: {loss:.4f} | Accuracy: {acc:.2f}%")


w = np.zeros((n, 1))
m_w = np.zeros((n, 1))  # العزم الأول (متوسط الجرادينت)
v_w = np.zeros((n, 1))  # العزم الثاني (متوسط مربع الجرادينت)
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

for t in range(1, epochs + 1):
  grad = (2 / m) * X.T @ (X @ w - y)

  # تحديث العزم الأول والثاني
  m_w = beta1 * m_w + (1 - beta1) * grad
  v_w = beta2 * v_w + (1 - beta2) * (grad ** 2)

  # تصحيح التحيز (Bias Correction)
  m_hat = m_w / (1 - beta1 ** t)
  v_hat = v_w / (1 - beta2 ** t)

  # تحديث الأوزان
  w -= lr * m_hat / (np.sqrt(v_hat) + eps)

print_metrics("Adam", w)
