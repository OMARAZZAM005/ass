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
lr = 0.001  # معدل تعلم أصغر لتفادي التذبذب الحاد عند التحديث بعينة واحدة
epochs = 50

# دالة بسيطة لطباعة النتيجة
def print_metrics(name, w):
  preds = X @ w
  loss = np.mean((preds - y) ** 2)
  acc = r2_score(y, preds) * 100
  print(f"[{name}] -> Loss: {loss:.4f} | Accuracy: {acc:.2f}%")


np.random.seed(42)
w = np.zeros((n, 1))

for epoch in range(epochs):
  # خلط ترتيب العينات في كل Epoch
  indices = np.random.permutation(m)

  # حساب الجرادينت وتحديث الأوزان لعينة واحدة فقط في كل خطوة
  for i in indices:
    xi = X[i:i + 1]
    yi = y[i:i + 1]
    grad = 2 * xi.T @ (xi @ w - yi)
    w -= lr * grad

print_metrics("SGD", w)
