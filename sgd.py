import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

X, y = load_breast_cancer(return_X_y=True)
y_svm = y * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y_svm, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = Sequential([
    Dense(
        1,
        input_shape=(X_train_s.shape[1],),
        activation="linear",
        kernel_regularizer="l2",
    )
])
model.compile(optimizer=SGD(learning_rate=0.01), loss="hinge")

history = model.fit(
    X_train_s, y_train, epochs=100, batch_size=1, verbose=0
)

y_pred = np.where(model.predict(X_test_s, verbose=0).flatten() >= 0, 1, -1)
print("SGD Accuracy:", accuracy_score(y_test, y_pred))

plt.plot(history.history["loss"])
plt.title("SVM Loss - SGD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
