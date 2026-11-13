from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
X,Y=load_iris(return_X_y=True)
X=X[:100]
Y=Y[:100]
scaler=StandardScaler()
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_Train=scaler.fit_transform(X_Train)
X_Test=scaler.fit_transform(X_Test)

model=Sequential([Dense(1,activation='sigmoid')])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(X_Train,Y_Train,epochs=50,batch_size=len(X_Train),verbose=0)
pre=model.predict(X_Test)

test_loss, test_acc = model.evaluate(X_Test, Y_Test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

