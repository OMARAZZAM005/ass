from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score 

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

X,Y=load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)


model = Sequential([Dense(1, input_shape=(X_train.shape[1],))])
model.compile(optimizer=SGD(learning_rate=0.1), loss='mse')
model.fit(X_train_s, y_train, epochs=100, batch_size=len(X_train_s), verbose=0)
lin_preds = model.predict(X_test_s, verbose=0).flatten()
voting_preds = (rf_preds + lin_preds) / 2
print(f"Voting Accuracy (GD): {r2_score(y_test, voting_preds) * 100:.2f}%")

