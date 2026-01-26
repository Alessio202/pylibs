import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

n_days = 100
seed = np.random.seed(67)

visite = np.round(np.random.normal(1500, 100, n_days)).astype(int)
pagine_sessione = np.round(np.random.normal(15, 2, n_days)).astype(int)
durata_sessione = np.round(np.random.normal(180, 20, n_days)).astype(int)
bounce_rate = np.round(np.random.normal(50, 10, n_days)).astype(int)


visite = np.clip(visite, 1200, 1800)
pagine_sessione = np.clip(pagine_sessione, 8, 22)
durata_sessione = np.clip(durata_sessione, 120, 240)
bounce_rate = np.clip(bounce_rate, 20, 80)


anomalies = [20, 50, 70]
visite[anomalies] = [2000, 900, 2100]             
pagine_sessione[anomalies] = [10, 22, 8]          
durata_sessione[anomalies] = [300, 90, 350]     
bounce_rate[anomalies] = [30, 68, 25] 
dataframe = pd.DataFrame({
    "visite": visite,
    "pagine_sessione": pagine_sessione,
    "durata_sessione_sec": durata_sessione,
    "bounce_rate": bounce_rate
})

input_dim = dataframe.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(input_dim)
])


X_train = dataframe.drop(index=anomalies).values
X_test = dataframe.values
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

model.compile(optimizer="adam", loss="mse", metrics=["mse"])
model.summary()
history=model.fit(X_train_scaled, X_train_scaled, epochs=50)
pred=model.predict(X_test_scaled)

plt.plot(history.history['loss'])
plt.ylabel("LOSS")
plt.xlabel("EPOCA")
plt.show()

mse = np.mean((X_test_scaled - pred)**2, axis=1)

mse_train = np.mean((X_train_scaled - model.predict(X_train_scaled))**2, axis=1)
#threshold = mse_train.mean() + 3 * mse_train.std() # deviazione 3 sigma
threshold = np.percentile(mse_train, 99)
print("Soglia:", np.round(np.sqrt(threshold), 2))

anomalies_detected = mse > threshold
anomaly_indices = np.where(anomalies_detected)[0]
print("Righe anomale:", anomaly_indices)

print("--Dettaglio--VISITE|PAGINE|DURATA|%USCITE")
for i in anomaly_indices[:]:  
    print(f"Riga {i}: input={scaler.inverse_transform([X_test_scaled[i]])[0].astype(int)}, MSE={np.round(np.sqrt(mse[i]),2)}")
