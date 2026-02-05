from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt

X, y = make_blobs(n_samples=300, centers=3, n_features=3,
                  random_state=67)

scaler = StandardScaler()
minmax = MinMaxScaler()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_minmaxxed = minmax.fit_transform(X_train)
X_test_minmaxxed = minmax.transform(X_test)

fig, axs = plt.subplots(2, 2, figsize=(10,8))

axs[0,0].scatter(X_train_scaled[:,0], X_train_scaled[:,1], cmap='Set1', c=y_train)
axs[0,0].set_title("Train StandardScaler")

axs[0,1].scatter(X_train_minmaxxed[:,0], X_train_minmaxxed[:,1], cmap='Set1', c=y_train)
axs[0,1].set_title("Train MinMaxScaler")

axs[1,0].scatter(X_test_scaled[:,0], X_test_scaled[:,1],cmap='plasma',c=y_test) # alpha per opacità
axs[1,0].set_title("Test StandardScaler")

axs[1,1].scatter(X_test_minmaxxed[:,0], X_test_minmaxxed[:,1], cmap='plasma',c=y_test)
axs[1,1].set_title("Test MinMaxScaler")

#plt.tight_layout()
plt.show()
