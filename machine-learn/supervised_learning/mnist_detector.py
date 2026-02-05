import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import datasets
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#Conv2D → ReLU → MaxPooling → Flatten → Dense → Softmax
model = tf.keras.Sequential([layers.Input(shape=(28,28,1)),
                             layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)),
                             layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
                             layers.Flatten(),
                             layers.Dense(10, activation="softmax")
                             ])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(x_train, y_train, epochs=1, batch_size=32)
img_test = Image.open("C:/Users/Academy/Downloads/aa/images/testtest.png").convert("L")
img_test = img_test = img_test.resize((28,28))
img_test = 1 - (np.array(img_test) / 255.0)
img_test = img_test.reshape((1, 28, 28, 1))
test = model.predict(img_test)
risultato = np.argmax(test)
print("RISULTATO: ", [0,1,2,3,4,5,6,7,8,9][risultato])

plt.plot(history.history['loss'])
plt.ylabel("LOSS")
plt.xlabel("EPOCH")

