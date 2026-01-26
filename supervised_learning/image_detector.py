import json
import tensorflow as tf
from tensorflow import keras
from keras import layers, utils
from matplotlib import pyplot as plt
import numpy as np
import images
from keras.applications import MobileNetV2


def model_giver():
    base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3))

    base_model.trainable = False  # congeli il backbone
    return base_model

def data_augment():
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    return data_augmentation # serve a far finta di avere piu dati di quante ne hai veramente

x = []
y = []

with open("images/immagini.json", "r") as fileRead:
    data = json.load(fileRead)

# creo una mappatura da classi (stringhe) a interi
class_mapping = {cls: i for i, cls in enumerate(data.keys())}

for cls_name, imgs in data.items():
    cls_index = class_mapping[cls_name]  # intero per sparse_categorical_crossentropy
    for img_name in imgs:
        img_loaded = utils.load_img("images/" + img_name, target_size=(224, 224))
        img_array = utils.img_to_array(img_loaded)
        x.append(img_array)
        y.append(cls_index)

model = keras.Sequential([
    layers.Input(shape=(224,224,3)),  # per vedere nel model.summary()
    layers.Rescaling(scale=1./255.), # come normalizzare i pixel, in questo caso nel range di [0, 1]
    layers.Resizing(height=224, width=224),
    data_augment(),
    model_giver(),
    layers.GlobalAveragePooling2D(),
    # layers.Conv2D(filters=32, kernel_size=3, activation="relu"), # filters quante unità per analizzare immagini, kernel size grandezza finestra iniziale. NON QUA, modello già ha
    # layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)), # pool size di quanto riduce, tot finestra. stride di quanti pixel si muove. NON QUA
    # layers.Flatten(), non qua, rischi overfitting
    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
history = model.fit(np.array(x), np.array(y), epochs=5) #primoparam, input, secondoparam, classegiustaperinput,

test = utils.load_img("images/test.jpeg", target_size=(224, 224))
test = utils.img_to_array(test)
test = np.expand_dims(test, axis=0)
test = model.predict(test)
risultato = np.argmax(test)
inverse_class_mapping = {v: k for k, v in class_mapping.items()}
classe = inverse_class_mapping[risultato]
print("QUESTA E' L'IMMAGINE DI UN/UNA:", classe.upper())

plt.plot(history.history['loss'])
plt.ylabel("LOSS")
plt.xlabel("EPOCA")
plt.show()
