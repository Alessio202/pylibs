import json
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras import Model
import numpy as np


def x_y_giver():
    x = []
    y = []
    with open("dati.json", "r") as file:
        data = json.load(file)

    print("Conteggio frasi per lingua: ")
    for lang, sentences in data.items():
        print(f"{lang}: {len(sentences)} frasi")
        x.extend(sentences)
        y.extend([lang] * len(sentences))
    print(f"Totale frasi: {len(x)}\n")

    return np.array(x), np.array(y)



def char_tokenize(text, char2idx, maxlen=60):
    seq = [char2idx.get(c, char2idx["<UNK>"]) for c in text]

    if len(seq) < maxlen:
        seq += [char2idx["<PAD>"]] * (maxlen - len(seq))
    else:
        seq = seq[:maxlen]

    return np.array(seq)


def build_char_vocab(texts):
    chars = set()
    for t in texts:
        for c in t:
            chars.add(c)

    char2idx = {c: i + 2 for i, c in enumerate(sorted(chars))}
    char2idx["<PAD>"] = 0
    char2idx["<UNK>"] = 1
    return char2idx


class MyModule(Model):
    def __init__(self, vocab_size, embed_dim=64, rnn_units=32, num_classes=5):
        super().__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.gru = GRU(rnn_units)
        self.dense = Dense(num_classes, activation="softmax")
        self.char2idx = None

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.gru(x)
        x = self.dense(x)
        return x

    def char_embeddings(self, texts):
        return np.array([char_tokenize(t, self.char2idx) for t in texts])

    def train_lan(self, x, y):
        X = self.char_embeddings(x)

        labels_map = {
            "italiano": 0,
            "francese": 1,
            "inglese": 2,
            "spagnolo": 3,
            "tedesco": 4
        }

        y_num = np.array([labels_map[i] for i in y])

        return self.fit(
            X,
            y_num,
            epochs=25,
            batch_size=8,
            shuffle=True
        )

    def predict_sentence(self, sentence):
        X = self.char_embeddings([sentence])
        return self.predict(X)


if __name__ == "__main__":
    x, y = x_y_giver()
    char2idx = build_char_vocab(x)
    vocab_size = len(char2idx)

    model = MyModule(vocab_size=vocab_size, embed_dim=64, rnn_units=32, num_classes=5)
    model.char2idx = char2idx

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.train_lan(x, y)

    pred = model.predict_sentence("La mélancolie diffuse de la scène évoquait une nostalgie indéfinissable")
    np.set_printoptions(precision=4, suppress=True)

    idx2label = {
        0: "italiano",
        1: "francese",
        2: "inglese",
        3: "spagnolo",
        4: "tedesco"
    }

    pred_idx = np.argmax(pred)
    confidence = pred[0][pred_idx]

    print("Lingua predetta:", idx2label[pred_idx])
    print("Confidenza:", round(confidence, 3))

    plt.plot(history.history['loss'])
    plt.ylabel("LOSS -> ALTO = MALE")
    plt.xlabel("EPOCA")
    plt.show()
