import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

tf.random.set_seed(42)

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_full = tf.expand_dims(x_train_full, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

val_split = 0.1
n_val = int(len(x_train_full) * val_split)

x_val, y_val = x_train_full[:n_val], y_train_full[:n_val]
x_train, y_train = x_train_full[n_val:], y_train_full[n_val:]

print(f"Treino: {x_train.shape[0]} amostras | Validação: {x_val.shape[0]} amostras | Teste: {x_test.shape[0]} amostras")

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=2,
)

val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"\nAcurácia final de validação: {val_accuracy:.4f}")
print(f"Perda final de validação: {val_loss:.4f}")

model.save("model.h5")
print("Modelo salvo em model.h5")