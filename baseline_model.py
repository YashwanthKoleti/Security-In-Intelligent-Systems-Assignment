import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import time
import seaborn as sns

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
datasets = tf.keras.datasets
callbacks = tf.keras.callbacks

NUM_CLASSES = 10
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 32
DATA_PATH = "archive"

def load_images(path,filename):
        filepath = os.path.join(path, filename)
        # Handle case where file might not exist or has different separator
        if not os.path.exists(filepath):
            filepath = filepath.replace('.', '-')
            
        with open(filepath, 'rb') as f:
            # Offset 16 bytes for header (magic number, count, rows, cols)
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)


def load_labels(path,filename):
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            filepath = filepath.replace('.', '-')
            
        with open(filepath, 'rb') as f:
            # Offset 8 bytes for header (magic number, count)
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

def load_local_mnist(path):
    x_train = load_images(path,'train-images.idx3-ubyte')
    y_train = load_labels(path,'train-labels.idx1-ubyte')
    x_test = load_images(path,'t10k-images.idx3-ubyte')
    y_test = load_labels(path,'t10k-labels.idx1-ubyte')
    return (x_train, y_train), (x_test, y_test)


def load_and_split_data():
    (x_full, y_full), (x_test, y_test) = load_local_mnist(DATA_PATH)
    x_full = x_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_full = np.expand_dims(x_full, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=0.1, random_state=42
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_model():
    model = models.Sequential([
        layers.Input(shape=IMG_SHAPE),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()
model = create_model()
            
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True,verbose=1)
            
history = model.fit(
                x_train, y_train,
                epochs=10, 
                batch_size=BATCH_SIZE,
                validation_data=(x_val, y_val),
                callbacks=[early_stop]
            )
            
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

start = time.time()
y_pred = model.predict(x_test)
end = time.time()

total_time = end - start
per_sample = total_time / len(x_test)

y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.savefig("confusion_matrix_baseline.png", dpi=300, bbox_inches="tight")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print('\n')
print('-'*50)
print(f"Training Complete")
print(f"Stopped at Epoch {len(history.history['loss'])}")
print(f"Training Accuracy {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy {test_acc:.4f}")
print("Per-sample inference time:", per_sample, "seconds")

model.save("baseline_model.h5")

