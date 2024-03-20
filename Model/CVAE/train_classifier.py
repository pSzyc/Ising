import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import time
from pathlib import Path
data_path = Path("../../GetData/Python/Data")
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from input_pipeline import dataset_tfrecord_pipeline
from tensorflow.keras.callbacks import EarlyStopping

class DataIterator:
    def __init__(self, datasets):
        self.datasets = datasets

    def __iter__(self):
        self.data_iterators = [iter(data) for data in self.datasets]
        return self

    def __next__(self):
        data_list = []
        temp_list = []
        for index, data_iterator in enumerate(self.data_iterators):
            data = next(data_iterator)
            data_list.append(data)
            temp = np.zeros((data.shape[0], len(self.data_iterators)))
            temp[:, index] = 1
            temp_list.append(temp)
        temps = tf.concat(temp_list, axis=0)
        data = tf.concat(data_list, axis=0)
        return data, temps

def make_dataset(data_dir, temps, batch_size=100, flatten=False):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    assert batch_size % len(temps) == 0, "Batch size must be divisible by the number of temperatures"

    trainset = []
    testset = []
    for temp in temps:
        dataset = dataset_tfrecord_pipeline(data_dir / f"Data{temp:.1f}.tfrecord", flatten=flatten, batch_size=batch_size // len(temps))
        trainset.append(dataset)
        dataset = dataset_tfrecord_pipeline(data_dir / f"TestData{temp:.1f}.tfrecord", flatten=flatten, batch_size=batch_size // len(temps))
        testset.append(dataset)

    gen_test = DataIterator(testset)
    test_dataset = tf.data.Dataset.from_generator(lambda: gen_test, output_signature = (tf.TensorSpec(shape=(None, 32, 32, 1), dtype=tf.float32), tf.TensorSpec(shape=(None, len(temps)), dtype=tf.float32)))
    gen_train = DataIterator(trainset)
    train_dataset = tf.data.Dataset.from_generator(lambda: gen_train, output_signature = (tf.TensorSpec(shape=(None, 32, 32, 1), dtype=tf.float32), tf.TensorSpec(shape=(None, len(temps)), dtype=tf.float32)))
    return train_dataset, test_dataset

def create_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ])
    return model

def plot(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/results.png")

temps = [1.5, 2.0, 2.5, 2.7, 3.0, 4.0]
batch_size = 120
trainset, testset = make_dataset(data_path, temps, batch_size=batch_size, flatten=False)

classifier = create_model(len(temps))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Add the early stopping callback to the fit function
history = classifier.fit(trainset, epochs=50, validation_data=testset, callbacks=[early_stopping])

classifier.save("results/classifier")
plot(history)