import csv
import numpy as np
import tensorflow as tf
import time

feature_description = {
    'data': tf.io.FixedLenFeature([], tf.string), 
}

def load_npy_file(file_path):
    return np.load(file_path)

def load_numpy_file_wrapper(file_path):
    return tf.numpy_function(load_npy_file, [file_path], tf.bool)

def dataset_colab_pipeline(path, flatten=True, batch_size=1):
    data = 2 * np.load(path) - 1
    print(f"Got {len(data)} samples")
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if flatten:
        dataset = dataset.map(lambda x: tf.reshape(x, [batch_size, -1]), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: tf.expand_dims(x, 3), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def _parse_function(example_proto):
    spins = tf.io.parse_single_example(example_proto, feature_description)['data']
    spins = tf.io.parse_tensor(spins, tf.bool)
    return  spins


def process_dataset(flatten, batch_size, dataset):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if flatten:
        dataset = dataset.map(lambda x: tf.reshape(x, [batch_size, -1]), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: tf.expand_dims(x, 3), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def dataset_tfrecord_pipeline(path, flatten=True, batch_size=1):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = process_dataset(flatten, batch_size, dataset)
    return dataset

def dataset_pipeline(path, flatten=True, batch_size=1):
    dataset = tf.data.Dataset.list_files(f"{path}/*/*.npy")
    print(f"Got {len(dataset)} samples")
    dataset = dataset.map(load_numpy_file_wrapper)
    dataset = process_dataset(flatten, batch_size, dataset)
    return dataset


def get_param_dict(path):
    try:
        reader = csv.DictReader(open(f"{path}/parameters.csv"))
        par_dict = next(reader) 
    except:
        raise ValueError("Invalid folder provided")
    return par_dict

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    count = 0
    for _ in range(num_epochs):
        for sample in dataset:
            count+=sample.shape[0]
            # Performing a training step
            time.sleep(1E-10)
    tf.print("Number of examples: ",count)       
    tf.print("Execution time:", time.perf_counter() - start_time)