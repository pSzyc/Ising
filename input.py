import csv
import numpy as np
import tensorflow as tf

def load_npy_file(file_path):
    return np.load(file_path)

def load_numpy_file_wrapper(file_path):
    return tf.numpy_function(load_npy_file, [file_path], tf.int32)

def dataset_pipeline(path):
    print("Getting data from " + path)
    dataset = tf.data.Dataset.list_files(f"{path}/*/*.npy")
    print(f"Got {len(dataset)} samples")
    dataset = dataset.map(load_numpy_file_wrapper)
    dataset = dataset.map(lambda x: tf.reshape(x, [-1]))
    dataset = dataset.map(lambda x: tf.cast(x, tf.float32))
    return dataset

def get_param_dict(path):
    try:
        reader = csv.DictReader(open(f"{path}/parameters.csv"))
        par_dict = next(reader) 
    except:
        raise ValueError("Invalid folder provided")
    return par_dict