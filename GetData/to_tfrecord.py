import tensorflow as tf
import numpy as np
import os
import sys

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def array_to_example(data):
    data = _bytes_feature(tf.io.serialize_tensor(data))
    feature = {
        'data': data
    }    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

input_folder = sys.argv[1]
output = sys.argv[2]
with tf.io.TFRecordWriter(output) as writer:
    for dir_path, folders, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".npy"):
                example = array_to_example(np.load(os.path.join(dir_path, file)))
                writer.write(example.SerializeToString())