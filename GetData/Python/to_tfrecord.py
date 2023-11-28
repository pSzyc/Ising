import tensorflow as tf
import numpy as np
import os
import sys

def array_to_example(data, output):
    # Convert boolean array to int array
    data = data.astype(np.int64)
    
    feature = {
        'data': tf.train.Feature(int64_list=tf.train.Int64List(value=data.flatten()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    with tf.io.TFRecordWriter(output) as writer:
        writer.write(example.SerializeToString())


input_folder = sys.argv[1]
output = sys.argv[2]

for dir_path, folders, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".npy"):
          array_to_example(np.load(os.path.join(dir_path, file)), output=output)