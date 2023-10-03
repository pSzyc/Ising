import numpy as np
import tensorflow as tf

def get_neighbour_sum_matrix(mat):
    """Matrix of the sum of spin values for all neighboring cells."""
    # Define shifts for different directions
    shifts = [
        (1, 1),  # Right
        (-1, 1),  # Left
        (-1, 0),  # Up
        (1, 0),  # Down
        (-1, 1),  # Up-Right
        (1, 1),  # Down-Right
        (-1, -1),  # Up-Left
        (1, -1),  # Down-Left
    ]

    # Initialize an empty matrix for the sum of neighboring cells
    neighbor_sum = np.zeros_like(mat)

    # Iterate through each shift and accumulate the values in neighbor_sum
    for shift in shifts:
        shifted_mat = np.roll(mat, shift=shift, axis=(0, 1))
        neighbor_sum += shifted_mat

    return neighbor_sum


def __get_neighb_filter(shape):
    kernel = np.array([[1.0, 1.0, 1.0], 
       [1.0, 0.0, 1.0], 
       [1.0, 1.0, 1.0]]) 
    conv = tf.keras.layers.Conv2D(1, (3, 3), 
        activation='linear',
        input_shape=shape,
        padding="same", 
        kernel_initializer = tf.keras.initializers.Constant(kernel), use_bias=False
    ) 
    return conv

def get_neighbour_sum_matrix_conv(mat):
    conv_filter = __get_neighb_filter(mat.shape)
    sum_mat = conv_filter(mat.reshape(1,mat.shape[0],mat.shape[1], 1).astype(np.float32))
    return sum_mat[0,:,:,0]


def calcEnergy(mat):
    '''Energy of a given configuration'''
    matrix_sum = get_neighbour_sum_matrix(mat)
    return - np.sum(np.multiply(matrix_sum, mat))

def calcMag(mat):
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag