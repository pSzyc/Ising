import numpy as np
import tensorflow as tf

def __get_neighbour_sum_matrix(mat):
    '''Matrix of sum of spin values for all neighbouring cells'''
    right = (1, 1)
    left = (-1, 1)
    up = (-1, 0)
    down = (1, 0)
    u_r = ((-1, 1), (0, 1))
    d_r = ((1, 1), (0, 1))
    u_l = ((-1, -1), (0, 1))
    d_l = ((1, -1), (0, 1))
    shift_axis_list = [right, left, u_r, d_r, u_l, d_l, up, down]
    return sum([np.roll(mat, shift=shift_axis[0], axis=shift_axis[1]) for shift_axis in shift_axis_list])

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
    matrix_sum = __get_neighbour_sum_matrix(mat)
    return - np.sum(np.multiply(matrix_sum, mat))

def calcMag(mat):
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag