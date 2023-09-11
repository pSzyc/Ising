import numpy as np

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

def calcEnergy(mat):
    '''Energy of a given configuration'''
    energy = 0
    matrix_sum = __get_neighbour_sum_matrix(mat)
    for i in range(len(mat)):
        for j in range(len(mat)):
            spin = mat[i,j]
            energy += -matrix_sum[i,j] * spin
    return energy

def calcMag(mat):
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag