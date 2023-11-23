import numpy as np
#import tensorflow as tf
import pandas as pd
import csv
from matplotlib import pyplot as plt
from pathlib import Path

def get_neighbour_sum_matrix(mat):
    """Matrix of the sum of spin values for all neighboring cells."""
    # Define shifts for different directions
    shifts = [
        (0, 1),  # Right
        (0, -1),  # Left
        (-1, 0),  # Up
        (1, 0),  # Down
    ]

    # Initialize an empty matrix for the sum of neighboring cells
    neighbor_sum = np.zeros_like(mat)

    # Iterate through each shift and accumulate the values in neighbor_sum
    for shift in shifts:
        shifted_mat = np.roll(mat, shift=shift, axis=(0, 1))
        neighbor_sum += shifted_mat

    return neighbor_sum


#def __get_neighb_filter(shape):
#    kernel = np.array([[1.0, 1.0, 1.0], 
#       [1.0, 0.0, 1.0], 
#       [1.0, 1.0, 1.0]]) 
#    conv = tf.keras.layers.Conv2D(1, (3, 3), 
#        activation='linear',
#        input_shape=shape,
#        padding="same", 
#        kernel_initializer = tf.keras.initializers.Constant(kernel), use_bias=False
#    ) 
#    return conv
#
#def get_neighbour_sum_matrix_conv(mat):
#    conv_filter = __get_neighb_filter(mat.shape)
#    sum_mat = conv_filter(mat.reshape(1,mat.shape[0],mat.shape[1], 1).astype(np.float32))
#    return sum_mat[0,:,:,0]


def calcEnergy(mat):
    '''Energy of a given configuration'''
    matrix_sum = get_neighbour_sum_matrix(mat)
    return - np.sum(np.multiply(matrix_sum, mat)/2)

def calcMag(mat):
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag


def time_series(folder):
    try:
        folder_path = Path(folder)
        reader = csv.DictReader(open(folder_path / "parameters.csv"))
        par_dict = next(reader) 
    except:
        raise ValueError("Invalid folder provided")

    n_rows = int(par_dict['Simulatiton Number'])

    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    fig.suptitle(f"Temperature: {par_dict['Temperature']} External Field: {par_dict['Magnetic Field']}", fontsize=18)

    subfigs = fig.subfigures(nrows=n_rows, ncols=1)

    for index, subfig in enumerate(subfigs):
        subfig.suptitle(f'Simulation number: {index}')
        data = 2 *  np.load(folder_path / f"output{index+1}" / "final.npy") - 1

        axs = subfig.subplots(nrows=1, ncols=3)

        axs[0].imshow(data)
        axs[0].set_title(f'Maze')

        columns = ['iter','energy','mag']
        df = pd.read_csv(folder_path / f"output{index+1}" / "data.csv", header=None, names=columns).set_index('iter')

        axs[1].plot(df.energy)
        axs[1].set_title(f'Energy')

        axs[2].plot(df.mag)
        axs[2].set_title(f'Magnetization')



def get_distribution_data(folder):
    try:
        folder_path = Path(folder)
        reader = csv.DictReader(open(folder_path / "parameters.csv"))
        par_dict = next(reader) 
    except:
        raise ValueError("Invalid folder provided")
    print("File parameters:")
    print(par_dict)

    folders = [f"{folder}/output{i}/final.npy" for i in range(1, int(par_dict['Simulatiton Number']) + 1)]
    df = pd.DataFrame(folders, columns =['folder'])
    df['data'] = df['folder'].apply(lambda x: 2 * np.load(x) - 1)
    df['Magnetization'] = df['data'].apply(calcMag)
    df['Energy'] = df['data'].apply(calcEnergy)
    df['Steps'] = int(par_dict['Steps'])
    df['Algorithm'] = "wolff" if par_dict['Wolff'] == "True" else "metropolis"
    return df.drop(columns=['data'])