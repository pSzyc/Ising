import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from pathlib import Path
import os
os.path.insert("..")
from Plots import style
    
def get_list_of_files(folder_path):
    par_dict = get_parameters_dict(folder_path)
    folder_path = Path(folder_path)
    n = int(par_dict['Simulatiton Number'])
    data_list = []
    for i in range(1, n+1):
        data_list.append(folder_path / f"output{i}"/ "final.npy")
    return np.array(data_list)

def get_heat_capacity(df, t):
    return (1 / t )** 2 * df['energy'].var()
    
def get_magnetic_susceptibility(df, t):
    return df['magnetisation'].var() / t

def get_stat_df(files, L = 32):
    df = pd.DataFrame(files, columns=['file'])
    df['data'] = df['file'].apply(lambda x: 2 * np.load(x) - 1)
    df['energy'] = df['data'].apply(calcEnergy)
    df['magnetisation'] = df['data'].apply(calcMag)
    return df

def validation_stats(filename, t):
    files = get_list_of_files(filename)
    df = get_stat_df(files)
    heat_capacity =  get_heat_capacity(df, t)
    magnetic_susceptibility = get_magnetic_susceptibility(df, t)
    return heat_capacity, magnetic_susceptibility

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

def calcEnergy(mat):
    '''Energy of a given configuration'''
    matrix_sum = get_neighbour_sum_matrix(mat)
    return - np.sum(np.multiply(matrix_sum, mat)/2)

def calcMag(mat):
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag

def get_parameters_dict(folder_path):
    try:
        folder_path = Path(folder_path)
        reader = csv.DictReader(open(folder_path / "parameters.csv"))
        par_dict = next(reader)
        return par_dict
    except:
        raise ValueError("Invalid folder provided")

def get_list_of_files(folder_path):
    par_dict = get_parameters_dict(folder_path)
    folder_path = Path(folder_path)
    n = int(par_dict['Simulatiton Number'])
    data_list = []
    for i in range(1, n+1):
        data_list.append(folder_path / f"output{i}"/ "final.npy")
    return np.array(data_list)
    
def time_series(folder_path):
    par_dict = get_parameters_dict(folder_path)
    folder_path = Path(folder_path)
    n_rows = 3

    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
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
        
        if par_dict['Wolff'] == 'True':
            axs[2].plot(df.mag.abs())
            axs[2].set_title(f'Absolute magnetization')
        else:
            axs[2].plot(df.mag)
            axs[2].set_title(f'Magnetization')
    plt.plot()



def get_distribution_data(folder):
    par_dict = get_parameters_dict(folder)
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