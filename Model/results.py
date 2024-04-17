import numpy as np
import matplotlib.pyplot as plt
import contextlib
from PIL import Image
import glob
import pandas as pd
import seaborn as sns
import sys
sys.path.insert(0, __file__ + '/..')
from Plots.style import *

def make_gif(results, temp):
  fp_in = results / f"Images/*.png"
  fp_out = results /  f"train_{temp}.gif"

  # use exit stack to automatically close opened images
  with contextlib.ExitStack() as stack:

      # lazily load images
      imgs = (stack.enter_context(Image.open(f))
              for f in sorted(glob.glob(str(fp_in))))

      # extract  first image from iterator
      img = next(imgs)

      # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
      img.save(fp=str(fp_out), format='GIF', append_images=imgs,
              save_all=True, duration=1000, loop=0)

def calcEnergy(mat):
    '''Energy of a given configuration'''
    matrix_sum = get_neighbour_sum_matrix(mat)
    return - np.sum(np.multiply(matrix_sum, mat))

def calcMag(mat):
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag

def get_heat_capacity(df, t):
    return (1 / t )** 2 * df['Energy'].var()
    
def get_magnetic_susceptibility(df, t):
    return df['Magnetization'].var() / t

def get_stats(df, t):
    return [df['Magnetization'].abs().mean(), df['Energy'].mean(), get_magnetic_susceptibility(df, t), get_heat_capacity(df, t)]

def get_df(data):
    df = pd.DataFrame({'image': list(data)})
    df['Magnetization'] = df['image'].apply(calcMag) / 32 ** 2
    df['Energy'] = df['image'].apply(calcEnergy)
    df.drop(columns='image', inplace=True)
    return df

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

def cost_function_plot(loss_list, results):
  plt.figure()
  plt.plot(loss_list)
  plt.xlabel("Epoch")
  plt.ylabel("Cost function")
  plt.savefig(results / "loss_function.png")

def final_plot(df):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=0.92)
    for i, column in enumerate(['Magnetization', 'Energy', 'Magnetic susceptibility', 'Heat Capacity']):
        plt.subplot(2, 2, i + 1)
        sns.scatterplot(data=df, x='Temperature', y=column, hue='Method', style='Method', markers=['o', 's', 'v'], alpha=0.5)

def comparision_plot(data_list):
    df = pd.DataFrame(data_list, columns=['Magnetization', 'Energy', 'Magnetic susceptibility', 'Heat Capacity', 'Temperature', 'Method'])
    df.set_index(keys='Temperature', inplace=True)
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['Magnetization', 'Energy', 'Magnetic susceptibility', 'Heat Capacity']):
        plt.subplot(2, 2, i + 1)
        data = df[df['Method'] == 'Monte Carlo'][column] / df[df['Method'] != 'Monte Carlo'][column]
        temps = df.index.unique()
        plt.scatter(temps, data)
        plt.legend('MC/ML')