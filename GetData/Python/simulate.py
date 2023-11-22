import numpy as np
from numba import jit
import numpy.typing as npt
import os

def _createArray(L: int) -> npt.NDArray:
    mat = 2 * np.random.randint(0,2,size=(L,L), dtype=int) - 1
    return mat

@jit(nopython = True)
def _iterate(mat: npt.NDArray, T: float, H: float):
    """
    Parameters:
        mat: numpy array containing the current spin configuration
        T: Temperature for the simulation
        calcE: Flag to calculate and return the average spin energy for the final configuration.
    """
    L,_ = mat.shape
      
    order1 = np.arange(L)
    order2 = np.arange(L)
    np.random.shuffle(order1)
    np.random.shuffle(order2)

    for i in order1:
        for j in order2:
            spin_current = mat[i,j]
            spin_new = spin_current*(-1)
            neighbour_sum = mat[(i+1)%L,j]+mat[(i-1)%L,j]+mat[i,(j+1)%L]+mat[i,(j-1)%L]
            E_current = -spin_current * neighbour_sum - spin_current * H
            E_new = -spin_new*neighbour_sum - spin_new * H
            E_diff = E_new - E_current
            if E_diff < 0:
                mat[i,j] = spin_new
            elif np.random.random()<= np.exp(-float(E_diff)/T):
                mat[i,j] = spin_new

def get_neighbour_sum_matrix(mat: npt.NDArray) -> float:
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

def calcEnergy(mat: npt.NDArray) -> float:
    '''Energy of a given config uration'''
    matrix_sum = get_neighbour_sum_matrix(mat)
    return - np.sum(np.multiply(matrix_sum, mat))

def calcMag(mat: npt.NDArray) -> float:
    '''Magnetization of a given configuration'''
    mag = np.sum(mat)
    return mag

@jit(nopython = True)
def _iterateWolff(mat: npt.NDArray, T: float):
    # Wolff algorithm for a square lattice
    """
    Parameters:
        mat: numpy array containing the current spin configuration
        T: Temperature for the simulation
        calcE: Flag to calculate and return the average spin energy for the final configuration.
    """
    L,_ = mat.shape
    tracker = np.zeros((L,L)) # Keep track of which spins have already be added to the cluster.
    
    i,j = np.random.randint(0,L,size=2)
    spin = mat[i,j]
    stack = [(i,j)]
    tracker[i,j]=1
    
    cluster = [(i,j)]
    while len(stack)>0:
        i,j = stack.pop()
        neighbors = [(i,(j+1)%L),(i,(j-1)%L),((i+1)%L,j),((i-1)%L,j)]
        for pair in neighbors:
            l,m = pair
            if (mat[l,m]==spin and tracker[l,m]==0 and np.random.random()< (1.0-np.exp(-2.0/T))):
                cluster.append((l,m))
                stack.append((l,m))
                tracker[l,m]=1
            
    # flip cluster
    for pair in cluster:
        i,j=pair
        mat[i,j]*=-1

def iterate_numpy(m: npt.NDArray, T: float, H: float) -> npt.NDArray:
    '''
        Function with an error demonstrating why metropolis/wolff approach should be used.
    '''
    m_sum = get_neighbour_sum_matrix(m)
    e_diff = - (m_sum + H) * ( -2 * m)
    spin_flip_mask = np.random.rand(*m.shape)<np.where(e_diff>0,np.exp(-e_diff/T), 1.0) 
    m = m + m* (-2) *spin_flip_mask
    return m

def simulate(steps: int, L: int, T: float, H: float, output_file: os.PathLike, wolff_sim: bool, stats: bool):
    np.random.seed()
    mat = _createArray(L)
    if stats:
        stat_series = []

    if wolff_sim:
        for i in range(steps): 
            _iterateWolff(mat, T)
            if stats:
                get_stats(i, mat, stat_series)
    else: 
        for i in range(steps): 
            _iterate(mat, T, H)
            if stats:
                get_stats(i, mat, stat_series)
 

    output_file.mkdir(parents=True, exist_ok=True)
    np.save(output_file / "final.npy", mat == 1)
    if stats:
        np.savetxt(output_file / "data.csv", np.array(stat_series), delimiter=',')

def get_stats(index: int, mat: npt.NDArray, stat_series: list):
    energy = calcEnergy(mat)
    magnetization = calcMag(mat)
    stat_series.append((index, energy, magnetization))