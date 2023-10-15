import numpy as np

def __createArray(L):
    mat = 2 * np.random.randint(0,2,size=(L,L), dtype=int) - 1
    return mat

def __iterate(mat,T, H):
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
            neighbour_sum = mat[(i+1)%L,j]+mat[(i-1)%L,j]+mat[i,(j+1)%L]+mat[i,(j-1)%L]+mat[(i+1)%L,(j+1)%L]+mat[(i-1)%L,(j-1)%L]
            E_current = -spin_current*neighbour_sum - spin_current * H
            E_new = -spin_new*neighbour_sum - spin_new * H
            E_diff = E_new - E_current
            if E_diff < 0:
                mat[i,j] = spin_new
            elif np.random.random()<= np.exp(-float(E_diff)/T):
                mat[i,j] = spin_new


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

def iterate_numpy(m, T, H):
    m_sum = get_neighbour_sum_matrix(m)
    e_diff = - (m_sum + H) * ( -2 * m)
    spin_flip_mask = np.random.rand(*m.shape)<np.where(e_diff>0,np.exp(-e_diff/T), 1.0) 
    m = m + m*(-2)*spin_flip_mask
    return m

def simulate(steps, L, T, H, output_file, numpy_sim):
    np.random.seed()
    mat = __createArray(L)
    sim_fun = iterate_numpy if numpy_sim else __iterate
    for _ in range(steps): sim_fun(mat, T, H)
    np.save(output_file, mat)
