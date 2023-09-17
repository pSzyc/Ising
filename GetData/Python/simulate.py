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

    for i in range(L):
        for j in range(L):
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


def simulate(steps, L, T, H, output_file):
    np.random.seed()
    mat = __createArray(L)
    for _ in range(steps):
        __iterate(mat, T, H)
    np.save(output_file, mat)
