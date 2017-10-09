import psi4
import numpy as np
import molecule
import integrals
from scipy.linalg import block_diag

def spin_block_tei(gao):
    """
    Spin blocks two electron integral array
    """
    Identity = np.eye(2) #2x2 identity matrix
    gao = np.kron(Identity, gao) #basically tensor product of 2x2 Identity with gao
    return np.kron(Identity, gao.T) #transpose to make rows into columns

def spin_orbital(Ca, Cb, ea, eb, I):
    """
    Converts MO coefficients, orbital energies, two electron integrals to spin orbital form
    """
    #gather block diagonalized coefficent matrix to transform 2 e- integrals to MO basis later
    C_block  = block_diag(Ca, Cb)
    orb_energies = np.concatenate((ea,eb))
    C_block = C_block[:,orb_energies.argsort()]
    orb_energies = np.sort(orb_energies, None)
    I = spin_block_tei(I)
    I_phys = I.transpose(0,2,1,3) - I.transpose(0,2,3,1)
    return I_phys, C_block, orb_energies


