
import psi4
import numpy as np
import configparser
from scipy.linalg import block_diag
import scipy.linalg as la

config = configparser.ConfigParser()
config.read('Options1.ini') #pointing to our options.ini file for info on molecule
molecule = psi4.geometry(config['DEFAULT']['molecule'])
df_basis = psi4.core.BasisSet.build(molecule, 'DF_BASIS_MP2', config['DEFAULT']['df_basis'], puream=0)
#define zero basis to trick psi4 into making a 2 center or 3 center integral with mints.ao_eri later
zero = psi4.core.BasisSet.zero_ao_basis_set()


SCF_MAX_ITER = int(config['SCF']['max_iter'])
#get basis set
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
#set up integrals
mints = psi4.core.MintsHelper(basis)
molecule.update_geometry()

nalpha = int(config['DEFAULT']['nalpha'])
nbeta = int(config['DEFAULT']['nbeta'])
nocc = nalpha + nbeta
ntotal = 2*mints.basisset().nbf()



pqP = mints.ao_eri(basis, basis, zero, df_basis).to_array() 
def spin_blockpq(gao):
    Identity = np.eye(2)
    #np.kron is a tensor product. we are putting gao in the space of the 2x2 identity
    return np.kron(Identity, gao.T).T

print(np.shape(pqP))
pqP = spin_blockpq(pqP)
print(np.shape(pqP))


Int = mints.ao_eri().to_array()
def spin_block_tei(gao):
    Identity = np.eye(2) #2x2 identity matrix
    gao = np.kron(Identity, gao) #basically tensor product of 2x2 Identity with gao (first two dimensions only)
    return np.kron(Identity, gao.T) #transpose and tensor product again to spin block the 3rd and 4th dimensions

print(np.shape(Int))
Int = spin_block_tei(Int)
print(np.shape(Int))

