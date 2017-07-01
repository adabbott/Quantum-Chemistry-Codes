import psi4
import numpy as np
import configparser
from scipy.linalg import block_diag
import scipy.linalg as la
import time

config = configparser.ConfigParser()
config.read('Options.ini') #pointing to our options.ini file for info on molecule
molecule = psi4.geometry(config['DEFAULT']['molecule'])
df_basis = psi4.core.BasisSet.build(molecule, 'DF_BASIS_MP2', config['DEFAULT']['df_basis'], puream=0)
#define zero basis to trick psi4 into making a 2 center or 3 center integral with mints.ao_eri later
zero = psi4.core.BasisSet.zero_ao_basis_set()


SCF_MAX_ITER = int(config['SCF']['max_iter'])
CEPA0_MAX_ITER = int(config['CEPA0']['max_iter'])
#get basis set
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
#set up integrals
mints = psi4.core.MintsHelper(basis)
molecule.update_geometry()

nalpha = int(config['DEFAULT']['nalpha'])
nbeta = int(config['DEFAULT']['nbeta'])
nocc = nalpha + nbeta
ntotal = 2*mints.basisset().nbf()
nvirt = ntotal-nocc

# Overlap integrals 
I = mints.ao_eri().to_array()



print(I)

