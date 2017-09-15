import psi4
import numpy as np
import configparser

#Read in molecule (geometry, atoms), number of e-, 
#number of basis functions (orbitals), occupied orbitals, and scf maxiter 
config = configparser.ConfigParser()
config.read('Options.ini')
molecule = psi4.geometry(config['DEFAULT']['molecule'])
molecule.update_geometry()
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
mints = psi4.core.MintsHelper(basis)
nalpha = int(config['DEFAULT']['nalpha'])
nbeta = int(config['DEFAULT']['nbeta'])
nocc = nalpha
ndocc = nbeta
nsocc = nocc-ndocc
SCF_MAX_ITER = int(config['SCF']['max_iter'])
Enuc = molecule.nuclear_repulsion_energy()

#Integrals
S = mints.ao_overlap().to_array() # Overlap Integrals 
T = mints.ao_kinetic().to_array() # Kinetic Energy Integrals
V = mints.ao_potential().to_array() #Potential Energy Integrals
I = mints.ao_eri().to_array() #Two-Electron Integrals

norb = mints.nbf()
#construct the orthogonalizer S^-1/2, 
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)  #diagonalize S matrix and take to negative one half power 
A = A.to_array()

#form the one-electron hamiltonian
H = T + V

D = np.zeros((norb,norb))

E_new = 1.0
v = 0
E_old = 0.0
iteration = 1

#to be implemented later
