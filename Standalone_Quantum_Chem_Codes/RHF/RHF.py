import psi4
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('Options.ini') #pointing to our options.ini file for info on molecule

molecule = psi4.geometry(config['DEFAULT']['molecule'])

#number of alpha and beta electrons, as given in input file
nalpha = int(config['DEFAULT']['nalpha'])
nbeta = int(config['DEFAULT']['nbeta'])

#Max number of SCF iterations 
SCF_MAX_ITER = int(config['SCF']['max_iter'])

molecule.update_geometry()
#get basis set
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
#set up integrals
mints = psi4.core.MintsHelper(basis)

#Here the to array function is converting psi4 arrays to np arrays
# Overlap integrals 
S = mints.ao_overlap().to_array()
#Kinetic energy portion
T = mints.ao_kinetic().to_array()
# Potential energy portion
V = mints.ao_potential().to_array()
#Two-electron repulsion
I = mints.ao_eri().to_array()

norb = mints.basisset().nbf()
nocc = nalpha*1
#form the one-electron hamiltonian
H = T + V

#construct the orthogonalizer S^-1/2, 
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)  #diagonalize S matrix and take to negative one half power 
A = A.to_array()

#CONTRUCT INTIAL DENSITY MATRIX
#form transformed fock matrix 
Ft = A.dot(H).dot(A)
#extract eigenvales and eigenvectors, we wont use eigvals but it's included anyway
e, C = np.linalg.eigh(Ft)
C = A.dot(C)
C = C[:, :nocc]
D = np.einsum('pi,qi->pq', C, C)
E = 0.0
Eold = 0.0

for iteration in range(1, SCF_MAX_ITER+1):
    # Build the Fock matrix
    J = np.einsum('pqrs,rs->pq', I, D) 
    K = np.einsum('prqs,rs->pq', I, D) 
    v = 2*J - K
    F = H + v  
    
    # Calculate SCF energy
    E_SCF = np.einsum('pq,qp->', H + F  , D) + molecule.nuclear_repulsion_energy()
    print('RHF iteration %3d: energy %20.14f  dE %1.5E' % (iteration, E_SCF, (E_SCF - Eold)))
    if (abs(E_SCF - Eold) < 1.e-10):
        break
    
    Eold = E_SCF
    # Transform the Fock matrix
    Ft = A.dot(F).dot(A)
    # Diagonalize the Fock matrix 
    e, C = np.linalg.eigh(Ft)
    # Construct new SCF eigenvector matrix for this iteration
    C = A.dot(C)
    # Form the new density matrix for this iteration
    C = C[:, :nocc]
    D = np.einsum('pi,qi->pq', C, C)


