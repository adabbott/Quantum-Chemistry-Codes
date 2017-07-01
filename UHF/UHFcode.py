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
Ca = C[:, :nalpha]
Cb = C[:, :nbeta]
Da = np.einsum('pi,qi->pq', Ca, Ca)
Db = np.einsum('pi,qi->pq', Cb, Cb)



E = 0.0
Eold = 0.0
Dolda = np.zeros_like(Da)
Doldb = np.zeros_like(Db)

for iteration in range(1, SCF_MAX_ITER+1):
    
    # Build the Fock matrix
    Ja = np.einsum('pqrs,rs->pq', I, Da) #total repulsion of alpha e-
    Jb = np.einsum('pqrs,rs->pq', I, Db) #total repulsion of beta e-
    Ka = np.einsum('prqs,rs->pq', I, Da) #total exchange of alpha e-
    Kb = np.einsum('prqs,rs->pq', I, Db) #total exchange of beta e-
    Fa = H + Ja - Ka + Jb
    Fb = H + Jb - Kb + Ja
    # Calculate SCF energy
    E_SCF = (1/2)*(np.einsum('pq,pq->', Fa+H, Da) + np.einsum('pq,pq->', Fb+H, Db))  + molecule.nuclear_repulsion_energy()
    print('UHF iteration %3d: energy %20.14f  dE %1.5E' % (iteration, E_SCF, (E_SCF - Eold)))
    
    if (abs(E_SCF - Eold) < 1.e-10):
        break
        
    Eold = E_SCF
    Dolda = Da
    Doldb = Db
    
    # Transform the Fock matrix
    Fta = A.dot(Fa).dot(A)
    Ftb = A.dot(Fb).dot(A)
    
    # Diagonalize the Fock matrix 
    ea, Ca = np.linalg.eigh(Fta)
    eb, Cb = np.linalg.eigh(Ftb)
    
    # Construct new SCF eigenvector matrix for this iteration
    Ca = A.dot(Ca)
    Cb = A.dot(Cb)
    # Form the new density matrix for this iteration
    Ca = Ca[:, :nalpha]
    Cb = Cb[:, :nbeta]
    Da = np.einsum('pi,qi->pq', Ca, Ca)
    Db = np.einsum('pi,qi->pq', Cb, Cb)


