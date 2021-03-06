import psi4
import numpy as np
import molecule
import integrals

def UHF(mol, maxiter = 50):
    nalpha = mol.ndocc + mol.nsocc
    nbeta = mol.ndocc 
    # Get integral arrays
    S, T, V, I = integrals.compute_integrals(mol)
    # Form orthogonalizer
    mints = psi4.core.MintsHelper(mol.basis)
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-16)  #diagonalize S matrix and take to negative one half power 
    A = A.to_array()
    #A = np.power(S, -0.5)   
    # Form one electron hamiltonian
    H = T + V      
    #Construct initial density matrices
    Ft = A.dot(H).dot(A)
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
    
    for iteration in range(1, maxiter+1):
        
        # Build the Fock matrix
        Ja = np.einsum('pqrs,rs->pq', I, Da) #total repulsion of alpha e-
        Jb = np.einsum('pqrs,rs->pq', I, Db) #total repulsion of beta e-
        Ka = np.einsum('prqs,rs->pq', I, Da) #total exchange of alpha e-
        Kb = np.einsum('prqs,rs->pq', I, Db) #total exchange of beta e-
        Fa = H + Ja - Ka + Jb
        Fb = H + Jb - Kb + Ja
        # Calculate SCF energy
        E_SCF = (1/2)*(np.einsum('pq,pq->', Fa+H, Da) + np.einsum('pq,pq->', Fb+H, Db))  + mol.nuclear_repulsion_energy
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
        ea, C_a = np.linalg.eigh(Fta)
        eb, C_b = np.linalg.eigh(Ftb)
        
        # Construct new SCF eigenvector matrix for this iteration
        C_a = A.dot(C_a)
        C_b = A.dot(C_b)
        # Form the new density matrix for this iteration
        Ca = C_a[:, :nalpha]
        Cb = C_b[:, :nbeta]
        Da = np.einsum('pi,qi->pq', Ca, Ca)
        Db = np.einsum('pi,qi->pq', Cb, Cb)
    return E_SCF, C_a, C_b, ea, eb
