import psi4
import numpy as np
import configparser
config = configparser.ConfigParser()
config.read('Options3.ini') #pointing to our options.ini file for info on molecule

molecule = psi4.geometry(config['DEFAULT']['molecule'])

nalpha = int(config['DEFAULT']['nalpha'])
nbeta = int(config['DEFAULT']['nbeta'])
SCF_MAX_ITER = int(config['SCF']['max_iter'])
molecule.update_geometry()
#get basis set
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
#set up integrals
mints = psi4.core.MintsHelper(basis)


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


#for DIIS 
a_focks = []  #space we collect our focks
b_focks = []
a_errs = []   #space we collect our error matrices
b_errs = [] 

for iteration in range(1, SCF_MAX_ITER+1):
    # Build the Fock matrix
    Ja = np.einsum('pqrs,rs->pq', I, Da) #total repulsion of alpha e-
    Jb = np.einsum('pqrs,rs->pq', I, Db) #total repulsion of beta e-
    Ka = np.einsum('prqs,rs->pq', I, Da) #total exchange of alpha e-
    Kb = np.einsum('prqs,rs->pq', I, Db) #total exchange of beta e-
    #form the fock matrix
    Fa = H + Ja - Ka + Jb
    Fb = H + Jb - Kb + Ja
    #conditions to turn on DIIS 
    if int(config['SCF']['diis']) == 1 and int(config['SCF']['diis_start']) <= iteration:  
    #do DIIS
        err_a = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
        err_b = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
        #limit the number of error vectors
        if len(a_errs) == int(config['SCF']['diis_nvector']):
            a_errs.pop(0)
            b_errs.pop(0)
            a_focks.pop(0)
            b_focks.pop(0)
        a_focks.append(Fa) 
        b_focks.append(Fb) 
        a_errs.append(err_a)
        b_errs.append(err_b)
        n = len(a_focks) 
        Pa = np.zeros((n, n)) 
        Pb = np.zeros((n, n))
        if n >= 2:
            #build P matrices
            #This is a 4x4 of zeroes for first iteration print(Pa) 
            for i in range(n):   #a_errs and b_errs will always be the same size 
                for j in range(n):
                   Pa[i,j] = np.vdot(a_errs[i], a_errs[j])
                   Pb[i,j] = np.vdot(b_errs[i], b_errs[j])
            #ALERT NOT BUILDING THE 1X1 PEES
            # build f (do i need to specify column? do numpy.stack?)
            fa = np.zeros(np.size(Pa, 0)) #size of Pa along axis 0
            fa = np.append(fa, -1)
            fb = np.zeros(np.size(Pb, 0))
            fb = np.append(fb, -1)
            #add a row/column of -1's and final element 0
            add_rowa    = np.zeros(np.size(Pa, 1)) + -1
            add_cola = np.zeros(np.size(Pa, 1)) + -1
            add_cola = np.append(add_cola, 0)
            add_rowb    = np.zeros(np.size(Pb, 1)) + -1
            add_colb = np.zeros(np.size(Pb, 1)) + -1
            add_colb = np.append(add_colb, 0)
            Pa = np.insert(Pa, np.size(Pa, 0), add_rowa, axis=0)
            Pa = np.insert(Pa, np.size(Pa, 1), add_cola , axis=1)
            Pb = np.insert(Pb, np.size(Pb, 0), add_rowb, axis=0)
            Pb = np.insert(Pb, np.size(Pb, 1), add_colb , axis=1)
            #P matrices are now built, and we have our f's, solve for q's
            P = (Pa + Pb)/2
            qa = np.linalg.solve(P, fa)
            qb = np.linalg.solve(P, fb)
            q = (qa + qb)/2
            #optimized F's
            Fa = sum(q[i]*a_focks[i] for i in range(n))
            Fb = sum(q[i]*b_focks[i] for i in range(n))
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
