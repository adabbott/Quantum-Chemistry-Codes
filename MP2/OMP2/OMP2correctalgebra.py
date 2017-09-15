import psi4
import numpy as np
import configparser
from scipy.linalg import block_diag
import time
import scipy as sp
config = configparser.ConfigParser()
config.read('Options.ini') #pointing to our options.ini file for info on molecule
#reads in geometry given by input
molecule = psi4.geometry(config['DEFAULT']['molecule'])

SCF_MAX_ITER = int(config['SCF']['max_iter'])
OMP2_MAX_ITER = int(config['OMP2']['max_iter'])
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
S = mints.ao_overlap().to_array()
#Kinetic energy portion, i.e. the kinetic enregy of attraction to the nuclei
T = mints.ao_kinetic().to_array()
# Potential energy portion, i.e. the potential energy of the electron attraction to the nuclei
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
#Commentary:
#The Roothan-Hall Equation: FC = SCe, is not an ordinary eigenvalue equation,
#but by multiplying both sides by S^(-1/2) we can reduce it to F'C'=C'e where F'=S^(-1/2)(F)S^(-1/2) and C' = S^(1/2)C
#this is effectively transforming to an orthogonalized atomic orbital basis. We can diagonalize F' easily, andd then transfrom the coefficients
#back into the MO representation.
#form transformed fock matrix 
Ft = A.dot(H).dot(A)
#extract eigenvales and eigenvectors, we wont use eigvals but it's included anyway
junk, C = np.linalg.eigh(Ft)
C = A.dot(C)
#here column vectors of C are the expansion coefficients of the AO's for a molecular orbital.
#Since we are doing UHF we parse these into an alpha and beta part
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
    Fa = H + Ja + Jb - Ka
    Fb = H + Ja + Jb - Kb
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
        pa = np.zeros((n,n)) 
        pb = np.zeros((n,n))
        if n >= 2:
            #build P matrices
            #This is a 4x4 of zeroes for first iteration 
            for i in range(len(a_errs)):   #a_errs and b_errs will always be the same size 
                for j in range(len(a_errs)):
                   pa[i,j] = np.vdot(a_errs[i], a_errs[j])
                   pb[i,j] = np.vdot(b_errs[i], b_errs[j])
            #ALERT NOT BUILDING THE 1X1 PEES
            # build f (do i need to specify column? do numpy.stack?)
            f = np.zeros(np.size(pa, 0)) #size of Pa along axis 0
            f = np.append(f, -1)
            #add a row/column of -1's and final element 0
            Pa = -np.ones((n+1,n+1))
            Pa[:n,:n] = pa
            Pa[n,n] = 0 
            Pb = -np.ones((n+1,n+1))
            Pb[:n,:n] = pb
            Pb[n,n] = 0 
            #P matrices are now built, and we have our f's, solve for q's
            P = (Pa + Pb)/2
            q = np.linalg.solve(P, f)
            #optimized F's
            Fa = sum(q[i]*a_focks[i] for i in range(n))
            Fb = sum(q[i]*b_focks[i] for i in range(n))
    # Calculate SCF energy
    E_SCF = (1/2)*(np.einsum('pq,pq->', Fa+H, Da) + np.einsum('pq,pq->', Fb+H, Db))  + molecule.nuclear_repulsion_energy()
    #print('UHF iteration %3d: energy %20.14f  dE %1.5E' % (iteration, E_SCF, (E_SCF - Eold)))
    
    if (abs(E_SCF - Eold) < 1.e-10):
        break
        
    Eold = E_SCF
    Dolda = Da
    Doldb = Db
    
    # Transform the Fock matrix
    Fta = A.dot(Fa).dot(A)
    Ftb = A.dot(Fb).dot(A)
    
    # Diagonalize the Fock matrix to get new coefficients
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

#do MP2
#gather block diagonalized coefficent matrix to transform 2 e- integrals to MO basis later
C_block  = block_diag(C_a, C_b)
orb_energies = np.concatenate((ea,eb))
#sort eigenvectors in the block diag C matrix according to the order in which they appear in the list of orbital energies
#argsort returns the indices of orb_energies in terms of their corresponding array value in increasing order,
# for example np.argsort([5,-1,0]) returns an array of indices [1,2,0]
#move the columns of C_block according to the value-sorted indices of orb_energies 
C_block = C_block[:,orb_energies.argsort()]
orb_energies = np.sort(orb_energies, None)
#now C is sorted and spin blocked

def spin_block_tei(gao):
    Identity = np.eye(2) #2x2 identity matrix
    gao = np.kron(Identity, gao) #basically tensor product of 2x2 Identity with gao
    return np.kron(Identity, gao.T) #transpose to make rows into columns
#What this is doing here is taking our 4 dimensional array of 2 electron integrals
#and putting them into the space of the 2x2 identity. Basically makes a 4d block of I, 2 4d blocks of 0's, and a 4d block of I
#then tranpose and do it again. The progression is I = (7,7,7,7) to (7,7,14,14) to (14,14,14,14) it doubles the dimensions
#for (14,14,14,14), for the first and third dimension, the first 7 entries have values, the last 7 have all zeroes
#for second and fourth dimension, the last 7 entries have values and the first 7 are all zeroes
I = spin_block_tei(I)
#I is now spin blocked as well
#antisymmetrize and convert to physicists notation
#given (ab|cd), convert to <ac|bd> - <ac|db>
I_phys = I.transpose(0,2,1,3) - I.transpose(0,2,3,1)

#individucal axis einsums are more efficient
#transform the two electron integrals from the atomic orbital basis to the molecular orbital basis
#using our coefficients
#here each dimension is getting the full treatment over all occupied and virtual orbitals
#we could parse our C_blocks in these einsums to get only certain terms, like gijab

t0 = time.time()
O = slice(None, nocc)
V = slice(nocc, None)
x = np.newaxis
I_mo_MP2 = np.einsum('pQRS, pP -> PQRS', 
       np.einsum('pqRS, qQ -> pQRS', 
       np.einsum('pqrS, rR -> pqRS', 
       np.einsum('pqrs, sS -> pqrS', I_phys, C_block[:,V]), C_block[:,V]), C_block[:,O]), C_block[:,O]) 
fast_EMP2 = 0.0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvirt):
            for b in range(nvirt):
                fast_EMP2 += ((1/4)*I_mo_MP2[i, j, a, b]**2)/(orb_energies[i]+orb_energies[j]-orb_energies[a+nocc]-orb_energies[b+nocc])
t1 = time.time()
print(fast_EMP2)
print('Fast MP2 took %7.5f seconds' % ((t1-t0)))

#do OMP2
def spinblockoei(hao):
    return np.kron(np.eye(2), hao)

#def transformoei(hao, C):       #C has dimensions AO x MO, to convert each axis of h, g to MO basis by dotting it with 
#    return np.einsum('pQ,pP->PQ',
#           np.einsum('pq,qQ->pQ', hao, C), C)  
#def transformoei(hao, C):       #C has dimensions AO x MO, to convert each axis of h, g to MO basis by dotting it with 
#    return np.einsum('Qp,pP->PQ',
#           np.einsum('pq,qQ->Qp', hao, C), C)  
def transformoei(hao, C):       #C has dimensions AO x MO, to convert each axis of h, g to MO basis by dotting it with 
    return np.einsum('Qp,pP->PQ',
           np.einsum('pq,qQ->Qp', hao, C), C)  
print(np.shape(C_block))
def transformtei(gao, C):
    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)

#initialize amplitudes, density matrices, newton raphson steps, orbital rotation matrix,
tamp = np.zeros((nvirt, nvirt, nocc, nocc)) #t is tabij! bra then ket!
DM1ref = np.zeros((ntotal,ntotal))
DM1ref[O,O] = np.identity(nocc) #1 particle density matrix should be 1's along the occupied diagonal, 0's elsewhere
DM1cor = np.zeros((ntotal,ntotal))
DM2cor = np.zeros((ntotal,ntotal,ntotal,ntotal))

EOMP2_old = 0.0
Co = C_block.copy()
h1 = transformoei(spinblockoei(H),Co) #spin blocks our 1e- hamiltonian and transforms to MO basis
I_mo = transformtei(I_phys,Co) #Our antisymmetrized physicists notation integrals is transfomed to the MO basis
 
for i in range(OMP2_MAX_ITER):    
    fock = h1 + np.einsum('piqi->pq', I_mo[:,O,:,O]) #
    energies = fock.diagonal().copy()
    fprime = np.copy(fock) #fprime is the off-diagonal part of fock operator
    np.fill_diagonal(fprime,0)  #so it has zeros on the diagonal
    term2 = np.einsum('ac,cbij->abij',fprime[V,V], tamp) #pre-define terms for our t amplitudes so permutations are easy
    term3 = np.einsum('ki,abkj->abij',fprime[O,O], tamp)  
    tamp = I_mo[V,V,O,O] + (term2 - term2.transpose(1,0,2,3))                          \
                    - (term3 - term3.transpose(0,1,3,2))
    tamp /=   energies[x,x,O,x] + energies[x,x,x,O]                                    \
            - energies[V,x,x,x] - energies[x,V,x,x]
    #build virtual and occupied space of our 1 and 2 e- density matrices
    DM1cor[V,V] =  (1/2)*np.einsum('acij,bcij->ba',tamp,tamp)
    DM1cor[O,O] = -(1/2)*np.einsum('abjk,abik->ji',tamp,tamp)
    DM2cor[O,O,V,V] = tamp.T #these are just the same as our t amplitudes
    DM2cor[V,V,O,O] = tamp
    #build 1-particle density matrix D1 and 2-particle density matrix D2
    D1 = DM1cor + DM1ref
    D2_term2 = np.einsum('rp,sq->rspq',DM1cor,DM1ref)
    D2_term3 = np.einsum('rp,sq->rspq',DM1ref,DM1ref)
    D2 = DM2cor + (D2_term2 - D2_term2.transpose(1,0,2,3) - D2_term2.transpose(0,1,3,2) \
         + D2_term2.transpose(1,0,3,2)) + (D2_term3 - D2_term3.transpose(1,0,2,3))
    #Form the initial generalized fock matrix, our 1e- and 2e- hamiltonians traced with our density matrices
    F = np.einsum('pr,rq->pq',h1,D1) + (1/2)*np.einsum('prst,stqr->pq',I_mo,D2) 
    #Build the Newton Raphson Orbital Rotation Matrix U and rotate the orbital coefficients Co
    X = np.zeros((ntotal,ntotal))
    X[O,V] = (F - F.T)[O,V]/(energies[O,x]-energies[x,V])
    U = sp.linalg.expm(X-X.T)
    Co = Co.dot(U)
    #transform the 1e- and 2e- hamiltonians to the spin-orbital basis using the new coefficient matrix
    h1  = transformoei(spinblockoei(H), Co)
    I_mo = transformtei(I_phys, Co)
    EOMP2 = molecule.nuclear_repulsion_energy()      \
        + np.einsum('pq,qp->',h1,D1)                 \
        + (1/4)*np.einsum('pqrs,rspq->',I_mo,D2)
    print(EOMP2)
    if (abs(EOMP2 - EOMP2_old) < 1.e-10):
        break
    EOMP2_old = EOMP2
     
