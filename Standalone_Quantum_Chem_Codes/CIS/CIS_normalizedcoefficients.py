import psi4
import numpy as np
import configparser
from scipy.linalg import block_diag
import time
config = configparser.ConfigParser()
config.read('Options.ini') #pointing to our options.ini file for info on molecule
#reads in geometry given by input
molecule = psi4.geometry(config['DEFAULT']['molecule'])

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
nvirt = ntotal-nocc


# Overlap integrals 
S = mints.ao_overlap().to_array()
#Kinetic energy portion, i.e. the kintetic enregy of attraction to the nuclei
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
e, C = np.linalg.eigh(Ft)
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
    print('UHF iteration %3d: energy %20.14f  dE %1.5E' % (iteration, E_SCF, (E_SCF - Eold)))
    
    if (abs(E_SCF - Eold) < 1.e-10 and (np.linalg.norm((Dolda+Doldb) - (Da+Db))) < 1.e-10):
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

#print(Da)
#wval, wvec = np.linalg.eigh(Da)
#print(wvec)
#does this diagonlize Da?
#Da = np.linalg.inv(wvec).dot(Da).dot(wvec)
#print(Da)
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
ta = time.time()
I_mo = np.einsum('pQRS, pP -> PQRS', 
       np.einsum('pqRS, qQ -> pQRS', 
       np.einsum('pqrS, rR -> pqRS', 
       np.einsum('pqrs, sS -> pqrS', I_phys, C_block), C_block), C_block), C_block) 
E_MP2 = 0.0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nocc,ntotal):
            for b in range(nocc,ntotal):
                E_MP2 += ((1/4)*I_mo[i, j, a, b]**2)/(orb_energies[i]+orb_energies[j]-orb_energies[a]-orb_energies[b])
tb = time.time()
print(E_MP2)
print('Regular MP2 took %7.5f seconds' % ((tb-ta)))















#do CIS
HCIS = np.zeros((nvirt*nocc,nvirt*nocc))
w = -1
strings = []
str2 = '->'
for i in range(nocc):
    for a in range(nocc, ntotal):
        w = w+1
        v = -1
        str1 = str(i)
        str3 = str(a)
        strings.append(''.join((str1, str2, str3)))
        for j in range(nocc):
            for b in range(nocc, ntotal):
                v = v+1
                HCIS[w,v] = (orb_energies[a] - orb_energies[i]) * (i==j) * (a==b) + I_mo[a, j, i, b]
print(HCIS)
ECIS, CCIS = np.linalg.eig(HCIS)
print(CCIS)
print('CIS:')
for state in range(nvirt*nocc):
    coeffs = CCIS[:,state]
    print('State %3d Energy (Eh) %15.14f' % (state, ECIS[state]), end=' ')
    for element in range(nvirt*nocc):
        q = (coeffs[element]**2)*100
        if q > 10:
            print('%d%% %s' % (q, strings[element]), end=' ')
    print()




