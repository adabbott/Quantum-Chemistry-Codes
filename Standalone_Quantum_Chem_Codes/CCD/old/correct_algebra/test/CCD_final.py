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


#gather block diagonalized coefficent matrix to transform 2 e- integrals to MO basis later
C_block  = block_diag(C_a, C_b)
orb_energies = np.concatenate((ea,eb))
C_block = C_block[:,orb_energies.argsort()]
orb_energies = np.sort(orb_energies, None)

def spin_block_tei(gao):
    Identity = np.eye(2) #2x2 identity matrix
    gao = np.kron(Identity, gao) #basically tensor product of 2x2 Identity with gao
    return np.kron(Identity, gao.T) #transpose to make rows into columns

I = spin_block_tei(I)

I_phys = I.transpose(0,2,1,3) - I.transpose(0,2,3,1)

O = slice(None, nocc)
V = slice(nocc, None)
#for CCD terms
G_vvoo = np.einsum('pQRS, pP -> PQRS', 
         np.einsum('pqRS, qQ -> pQRS', 
         np.einsum('pqrS, rR -> pqRS', 
         np.einsum('pqrs, sS -> pqrS', I_phys, C_block[:,O]), C_block[:,O]), C_block[:,V]), C_block[:,V]) 

G_vvvv = np.einsum('pQRS, pP -> PQRS', 
         np.einsum('pqRS, qQ -> pQRS', 
         np.einsum('pqrS, rR -> pqRS', 
         np.einsum('pqrs, sS -> pqrS', I_phys, C_block[:,V]), C_block[:,V]), C_block[:,V]), C_block[:,V]) 

G_oooo = np.einsum('pQRS, pP -> PQRS', 
         np.einsum('pqRS, qQ -> pQRS', 
         np.einsum('pqrS, rR -> pqRS', 
         np.einsum('pqrs, sS -> pqrS', I_phys, C_block[:,O]), C_block[:,O]), C_block[:,O]), C_block[:,O]) 

G_ovvo = np.einsum('pQRS, pP -> PQRS', 
         np.einsum('pqRS, qQ -> pQRS', 
         np.einsum('pqrS, rR -> pqRS', 
         np.einsum('pqrs, sS -> pqrS', I_phys, C_block[:,O]), C_block[:,V]), C_block[:,V]), C_block[:,O]) 

G_oovv = np.einsum('pQRS, pP -> PQRS', 
         np.einsum('pqRS, qQ -> pQRS', 
         np.einsum('pqrS, rR -> pqRS', 
         np.einsum('pqrs, sS -> pqrS', I_phys, C_block[:,V]), C_block[:,V]), C_block[:,O]), C_block[:,O])

#perform CCD
N = np.newaxis #dummy axis
e_abij =  (-orb_energies[V, N, N, N] - orb_energies[N, V, N, N] + orb_energies[N,N,O,N] + orb_energies[N,N,N,O])
t_amp = np.zeros((nvirt, nvirt, nocc, nocc))
Energy = 0.0

#need to generalize, only works for 4d arrays, 2 sub permutations
def permute(array, lower = [0,1] , upper = [2,3]):
    #what is this bullshit
    new_array = 1*array
    if lower == [1,0] and upper == [2,3]:
        new_array -= array.transpose(tuple(lower + upper))
    if lower == [0,1] and upper == (3,2):
        new_array -= array.transpose(tuple( [0,1] + upper))
    if lower == [1,0] and upper == [3,2]:
        new_array -= array.transpose(tuple( [0,1] + upper))
        new_array -= array.transpose(tuple(lower + [2,3]))
        new_array += array.transpose(tuple(lower + upper)) 
    return new_array  

#tccd3    =         np.einsum('kbcj, acik -> abij', G_ovvo, t_amp) #NEEDS PERMUTATIONS _ab ^ij
#print((tccd3 - tccd3.transpose((1,0,2,3)) - tccd3.transpose((0,1,3,2)) + tccd3.transpose((1,0,3,2))) - permute(tccd3,(1,0),(3,2)) )

#tccd6    = -(1/2) * np.einsum('klcd, cakl, dbij -> abij', G_oovv, t_amp, t_amp) #NEEDS PERMUTATIONS _ab
#print( (tccd6 - tccd6.transpose((1,0,2,3))) - permute(tccd6,(1,0),(0,1)) )
#tccd7    = -(1/2) * np.einsum('klcd, cdki, ablj -> abij', G_oovv, t_amp, t_amp) #NEEDS PERMUTATIONS ^ij

t0 = time.time()
#all indices read "bra, bra, ket, ket", or "lower, lower, upper, upper", einstein summation convention
for iteration in  range(CEPA0_MAX_ITER+1):
    #eight diagrams in t_amp expansion
    mp2_term = G_vvoo
    tccd1    = (1/2) * np.einsum('abcd, cdij -> abij', G_vvvv, t_amp)
    tccd2    = (1/2) * np.einsum('klij, abkl -> abij', G_oooo, t_amp)
    tccd3    =         np.einsum('kbcj, acik -> abij', G_ovvo, t_amp) #NEEDS PERMUTATIONS _ab ^ij
    full_tccd3 = tccd3 - tccd3.transpose((1,0,2,3)) - tccd3.transpose((0,1,3,2)) + tccd3.transpose((1,0,3,2))
#    print( full_tccd3 - permute(tccd3, [1,0], [3,2] ) )
    tccd4    = (1/4) * np.einsum('klcd, abkl, cdij -> abij', G_oovv, t_amp, t_amp)
    tccd5    = (1/2) * np.einsum('klcd, acik, dblj -> abij', G_oovv, t_amp, t_amp) #NEEDS PERMUTATIONS _ab ^ij
    full_tccd5 = tccd5 - tccd5.transpose((1,0,2,3)) - tccd5.transpose((0,1,3,2)) + tccd5.transpose((1,0,3,2))
    tccd6    = -(1/2) * np.einsum('klcd, cakl, dbij -> abij', G_oovv, t_amp, t_amp) #NEEDS PERMUTATIONS _ab
    full_tccd6 = tccd6 - tccd6.transpose((1,0,2,3))
    print( full_tccd6 - permute(tccd6,(1,0),(2,3)) )
    tccd7    = -(1/2) * np.einsum('klcd, cdki, ablj -> abij', G_oovv, t_amp, t_amp) #NEEDS PERMUTATIONS ^ij
    full_tccd7 = tccd7 - tccd7.transpose((0,1,3,2))
    #print( ( full_tccd7 - permute(tccd7, (0,1), (3,2)) ) )
     
    #t_amp_new = (1/e_abij)*(mp2_term + tccd1 + tccd2 + full_tccd3 + tccd4 + full_tccd5 + full_tccd6 + full_tccd7)
    t_amp_new = (1/e_abij)*(mp2_term + tccd1 + tccd2 + permute(tccd3,[1,0],[3,2]) + \
                tccd4 + permute(tccd3,[1,0],[3,2]) + permute(tccd6,[1,0],[2,3]) + permute(tccd7,[0,1],[3,2]) )
    E_CCD = (1/4)*np.einsum('ijab, abij ->', G_oovv, t_amp_new) 
    print('CCD iteration %3d: energy %20.14f  dE %1.5E |dT| %1.5E ' % (iteration, E_CCD, E_CCD - Energy, np.linalg.norm(t_amp_new-t_amp)))
    if  (abs(Energy - E_CCD)) < 1.e-10 and (abs(np.linalg.norm(t_amp_new-t_amp)) < 1.e-10): 
        break
    t_amp = t_amp_new
    Energy = E_CCD
t1 = time.time() 
print('CCD correlation energy:')
print(E_CCD)
print('Total UCCD energy')
print(E_SCF + E_CCD)
print('CCD took %7.5f seconds' % ((t1-t0)))

#psi4.set_options({'basis': 'sto-3g', 'reference': 'uhf',
#                   'scf_type': 'pk','guess': 'core', 'mp2_type': 'conv',
#                   'e_convergence': 1e-10})
#psi4.energy('bccd')
