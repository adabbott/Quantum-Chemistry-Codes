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

def spin_block_tei(gao):
    Identity = np.eye(2) #2x2 identity matrix
    gao = np.kron(Identity, gao) #basically tensor product of 2x2 Identity with gao
    return np.kron(Identity, gao.T) #transpose to make rows into columns

I = spin_block_tei(I)

I_phys = I.transpose(0,2,1,3) - I.transpose(0,2,3,1)
#C_block = C
#I_mo = np.einsum('pqrs, pP, qQ, rR, sS -> PQRS', I_phys, C_block, C_block, C_block, C_block)
#individucal axis einsums are more efficient
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

#do df-MP2
#build J (2 center auxilliary basis integrals)
Jaux = mints.ao_eri(df_basis, zero, df_basis, zero).to_array()
Jaux = np.squeeze(Jaux)
Jaux = la.inv(la.sqrtm(Jaux))

pqP = mints.ao_eri(basis, basis, zero, df_basis).to_array() 
def spin_blockpq(gao):
    Identity = np.eye(2)
    #np.kron is a tensor product. we are putting only the first two indices of our 2 electron integrals
    #into the space of the 2x2 identity,
    return np.kron(Identity, gao.T).T

#spin block pqP and squeeze out the dimensionless indice we created initially
pqP = spin_blockpq(pqP)
pqP = np.squeeze(pqP)
#build B
Bpq = np.einsum('pqQ, QR -> pqR', pqP, Jaux)
#perform the integral transformation, transforming our b's from AO basis to MO basis using our UHF MO coefficients, C_block
Biq = np.einsum('pqR, pi -> iqR', Bpq, C_block) 
Bia = np.einsum('iqR, qa -> iaR', Biq, C_block)

#solve for the energy
E_dfMP2 = 0.0
for i in range(nocc):
    for j in range(nocc):
        MP2_relevant_2_e_integrals = np.einsum('aR, bR -> ab', Bia[i], Bia[j])
        print(MP2_relevant_2_e_integrals)
        for a in range(nocc,ntotal):
            for b in range(nocc,ntotal):
                E_dfMP2 += ((1/4)*(MP2_relevant_2_e_integrals[a, b] - MP2_relevant_2_e_integrals[b, a])**2)/(orb_energies[i]+orb_energies[j]-orb_energies[a]-orb_energies[b])

print("Density fitted MP2 energy:")        
print(E_dfMP2)
print('DF error:')
print(E_MP2-E_dfMP2)



"""psi4.set_options({'basis': 'sto-3g', 'reference': 'uhf',
                  'scf_type': 'pk','guess': 'core','mp2_type': 'conv',
                  'e_convergence': 1e-10})
psi4.energy('mp2')"""


