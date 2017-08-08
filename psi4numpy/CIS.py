import psi4
import numpy as np


# Set Psi4 Options
psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
0 1
H
H 1 1.1
symmetry c1
""")
psi4.set_options({'basis':        'sto-3g',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

#H 1 1.1 2 104
scf_e, scf_wfn = psi4.energy('scf', return_wfn = True)

# Check memory requirements
nmo = scf_wfn.nmo()
I_size = (nmo**4) * 8e-9
print('\nSize of the ERI tensor will be %4.2f GB.\n' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted \
                     memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Get basis and orbital information
mints = psi4.core.MintsHelper(scf_wfn.basisset())
nbf = mints.nbf()
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nocc = nalpha + nbeta
nvirt = 2 * nbf - nocc
nso = 2 * nbf 

def spin_block_tei(I):
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)
gao = I_spinblock.transpose(0, 2, 1, 3) - I_spinblock.transpose(0, 2, 3, 1)

# Sort orbital energies in increasing order 
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)

# Transform two-electron integrals to the spin-MO basis 
Ca = np.asarray(scf_wfn.Ca()) 
Cb = np.asarray(scf_wfn.Cb()) 
C = np.block([ 
             [      Ca           ,   np.zeros_like(Cb) ], 
             [np.zeros_like(Ca)  ,          Cb         ] 
            ]) 
 
# Sort the columns of C according to the order of orbital energies 
C = C[:, eps.argsort()] 

# Sort orbital energies
eps = np.sort(eps) 
 
# Transform the physicist's notation, antisymmetrized, spin-blocked  
# two-electron integrals to the MO basis 
gmo = np.einsum('pQRS, pP -> PQRS',  
      np.einsum('pqRS, qQ -> pQRS',  
      np.einsum('pqrS, rR -> pqRS',  
      np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)  


# Initialize CIS matrix, dimensions are the number of possible single excitations
HCIS = np.zeros((nvirt * nocc, nvirt * nocc))
w = -1
strings = []
str2 = '->'

# Build the possible exciation indices
excitations = []
for i in range(nocc):
    for a in range(nocc, nso):
        excitations.append((i,a))

for p, left_excitation in enumerate(excitations):
    i, a = left_excitation
    for q, right_excitation in enumerate(excitations):
        j, b = right_excitation
        HCIS[p, q] = (eps[a] - eps[i]) * (i == j) * (a == b) + gmo[a, j, i, b]


ECIS, CCIS = np.linalg.eigh(HCIS)
print(ECIS)
print(CCIS)


print('CIS:')
for state in range(len(ECIS)):
    # Print state, energy
    print('State %3d Energy (Eh) %10.10f' % (state, ECIS[state]) , end = ' ')
    for p, excitation in enumerate(excitations):
        if CCIS[p, state]**2 * 100 >= 10:
            i, a = excitation
            print('%d%% %d -> %d' % (CCIS[p, state]**2 * 100, i, a), end = ' ')
    print() 

#stupid eig doesnt sort energies/vectors in increasing order, so we sort
#CCIS = CCIS[:,ECIS.argsort()]

#ECIS = np.sort(ECIS)

#print('CIS:')
#for state in range(nvirt*nocc):
#    coeffs = CCIS[:,state]
#    print('State %3d Energy (Eh) %15.14f' % (state, ECIS[state]), end=' ')
#    for element in range(nvirt*nocc):
#        q = (coeffs[element]**2)*100
#        if q > 10:
#            print('%d%% %s' % (q, strings[element]), end=' ')
#    print()
#
#


