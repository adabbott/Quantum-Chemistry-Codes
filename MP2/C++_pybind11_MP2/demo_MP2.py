import numpy as np
import psi4
import mp2
import time

psi4.core.be_quiet()

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

e, wfn = psi4.energy('scf/aug-cc-pVTZ', return_wfn = True)

# Get orbital energies and MO coefficients
epsilon = wfn.epsilon_a().to_array()
C = wfn.Ca().to_array()

# Get two electron integrals
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVTZ", key='basis')
mints = psi4.core.MintsHelper(bas)
g = np.array(mints.ao_eri())

nocc = 5
nbf = mints.nbf()
nvirt = nbf - nocc

# Transform two electron integrals
O = slice(None, nocc)
V = slice(nocc, None)
g_iajb = np.einsum('pQRS, pP -> PQRS',
         np.einsum('pqRS, qQ -> pQRS',
         np.einsum('pqrS, rR -> pqRS',
         np.einsum('pqrs, sS -> pqrS', g, C[:,V]), C[:,O]), C[:,V]), C[:,O]) 

# a naive, slow MP2 energy algorithm
def mp2_energy_slow(g_iajb = g_iajb, epsilon = epsilon, nocc = nocc, nvirt = nvirt):
    E_mp2 = 0.0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    e_denom = 1 / (epsilon[i] + epsilon[j] - epsilon[nocc + a] - epsilon[nocc + b])
                    iajb = g_iajb[i, a, j, b]
                    ibja = g_iajb[i, b, j, a]
                    E_mp2 += iajb * (2*iajb - ibja) * e_denom
    return E_mp2 

# a fast, tensor contraction MP2 energy algorithm (requires more memory)
def mp2_energy_fast(g_iajb = g_iajb, epsilon = epsilon, nocc = nocc):
    eocc = epsilon[:nocc]
    evir = epsilon[nocc:]
    e_iajb = 1 / (eocc.reshape(-1, 1, 1, 1) - evir.reshape(-1, 1, 1) + eocc.reshape( -1, 1) - evir)    
    E_mp2 = np.einsum('iajb, iajb->', g_iajb * (2 * g_iajb - g_iajb.swapaxes(1,3)), e_iajb)
    return E_mp2

# call C++ mp2 energy with mp2.mp2_energy(g_iajb, epsilon, nocc)
# The same naive MP2 for loop algorithm, but in C++ with parallelization and 
# vectorization. 350x faster than the Python counterpart, and a ~30% speedup
# over the Python tensor contraction code
E = mp2.mp2_energy(g_iajb, epsilon, nocc)

