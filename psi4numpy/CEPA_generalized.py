import psi4
import numpy as np
import scipy.linalg

psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
0 2
O
H 1 1.1
symmetry c1
""")
psi4.set_options({'basis':        '6-31g',
                  'scf_type':     'pk',
                  'reference':    'uhf',
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

mints = psi4.core.MintsHelper(scf_wfn.basisset())
nbf = mints.nbf()
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nocc = nalpha + nbeta
nvirt = 2*nbf - nocc
MAXITER = 40
   
def spin_block_tei(I):
    identity = np.eye(2)
    g = np.kron(identity, I)
    return np.kron(identity, g.T)

I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)
gao = I_spinblock.transpose(0, 2, 1, 3) - I_spinblock.transpose(0, 2, 3, 1)

eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)
# NO!
#eps = np.sort(eps)

# Transform two-electron integrals to the spin-MO basis
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = np.block([
             [      Ca           ,   np.zeros_like(Cb) ],
             [np.zeros_like(Ca)  ,          Cb         ]
            ])

C = C[:, eps.argsort()]
eps = np.sort(eps)
# Transform antisymmetrized spin blocked two electron integrals
# to the MO basis
gmo = np.einsum('pQRS, pP -> PQRS', 
      np.einsum('pqRS, qQ -> pQRS', 
      np.einsum('pqrS, rR -> pqRS', 
      np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C) 

# Convert integrals to physicists notation and antisymmetrize
# (pq | rs) ----> <pr | qs> - <pr | sq>

# Form 4-index tensor of orbital energy denominators


n = np.newaxis
o = slice(None, nocc)
v = slice(nocc, None)
e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])

# Create space for t amplitudes
t_amp = np.zeros((nvirt, nvirt, nocc, nocc))

E_CEPA0 = 0.0

for cc_iter in range(MAXITER):
    E_old = E_CEPA0
    # Collect terms
    mp2   = gmo[v, v, o, o]
    cepa1 = 0.5 * np.einsum('abcd, cdij -> abij', gmo[v, v, v, v], t_amp)
    cepa2 = 0.5 * np.einsum('klij, abkl -> abij', gmo[o, o, o, o], t_amp)
    cepa3a = np.einsum('kbcj, acik -> abij', gmo[o, v, v, o], t_amp)
    cepa3b = -cepa3a.transpose(1, 0, 2, 3) 
    cepa3c = -cepa3a.transpose(0, 1, 3, 2) 
    cepa3d = cepa3a.transpose(1, 0, 3, 2) 

    t_amp_new = e_abij * (mp2 + cepa1 + cepa2 + cepa3a + cepa3b + cepa3c + cepa3d)
    E_CEPA0 = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v], t_amp_new)
    t_amp = t_amp_new
    dE = E_CEPA0 - E_old
    print('CEPA0 Iteration %3d: Energy = %4.12f dE = %1.5E' % (cc_iter, E_CEPA0, dE))

    if abs(dE) < 1.e-10:
        print("\nCEPA0 Iterations have converged!")
        break
    
    if (cc_iter == MAXITER):
        psi4.core.clean()
        raise Exception("\nMaximum number of iterations exceeded.")


print('\nCEPA0 Correlation Energy: %5.15f' % (E_CEPA0))
print('CEPA0 Total Energy: %5.15f' % (E_CEPA0 + scf_e))

psi4_cepa = psi4.energy('lccd', ref_wfn = scf_wfn)
print('\nPsi4 CEPA0 Correlation Energy:     ', psi4_cepa - scf_e)
print('Psi4 CEPA0 Total Energy:     ', psi4_cepa)


