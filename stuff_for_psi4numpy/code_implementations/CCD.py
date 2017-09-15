import psi4
import numpy as np


psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")
psi4.set_options({'basis':        '6-31g',
                  'scf_type':     'pk',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

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
ndocc = scf_wfn.nalpha()
nocc = 2*ndocc
nvirt = 2*nbf - nocc
MAXITER = 40
   
# Transform two-electron integrals to the spin-MO basis
C = scf_wfn.Ca()
I_mo = np.asarray(mints.mo_spin_eri(C, C))

# Convert integrals to physicists notation and antisymmetrize
# (pq | rs) ----> <pr | qs> - <pr | sq>
g = I_mo.transpose(0, 2, 1, 3) - I_mo.transpose(0, 2, 3, 1)

# Form 4-index tensor of orbital energy denominators
eps = np.asarray(scf_wfn.epsilon_a())
eps = np.repeat(eps, 2)
n = np.newaxis
o = slice(None, nocc)
v = slice(nocc, None)
e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])

# Create space for t amplitudes
t_amp = np.zeros((nvirt, nvirt, nocc, nocc))

E_CCD = 0.0

for cc_iter in range(MAXITER):
    E_old = E_CCD
    # Collect terms
    mp2    = g[v, v, o, o]
    cepa1  = 0.5 * np.einsum('abcd, cdij -> abij', g[v, v, v, v], t_amp)
    cepa2  = 0.5 * np.einsum('klij, abkl -> abij', g[o, o, o, o], t_amp)
    cepa3a = np.einsum('kbcj, acik -> abij', g[o, v, v, o], t_amp)
    cepa3b = -cepa3a.transpose(1, 0, 2, 3) 
    cepa3c = -cepa3a.transpose(0, 1, 3, 2) 
    cepa3d =  cepa3a.transpose(1, 0, 3, 2) 
    cepa3  =  cepa3a + cepa3b + cepa3c + cepa3d

    ccd1   = 0.25 * np.einsum('klcd, abkl, cdij -> abij', g[o, o, v, v], t_amp, t_amp)

    ccd2a  =  np.einsum('klcd, acik, dblj -> abij', g[o, o, v, v], t_amp, t_amp)  
    ccd2b  = -ccd2a.transpose(1, 0, 2, 3) 
    ccd2c  = -ccd2a.transpose(0, 1, 3, 2) 
    ccd2d  =  ccd2a.transpose(1, 0, 3, 2) 
    ccd2   =  0.5 * (ccd2a + ccd2b + ccd2c + ccd2d)

    ccd3a  =  np.einsum('klcd, cakl, dbij -> abij', g[o, o, v, v], t_amp, t_amp)
    ccd3b  = -ccd3a.transpose(1, 0, 2, 3) 
    ccd3   = -0.5 * (ccd3a + ccd3b)

    ccd4a  =  np.einsum('klcd, cdki, ablj -> abij', g[o, o, v, v], t_amp, t_amp)
    ccd4b  = -ccd4a.transpose(0, 1, 3, 2) 
    ccd4   = -0.5 * (ccd4a + ccd4b)

    t_amp_new = e_abij * (mp2 + cepa1 + cepa2 + cepa3 + ccd1 + ccd2 + ccd3 + ccd4)
    E_CCD = (1 / 4) * np.einsum('ijab, abij ->', g[o, o, v, v], t_amp_new)
    t_amp = t_amp_new
    dE = E_CCD - E_old
    print('CCD Iteration %3d: Energy = %4.12f dE = %1.5E' % (cc_iter, E_CCD, dE))

    if abs(dE) < 1.e-10:
        print("\nCCD Iterations have converged!")
        break
    
    if (cc_iter == MAXITER):
        psi4.core.clean()
        raise Exception("\nMaximum number of iterations exceeded.")


print('\nCCD Correlation Energy: %5.15f' % (E_CCD))
print('CCD Total Energy: %5.15f' % (E_CCD + scf_e))

psi4_bccd = psi4.energy('bccd', ref_wfn = scf_wfn)
print('\nPsi4 BCCD Correlation Energy:     ', psi4_bccd - scf_e)
print('Psi4 BCCD Total Energy:     ', psi4_bccd)

