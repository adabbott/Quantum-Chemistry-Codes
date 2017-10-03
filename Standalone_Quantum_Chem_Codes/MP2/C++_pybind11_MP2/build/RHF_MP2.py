import numpy as np
import timeit
import psi4
import mp2
import time

psi4.core.be_quiet()
np.set_printoptions(suppress=True, precision=4)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()
mol.print_out()

e_conv = 1.e-8
d_conv = 1.e-8
nocc = 5
damp_value = 0.20
damp_start = 5

# Build a basis
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVTZ", key='basis')
bas.print_out()

# Build a MintsHelper
mints = psi4.core.MintsHelper(bas)
nbf = mints.nbf()
nvirt = nbf - nocc


V = np.array(mints.ao_potential())
T = np.array(mints.ao_kinetic())

# Core Hamiltonian
H = T + V

S = np.array(mints.ao_overlap())
g = np.array(mints.ao_eri())


A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)



# Diagonalize Core H
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


eps, C = diag(H, A)
Cocc = C[:, :nocc]
D = Cocc @ Cocc.T

E_old = 0.0
F_old = None
for iteration in range(25):
    # F_pq = H_pq + 2 * g_pqrs D_rs - g_prqs D_rs

    # g = (7, 7, 7, 7)
    # D = (1, 1, 7, 7)
    # Jsum = np.sum(g * D, axis=(2, 3))
    J = np.einsum("pqrs,rs->pq", g, D)
    K = np.einsum("prqs,rs->pq", g, D)

    F_new = H + 2.0 * J - K

    # conditional iteration > start_damp
    if iteration >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new

    F_old = F_new
    # F = (damp_value) Fold + (??) Fnew

    # Build the AO gradient
    grad = F @ D @ S - S @ D @ F

    grad_rms = np.mean(grad ** 2) ** 0.5

    # Build the energy
    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()

    E_diff = E_total - E_old
    E_old = E_total

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    eps, C = diag(F, A)
    Cocc = C[:, :nocc]
    D = Cocc @ Cocc.T

print("SCF has finished!\n")

psi4.set_options({"scf_type": "pk", "mp2_type": "conv"})
psi4_energy = psi4.energy("SCF/aug-cc-pVTZ", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))

# do MP2
def trans_MO(g):
    O = slice(None, nocc)
    V = slice(nocc, None)
    g_iajb = np.einsum('pQRS, pP -> PQRS',
        np.einsum('pqRS, qQ -> pQRS',
        np.einsum('pqrS, rR -> pqRS',
        np.einsum('pqrs, sS -> pqrS', g, C[:,V]), C[:,O]), C[:,V]), C[:,O]) 
    return g_iajb


g_iajb = trans_MO(g)

def mp2_energy_v2(g_iajb, eps, nocc):
    eocc = eps[:nocc]
    evir = eps[nocc:]
    e_iajb = 1 / (eocc.reshape(-1, 1, 1, 1) - evir.reshape(-1, 1, 1) + eocc.reshape( -1, 1) - evir)    
#    os = np.einsum("iajb,iajb,iajb->", g_iajb, g_iajb, e_iajb) 
#    g_ibja = g_iajb.swapaxes(1, 3)    
#    ss = np.einsum("iajb, iajb, iajb->", (g_iajb - g_ibja), g_iajb, e_iajb)
#    E_mp2 = os + ss
    E_mp2 = np.einsum('iajb, iajb->', g_iajb * (2 * g_iajb - g_iajb.swapaxes(1,3)), e_iajb)
    return E_mp2
start = timeit.default_timer()
v2 = mp2_energy_v2(g_iajb, eps, nocc)
print(timeit.default_timer() - start)

start2 = timeit.default_timer()
v3 = mp2.mp2_e(g_iajb, eps, nocc)
print(timeit.default_timer() - start2)

print('MP2 correlation energy: %10.10f ' % (v2) )
print('MP2 correlation energy: %10.10f ' % (v3) )

psi_mp2 = psi4.energy("mp2/aug-cc-pVTZ", molecule=mol)
print('psi4 mp2 energy', psi_mp2-psi4_energy)
