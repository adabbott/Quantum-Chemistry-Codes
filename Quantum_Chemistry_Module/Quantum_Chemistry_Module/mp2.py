import hartree_fock
import numpy as np
import integrals
import spin_orbital_setup

def mp2(mol):
    """MP2 energy using conventional ERIs

    Parameters
    ----------
    mol: a molecule object
    Returns
    --------
    mp2_total_energy: float
        The mp2 energy, scf_energy + mp2 correlation energy
    """
    E_SCF, C_a, C_b, ea, eb = hartree_fock.UHF(mol)
    S, T, V, g_ao = integrals.compute_integrals(mol)
    I_phys, C, eps = spin_orbital_setup.spin_orbital(C_a, C_b, ea, eb, g_ao)
    nocc  = mol.ndocc * 2 + mol.nsocc

    gmo = np.einsum('pQRS, pP -> PQRS', 
          np.einsum('pqRS, qQ -> pQRS', 
          np.einsum('pqrS, rR -> pqRS', 
          np.einsum('pqrs, sS -> pqrS', I_phys, C), C), C), C) 
    
    # Form 4-index tensor of orbital energy denominators
    n = np.newaxis
    o = slice(None, nocc)
    v = slice(nocc, None)
    eps = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    # Compute energy
    E_mp2 = (1 / 4) * np.einsum('ijab, abij ->', gmo[o, o, v, v], gmo[v, v, o, o] * eps)

    mp2_total_energy = E_mp2 + E_SCF
    print("MP2 Correlation Energy: " + str(E_mp2))
    print("MP2 Total Energy: " + str(mp2_total_energy))
    return mp2_total_energy
