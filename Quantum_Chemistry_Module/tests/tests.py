"""
Testing module functions
"""

import Quantum_Chemistry_Module as qcm
import psi4
import pytest
import numpy as np


geom = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

my_mol = molecule.Molecule(0, 1, geom, "sto-3g")

def test_integrals(molecule = my_mol):
    S, T, V, I = integrals.compute_integrals(molecule)    
    mints = psi4.core.MintsHelper(molecule.basis)
     
    Spsi4 = np.asarray(mints.ao_overlap())
    assert np.allclose(S, Spsi4, 6)
    
    Tpsi4 = np.asarray(mints.ao_kinetic())
    assert np.allclose(T, Tpsi4, 6)
    
    Vpsi4 = np.asarray(mints.ao_potential())
    assert np.allclose(V, Vpsi4, 6)

    Ipsi4 = np.asarray(mints.ao_eri())
    assert np.allclose(I, Ipsi4, 6)
