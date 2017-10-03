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
mol.set_molecular_charge(0)
mol.set_multiplicity(1)
mol.update_geometry()
print(mol.charge(0))
