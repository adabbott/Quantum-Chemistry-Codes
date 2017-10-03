import psi4
import numpy as np

import molecule
geom = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")


my_mol = molecule.Molecule(0, 1, geom, "sto-3g")

print("The charge of the molecule is")
print(my_mol.charge)
print("The multiplicity of the molecule is")
print(my_mol.mult)
print("The geometry of the molecule is")
my_mol.geometry.print_out()
print("The basis of the molecule is")
my_mol.basis.print_out()
print("The nuclear repulsion energy of the molecule is")
print(my_mol.nuclear_repulsion_energy)
print("The number of electrons in the molecule is")
print(my_mol.number_of_electrons)
print("The number of doubly occupied orbitals in the molecule is")
print(my_mol.ndocc)
print("The number of singly occupied orbitals in the molecule is")
print(my_mol.nsocc)