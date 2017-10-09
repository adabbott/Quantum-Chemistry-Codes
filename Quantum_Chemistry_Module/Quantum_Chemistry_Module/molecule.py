# This file defines a molecule class that stores information about the molecule.

import psi4
import numpy as np
psi4.core.be_quiet()

class Molecule(object):
    """
    A molecule class that is stores basic parameters, including
    psi4 geometry, basis set, charge, multiplicty, number of electrons, and electron occupations
    """

    def __init__(self, charge=None, mult=None, geometry=None, basis=None):
        """
        Initializes molecule with geometry, basisset, charge, multiplicity
        Calculates number of electrons and occupations 

        Parameters
        ----------
        charge : int 
            The molecular charge
        mult : int  
            The spin multiplicity (1 = singlet, 2 = doublet, 3 = triplet)
        geometry : str or psi4.core.Molecule
              psi4 geometry
        basis : str
              basis set string recognizable by Psi4 basis library

        Returns
        -------
        A built molecule

        """

        self.charge = charge
        self.mult = mult
        self.geometry = geometry
        self.basis = basis

        self.nuclear_repulsion_energy = None
        self.number_of_electrons = None
        self.ndocc = None
        self.nsocc = None

#        # Sets molecule data
        if geometry is not None:
            self.set_geometry(geometry)
#
        if basis is not None:
            self.set_basis(basis)
#
#        if charge is not None:
#            self.set_charge(charge)
#    
#        if mult is not None:
#            self.set_mult(mult)

        if ((geometry is not None) and (charge is not None) and (mult is not None)): 
            self.set_electron_occupations()
            
    def set_geometry(self, geometry_string):
        """
        Set the molecules geometry.
        Parameters 
        ----------
        geometry_string : str
        Geometry string in the form of any valid psi4 geometry 
        """
        if isinstance(geometry_string, str):
            self.geometry = psi4.geometry(geometry_string) 
        elif isinstance(geometry_string, psi4.core.Molecule):
            self.geometry = geometry_string
        else:
            raise TypeError("Input must either be a geometry array or Psi4 geometry string")
        self.geometry.update_geometry()
        # grab nuclear repulsion while convenient
        self.nuclear_repulsion_energy = self.geometry.nuclear_repulsion_energy() 
        
       
    def set_basis(self, basis_str):
        """
        Set the basis set.
        Parameters 
        ----------
        basis_str : str
        Psi4 basis set name 
        """
        
        if (self.geometry is None):
            raise Exception("Error: Geometry not defined") 
        else:
            self.basis = psi4.core.BasisSet.build(self.geometry, "BASIS", basis_str, puream=0) # our hand coded integrals use cartesian Angular momentum

    def set_charge(self, charge_int):
        if (self.geometry is None):
            raise Exception("Error: Geometry not defined") 
        if isinstance(charge_int, int):
            self.geometry.set_molecular_charge(charge_int)
            self.charge = int(charge_int)
        else:
            raise TypeError("Input must be an integer for charge")

    def set_mult(self, mult_int):
        if (self.geometry is None):
            raise Exception("Error: Geometry not defined") 
        if isinstance(mult_int, int):
            self.geometry.set_multiplicity(self.mult)
            self.mult = int(mult_int) 
        else:
            raise TypeError("Input must be an integer for charge")
    
    def set_electron_occupations(self):
        if (self.geometry is None):
            raise Exception("Error: Geometry not set") 
        if (self.charge is None):
            raise Exception("Error: Molecular charge not set") 
        if (self.mult is None):
            raise Exception("Error: Multiplicity not set") 

        protons = 0
        for atom_index in range(self.geometry.natom()):
            protons += self.geometry.charge(atom_index)

        self.number_of_electrons = int(-self.charge + protons)
        self.ndocc = int(np.round(self.number_of_electrons * 0.5) - self.mult + 1) # Number of doubly occupied orbitals
        if self.ndocc < 0: # cover cases with more unpaired electrons than paired
            self.ndocc = 0
        self.nsocc = int(self.mult - 1)      # Number of singly occupied orbitals
        # check for sane charge and multiplicity combination
        if ((2 * self.ndocc + self.nsocc) != self.number_of_electrons):
            raise Exception("Error: Charge and multiplicity combination is impossible") 
