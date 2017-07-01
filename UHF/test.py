import psi4
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('newoptions.ini')


molecule = psi4.geometry(config['DEFAULT']['molecule'])
psi4.set_options({'basis': 'sto-3g', 'scf_type': 'pk','guess': 'core','reference': 'uhf','e_convergence': 1e-10})
psi4.energy('scf')
