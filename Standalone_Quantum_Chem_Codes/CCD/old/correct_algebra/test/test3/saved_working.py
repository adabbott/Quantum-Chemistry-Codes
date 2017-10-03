import psi4
import numpy as np
import configparser
from scipy.linalg import block_diag
import scipy.linalg as la
import time
import itertools as it
from sympy.combinatorics.permutations import Permutation

config = configparser.ConfigParser()
config.read('Options.ini') #pointing to our options.ini file for info on molecule
molecule = psi4.geometry(config['DEFAULT']['molecule'])

#get basis set
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
#set up integrals
mints = psi4.core.MintsHelper(basis)
molecule.update_geometry()

nalpha = int(config['DEFAULT']['nalpha'])
nbeta = int(config['DEFAULT']['nbeta'])
nocc = nalpha + nbeta
ntotal = 2*mints.basisset().nbf()
nvirt = ntotal-nocc

# Overlap integrals 
I = mints.ao_eri().to_array()

#dim = np.shape(I)
#ndim = len(dim)
#std_indices = list(range(ndim))
#
#my_p = [[0,1],[2,3]]
#p2 = [[0,1]]
#print(Permutation(p2).list() )

#print(Permutation(p2).inversions() )
#p = Permutation( my_p )
#print(p)
#print(p.list())
#print(p.inversions())

#print(new_array)
##each 1 permutation subset
#for i in range(len(my_p)):
#    p = Permutation([my_p[i]])
#    perm = p.list(4)    
#    new_array +=  (-1)**(p.inversions()) * I.transpose(tuple(perm))
#     
##full permutation
#full_p = [[0,1],[2,3]]
#full_p_perm = Permutation(full_p)
#full_p_perm_list = full_p_perm.list(4)
#new_array += (-1)**(full_p_perm.inversions()) * I.transpose(tuple(full_p_perm))


#print(new_array)
#def findsubsets(S,m):
#    return list(it.combinations(S,m))


#test_list = [[0,1], [2,3], [4,5] ] 
#for i in list(it.combinations(test_list,2)):
#    print(list(i))


#def permute(array, permutations = [[0,1],[2,3]]
array = I
dim = np.shape(I)
ndim = len(dim)
new_array = I*1

my_list = [[0,1], [2,3]]
# create every possible subset of my_list in the form [[A1,..,An]]
for i in range(1, len(my_list) + 1):
    for j in list(it.combinations(my_list, i)):
        #print(list(j))
        PERMUTATION = Permutation(list(j))
        print(PERMUTATION)
        PERMUTATION_LIST = PERMUTATION.list(ndim)
        print(PERMUTATION_LIST)
        new_array +=  (-1)**(PERMUTATION.inversions()) * I.transpose(tuple(PERMUTATION_LIST))


#print(new_array)

#create functional form

