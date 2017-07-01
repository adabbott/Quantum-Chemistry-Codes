import numpy as np

print('Matrix C:')
C = np.array([[1,2],[3,4]])
print(C)
print('Matrix D:')
D = np.array([[0,5],[10,15]])
print(D)
Y = np.einsum('ij,jk->ik', C, D)
print('einsum(ij,jk->ik, C, D')
print(Y)

A = np.array([0, 1, 2])

B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
print('Matrix A:')
print(A)
print('Matrix B:')
print(B)
Z = np.einsum('i,ij->ij', A, B)
print('einsum(i,ij->ij, A, B')
print(Z)
Z = np.einsum('i,ij->j', A, B)
print('einsum(i,ij->j, A, B')
print(Z)
