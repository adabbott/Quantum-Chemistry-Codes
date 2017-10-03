import numpy as np

A = np.ones((3,4,5,6))
B = np.ones((7, 8, 9))
identity = np.eye(2)
C = np.ones((11, 12, 13 , 14, 15, 16))
print(np.shape(C))
print(np.shape(C.T))

print('Initial 4d shape:')
print(np.shape(A))
Aa = np.kron(A, identity)
print('Shape after tensor product with 2x2 Identity:')
print(np.shape(Aa))
print('Shape after transposing the tensor product:')
print(np.shape(Aa.T))
print('Shape after tensoring the transpose with the 2x2 Identity:')
Ab = np.kron(Aa.T, identity)
print(np.shape(Ab))
print('BREAK')
print('Shape after tensor product with the 4x4 Identity:')
print(np.shape(np.kron(A, np.eye(4))))