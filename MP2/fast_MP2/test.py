import numpy as np

A = np.array([1,2,3,4,5])
B = np.array([1,2,3,4,5])

print(A)
print(B)
#print(np.kron(A,B))
#print(np.einsum('p,p,p,p->p',A,A,A,A))


print(np.tensordot(A,B.T,0))

