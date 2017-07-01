import psi4
import numpy as np
import math
from collections import namedtuple
import configparser
config = configparser.ConfigParser()
config.read('Options.ini')

molecule = psi4.geometry(config['DEFAULT']['molecule'])
basis = psi4.core.BasisSet.build(molecule, 'BASIS', config['DEFAULT']['basis'], puream=0)
mints = psi4.core.MintsHelper(basis)

def os_recursion(PA, PB, alpha, AMa, AMb, x, y, z):
    if len(PA) != 3 or len(PB) != 3:
        raise ""
        
    # Allocate the x, y, and z matrices.
    #  Why do I add 1 here? so ang mom of 0 will still give a matrix
    # 
#    x = np.zeros((AMa+1, AMb+1))
#    y = np.zeros((AMa+1, AMb+1))
#    z = np.zeros((AMa+1, AMb+1))
    
    # Perform the recursion
#    x[0,0] = 1
#    y[0,0] = 1
#    z[0,0] = 1
    for i in range(0,AMa): #First column
        x[i+1,0] = PA[0]*x[i,0] + (1/(2*alpha))*i*x[i-1,0]
        y[i+1,0] = PA[1]*y[i,0] + (1/(2*alpha))*i*y[i-1,0] 
        z[i+1,0] = PA[2]*z[i,0] + (1/(2*alpha))*i*z[i-1,0]
    for i in range(0,AMb): #first row
        x[0,i+1] = PB[0]*x[0,i] + (1/(2*alpha))*i*x[0,i-1]  
        y[0,i+1] = PB[1]*y[0,i] + (1/(2*alpha))*i*y[0,i-1] 
        z[0,i+1] = PB[2]*z[0,i] + (1/(2*alpha))*i*z[0,i-1]
   
        for j in range(0,AMa):
            if j == 0:
               x[j+1,i+1] = PA[0]*x[j,i+1] + (1/(2*alpha))*(j)*x[j,i] + (1/(2*alpha))*(i+1)*x[j,i]    
               y[j+1,i+1] = PA[1]*y[j,i+1] + (1/(2*alpha))*(j)*y[j,i] + (1/(2*alpha))*(i+1)*y[j,i]
               z[j+1,i+1] = PA[2]*z[j,i+1] + (1/(2*alpha))*(j)*z[j,i] + (1/(2*alpha))*(i+1)*z[j,i]
            elif j > 0:
                x[j+1,i+1] = PA[0]*x[j,i+1] + (1/(2*alpha))*j*x[j-1,i+1] + (1/(2*alpha))*(i+1)*x[j,i]   
                y[j+1,i+1] = PA[1]*y[j,i+1] + (1/(2*alpha))*j*y[j-1,i+1] + (1/(2*alpha))*(i+1)*y[j,i]
                z[j+1,i+1] = PA[2]*z[j,i+1] + (1/(2*alpha))*j*z[j-1,i+1] + (1/(2*alpha))*(i+1)*z[j,i]
         
    # Return the results
    #return RecursionResults(x, y, z)
    #print(x)

def integrals():
    S = np.zeros((basis.nao(),basis.nao()))
    T = np.zeros((basis.nao(),basis.nao()))
    D = np.zeros((basis.nao(),basis.nao()))
    Dx = np.zeros((basis.nao(),basis.nao()))
    Dy = np.zeros((basis.nao(),basis.nao()))
    Dz = np.zeros((basis.nao(),basis.nao()))
    for i in range(basis.nshell()):
        for j in range(basis.nshell()):
             ishell = basis.shell(i) #basis.shell is a basis function of a shell. Doesn't pick out px py pz.
             jshell = basis.shell(j) 
             nprimi = ishell.nprimitive # and finding the number of primitives in each shell(1s, 2s, 2p, etc)which is 3 for STO-3G
             nprimj = jshell.nprimitive 
             for p in range(nprimi):  #for each primitive, grab the orbital exponent of that primitive
                 for q in range(nprimj):
                    expp = ishell.exp(p)
                    expq = jshell.exp(q)
                    alpha = expp + expq #alpha is the sum of primitive exponents
                    zeta = (expp*expq)/alpha #zeta is just for (s|s) to set your 0,0 elements HERE ZETA is changing is that supposed tobe?
                    #defining centers for each basis function, ishell
                    A = [molecule.x(ishell.ncenter), molecule.y(ishell.ncenter), molecule.z(ishell.ncenter)]           
                    B = [molecule.x(jshell.ncenter), molecule.y(jshell.ncenter), molecule.z(jshell.ncenter)]           
                    A = np.array(A) #putting them into an array
                    B = np.array(B)   
                    P = (expp*A + expq*B)/(alpha) 
                    PA = P-A   #get PA, PB for our recursion
                    PB = P-B
                    AMa = ishell.am #grab the total angular momentum of each basis function
                    AMb = jshell.am
                    x = np.zeros((AMa+2, AMb+2)) #changed to +2 to increase the number of generated terms to include 
                    y = np.zeros((AMa+2, AMb+2)) # this covers the +1|-1 in our kinetic equation 
                    z = np.zeros((AMa+2, AMb+2)) # we have to set a +1|-1 element to 
                    #set the (0 | 0) element to the first element of our matrices
                    #exponent is 1/2 since the x y and z components are multiplied together to get 3/2
                    x[0,0] = (math.pi/alpha)**(1/2)*np.exp(-zeta*(A[0]-B[0])**2)  
                    y[0,0] = (math.pi/alpha)**(1/2)*np.exp(-zeta*(A[1]-B[1])**2)  
                    z[0,0] = (math.pi/alpha)**(1/2)*np.exp(-zeta*(A[2]-B[2])**2)  
                    os_recursion(PA,PB, alpha, AMa+1, AMb+1, x, y, z)
                    #generated recursive x, recursive y, and recursive z matrices
                    #currently iterating over shells, without distinguishing px py pz 
                    counter1 = 0
                    for ii in range(AMa+1):
                        L1 = AMa - ii   #AMa is the total angular momentum of orbital a, let L1 take on all values, and ii be whats left over
                        for jj in range(ii+1): #allocate the leftover angular momentum to M1 and N1 to get all possible angular momentum combinations
                            M1 = ii - jj
                            N1 = jj
                    #each time this iterates through, we get a unique tuple
                    #we can just do this loop again for the other dimension of our S matrix         
                            counter2 = 0
                            for aa in range(AMb+1):
                                L2 = AMb - aa 
                                for bb in range(aa+1): 
                                    M2 = aa-bb
                                    N2 = bb
                                    S[ishell.function_index+counter1,jshell.function_index+counter2] += (ishell.coef(p))*(jshell.coef(q))*x[L1,L2]*y[M1,M2]*z[N1,N2] 
                                    Tx = (1/2)*(L1*L2*x[L1-1,L2-1] + 4*expp*expq*x[L1+1,L2+1] - 2*expp*L2*x[L1+1,L2-1] - 2*expq*L1*x[L1-1,L2+1])*y[M1,M2]*z[N1,N2]
                                    Ty = (1/2)*(M1*M2*y[M1-1,M2-1] + 4*expp*expq*y[M1+1,M2+1] - 2*expp*M2*y[M1+1,M2-1] - 2*expq*M1*y[M1-1,M2+1])*x[L1,L2]*z[N1,N2]
                                    Tz = (1/2)*(N1*N2*z[N1-1,N2-1] + 4*expp*expq*z[N1+1,N2+1] - 2*expp*N2*z[N1+1,N2-1] - 2*expq*N1*z[N1-1,N2+1])*x[L1,L2]*y[M1,M2]
                                    T[ishell.function_index+counter1,jshell.function_index+counter2] += (ishell.coef(p))*(jshell.coef(q))*(Tx + Ty + Tz)
                            
                                    dx = x[L1+1,L2]*y[M1,M2]*z[N1,N2] + A[0]*x[L1,L2]*y[M1,M2]*z[N1,N2] 
                                    dy = y[M1+1,M2]*x[L1,L2]*z[N1,N2] + A[1]*x[L1,L2]*y[M1,M2]*z[N1,N2] 
                                    dz = z[N1+1,N2]*x[L1,L2]*y[M1,M2] + A[2]*x[L1,L2]*y[M1,M2]*z[N1,N2] 
                                    Dx[ishell.function_index+counter1,jshell.function_index+counter2] += dx*(ishell.coef(p))*(jshell.coef(q))
                                    Dy[ishell.function_index+counter1,jshell.function_index+counter2] += dy*(ishell.coef(p))*(jshell.coef(q))
                                    Dz[ishell.function_index+counter1,jshell.function_index+counter2] += dz*(ishell.coef(p))*(jshell.coef(q))
                                    counter2 += 1
                            counter1 += 1    
    return (S, T, Dx, Dy, Dz)

if __name__ == '__main__':
    integrals()

