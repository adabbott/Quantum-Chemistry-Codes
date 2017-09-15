import numpy as np
from collections import namedtuple
RecursionResults = namedtuple('RecursionResults', ['x', 'y', 'z'])

def os_recursion(PA, PB, alpha, AMa, AMb):
    if len(PA) != 3 or len(PB) != 3:
        raise ""
        
    # Allocate the x, y, and z matrices.
    #  Why do I add 1 here? so ang mom of 0 will still give a matrix
    # 
    x = np.zeros((AMa+1, AMb+1))
    y = np.zeros((AMa+1, AMb+1))
    z = np.zeros((AMa+1, AMb+1))
    
    # Perform the recursion
    x[0,0] = 1
    y[0,0] = 1
    z[0,0] = 1
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
    return RecursionResults(x, y, z)

#results = os_recursion([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1, 1)
results = os_recursion([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 5.0, 3, 3)
print(results.x)
print(results.y)
print(results.z)
