import psi4
import numpy as np
import molecule

def obara_saika_recursion(PA, PB, alpha, AMa, AMb):
    """
    Performs Obara-Saika recursion routine to fill in all integral cartesian components recursively
    Used to construct overlap, kinetic, and dipole integral x, y, z component arrays
    Parameters
    ----------
    PA: wighted coordinate vector on atom A
    PB: weighted coordinate vector on atom B
    alpha: orbital exponent
    AMa: angular momentum of A
    AMb: angular momentum of B 
    """
    if len(PA) != 3 or len(PB) != 3:
        raise "PA and PB must be xyz coordinates."
   
    # Allocate space x, y, and z matrices
    # We add one because the equation for the kinetic energy
    # integrals require terms one beyond those in the overlap
    x = np.zeros((AMa + 1, AMb + 1))
    y = np.zeros((AMa + 1, AMb + 1))
    z = np.zeros((AMa + 1, AMb + 1))

    # Define 1/2alpha factor for convenience
    oo2a = 1.0 / (2.0 * alpha)

    # Set initial conditions (0a|0b) to 1.0 for each cartesian component
    x[0, 0] = y[0, 0] = z[0, 0] = 1.0

    
    # BEGIN RECURSION
    # Fill in the [0,1] position with PB
    if AMb > 0:
        x[0, 1] = PB[0]
        y[0, 1] = PB[1]
        z[0, 1] = PB[2]

    # Fill in the rest of row zero
    for b in range(1, AMb):
        x[0, b + 1] = PB[0] * x[0, b] + b * oo2a * x[0, b - 1]
        y[0, b + 1] = PB[1] * y[0, b] + b * oo2a * y[0, b - 1]
        z[0, b + 1] = PB[2] * z[0, b] + b * oo2a * z[0, b - 1]
    
    # Now, we have for each cartesian component
    # | 1.0  PB #  #|
    # |  0   0  0  0|
    # |  0   0  0  0| 
    # |  0   0  0  0|

    # Upward recursion in a for all b's
    # Fill in the [1,0] position with PA
    if AMa > 0:                                                 
        x[1, 0] = PA[0]
        y[1, 0] = PA[1]
        z[1, 0] = PA[2]
        
    # Now, we have for each cartesian component
    # | 1.0  PB #  #|
    # |  PA  0  0  0|
    # |  0   0  0  0| 
    # |  0   0  0  0|

        # Fill in the rest of row one
        for b in range(1, AMb + 1):
            x[1, b] = PA[0] * x[0, b] + b * oo2a * x[0, b - 1]
            y[1, b] = PA[1] * y[0, b] + b * oo2a * y[0, b - 1]
            z[1, b] = PA[2] * z[0, b] + b * oo2a * z[0, b - 1]
            
        # Now, we have for each cartesian component
        # | 1.0  PB #  #|
        # |  PA  #  #  #|
        # |  0   0  0  0| 
        # |  0   0  0  0|

        # Fill in the rest of column 0
        for a in range(1, AMa):
            x[a + 1, 0] = PA[0] * x[a, 0] + a * oo2a * x[a - 1, 0]
            y[a + 1, 0] = PA[1] * y[a, 0] + a * oo2a * y[a - 1, 0]
            z[a + 1, 0] = PA[2] * z[a, 0] + a * oo2a * z[a - 1, 0]
            
        # Now, we have for each cartesian component
        # | 1.0  PB #  #|
        # |  PA  #  #  #|
        # |  #   0  0  0| 
        # |  #   0  0  0|
    
        # Fill in the rest of the a'th row
            for b in range(1, AMb + 1):
                x[a + 1, b] = PA[0] * x[a, b] + a * oo2a * x[a - 1, b] + b * oo2a * x[a, b - 1]
                y[a + 1, b] = PA[1] * y[a, b] + a * oo2a * y[a - 1, b] + b * oo2a * y[a, b - 1]
                z[a + 1, b] = PA[2] * z[a, b] + a * oo2a * z[a - 1, b] + b * oo2a * z[a, b - 1]

        # Now, we have for each cartesian component
        # | 1.0  PB #  #|
        # |  PA  #  #  #|
        # |  #   #  #  #| 
        # |  #   #  #  #|
        
    # Return the results
    return (x, y, z)

def compute_integrals(mol):
    """
    Computes the integrals for hartree fock
    Parameters
    ----------
    A Molecule object
    Returns
    ------- 
    Overlap, Kinetic, Potential, and 2-electron integrals in a tuple
    """
    
    basis = mol.basis
    
    # make space to store the overlap, kinetic, and dipole integral matrices
    S = np.zeros((basis.nao(),basis.nao()))
    T = np.zeros((basis.nao(),basis.nao()))
    Dx = np.zeros((basis.nao(),basis.nao()))
    Dy = np.zeros((basis.nao(),basis.nao()))
    Dz = np.zeros((basis.nao(),basis.nao()))
    
    # loop over the shells, basis.nshell is the number of shells
    for i in range(basis.nshell()):
        for j in range(basis.nshell()):
            # basis.shell is a shell (1s, 2s, 2p, etc.)
            # for water, there are 5 shells: (H: 1s, H: 1s, O: 1s, 2s, 2p)
            ishell = basis.shell(i) 
            jshell = basis.shell(j)
            # each shell has some number of primitives which make up each component of a shell
            # sto-3g has 3 primitives for every component of every shell.
            nprimi = ishell.nprimitive 
            nprimj = jshell.nprimitive
            # loop over the primitives within a shell
            for a in range(nprimi):  
                for b in range(nprimj):
                    expa = ishell.exp(a) # exponents
                    expb = jshell.exp(b)
                    coefa = ishell.coef(a)  # coefficients
                    coefb = jshell.coef(b)
                    AMa = ishell.am  # angular momenta
                    AMb = jshell.am
                    # defining centers for each basis function 
                    # mol.x() returns the x coordinate of the atom given by ishell.ncenter
                    # we use this to define a coordinate vector for our centers
                    A = np.array([mol.geometry.x(ishell.ncenter), mol.geometry.y(ishell.ncenter), mol.geometry.z(ishell.ncenter)])
                    B = np.array([mol.geometry.x(jshell.ncenter), mol.geometry.y(jshell.ncenter), mol.geometry.z(jshell.ncenter)])
                    alpha = expa + expb
                    zeta = (expa * expb) / alpha
                    P = (expa * A + expb * B) / alpha
                    PA = P - A
                    PB = P - B
                    AB = A - B
                    start = (np.pi / alpha)**(3 / 2) * np.exp(-zeta * (AB[0]**2 + AB[1]**2 + AB[2]**2))
                    # call the recursion
                    x, y, z = obara_saika_recursion(PA, PB, alpha, AMa+1, AMb+1)
    
                    
                    # Basis function index where the shell begins
                    i_idx = ishell.function_index  
                    j_idx = jshell.function_index
                    
                    # We use counters to keep track of which component (e.g., p_x, p_y, p_z)
                    # within the shell we are on
                    counta = 0
                    
                    for p in range(AMa + 1):
                        la = AMa - p                    # Let l take on all values, and p be the leftover a.m.
                        for q in range(p + 1):
                            ma = p - q                  # distribute all leftover a.m. to m and n
                            na = q
                            countb = 0
                            for r in range(AMb + 1):
                                lb = AMb - r            # Let l take on all values, and r the leftover a.m.
                                for s in range(r + 1):
                                    mb = r - s          # distribute all leftover a.m. to m and n
                                    nb = s
                                    
                                    S[i_idx + counta, j_idx + countb] += start    \
                                                                       * coefa    \
                                                                       * coefb    \
                                                                       * x[la,lb] \
                                                                       * y[ma,mb] \
                                                                       * z[na,nb] 
                                                        
                                    Tx = (1 / 2) * (la * lb * x[la - 1, lb - 1] + 4 * expa * expb * x[la + 1, lb + 1] \
                                           - 2 * expa * lb * x[la + 1, lb - 1] - 2 * expb * la * x[la - 1, lb + 1])   \
                                           * y[ma, mb] * z[na, nb]
    
                                    Ty = (1 / 2) * (ma * mb * y[ma - 1, mb - 1] + 4 * expa * expb * y[ma + 1, mb + 1] \
                                           - 2 * expa * mb * y[ma + 1, mb - 1] - 2 * expb * ma * y[ma - 1, mb + 1])   \
                                           * x[la, lb] * z[na, nb]
    
                                    Tz = (1 / 2) * (na * nb * z[na - 1, nb - 1] + 4 * expa * expb * z[na + 1, nb + 1] \
                                           - 2 * expa * nb * z[na + 1, nb - 1] - 2 * expb * na * z[na - 1, nb + 1])   \
                                           * x[la, lb] * y[ma, mb]
    
                                    T[i_idx + counta, j_idx + countb] += start * coefa * coefb * (Tx + Ty + Tz)
    
    
                                    dx = (x[la + 1, lb] + A[0] * x[la, lb]) * y[ma, mb] * z[na, nb]
                                    dy = (y[ma + 1, mb] + A[1] * y[ma, mb]) * x[la, lb] * z[na, nb]
                                    dz = (z[na + 1, nb] + A[2] * z[na, nb]) * x[la, lb] * y[ma, mb]
    
                                    Dx[i_idx + counta, j_idx + countb] += start * coefa * coefb * dx
                                    Dy[i_idx + counta, j_idx + countb] += start * coefa * coefb * dy
                                    Dz[i_idx + counta, j_idx + countb] += start * coefa * coefb * dz
                                    
                                    countb += 1
                            counta += 1
    mints = psi4.core.MintsHelper(basis)
    # Repulsion and two electron integrals are too hard to implement efficiently. Give up.
    V = mints.ao_potential().to_array()
    I = mints.ao_eri().to_array()
    return S, T, V, I


