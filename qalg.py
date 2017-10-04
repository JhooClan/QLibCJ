from qlibcj import *
def DJAlg(size, U_f): # Implementation of Deustch-Josza algorithm
    r = QRegistry(([QZero() for i in range(0, size - 1)] + [QOne()])) # Initializing registry with size-1 QZero and one QOne
    r.ApplyGate(Hadamard(size)) # Applying Hadamard to all QuBits
    r.ApplyGate(U_f) # Applying oracle
    r.ApplyGate(np.kron(Hadamard(size - 1), I(1))) # Applying Hadamard to all but the last QuBit
    return r.Measure([0], False) # Measuring the first QuBit

def Bal(n): # Creates an U_f for a balanced f
    b = I(n)
    for i in range(int((2**n)/2), (2**n) - 1, 2):
        t = np.copy(b[i,:])
        b[i], b[i+1] = b[i+1, :], t
    return b

def Teleportation(qbit):
    r = QRegistry([qbit, QZero(), QZero()])
    print ("Original registry:\n", r.state)
    r.ApplyGate(np.kron(I(1), np.kron(Hadamard(1), I(1))))
    r.ApplyGate(np.kron(I(1), CNOT()))
    print ("With Bell+ state:\n", r.state)
    r.ApplyGate(np.kron(CNOT(), I(1)))
    r.ApplyGate(np.kron(Hadamard(1), I(2)))
    print ("\nBefore measurement:\n", r.state)
    m = r.Measure([0,1], False)
    print ("q0 = ", m[0], "\nq1 = ", m[1])
    q0 = QZero()
    q1 = QZero()
    if (m[1] == 1):
        q1 = QOne()
        r.ApplyGate(np.kron(I(2), PauliX()))
    if (m[0] == 1):
        q0 = QOne()
        r.ApplyGate(np.kron(I(2), PauliZ()))
    er = QRegistry([q0, q1, qbit])
    print ("\nExpected result:\n", er.state, "\nResult:\n", r.state)
    return r
