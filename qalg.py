from qlibcj import *
def DJAlg(size, U_f):
    r = QRegistry(([QZero() for i in range(0, size - 1)] + [QOne()]))
    r.ApplyGate(Hadamard(size))
    r.ApplyGate(U_f)
    r.ApplyGate(np.kron(Hadamard(size - 1), I(1)))
    return r.Measure([0])

def Bal(n):
    b = I(n)
    for i in range(int((2**n)/2), (2**n) - 1, 2):
        t = np.copy(b[i,:])
        b[i], b[i+1] = b[i+1, :], t
    return b
