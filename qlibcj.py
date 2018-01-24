import math as m
import cmath as cm
import numpy as np
import random as rnd
# np.zeros((h,w), dtype=complex) Inicializa una matriz de numeros complejos con alto h y ancho w
# La suma de matrices se realiza con +. A + B
# La multiplicacion por un escalar se hace con *. n * A
# Para multiplicar las matrices A y B se usa np.dot(A,B)
# El producto Kronecker de A y B esta definido con np.kron(A,B)

class QRegistry:
    def __init__(self, qbits, **kwargs):    # QuBit list. Seed for the Pseudo Random Number Generation can be specified with seed = <seed> as an argument.
        if (type(qbits) != list or \
            not qbits or \
            not all(type(qbit) == np.ndarray and \
                    qbit.shape == (1,2) and \
                    qbit.dtype == 'complex128' for qbit in qbits)):
            raise ValueError('Impossible QuBit Registry')
        qbs = qbits[:]

        self.state = qbs[0]
        Normalize(self.state)
        del qbs[0]
        for qbit in qbs:
            Normalize(qbit)
            self.state = np.kron(self.state, qbit)
        Normalize(self.state)
        if (kwargs.get('seed', None) != None):
            rnd.seed(kwargs['seed'])

    def Measure(self, mask, remove = False): # List of numbers with the QuBits that should be measured, numerated as q0..qn. If you want to measure q2 and q4, the imput will be [2,4]. remove = True if you want to remove a QuBit from the registry after measuring
        if (type(mask) != list or \
            not all(type(num) == int for num in mask)):
            raise ValueError('Not valid mask')
        tq = m.log(self.state.size,  2)
        if (not all(num < tq and num > -1 for num in mask)):
            raise ValueError('Out of range')
        measure = []
        for qbit in mask:
            r = rnd.random()
            p = 0
            max = 2**(tq - (qbit + 1))
            cnt = 0
            rdy = True
            for i in range(0, self.state.size):
                if (cnt == max):
                    rdy = not rdy
                    cnt = 0
                if (rdy):
                    p += cm.polar(self.state[0,i])[0]**2
                cnt += 1
            if (r < p):
                me = 0
            else:
                me = 1
            measure.append(me)
            self.Collapse((tq - (qbit + 1)), me, remove)
        return measure

    def ApplyGate(self, *gates): # Applies a quantum gate to the registry.
        gate = gates[0]
        for i in range(1, len(gates)):
            gate = np.kron(gate, gates[i])
        self.state = np.dot(Bra(self.state), gate)
    
    def Collapse(self, qbit, measure, remove): # Collapses a qubit from the registry. qbit is the id of the qubit, numerated as q0..qn in the registry. measure is the value obtained when measuring it. remove indicates whether it should be removed from the registry.
        max = 2**qbit
        cnt = 0
        rdy = measure == 1
        mfd = []
        for i in range(0, self.state.size):
            if (cnt == max):
                rdy = not rdy
                cnt = 0
            if (rdy):
                self.state[0, i] = 0
                mfd.append(i)
            cnt += 1
        if (remove):
            for qbit in mfd[::-1]:
                self.state = np.delete(self.state, qbit, 1)
        Normalize(self.state)
    def DensityMatrix(self):
        return np.dot(Ket(self.state), Bra(self.state))
    def VNEntropy(self):
        #dm = self.DensityMatrix()
        #evalues, m = np.linalg.eig(dm)
        entropy = 0
        #for e in evalues:
        #    if e != 0:
        #        entropy += e * np.log(e)
        for amp in self.state[0]:
            p = cm.polar(amp)[0]**2
            if p > 0:
                entropy += (p * np.log(p))
        return -entropy


def Prob(q, x): # Devuelve la probabilidad de obtener x al medir el qbit q
    p = 0
    if (x < q.size):
        p = cm.polar(q[0,x])[0]**2
    return p

def Hadamard(n): # Devuelve una puerta Hadamard para n QuBits
    H = 1 / m.sqrt(2) * np.ones((2,2), dtype=complex)
    H[1,1] = -1 / m.sqrt(2)
    if n > 1:
        H = np.kron(H, Hadamard(n - 1))
    return H

def QBit(a,b): # Devuelve un QuBit con a y b. q = a|0> + b|1>, con a y b complejos
    q = np.array([a,b], dtype=complex)
    q.shape = (1,2)
    Normalize(q)
    return q

def QZero(): # Devuelve un QuBit en el estado 0
    q = np.array([complex(1,0),complex(0,0)])
    q.shape = (1,2)
    return q

def QOne(): # Devuelve un QuBit en el estado 1
    q = np.array([complex(0,0),complex(1,0)])
    q.shape = (1,2)
    return q

def PauliX(): # Also known as NOT
    px = np.array([0,1,1,0], dtype=complex)
    px.shape = (2,2)
    return px

def PauliY():
    py = np.array([0,-1j,1j,0], dtype=complex)
    py.shape = (2,2)
    return py

def PauliZ():
    pz = np.array([1,0,0,-1], dtype=complex)
    pz.shape = (2,2)
    return pz

def Bra(v): # Devuelve el QuBit pasado como parametro en forma de fila. <q|
    b = v[:]
    s = v.shape
    if s[0] != 1:
        b = np.transpose(b)
    return b

def Ket(v): # Devuelve el QuBit pasado como parametro en forma de columna. |q>
    k = v[:]
    s = v.shape
    if s[1] != 1:
        k = np.transpose(k)
    return k

def ApplyGate(state, gate): # Devuelve el resultado de aplicar una puerta logica sobre una serie de estados. El QuBit mas significativo a la izquierda de la lista.
    return np.dot(Bra(state), gate)

def Superposition(x, y): # Devuelve el estado compuesto por los dos QuBits.
    z = np.kron(x, y)
    Normalize(z)
    return z

def Normalize(state): # Funcion que asegura que se cumpla la propiedad que dice que |a|^2 + |b|^2 = 1 para cualquier QuBit. Si no se cumple, modifica el QuBit para que la cumpla si se puede.
    sqs = 0
    for i in range(0, state.size):
        sqs += cm.polar(state[0, i])[0]**2
    sqs = m.sqrt(sqs)
    if (sqs == 0):
        raise ValueError('Impossible QuBit')
    if (sqs != 1):
        for bs in state:
            bs /= sqs

def CNOT(): # Devuelve una compuerta CNOT para dos QuBits
    cn = np.zeros((4,4), dtype=complex)
    cn[0,0] = 1
    cn[1,1] = 1
    cn[2,3] = 1
    cn[3,2] = 1
    return cn

def I(n): # Devuelve la matriz identidad
    IM = np.array([[1,0],[0,1]], dtype=complex)
    if n > 1:
        IM = np.kron(IM, I(n - 1))
    return IM

def hopfCoords(qbit):
    alpha = qbit[0][0]
    beta = qbit[0][1]
    theta = np.arccos(alpha) * 2
    print (theta)
    s = np.sin(theta/2)
    if (s != 0):
        phi = np.log(beta/np.sin(theta/2)) / 1j
    else:
        phi = 0j
    return (theta, phi)

def QEq(q1, q2):
    return np.array_equal(q1,q2) and str(q1) == str(q2)
