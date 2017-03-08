import math as m
import cmath as cm
import numpy as np
import sympy as sp
import random as rnd
# np.zeros((h,w), dtype=complex) Inicializa una matriz de numeros complejos con alto h y ancho w
# La suma de matrices se realiza con +. A + B
# La multiplicacion por un escalar se hace con *. n * A
# Para multiplicar las matrices A y B se usa np.dot(A,B)
# El producto Kronecker de A y B esta definido con np.kron(A,B)

class QRegistry:
    def __init__(self, qbits):    # QuBit list and seed for the Pseudo Random Number Generation.
        if (type(qbits) != list or \
            not qbits or \
            not all(type(qbit) == np.ndarray and \
                    qbit.shape == (1,2) and \
                    qbit.dtype == 'complex128' for qbit in qbits)):
            raise ValueError('Impossible QuBit Registry')
        qbs = qbits[:]

        self.state = qbs[0]
        Normalize(self.state)
        self.measure = [-1]
        del qbs[0]
        for qbit in qbs:
            Normalize(qbit)
            self.state = np.kron(self.state, qbit)
            self.measure.append(-1)
        Normalize(self.state)

    def Measure(self, mask):
        if (type(mask) != list or \
            not all(type(num) == int for num in mask)):
            raise ValueError('Not valid mask')
        measure = []
        for qbit in mask:
            r = rnd.random()
            p = 0
            max = 2**qbit
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
                m = 0
            else:
                m = 1
            measure.append(m)
            Collapse(qbit, m)

    def ApplyGate(self, gate):
        self.state = np.dot(Bra(self.state), gate)
    
    def Collapse(self, qbit, measure):
        max = 2**qbit
        cnt = 0
        rdy = measure == 1
        for i in range(0, self.state.size):
            if (cnt == max):
                rdy = not rdy
                cnt = 0
            if (rdy):
                self.state[0,i] = 0j
            cnt += 1


def Prob(q, x): # Devuelve la probabilidad de obtener x al medir el qbit q
    p = 0
    if (x < q.size):
        p = cm.polar(q[0,x])[0]**2
    return p

def Measure(q): # Toma una medida del QuBit pasado como parametro
    r = rnd.random()
    ms = 0
    for i in range(0, q.size):
        p = Prob(q, i)
        if (r < p):
            ms = i
            break
        r -= p
    return ms

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
    for bs in np.nditer(state):
        sqs += cm.polar(bs)[0]**2
    sqs = cm.sqrt(sqs)
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

#def DJOracle(x, y, f): # Implementa el oraculo Uf descrito en el algoritmo de Deutsch-Jozsa, colocando en y el resultado de y XOR f(x)
#    return (x, ApplyGate(Superposition(y, f(x)), CNOT()))

def Separate(state):
    sol = [state]
    ukw = sp.symbols('x1, x2')
    eq = ()
    ss = state.size
    if (ss > 2):
        ukws = int(m.log(ss,2)) * 2 # Numero de incognitas, siempre es entero
        for i in range(0, ukws - 2): # Creacion de la tupla de incognitas
            ukw += (sp.symbols('y' + str(i)),)
        for i in range(2, ukws): # Creacion de la tupla de funciones
            eq += (ukw[0] * ukw[i] - state[0, i - 2], ukw[1] * ukw[i] - state[0, i - 2 + int(ss/2)],)
        f = -1
        for i in range(2, ukws):
            f += ukw[i]**2
        eq += (ukw[0]**2 + ukw[1]**2 - 1, f)
        sols = sp.solvers.solve(eq, ukw)
        auxs = []
        for s in sols:
            auxa = np.array([s[0], s[1]], dtype=complex)
            auxa.shape = (1,2)
            auxb = np.array([s[i] for i in range(2, ukws)], dtype=complex)
            auxb.shape = (1,auxb.size)
            if (QEq(Superposition(auxa, auxb), state)):
                sol = Separate(auxa) + Separate(auxb)
                break
    return sol

def QEq(q1, q2):
    return np.array_equal(q1,q2) and str(q1) == str(q2)
