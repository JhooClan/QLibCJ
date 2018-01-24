from qlibcj import *
def DJAlg(size, U_f): # U_f es el oraculo, que debe tener x1..xn e y como qubits. Tras aplicarlo el qubit y debe valer f(x1..xn) XOR y. El argumento size es n + 1, donde n es el numero de bits de entrada de f.
    r = QRegistry(([QZero() for i in range(0, size - 1)] + [QOne()])) # Los qubits se inicializan a cero (x1..xn) excepto el ultimo (y), inicializado a uno
    r.ApplyGate(Hadamard(size)) # Se aplica una compuerta hadamard a todos los qubits
    r.ApplyGate(U_f) # Se aplica el oraculo
    r.ApplyGate(Hadamard(size - 1), I(1)) # Se aplica una puerta Hadamard a todos los qubits excepto al ultimo
    return r.Measure([0])[0] # Se mide el qubit x1, si es igual a 0 la funcion es constante. En caso contrario no lo es.

'''
Crea un oraculo U_f tal y como viene definido en el algoritmo de Deutsch-Josza para una funcion balanceada f: {0,1}^n ---> {0,1}, f(x) = msb(x) (bit mas significativo de x).
El argumento n no es el numero de bits de la entrada de f, sino dicho numero mas 1 (para el qubit de "salida").
'''
def Bal(n): 
    b = I(n)
    '''
    Se invierte el valor del qubit y en los casos en los que el bit mas significativo sea 1.
    Una puerta C-NOT serviria de U_f con la definicion de f dada con n = 2. Bal(2) = CNOT().
    '''
    for i in range(int((2**n)/2), (2**n) - 1, 2):
        t = np.copy(b[i,:])
        b[i], b[i+1] = b[i+1, :], t
    return b
'''
U_f generada con n = 3:
1 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0
0 0 1 0 0 0 0 0
0 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0
0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 1
0 0 0 0 0 0 1 0
Las entradas son, empezando por el qubit mas significativo: x1, x2 e y.
Al aplicar el oraculo lo que hara es intercambiar la probabilidad asociada a |100> con la de |101> y la de |110> con |111>.
De forma mas general, la funcion Bal se observa que devuelve siempre una puerta que al ser aplicada a un conjunto x1, ..., xn, y
de qubits aplicara una C-NOT sobre x1 (control) e y (objetivo), dejando el resto de qubits intactos.
De esta forma el oraculo pondra en el qubit y el valor de x1 XOR y. Como para la mitad de las posibles entradas x1 valdra 0
y para la otra mitad 1, la funcion f es balanceada ya que devuelve 0 para la mitad de las posibles entradas y 1 para la otra mitad.
El oraculo U_f a su vez se comporta como se indica en el algoritmo, teniendo que y <- f(x) XOR y.
'''

def Teleportation(qbit, **kwargs): # El qubit que va a ser teleportado. Aunque en un computador cuantico real no es posible ver el valor de un qubit sin que colapse, al ser un simulador se puede. Puede especificarse semilla con seed = <seed>.
    r = QRegistry([qbit, QZero(), QZero()], seed=kwargs.get('seed', None)) # Se crea un registro con el qubit que debe ser enviado a Alice, el qubit de Bob y el de Alice, en adelante Q, B y A. B y A estan inicializados a |0>.
    print ("Original registry:\n", r.state) # Se muestra el estado del registro de qubits.
    r.ApplyGate(I(1), Hadamard(1), I(1)) # Se aplica la puerta Hadamard a B, ahora en una superposicion de los estados |0> y |1>, ambos exactamente con la misma probabilidad.
    r.ApplyGate(I(1), CNOT()) # Se aplica una puerta C-NOT sobre B (control) y A (objetivo).
    print ("With Bell+ state:\n", r.state) # Tras la aplicacion de las anteriores dos puertas tenemos un estado de Bell +, B y A estan entrelazados. Se muestra el valor del registro.
    r.ApplyGate(CNOT(), I(1)) # Se aplica una puerta C-NOT sobre Q (control) y B (objetivo).
    r.ApplyGate(Hadamard(1), I(2)) # Se aplica una puerta Hadamard sobre Q.
    print ("\nBefore measurement:\n", r.state) # Se muestra el valor del registro antes de la medida.
    m = r.Measure([0,1]) # Se miden los qubits Q y B.
    print ("q0 = ", m[0], "\nq1 = ", m[1]) # Se muestra el resultado de la medida
    q0 = QZero() # Se crean para ver que la teleportacion se realiza con exito dos qubits, q0 y q1.
    q1 = QZero() # Usandolos crearemos un registro con los valores que debe tener si la teleportacion se ha realizado con exito.
    if (m[1] == 1):
        q1 = QOne()
        r.ApplyGate(I(2), PauliX()) # Si al medir B obtuvimos un 1, rotamos A en el eje X (Pauli-X o NOT)
    if (m[0] == 1):
        q0 = QOne()
        r.ApplyGate(I(2), PauliZ()) # Si al medir Q obtuvimos un 1, rotamos A en el eje Z (Pauli-Z).
    er = QRegistry([q0, q1, qbit]) # Se crea el registro para testeo mencionado anteriormente.
    print ("\nExpected result:\n", er.state, "\nResult:\n", r.state) # Se muestra el contenido de los registros, tanto el del resultado esperado como el obtenido.
    return r # Se devuelve el registro obtenido tras aplicar el algoritmo.
