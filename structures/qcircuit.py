from structures.qgate import QGate
from structures.qregistry import *
import gc

class Measure(object):
	def __init__(self, mask, tasks=[], conds=[], remove=False):
		self.mask = mask
		self.conds = conds
		self.tasks = tasks
		self.remove = remove
	
	def __repr__(self):
		return ["Measure" if i == 1 else "I" for i in self.mask]
	
	def __str__(self):
		return self.__repr__()
	
	def _mesToList(self, mresults):
		lin = 0
		mres = []
		for m in self.mask:
			tap = None
			if m == 1:
				tap = mresults[lin]
				lin += 1
			mres.append(tap)
		return mres
	
	def Check(self, qregistry):
		res = qregistry.Measure(self.mask, remove=self.remove)
		res = self._mesToList(res)
		r = qregistry
		for cond in self.conds:
			r = cond.Evaluate(r, res)
		for task in self.tasks:
			task(r, res)
		return r
		

class Condition(object):
	def __init__(self, cond, ifcase, elcase = None):
		# cond is an array of what we expect to have measured in each QuBit. None if we don't care about a certain value. Example: [0, 1, None, None, 1].
		# ifcase and elcase can be Conditions or QCircuits to be applied to the registry. They can also be functions that take the registry and the result as a parameter.
		self.cond = cond
		self.ifcase = ifcase
		self.elcase = elcase
	
	def Evaluate(self, qregistry, mresults):
		case = self.elcase
		if _specialCompare(self.cond, mresults):
			case = self.ifcase
		t = type(case)
		if t == Condition:
			r = case.Evaluate(qregistry, mresults)
		elif t == QGate:
			r = qregistry
			r.ApplyGate(case)
		elif t == QCircuit:
			r = case._executeOnce(qregistry)
		elif t != type(None):
			r = case(qregistry, mresults)
		else:
			r = qregistry
		return r

class QCircuit(object):
	def __init__(self, name="UNNAMED", ancilla=[], save=True): # You can choose whether to save the circuit and apply gates separately on each computation (faster circuit creation) or to precompute the matrixes (faster execution)
		self.name = name
		self.matrix = [1]
		self.measure = []
		self.lines = []
		self.plan = [0]
		self.ancilla = ancilla
		self.save = save
	
	def AddLine(self, *args):
		try:
			if self.save:
				self.lines.append(list(args))
			else:
				if type(args[0]) != Measure:
					mlen = len(self.measure)
					aux = getMatrix(args[0])
					for gate in list(args)[1:]:
						aux = np.kron(aux, getMatrix(gate))
					del args
					self.matrix[mlen] = np.dot(aux, self.matrix[mlen])
					del aux
					if self.plan[-1] != 0:
						self.plan.append(0)
				else:
					self.measure.append(args[0])
					self.plan.append(1)
		finally:
			gc.collect()
	
	def _executeOnce(self, qregistry, iterations = 1): # You can pass a QRegistry or an array to build a new QRegistry. When the second option is used, the ancilliary qubits will be added to the specified list.
		if type(qregistry) != QRegistry:
			r = QRegistry(qregistry + self.ancilla)
			ini = qregistry[:]
		else:
			r = QRegistry([0])
			ini = qregistry.state[:]
			r.state = ini[:]
			if self.ancilla is not None and len(self.ancilla) > 0:
				print (self.ancilla)
				aux = QRegistry(self.ancilla)
				r.state = np.kron(r.state, aux.state)
		try:
			if self.save:
				for line in self.lines:
					g = line[0]
					if type(g) != Measure:
						g = getMatrix(g)
						for gate in line[1:]:
							g = np.kron(g, getMatrix(gate))
						r.ApplyGate(g)
						del g
					else:
						r = g.Check(r)
					gc.collect()
			else:
				gid = 0
				mid = 0
				for task in self.plan:
					if task == 0:
						r.ApplyGate(self.matrix[gid])
						gid += 1
					else:
						r = self.measure[mid].Check(r)
						mid += 1
					gc.collect()
		finally:
			gc.collect()
		return r
	
	def Execute(self, qregistry, iterations = 1):
		sol = [self._executeOnce(qregistry) for i in range(iterations)]
		if iterations == 1:
			sol = sol[0]
		return sol

def _specialCompare(a, b):
	same = len(a) == len(b)
	if (same):
		for i in range(len(a)):
			if a[i] is not None and a[i] != b[i]:
				same = False
				break
	return same
