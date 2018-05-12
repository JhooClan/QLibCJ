from structures.qgate import QGate
from structures.qregistry import *

class Measure(object):
	def __init__(self, mask, tasks=[], conds=[], remove=False):
		self.mask = mask
		self.conds = conds
		self.tasks = tasks
		self.remove = remove
	
	def mesToList(self, mresults):
		lin = 0
		mres = []
		for m in self.mask:
			tap = None
			if m == 1:
				tap = mresults[lin]
				lin += 1
			mres.append(tap)
		return mres
	
	def measure(self, qregistry):
		res = qregistry.Measure(self.mask, remove=self.remove)
		for task in self.tasks:
			task(res)
		res = self.mesToList(res)
		r = qregistry
		for cond in self.conds:
			r = cond.evaluate(r, res)
		return r
		

class Condition(object):
	def __init__(self, cond, ifcase, elcase):
		# cond is an array of what we expect to have measured in each QuBit. None if we don't care about a certain value. Example: [0, 1, None, None, 1].
		# ifcase and elcase can be Conditions or QCircuits to be applied to the registry. They can also be functions that take the registry and the result as a parameter.
		self.cond = cond
		self.ifcase = ifcase
		self.elcase = elcase
	
	def evaluate(self, qregistry, mresults):
		case = elcase
		if mresults == cond:
			case = ifcase
		t = type(case)
		if t == Condition:
			r = case.evaluate(qregistry, mresults)
		elif t == QCircuit:
			r = case.execute(qregistry)
		else:
			r = case(qregistry, mresults)
		return r

class QCircuit(object):
	def __init__(self, name="UNNAMED", save=False):
		self.matrix = [1]
		self.measure = []
		self.save = save
		self.lines = []
		self.plan = [0]
		self.name = name
	
	def addLine(self, *args):
		if self.save:
			self.lines.append(list(args))
		if type(args[0]) != Measure:
			mlen = len(self.measure)
			aux = getMatrix(args[0])
			for gate in list(args)[1:]:
				aux = np.kron(aux, getMatrix(gate))
			self.matrix[mlen] = np.dot(self.matrix[mlen], aux)
			if self.plan[-1] != 0:
				self.plan.append(0)
		else:
			self.measure.append(args[0])
			self.plan.append(1)
	
	def execute(self, qregistry):
		r = qregistry
		gid = 0
		mid = 0
		for task in self.plan:
			if task == 0:
				r.ApplyGate(self.matrix[gid])
				gid += 1
			else:
				r = self.measure[mid].measure(r)
				mid += 1
		return r
