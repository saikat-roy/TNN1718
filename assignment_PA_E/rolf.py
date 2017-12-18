import numpy as np

def load_data(filepath):
	'''
	Loads the training file for training as a ndarray.
	
	Input:
	-----
	filepath: The path of the file to load.
	'''
	print 'Using data from given training file'
	return np.loadtxt(filepath, comments='#')


def generate_data(pts=100):
	'''
	Generates data from 3 non-overlapping areas in a unit cube
	
	In this case regions used (based on 2 corners) are:
	1) (0.2,0.1), (0.5,0.4)
	2) (0.2,0.45),(0.5,0.75)
	3) (0.2,0.8), (0.5,0.99)
	'''
	print 'Using data drawn from non-overlapping regions in unit cube'
	x1 = np.random.uniform(0.2, 0.5, pts)
	x2_1 = np.random.uniform(0.1, 0.4, pts)
	x2_2 = np.random.uniform(0.45, 0.75, pts)
	x2_3 = np.random.uniform(0.8, 0.99, pts)
	
	arr = np.append(np.array([x1,x2_1]),np.array([x1,x2_2]),axis=1)
	arr = np.append(arr, np.atleast_2d(np.array([x1,x2_3])),axis=1)
	return arr.T
	

class Neuron:
	
	def __init__(self, c, sigma, rho, eta_c, eta_sigma):
		'''
		initializes a ROLF Neuron
		'''
		self.c = c
		self.sigma = sigma
		self.rho = rho
		self.eta_c = eta_c
		self.eta_sigma = eta_sigma
		
		
	def _update_c(self, x):
		'''
		Updates the c value of the ROLF Neuron
		'''
		self.c = self.c+(self.eta_c*np.linalg.norm(self.c-x))
		
		
	def _update_sigma(self, x):
		'''
		Updates sigma of the ROLF neuron
		'''
		self.sigma = self.sigma+(self.eta_sigma*\
								(self.sigma-np.linalg.norm(self.c-x)))
	
	
	def check_acceptance(self, x):
		'''
		Checks whether a ROLF Neuron accepts a data point. Returns
		the distance of the point (x) from the center if it does
		otherwise returns None
		'''
		return np.linalg.norm(self.c-x) if np.linalg.norm(self.c-x)<= \
							(self.sigma*self.rho) else None


class Rolf:
	
	def __init__(self, sigma, rho, eta_c, eta_sigma, \
				 new_sigma_type='init'):
		'''
		Initializes a ROLF networks with parameters as in standard 
		literature. eta_c and eta_sigma are the learning rates for
		eta and sigma of each neuron respectively. new_sigma_type
		gives the type of initialization for the sigma values of new
		neurons. Two types are implemented 'init' and 'mean'
		'''
		self.sigma = sigma
		self.rho = rho
		self.eta_c = eta_c
		self.eta_sigma = eta_sigma
		self.n_list = []
		self.new_sigma_type = new_sigma_type
	
	
	def train(self, trainX):
		'''
		Trains the ROLF with samples from the training set. Periodically
		prints the number of neurons in the network during training. 
		The total number of neurons are limited to 10000 as defined in
		the question.
		'''
		i=0
		for x in trainX:
			if i%100==0:
				print '%d data points done.'%i
				print 'Number of neurons in ROLF=%d\n'%len(self.n_list)
			if len(self.n_list)==10000:
				raise Exception('number of neurons>10000')
			self._update(x)
			i+=1
		self.write_centers_to_file()
	
			
	def _update(self, x):
		'''
		Based on if there is an accepting neuron, the ROLF either
		updates the weight of the accepting neuron or adds a new neuron
		to accept the data point in the absence of an acceptor neuron.
		'''
		acceptor = [None, None] # (idx, dist)
		
		if len(self.n_list) == 0:
			self.n_list.append(Neuron(x, self.sigma, self.rho,\
								self.eta_c, self.eta_sigma))
			return

		for i in range(len(self.n_list)):
			d = self.n_list[i].check_acceptance(x)
			if d is not None:
				if acceptor[1] is None:
					acceptor[0] = i
					acceptor[1] = d 
				elif acceptor[1]>d:
					acceptor[0] = i
					acceptor[1] = d
		
		if acceptor[1] is None:
			if self.new_sigma_type == 'init': # init sigma
				sigma = self.sigma
			elif self.new_sigma_type == 'mean': # mean sigma
				sigma = reduce(lambda x,y: x+y, [i.sigma for 
							i in self.n_list])/float(len(self.n_list))
			else:
				raise Exception('Other sigma init not implemented')
																				 
			self.n_list.append(Neuron(x, sigma, self.rho,\
									self.eta_c, self.eta_sigma))
		else:
			self.n_list[acceptor[0]]._update_c(x)
			self.n_list[acceptor[0]]._update_sigma(x)
	
	
	def write_centers_to_file(self, fname='PA-E.out'):
		'''
		Writes Centers to a file with given filename, GNUPLOT compatible
		'''
		with open(fname,'a') as f:
			f.truncate()
			for n in self.n_list:
				for ci in range(n.c.shape[0]):
					f.write(str(n.c[ci])+'\t')
				#np.savetxt(f, n.c.T, fmt='%.4f')
				f.write('\n')
		

if __name__ == '__main__':
	
	#d = generate_data(1000) # Please use to randomly generate dataset 
							 # from unit cube (SAME AS LAST EXERCISE)
						 
	d = load_data('train_PA-D.dat.txt') 
	# Using same data file as the last programming exercise as sample
	# training set
	
	rolf = Rolf(sigma=0.05, rho=2.0, eta_c=0.25, eta_sigma=0.25,\
				new_sigma_type='mean') 
	#must provide sigma even in case of 'mean' for 1st neuron
	
	rolf.train(d)
	
