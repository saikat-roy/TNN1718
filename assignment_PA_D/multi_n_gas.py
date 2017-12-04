import numpy as np
import math

def load_data(filepath):
	'''
	Loads the training file for training as a ndarray.
	
	Input:
	-----
	filepath: The path of the file to load.
	'''
	print 'Using data from given training file'
	return np.loadtxt(filepath, comments='#')


def generate_data(N, pts=100):
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
		
	
class NeuralGas:
	
	def __init__(self, n_neurons, d_input):
		'''
		Initializes a NeuralGas object by initializing the centers of 
		the neurons.
		
		Input:
		-----
		n_neurons: No. of neurons in the Neural Gas
		d_input  : The dimension of the input
		'''
		self.n_centers = n_neurons
		self.d_centers = d_input
		self.centers = np.random.rand(self.n_centers, self.d_centers)
	
	
	def _update(self, X, s, lr):
		'''
		Updates the centers of the neurons of the NeuralGas object
		
		Input:
		-----
		X  : The input sample
		s  : The s.d. of the neighbourhood function (precalculated)
		lr : the effective learning rate (precalculated)
		'''
		dist = np.sum(self.centers-np.tile(X,(self.n_centers,1)),axis=1)
		h = map(lambda x: lr*math.exp(-0.5*math.pow(x/float(s),2)),\
				np.argsort(dist))
		for i in range(self.centers.shape[0]):
			self.centers[i]=self.centers[i]+(lr*h[i]*(X-self.centers[i]))
		
		return 
	
	
	def return_winner(self, X):
		'''
		Returns the distance of the closest neuron to the sample
		
		Input:
		-----
		X: The input sample
		'''
		return min(np.absolute(np.sum(self.centers-np.tile(X, \
			(self.n_centers,1)),axis=1)))
			

class Multi_NeuralGas:
	
	def __init__(self, M, K, N):
		'''
		Initializes the M-Neural Gas by initializing a list of NeuralGas
		objects
		
		Input:
		-----
		M : The number of partner Neural Gas objects
		K : The number of neurons in each partner neurons
		N : The dimension of the input
		'''
		self.NG_list = [NeuralGas(K, N) for i in range(M)]
	
	
	def train(self, trainX, lr_0=0.5, lr_end = 0.1, decay_to_s=0.25,
			  epochs=1000, s=0.3):
		'''
		Train a Multi-N-Gas by training only the requisite winner neural
		gas model.
		
		Input:
		-----
		trainX : The training data set
		lr_0   : Initial learning rate
		lr_end : Learning rate after decay over total number of epochs
		decay_to_s : The fraction to which s is decayed
		epochs : The total number of epochs
		s      : The spread of the neighbourhood function
		'''
		assert lr_0 >= lr_end
		
		t=0
		# Calculate lambda for learning rate decay
		# Designed to decays to lr_end by final epoch
		lambda_t = (1.0/epochs)*math.log(lr_0/lr_end)
		
		for e in range(epochs):
			
			#Periodic epoch counter
			if (e+1)%100==0:
				print 'Running epoch %d/%d.'%(e+1, epochs)
				
			# Calculate new learning rate based on exponential decay
			_lr = lr_0 * math.exp(-lambda_t*t) 
			
			if decay_to_s is not None:
				# Decays s.d. linearly of neighbourhood function to 
				# reach decay_to_s percentage of original by final epoch
				_s = s-(s*(1-decay_to_s)*t/epochs)
			else:
				_s = s
				
			for d in trainX:
				winner_idx = np.argmin([i.return_winner(d) for i in \
							 self.NG_list])
				self.NG_list[winner_idx]._update(d, _s, _lr)
			t+=1
			
		#self.plot2D(trainX)
		self.write_centers_to_file()
		return
	
	
	def print_centers(self):
		'''
		Simply prints all the centers of all NeuralGas objects in the
		M-NeuralGas
		'''
		for i in self.NG_list:
			print i.centers
		
	
	def write_centers_to_file(self, fname='PA-D.net'):
		'''
		Writes Centers to a file with given filename
		'''
		with open(fname,'a') as f:
			for i in self.NG_list:
				np.savetxt(f, i.centers)
				f.write('\n')
		
	
	def plot2D(self, trainX):
		'''
		Function to plot the training file as a graph.
		ONLY FOR DEMO. DOES NOT RUN FOR M!=4
		'''
		import matplotlib.pyplot as plt
				
		colors = ("red", "green", "blue", "yellow")
		groups = ("NG1", "NG2", "NG3", "NG4") 
		 
		# Create plot
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
		#print trainX
		x, y = trainX[:,0], trainX[:,1] 
		ax.scatter(x, y, alpha=0.8, c='black')
		
		for i in range(len(self.NG_list)):
			_NG = self.NG_list[i].centers
			x, y = _NG[:,0], _NG[:,1] 
			#print x, y
			ax.scatter(x, y, c=colors[i], edgecolors='none',\
					   s=30, label=groups[i])
		 
		plt.title('Matplot scatter plot')
		plt.legend(loc=2)
		plt.show()
		
if __name__ == '__main__':
	
	#d = load_data('train_PA-D.dat.txt') # Use to read given dataset
	
	d = generate_data(2) # Please use to randomly generate dataset from 
						 # unit cube
	
	M_NG = Multi_NeuralGas(M=4,K=2,N=2)
	M_NG.train(d)
	
