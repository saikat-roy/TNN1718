Multi-Neural Gas

Dependencies:
------------
Python 2.7
Numpy

The program may be controlled by calling the train(..) method of the 
Multi_NeuralGas. There is choice of trainset creation from either
given training data file or from non-overlapping regions of the unit
cube.

The value of the s.d of the Gaussian neighbourhood function has an
option of remaining unchanged or having linear decay to a predefined
fraction of original value.


As required in the question:

Non Overlapping regions in unit cube used, as defined by their bottom
left and top right corners, are:

In this case regions used (based on 2 corners) are:
	1) (0.2,0.1), (0.5,0.4)
	2) (0.2,0.45),(0.5,0.75)
	3) (0.2,0.8), (0.5,0.99)
	
A trial run with 1000 epochs and default values as mentioned in program
for hyperparameters generates centres as provided inthe PA-D.net.


Note: The plot2D method is incomplete and only given as a reference
(Only works for M=4 currently). Uncomment in train(..) to use. Uses
matplotlib.
