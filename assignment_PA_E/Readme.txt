Regional and Online Learnable Fields

Dependencies:
------------
Python 2.7
Numpy

The program may be controlled by calling the train(..) method of the 
ROLF after initialization. There is choice of trainset creation from 
either given training data file or from non-overlapping regions of the 
unit cube.

As an example the training file provided in the previous programming
exercise as well as the non-overlapping regions has been reused.

As required in the question:

The 'mean' and 'init' techniques of initializing sigma have been 
implemented. Also the train(..) method calls the write_centers_to_file(.)
method which writes the neuron centers to a file (GNUPlot compatible).
Also, the number of neurons are designed not to exceed 10000 in the 
ROLF.
