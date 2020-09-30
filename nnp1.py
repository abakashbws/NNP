
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

some_value = -0.5
weight = 0.7
bias = 0.7

print(some_value*weight)
print(some_value+bias)





bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)
layer_outputs = [] 
# o/p of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    # o/p of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt


nnfs.init()

X = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

'''
# x = inputs
weights = [[0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]

#To create a second layer of neurons, weights2 and biases2 have been used


biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]]


biases2 = [-1, 2, -0.5]
layer1_output  = np.dot(inputs, np.array(weights).T) + biases  
layer2_output  = np.dot(layer1_output, np.array(weights2).T) + biases2  

#did dot product to give final output to the matrix product 
#transposed the weights
print(layer2_output)
'''

np.random.seed(0)


class layer_dense:
        def __init__(self, n_inputs, n_neurons):
                self.weights = 0.10* np.random.randn(n_inputs, n_neurons)
                self.biases = np.zeros((1, n_neurons))
                # both return the shape of the matrix
        def forward(self, inputs):
                self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
        def forward(self, inputs):
                self.output = np.maximum(0, inputs)
                
layer1 = layer_dense(2,5) 
#layer_dense(x,y); x is the number of inputs and y is the number of neurons
#layer2 = layer_dense(5,2)
activation1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)




# layer2.forward(layer1.output)
# print(layer2.output)




# sigmoid activation function and step function is to be used
# optimisation required
# step function doesn't give granularity so sigmoid is preferred
# y = x if x > 0 or y = 0 if x <= 0
# rectfied linear function can be used too for hidden layers based upon optimisation parameters

'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
        output.append(max(0,i))


        # if i>0:
        #         output.append(i)
        # elif i<=0:
        #         output.append(0)
'''


N = 100 #no of points per class
D = 2 #dimensionality
K = 3 #number of classes
X = np.zeros((N*K, D)) #data matrix
y = np.zeros(N*K, dtype='uint8') #class labels
for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 #theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()
