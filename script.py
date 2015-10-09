# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:39:38 2015

@author: GlennGuo
"""

from neuralnetwork import NeuralNetwork
import numpy as np
import util

# Load MNIST with all label vectorized
train_x, train_y, valid_x, valid_y, test_x, test_y = util.MNIST_Load()

# Scale data
"""
train_x = util.MapMinMax(train_x, -1, 1)
valid_x = util.MapMinMax(valid_x, -1, 1)
test_x = util.MapMinMax(test_x, -1, 1)

"""
train_x = util.MNIST_Scale(train_x)
valid_x = util.MNIST_Scale(valid_x)
test_x = util.MNIST_Scale(test_x)

# determine network structure and hyperparameters    

layers = np.array([784, 784, 10])
learningRate = 0.01
momentum = 0.00
batch_size = 100
num_of_batch = len(train_y)/batch_size
nepoch = 2000

# initialize weights with U ~ [-sqrt(3)/sqrt(k), sqrt(3)/sqrt(k)]
#factor = np.sqrt(3.0) / np.sqrt(layers[0:-1])
factor = 0.05 *np.sqrt(layers[0:-1]) / np.sqrt(layers[0:-1])
#factor = 1.0*np.sqrt(layers[0:-1]) / np.sqrt(layers[0:-1])


w = []; b = []
for i_o_theta in range(len(layers)-1):
    w.append(np.random.uniform(-factor[i_o_theta], factor[i_o_theta], layers[i_o_theta:i_o_theta+2][::-1]))
    b.append(np.random.uniform(-factor[i_o_theta], factor[i_o_theta], (1,layers[i_o_theta + 1])))
    

nn = NeuralNetwork(layers, 'tanh', 'softmax', early_stop = 'off')
cost_trace, valid_err = nn.train(train_x, train_y, w, b, momentum, learningRate, batch_size, nepoch, valid_x, valid_y)
test_err = nn.predict(test_x, test_y)
print 'test_err:%.2f%%'%(test_err*100)


