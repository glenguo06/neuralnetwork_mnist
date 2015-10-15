# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:39:38 2015

@author: GlennGuo
"""

from neuralnetwork import NeuralNetwork
import numpy as np
import util
import matplotlib.pyplot as plt

"""
 Load MNIST with all label vectorized
"""
train_x, train_y, valid_x, valid_y, test_x, test_y = util.MNIST_Load()


"""
 Scale data
"""
train_x = util.MNIST_Scale(train_x)
valid_x = util.MNIST_Scale(valid_x)
test_x = util.MNIST_Scale(test_x)


"""
 Determine structure and parameters of the network
"""
layers = np.array([784, 300, 10])
learningRate = 0.01
momentum = 0.00
batch_size = 100
num_of_batch = len(train_y)/batch_size
nepoch = 100
regularizer = 0.0005

# initialize weights with U ~ [-sqrt(3)/sqrt(k), sqrt(3)/sqrt(k)]
#factor = np.sqrt(3.0) / np.sqrt(layers[0:-1])
factor = 0.05 *np.sqrt(layers[0:-1]) / np.sqrt(layers[0:-1])
#factor = 1.0*np.sqrt(layers[0:-1]) / np.sqrt(layers[0:-1])


w = []; b = []
for i_o_theta in range(len(layers)-1):
    w.append(np.random.uniform(-factor[i_o_theta], factor[i_o_theta], layers[i_o_theta:i_o_theta+2][::-1]))
    b.append(np.random.uniform(-factor[i_o_theta], factor[i_o_theta], (1,layers[i_o_theta + 1])))
    

"""
 train both networks with back-prop
"""
nn1 = NeuralNetwork(layers, 'tanh', 'softmax', early_stop = 'off', reg = 0)
cost_trace1, valid_err1 = nn1.train(train_x, train_y, w, b, momentum, learningRate, batch_size, nepoch, valid_x, valid_y)

nn2 = NeuralNetwork(layers, 'tanh', 'softmax', early_stop = 'off', reg = regularizer)
cost_trace2, valid_err2 = nn2.train(train_x, train_y, w, b, momentum, learningRate, batch_size, nepoch, valid_x, valid_y)

test_err1 = nn1.predict(test_x, test_y)
test_err2 = nn2.predict(test_x, test_y)

# print 'test_err:%.2f%%'%(test_err*100)

"""
Draw comparison
"""
plt.figure(1)
plt.plot(valid_err1[0:100], 'b-', linewidth=2, label='Typical Learning Process', hold='on')
plt.plot(valid_err2[100:], 'r-', linewidth=2, label='\"$Anti$-Bayesian" Learning', hold='off')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Test Error', fontsize=18)
plt.grid()
plt.show()

si = train_x[1,:]
si.shape += (1,)

plt.imshow(si.reshape(28,28), cmap='gray')
plt.show()