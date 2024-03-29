# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:03:32 2015

@author: GlennGuo
"""

import numpy as np
import time

class NeuralNetwork:
    
    __units = []
    
    __w = []
    __b = []
    
    __f_out = []
    __f_act = []
    __f_cost = []
    __f_act_prime = []
    
    __stop_counter = 100 # No-improvement in N, N=40
    __tmp_valid_record = []
    __best_valid_err = 1.0
    
    __best_w = []
    __best_b = []
    
    __h_lr = [] # handler of function which computes learning rate
    
    __early_stop = 'on'
    
    __weight_trace = []
    __bias_trace = []
    
    cost_trace = []
    cost_epoch = []
    valid_error = []
    
    __maxEpoch = 100
    
    __reg = 0
    
    def __init__(self, layers, activation = 'logistic', out = 'softmax', early_stop = 'on', reg = 0):
        """
        Construct an instance of NeuralNetwork with personalized stucture.
        
        Parameters:
        -----------
        layers : A list containing the number of hidden units of each layers.
                (including input layer and output layer)
                `eg. layers=[784,1000,10]`
        activation : Activation function of each hidden units.
                [option]:
                (1)'logistic'
                (2)'tanh' : a rectified version given by Yann LeCun's publication
                (3)'tanh_old' : the typical tanh function
                (4)'Relu' : rectified linear units proposed by Geoff Hinton for 
                            deeper networks.
        out : Activation function for output units.
                [option]:
                (1)'softmax'
        early_stop ： Enable/Disable early stopping strategy proposed in Dan's publication
            
        reg : The "Anti-Bayesian" regularization coefficient. The strategy is proposed
                based on Prof. Oommen's publications.
                
        Returns:
        --------
            The instance of a structured network
              
        """
        
        self.__units = layers
        self.__early_stop = early_stop
        self.__h_lr = self.__learning_rate        
        self.__reg = reg
        
        if activation == 'logistic':
            self.__f_act = self.__logistic
            self.__f_act_prime = self.__logistic_prime
            
        if activation == 'tanh':
            self.__f_act = self.__tanh
            self.__f_act_prime = self.__tanh_prime
        
        if activation == 'tanh_old':
            self.__f_act = self.__tanh_old
            self.__f_act_prime = self.__tanh_old_prime
            
        if activation == 'Relu':
            self.__f_act = self.__Relu
            self.__f_act_prime = self.__Relu_prime
            
        if out == 'softmax':
            self.__f_out = self.__softmax
            self.__f_cost= self.__softmax_costFunction

    def __tanh_old(self, x):
        """
        The original version of tanh function, without any motification
        
        Parameters:
        -----------
        x : Input ndarray of size (m,n)
            
        Returns:
        --------
            sinh(x)/cosh(x)
        """
        return np.tanh(x)
        
    def __tanh(self, x):
        """
        According to publication:
            LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). 
            'Gradient-Based Learning Applied to Document Recognition.'
            Proceedings of the IEEE, 86, 309 – 318.     
        """
        
        x = 0.6666 * x
        return 1.7159 * np.sinh(x)/np.cosh(x)
        
    def __logistic(self, x):
        
        return 1.0/(1+np.exp(-x))
        
    def __logistic_prime(self, x):
        """
        Prime of the logistic function
        """
        
        return self.__logistic(x)*(1.0-self.__logistic(x))
    
    def __tanh_prime(self, x):
        """
        Prime of the tanh function
        
            f(x) = 1.7159*tanh(0.6666*x)
            f'(x) = 1.1438*(1-f(x)**2)
        """       
        x = 0.6666*x
        
        return 1.1438*(1.0-self.__tanh_old(x)**2)
    
    def __tanh_old_prime(self, x):
        """
        Prime of the old tanh function
        
            f(x) = tanh(x)
            f'(x) = 1-f(x)**2
        
        """
        return 1.0-self.__tanh_old(x)**2
        
    def __Relu(self, x):
        """
        Rectified Linear Units
        """
        return x*(x>0)
        
    def __Relu_prime(self, x):
        """
        Prime of Relu function
        
        Relu'(x) = 1, if x>0
                 = 0, if x<=0
        """
        
        return x/x
        
    def __softmax(self, x):
        """
        softmax hopothesis function
        
        Para:
            x   :   input numpy.ndarray of m rows 10 columns
        
        Return:
            Transformed numpy.ndarray
        """
        
        x = np.exp(x)
        x_sum = np.sum(x,axis=1).reshape(len(x), 1) #shape = (nsamples,1)
        x_sum = np.tile(x_sum, (1, x.shape[1]))        
        x /= x_sum
        
        return x        
        
    def __forward_prop(self, X, w, b):
        """
        Forward propagation implemented with verctor manipulatation
        
        Para:
            X   :   input numpy.ndarray of m rows n columns
            w   :   'list' of weights values for each layer
            b   :   'list' of bias values
        
        Return:
            'list' of activation values
        """
        par_len = len(w)
        
        z = [] # linear product
        z.append([0]) # z0 = 0, will not be indexed
        
        act = []    # activations            
        act.append(X)   # a1 = input 

        for par_idx in range(par_len-1):
            z.append(np.dot(np.insert(act[-1],0,1,axis=1),np.insert(w[par_idx],0,b[par_idx],axis=1).T))
            act.append(self.__f_act(z[-1]))
        
        z.append(np.dot(np.insert(act[-1],0,1,axis=1),np.insert(w[-1],0,b[-1],axis=1).T))
        act.append(self.__f_out(z[-1]))
            
        return z, act
        
    def __back_prop(self, a, Y, w, b):
        """
        Backpropagation to compute gradient
        
        Para:
            X   :   training examples of m rows n columns
                    m = 1 for stochastic gradient descent
                    m = k for mini-batch
            Y   :   corresponding labels (unvectorized)
            w   :   'list' of weights values for each layer
            b   :   'list' of bias values
        
        Return:
            grad{w}    :   gradient to update w
            grad{b}    :   gradient to update b
            
        Example:
            grad = __back_prop(X = [[0.1 0 0.3 0],
                                    [0.4 0.8 0 0],
                                    [...],
                                    [...]],
                               Y = [[0 0 0 1],
                                    [0 0 1 0],
                                    [1 0 0 0],
                                    [0 1 0 0]], w = ..., b = ...)
        """
        
        batch_size = len(Y)
        
        d = []
        D = []
        grad = {}
        
        ndelta = len(a)-1
        
        # compute delta
        d.append(a[-1] - Y) # delta for output units
        D.append(np.dot(d[0].T,a[-2])/batch_size)
        
        for i in range(ndelta-1, 0, -1):
            d.append((np.dot(d[-1],w[i])+self.__reg)*self.__f_act_prime(a[i]))
            D.append(np.dot(d[-1].T,a[i-1])/batch_size)
            
        # reverse delta and grad
        d = d[::-1]
        D = D[::-1]
        
        for i in range(len(d)):
            d[i] = np.sum(d[i],axis=0)/batch_size
            
        grad['b'] = d
        grad['w'] = D
        
        return grad
        
    def __EarlyStop(self, valid_error):
        """
        Check the condition of eary-stopping with a simple and efficient criteria
        'No-improvement in n' where n is determined by __stop_counter
        
        In:
            valid_error: validation error of current epoch
        
        Out:
            [1] StopNow ? True/False
            [2] Record optimal parameters ? True/False
        """
        if(self.__stop_counter > 0):
            if(self.__best_valid_err > valid_error):
                self.__best_valid_err = valid_error
                self.__stop_counter = 40 # reset counter
                return False, True
            else:
                self.__stop_counter -= 1
                return False, False
            
        else:
            return True, False
            
        
    def __softmax_costFunction(self, h, y):
            
        return -np.sum(y*np.log(h))/len(y)
        
    def __learning_rate(self, lr0, epoch):
        """
        Adapt learning rate with O(1/t) schedule
        where lr = lr0*s/max{epoch, s}.
        Learning rate stays as a constant during the first 's' epoch
        
        Reference:
            Bergstra, J., Bengio, Y.: Eandom search for hyper-parameter optimization.
            J. Machine Learning  Res. 13, 281-305 (2012)
            
        * Feel free to modify and present your own implementation
        
            s = 20
        
            return np.float32(lr0)*s/np.max([epoch, s])        
        """
        
        """
        Dan's Methos
        """
        lrs = lr0 * 0.001
        c = np.power((lrs/lr0), 1.0/self.__maxEpoch)
        
        return lr0*np.power(c, epoch)
        
            
    def train(self, X, Y, w, b, momentum, learningRate, batch_size, nepoch, valid_set, valid_label):
        """
        Train the neural network with backpropagation algorithm
        
        Parameters:
        -----------
        X : Training samples arranged as (m,n) ndarray.
            m -- the number of samples
            n -- the number of features of each sample
        Y : Corrsponding output values of the training set arranged as (m,n) ndarray 
        w ：A list containing all weights vectors.
            `eg: w[0] is the weight matrix connecting layer 0 and layer 1`
        b : A list containing all bias vectors.
            `eg: b[0] is the bias vector for layer 0`
        momentum : A hyper-parameter to avoid over-fitting.
            See "bayesian interpretation of supervised learning" for reference.
        learningRate : A hyper-parameter controling the speed to update.
            The value of learning rate during each epoch is "Auto-adjusted"
            with the scheme proposed by Dan's publication.
            See "__learning_rate()" in this file for further reference.
        batch_size : The size of each mini batch.
        nepoch : The
        """
        
        Duration = 0
        
        num_of_batch = len(Y)/batch_size
        self.__maxEpoch = nepoch
        
        assert len(X) == len(Y)
        
        for epoch in range(nepoch):
            
            self.__weight_trace.append(w)
            self.__bias_trace.append(b)

            shuffle_idx = np.random.permutation(len(X))
            X = X[shuffle_idx]
            Y = Y[shuffle_idx]            
            
            cost_local_sum = 0
            
            start_time = time.time()
            
            for idx in range(num_of_batch):
                
                x = X[idx*batch_size:(idx+1)*batch_size,:]
                y = Y[idx*batch_size:(idx+1)*batch_size]
                
                z, a = self.__forward_prop(x, w, b)
                grad = self.__back_prop(a, y, w, b)
            
                # update parameters
                for i in range(len(w)):
                    w[i] -= self.__h_lr(learningRate, epoch+1) * (grad['w'][i] + momentum * w[i])
                    b[i] -= self.__h_lr(learningRate, epoch+1) * grad['b'][i]
                    
                cost = self.__f_cost(a[-1], y)
                cost_local_sum += cost                
                self.cost_trace.append(cost)
            
            # save parameters
            self.__w = w
            self.__b = b
                
            # proceed validation after each epoch
            self.cost_epoch.append(cost_local_sum / num_of_batch)
            self.valid_error.append(self.__predict(valid_set, valid_label))
            
            # Early-stoppiing ?
            if self.__early_stop == 'on':
                Stop, Save = self.__EarlyStop(self.valid_error[-1])
                if not Stop and Save:
                    # save optimal weights                
                    self.__best_w = w
                    self.__best_b = b
             
                if Stop and not Save:
                    # Stop Right Here
                    print '<Early top at epoch %d with optimal validation error: %f%% at epoch %d\tTotal duration: %.2f>' %(epoch+1, np.min(self.valid_error)*100, np.argmin(self.valid_error)+1, Duration)
                    return self.cost_trace, self.valid_error
            else:
                self.__best_w = w
                self.__best_b = b
            
            end_time = time.time()
            Duration += end_time - start_time
            
            print 'epoch %d\tE_avg: %.4f\tvalid_error: %.2f%%\tDuration: %.2f second' % (epoch+1, self.cost_epoch[-1], self.valid_error[-1]*100, end_time-start_time)
        
        print '<WARNING ! No early-stopping case detected, Total duration: %.2f>'%(Duration)
        return self.cost_trace, self.valid_error
        
    
        
    def __predict(self, X, Y):
        
        """
        Predict function privately for validation use
        In contrast to predict(self, X, Y), this function use 'temporary' weights instead of optimal weights
        """
        h = self.__forward_prop(X, self.__w, self.__b)[1][-1]
        
        re = np.argmax(h,axis=1) != np.argmax(Y, axis = 1)
                
        err_rate = np.float32(np.sum(re))/len(Y)
        
        return err_rate    
    
    def predict(self, X, Y):
        
        h = self.__forward_prop(X, self.__best_w, self.__best_b)[1][-1]
        
        re = np.argmax(h,axis=1) != np.argmax(Y, axis = 1)
                
        err_rate = np.float32(np.sum(re))/len(Y)
        
        return err_rate    
    
    def getWeights(self):
        
        """
        return weights
        """
        
        return self.__best_w, self.__best_b
        
    def getWeightsTrace(self):
        
        return self.__weight_trace
        
    def getBiasTrace(self):
        
        return self.__bias_trace
        
    
