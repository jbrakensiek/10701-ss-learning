import math
from random import random
import parser
import sys
import numpy as np

def sigmoid(x):
    if x > 200:
        return 1
    if x < -200:
        return 0    
    return 1.0 / (1.0 + math.exp(-x))

class WeightedDiGraph:
    '''
    Graph representation of neural network
    '''
    
    def __init__(self, n_vertices):
        ''' initializes with isolated vertices'''
        self.N = n_vertices # number of vertices
        self.adj = [None] * self.N # adjaceny list dictionary with weight
        self.rev = [None] * self.N # back edges
        self.grad = [None] * self.N # previous gradient, for momentum
        for i in range(self.N):
            self.adj[i] = dict()
            self.rev[i] = dict()
            self.grad[i] = dict()
    
        self.const = [0] * self.N # constant weights
        self.const_grad = [0] * self.N # previous gradient on constants
        
    def add_edge(self, v_in, v_out, w):
        '''adds the edge if it did not exist before'''
        self.adj[v_in][v_out] = w
        self.rev[v_out][v_in] = w
        self.grad[v_in][v_out] = 0

class NeuralNet:
    ''' A semi-supervised neural network implementation

    Based on 'Pseudo Label' Dong-Hyun Lee
    '''
    
    def __init__(self, nI, nB, nO, eta):
        '''initialize neural network with random weights'''
        self.numInput = nI # number of input nodes
        self.numBetween = nB
        self.numOutput = nO
        self.Net = WeightedDiGraph(nI + nB + nO + nI)
        self.EPS = 0.00
        
        # initialize constant weights
        for i in range(self.Net.N):
            self.Net.const[i] = random() * eta
        
        for i in range(nI):
            for j in range(nB):
                self.Net.add_edge(i, j + nI, random() * eta)
        for i in range(nB):
            for j in range(nO):
                self.Net.add_edge(i + nI, j + nI + nB, random() * eta)
            for j in range(nI):
                self.Net.add_edge(i + nI, j + nI + nB + nO, random() * eta)
                
    def compute(self, x, drop=False):
        '''compute the values of all the units on input x'''
        assert (len(x) == self.numInput)

        v = [0] * self.Net.N
        for i in range(self.numInput):
            v[i] = x[i]
            
        for i in range(self.numInput, self.Net.N):
            tot = self.Net.const[i]
            for key in self.Net.rev[i]:
                tot += v[key] * self.Net.rev[i][key]
            v[i] = sigmoid(tot)
            # dropout
            if i < self.numInput + self.numBetween and drop and random() > .5:
                v[i] = 0

        return v
        
                
    def classify(self, x):
        '''classifies a single data point x'''
        v = self.compute(x)
        #print x, v
        
        best = -1
        bestVal = -1
        
        for i in range(self.numOutput):
            newVal = v[self.numInput + self.numBetween + i]
            if newVal > bestVal:
                bestVal = newVal
                best = i

        assert (best != -1)
        return best

    def update(self, x, y, p, eta):
        '''perform a backpropogation step based on this training sample

        Implements backprop algorithm from Lecture 12
        '''
        
        v = self.compute(x)
        delta = [0] * self.Net.N

        for i in range(self.numOutput):
            ind = i + self.numInput + self.numBetween
            delta[ind] = (v[ind] * (1 - v[ind]) + self.EPS) * (int(i == y) - v[ind])

        for i in range(self.numInput):
            ''' auto encoder '''
            ind = i + self.numInput + self.numBetween + self.numOutput
            delta[ind] = 3 * (v[ind] * (1 - v[ind]) + self.EPS) * (x[i] - v[ind])
            
        for i in range(self.numBetween):
            ind = i + self.numInput
            for j in self.Net.adj[ind]:
                assert ind in self.Net.rev[j]
                delta[ind] += (v[ind] * (1 - v[ind]) + self.EPS) * self.Net.adj[ind][j] * delta[j]

        for i in range(self.Net.N):
            self.Net.const_grad[i] += eta * delta[i]
            for j in self.Net.adj[i]:
                assert i in self.Net.rev[j]
                self.Net.grad[i][j] += eta * delta[j] * v[i]
        
    def labeled_backprop_once(self, X, Y, p, eta):
        ''' train the neural net on the labeled data using back propogation'''
        for k in range(len(X)):
            for i in range(self.Net.N):
                self.Net.const_grad[i] *= p
                for j in self.Net.adj[i]:
                    self.Net.grad[i][j] *= p
        
            self.update(X[k], Y[k], p, eta)

            for i in range(self.Net.N):
                self.Net.const[i] += self.Net.const_grad[i]
                for j in self.Net.adj[i]:
                    self.Net.adj[i][j] += self.Net.grad[i][j]
                    self.Net.rev[j][i] = self.Net.adj[i][j]
            
    def unlabeled_backprop_once(self, X, p, eta):
        ''' train the neural net on the unlabeled data using pseudolabels'''
        for k in range(len(X)):
            for i in range(self.Net.N):
                self.Net.const_grad[i] *= p
                for j in self.Net.adj[i]:
                    self.Net.grad[i][j] *= p

            self.update(X[k], self.classify(X[k]), p, eta)

            for i in range(self.Net.N):
                self.Net.const[i] += self.Net.const_grad[i]
                for j in self.Net.adj[i]:
                    self.Net.adj[i][j] += self.Net.grad[i][j]
                    self.Net.rev[j][i] = self.Net.adj[i][j]
            
    def test(self, X_tes, Y_tes):
        ''' run on test data '''
        mat = [[0] * 10] * 10
        cor = 0
        for i in range(len(X_tes)):
            y = self.classify(X_tes[i])
            if (y == int(Y_tes[i])):
                cor += 1
            #print y, int(Y_tes[i])
            mat[int(Y_tes[i])][y] += 1
        return 1.0 * cor / len(X_tes), mat

    def error(self, X_tes, Y_tes):
        err = 0.0
        for j in range(len(X_tes)):
            v = self.compute(X_tes[j])
            for i in range(self.numOutput):
                err += .5 * (v[self.numInput + self.numBetween + i] - int (Y_tes[j] == i))**2
            for i in range(self.numInput):
                err += 1.5 * (v[self.numInput + self.numBetween + self.numOutput + i] - X_tes[j][i])**2

        return err

if __name__ == "__main__":
    frac = float(sys.argv[1])

    pen = parser.PenParser()
    X_lab, Y_lab, X_unlab, X_tes, Y_tes = pen.retrieve_pendigits_data(frac)

    nnet = NeuralNet(16, 14, 10, 0.1)
    
    print 'Supervised initial phase'

    print len(X_lab)
    
    for i in range(100):
        print "Round: " + str(i)
        nnet.labeled_backprop_once(X_lab, Y_lab, .9, (100.0 - i)*.002)
        print nnet.test(X_tes, Y_tes)[0], nnet.test(X_lab, Y_lab)[0]
        print nnet.error(X_tes, Y_tes), nnet.error(X_lab, Y_lab)
        
    print 'Semi-supervised phase'
        
    for i in range(500):
        print "Round: " + str(i + 100)
        nnet.unlabeled_backprop_once(X_unlab, 0.9, (i + 1) * .0000002)
        nnet.labeled_backprop_once(X_lab, Y_lab, 0.9, (600 - i) * .0004)
        print nnet.test(X_tes, Y_tes)[0], nnet.test(X_lab, Y_lab)[0]
        print nnet.error(X_tes, Y_tes), nnet.error(X_lab, Y_lab)
