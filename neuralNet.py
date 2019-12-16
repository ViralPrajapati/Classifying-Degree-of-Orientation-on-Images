#!/usr/bin/env python
# coding: utf-8
#Citations
#Refered below mentioned links to understand the flow and the intution of neural network
#https://cs231n.github.io/classification/
#https://towardsdatascience.com/understanding-neural-networks-19020b758230

import numpy as np
import random

np.random.seed(11)

class Model:

    #Initializing the class
    def __init__(self, noCategories, inputSize, hiddenUnits, l1, l2, epochs, 
                 lrate,noBatches):

        self.noCategories = noCategories
        self.inputSize = inputSize
        self.hiddenUnits = hiddenUnits
        self.w1, self.w2, self.w3, self.w4 = self.initWeights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.lrate = lrate
        self.noBatches = noBatches
        
    #To convert labeled data to categorical data
    def oneHotEncoding(self, y):
        labelEnc = []
        for label in y:
            if label == 0:
                labelEnc.append([1,0,0,0])
            elif label == 1:
                labelEnc.append([0,1,0,0])
            elif label == 2:
                labelEnc.append([0,0,1,0])
            elif label == 3:
                labelEnc.append([0,0,0,1])
        labelEnc = np.array(labelEnc, dtype = 'float')
        return labelEnc
    
#Inorder to improve accuracy we have used L1 L2 regularization function.
#Used below mentioned link to properly understand the regularizaion functions.
#http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
        
    #regularization functions
    def L1(self, lambdaa, w1, w2, w3, w4):
        l1Reg =  (lambdaa / 2.0) * (np.abs(w1).sum() + np.abs(w2).sum() + np.abs(w3).sum() + np.abs(w4).sum())
        return l1Reg
    
    def L2(self, lambdaa, w1, w2, w3, w4):
        l2Reg = (lambdaa / 2.0) * (np.sum(w1 ** 2) + np.sum(w2 ** 2) + np.sum(w3 ** 2) + np.sum(w4 ** 2))
        return l2Reg
    
    def crossEntropy(self, outputs, y_target):
        crssEnty = -np.sum(np.log(outputs) * y_target, axis=1)
        return crssEnty
    
    #Sigmoid activation function
    def sigmoid(self, x):
        sig = 1.0 / (1.0 + np.exp(-x))
        return sig
        #return expit(z)
    
    #Sigmoid derivative function for backward propogation
    def sigmoidDerivative(self, x):
        sig = self.sigmoid(x)
        sigDr = sig * (1 - sig)
        return sigDr
    
    #ReLu activation function
    def reLU(self, x):
        return np.maximum(0.0,x)
    
    #ReLU derivative function for backward propogation
    def relu_derivative(self, x):
        return np.greater(x,0).astype(int)

    #Softmax for calculating class probabilities
    def softmax(self, x):
        sfmax = (np.exp(x) / np.sum(np.exp(x), axis=0))
        return sfmax

    def initWeights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=(self.hiddenUnits, self.inputSize))        
        w2 = np.random.uniform(-1.0, 1.0, size=(self.hiddenUnits, self.hiddenUnits))
        w3 = np.random.uniform(-1.0, 1.0, size=(self.hiddenUnits, self.hiddenUnits))        
        w4 = np.random.uniform(-1.0, 1.0, size=(self.noCategories, self.hiddenUnits))
        return w1, w2, w3, w4
      
    def forwardPass(self, X):
        
        inputLayer = np.array(X.copy(), dtype = "float64")
        layer1 = self.w1.dot(inputLayer.T)
        actLayer1 = self.sigmoid(layer1)
        layer2 = self.w2.dot(actLayer1)
        actLayer2 = self.sigmoid(layer2)
        layer3 = self.w3.dot(actLayer2)
        actLayer3 = self.sigmoid(layer2)
        outputLayer = self.w4.dot(actLayer3)
        outputActivation = self.sigmoid(outputLayer)
        
        return inputLayer, layer1, actLayer1, layer2, actLayer2, layer3, actLayer3, outputLayer, outputActivation
    
    def backwardPass(self, inputLayer, layer1, actLayer1, layer2, actLayer2, layer3, actLayer3, outputActivation, y):
        sigma5 = outputActivation - y
        sigma4 = self.w4.T.dot(sigma5) * self.sigmoidDerivative(layer3)
        sigma3 = self.w3.T.dot(sigma4) * self.sigmoidDerivative(layer2)
        sigma2 = self.w2.T.dot(sigma3) * self.sigmoidDerivative(layer1)

        grad1 = sigma2.dot(inputLayer)
        grad2 = sigma3.dot(actLayer1.T)
        grad3 = sigma4.dot(actLayer2.T)
        grad4 = sigma5.dot(actLayer3.T)
        return grad1, grad2, grad3, grad4
    
    def calError(self, y, output):
        l1Val = self.L1(self.l1, self.w1, self.w2, self.w3, self.w4)
        l2Val = self.L2(self.l2, self.w1, self.w2, self.w3, self.w4)
        error = self.crossEntropy(output, y) + l1Val + l2Val
        return 0.5 * np.mean(error)

    def backPropogate(self, X, y):
        inputLayer, layer1, actLayer1, layer2, actLayer2, layer3, actLayer3, outputLayer, outputActivation = self.forwardPass(X)
        y = y.T

        grad1, grad2, grad3, grad4 = self.backwardPass(inputLayer, layer1, actLayer1, layer2, actLayer2, layer3, actLayer3, outputActivation, y)

        grad1 += (self.w1 * (self.l1 + self.l2))
        grad2 += (self.w2 * (self.l1 + self.l2))
        grad3 += (self.w3 * (self.l1 + self.l2))
        grad4 += (self.w4 * (self.l1 + self.l2))
        
        error = self.calError(y, outputActivation)
        
        return error, grad1, grad2, grad3, grad4

    def predict(self, X):
        X1 = X.copy()
        inputLayer, layer1, actLayer1, layer2, actLayer2, layer3, actLayer3, outputLayer, outputActivation = self.forwardPass(X1)
        return np.argmax((outputLayer.T), axis=1)
    
    def softmaxValues(self, X):
        X1 = X.copy()
        inputLayer, layer1, actLayer1, layer2, actLayer2, layer3, actLayer3, outputLayer, outputActivation = self.forwardPass(X1)
        return self.softmax(outputActivation.T)

    def fit(self, X, y):
        err = []
        X_data, y_data = X.copy(), y.copy()
        yEnc = self.oneHotEncoding(y_data)
                
        xBatches = np.array_split(X_data, self.noBatches)
        yBatches = np.array_split(yEnc, self.noBatches)     
        
        for i in range(self.epochs):
            
            epochError = []

            for Xi, yi in zip(xBatches, yBatches):
                
                error, grad1, grad2, grad3, grad4 = self.backPropogate(Xi, yi)
                epochError.append(error)
                self.w1 -= (self.lrate * grad1)
                self.w2 -= (self.lrate * grad2)
                self.w3 -= (self.lrate * grad3)
                self.w4 -= (self.lrate * grad4)
            err.append(np.mean(epochError))
        return self
    
    def score(self, X, y):
        predicted = self.predict(X)
        return np.sum(y == predicted, axis=0) / float(X.shape[0])

def train(data):
    pId = []
    label = []
    trainData = []
    for value in data:
        pId.append(value[0])
        label.append(value[1])
        trainData.append(value[2])
        
    pId = np.array(pId).reshape(len(pId),1)
    label = np.array(label, dtype = int).reshape(len(label),1)
    label = label//90
    trainData = np.array(trainData, dtype=int)

    trainDataset = np.concatenate((trainData, label), axis=1)
    random.shuffle(trainDataset)
    
    X_train, y_train = trainDataset[:,:-1], trainDataset[:,-1]
    X_train_normalized = X_train.astype(np.float64) / 224
    
    
    nn = Model(noCategories=4, inputSize=192, hiddenUnits=50, l2=0.5, l1=0.0, epochs=300,
                      lrate=0.001, noBatches=200).fit(X_train_normalized, y_train)
        
    return nn

def test(nn, test_data):
    pId = []
    label = []
    testData = []
    for value in test_data:
        pId.append(value[0])
        label.append(value[1])
        testData.append(value[2])
    
    X_test = np.array(testData, dtype = 'float64')
    X_test_normalized = X_test / 224
    y_test = np.array(label, dtype = 'float64') / 90
    yPred = nn.softmaxValues(X_test_normalized)
    pred = np.argmax(yPred, axis=1)
    accuracy = nn.score(X_test_normalized, y_test)
    return (pId, pred * 90), accuracy
