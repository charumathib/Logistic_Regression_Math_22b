import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, eta, lam, runs, basis="lin"):
        self.eta = eta
        self.lam = lam
        self.runs = runs
        self.basis = basis

    def fit(self, X, y):
        # adds another column to each datapoint x_n that is the first entry in x_n squared 
        # (used for inflation vs government expenditure dataset)
        if self.basis == "quad": self.X = np.c_[np.ones(len(X)), X, X[:,0]**2]
        # adds two more columns to each datapoint x_n that are both entires in x_n squared 
        # (used for inflation vs government borrowing)
        elif self.basis == "doublequad": self.X = np.c_[np.ones(len(X)), X, X[:,0]**2, X[:,1]**2]
        # no basis transformation
        else: self.X = np.c_[np.ones(len(X)), X]

        self.len, self.feat = self.X.shape
        self.y_target = y
        self.W = np.random.rand(self.feat)
        self.loss = []
        self.num_runs = 0

        # this implementation of gradient descent runs a fixed number of times, but an
        # epsilon-based approach may also be used
        while self.num_runs < self.runs:
            gradient = 0
            for i in range(self.len):
                z = np.dot(self.W, self.X[i])
                gradient += (self.__sigmoid(z) - self.y_target[i]) * self.X[i]
            gradient += self.lam * self.W
            self.W -= self.eta * gradient
            self.loss.append(self.__calculate_loss())
            self.num_runs += 1

        return self.W

    def __sigmoid (self, z):
        return 1/(1 + np.exp(-z))

    def __calculate_loss (self):
        loss = 0
        for i in range(self.len):
            z = np.dot(self.W, self.X[i])
            loss = loss + (self.y_target[i])*(np.log(self.__sigmoid(z))) + (1-self.y_target[i])*(np.log(1-self.__sigmoid(z)))
        return -loss + self.lam/2 * np.linalg.norm(self.W)**2

    # uses the values of parameters w calculated in the fit() function to predict the class
    # of each point x in the matrix X_pred
    def predict(self, X_pred):
        preds = []
        # for government expenditure
        if self.basis == "quad":
            X_pred = np.c_[np.ones(len(X_pred)), X_pred, X_pred[:,0]**2]
        # for government borrowing
        elif self.basis == "doublequad":
            X_pred = np.c_[np.ones(len(X_pred)), X_pred, X_pred[:,0]**2, X_pred[:,1]**2]
        # regular
        else:
            X_pred = np.c_[np.ones(len(X_pred)), X_pred]  
        for i in range(len(X_pred)):
            z = np.dot(self.W, X_pred[i])
            sigmoid = self.__sigmoid(z)
            if sigmoid > 0.5: preds.append(1) 
            else: preds.append(0)
        return np.array(preds)
    
    # accuracy
    def score(self, X, y_true):
        return np.count_nonzero((self.predict(X) - y_true) == 0)/len(y_true)
    
    # plots the value of the loss function at every run
    def plot_loss(self):
        plt.plot(np.linspace(0, self.num_runs, self.num_runs), self.loss)
        plt.show()