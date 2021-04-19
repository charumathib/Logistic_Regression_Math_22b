import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:
    def __init__(self, eta, lam, runs):
        self.eta = eta
        self.lam = lam
        self.runs = runs

    def fit(self, X, y):
        self.len, self.feat = X.shape
        self.X = np.c_[np.ones(self.len), X]
        self.y_target = y
        self.W = np.random.rand(self.feat + 1)
        self.loss = []

        self.num_runs = 0


        while self.num_runs < self.runs:
            gradient_wrt_i = 0
            for i in range(self.len):
                z = np.dot(self.W, self.X[i])
                gradient_wrt_i += (self.__sigmoid(z) - self.y_target[i]) * self.X[i]
            gradient_wrt_i += self.lam * self.W
            self.W -= self.eta * gradient_wrt_i
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

    def predict(self, X_pred):
        preds = []
        X_pred = np.c_[np.ones(len(X_pred)), X_pred]
        for i in range(len(X_pred)):
            z = np.dot(self.W, X_pred[i])
            sigmoid = self.__sigmoid(z)
            if sigmoid > 0.5: preds.append(1) 
            else: preds.append(0)
        return np.array(preds)
    
    def plot_loss(self):
        plt.plot(np.linspace(0, self.num_runs, self.num_runs), self.loss)
        plt.show()