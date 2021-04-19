import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, eta, lam, runs):
        self.eta = eta
        self.lam = lam
        self.runs = runs

    def fit(self, X, y):
        # for government expenditure
        # self.X = np.c_[np.ones(len(X)), X, X[:,0]**2]
        # for government borrowing
        self.X = np.c_[np.ones(len(X)), X, X[:,0]**2, X[:,1]**2]
        #regular
        # self.X = np.c_[np.ones(len(X)), X]
        self.len, self.feat = self.X.shape
        self.y_target = y
        self.W = np.random.rand(self.feat)
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
        # for government expenditure
        # X_pred = np.c_[np.ones(len(X_pred)), X_pred, X_pred[:,0]**2]
        # for government borrowing
        X_pred = np.c_[np.ones(len(X_pred)), X_pred, X_pred[:,0]**2, X_pred[:,1]**2]
        # regular
        # X_pred = np.c_[np.ones(len(X_pred)), X_pred]
        for i in range(len(X_pred)):
            z = np.dot(self.W, X_pred[i])
            sigmoid = self.__sigmoid(z)
            if sigmoid > 0.5: preds.append(1) 
            else: preds.append(0)
        return np.array(preds)
    
    def score(self, X, y_true):
        return np.count_nonzero((self.predict(X) - y_true) == 0)/len(y_true)
    
    def plot_loss(self):
        plt.plot(np.linspace(0, self.num_runs, self.num_runs), self.loss)
        plt.show()

# data = pd.read_excel(r"./math_proj_3.xlsx")
# x1 = data["Maternity"].to_numpy()
# x2 = data["Government Expenditure"].to_numpy()
# scaler = StandardScaler()
# x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
# scaler.fit(x_s)
# x_s = scaler.transform(x_s)
# y_s = data["Class"].to_numpy()
# lr = LogisticRegression(eta=0.001, lam=0.00, runs=1000)
# lr.fit(x_s, y_s)
# print(lr.score(x_s, y_s))
# lr.plot_loss()