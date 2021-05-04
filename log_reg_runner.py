import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches
from logistic_regression import LogisticRegression
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

# Logistic Regression hyperparameters
eta = 0.001 # Learning rate
lam = 0.0001 # Lambda for regularization

def visualize_boundary(model, X, y, title, xlabel, ylabel, imgname, show, width=2):
    # Create a grid of points
    if show:
        weights = model.fit(x_s, y_s)
        ax = plot_decision_regions(X, y, clf=model)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, 
                ['Developed', 'Developing'], 
                framealpha=0.3, scatterpoints=1)
        plt.show()
        print("Accuracy for " + imgname + ": ", model.score(x_s, y_s))


data = pd.read_excel(r"./math_proj.xlsx")
x1 = data["Real GDP growth"].to_numpy()
x2 = data["Human Development Index"].to_numpy()
x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
y_s = data["Class"].to_numpy()
lr = LogisticRegression(eta=eta, lam=lam, runs=5000)
visualize_boundary(lr, x_s, y_s, 
                    "Using Logistic Regression to Predict Whether a Nation is Developed or Developing\n Based on GDP Growth and Human Development Index", 
                    "Real GDP Growth",
                    "Human Development Index",
                    'gdp_vs_hdi', True)

data = pd.read_excel(r"./math_proj_2.xlsx")
x1 = data["Inflation (% change) "].to_numpy()
x2 = data["General Government Total Expenditure (%GDP)"].to_numpy()
scaler = StandardScaler()
x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
scaler.fit(x_s)
x_s = scaler.transform(x_s)
y_s = data["Class"].to_numpy()
lr = LogisticRegression(eta=eta, lam=lam, runs=5000, basis="quad")
visualize_boundary(lr, x_s, y_s, 
                    "Using Logistic Regression to Predict Whether a Nation is Developed or Developing\n Based on Inflation and Government Expenditure", 
                    "Inflation (% change) (Standardized)",
                    "General Government Total Expenditure (%GDP) (Standardized)",
                    'inflation_vs_expenditure', True)

data = pd.read_excel(r"./math_proj_2.xlsx")
x1 = data["Inflation (% change) "].to_numpy()
x2 = data["General government net lending/borrowing (% GDP)"].to_numpy()
x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
y_s = data["Class"].to_numpy()
lr = LogisticRegression(eta=eta, lam=lam, runs=5000, basis="doublequad")
visualize_boundary(lr, x_s, y_s, 
                    "Using Logistic Regression to Predict Whether a Nation is Developed or Developing\n Based on Inflation and Government Net Lending and Borrowing", 
                    "Inflation (% change)",
                    "General government Net Lending/Borrowing (% GDP)",
                    'inflation_vs_borrowing', True)

data = pd.read_excel(r"./math_proj_3.xlsx")
x1 = data["Maternity"].to_numpy()
x2 = data["Government Expenditure"].to_numpy()
scaler = StandardScaler()
x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
scaler.fit(x_s)
x_s = scaler.transform(x_s)
y_s = data["Class"].to_numpy()
lr = LogisticRegression(eta=eta, lam=lam, runs=5000)
visualize_boundary(lr, x_s, y_s, 
                    "Using Logistic Regression to Predict Whether a Nation is Developed or Developing\n Based on Duration of Paid Maternity Leave and Government Expenditure", 
                    "Duration of Paid Maternity Leave (Standardized)",
                    "Government Expenditure (Standardized)",
                    'maternity_vs_expenditure', True)

data = pd.read_excel(r"./math_proj_4.xlsx")
x1 = data["Tech"].to_numpy()
x2 = data["Births"].to_numpy()
scaler = StandardScaler()
x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
scaler.fit(x_s)
x_s = scaler.transform(x_s)
y_s = data["Class"].to_numpy()
lr = LogisticRegression(eta=eta, lam=lam, runs=5000)
visualize_boundary(lr, x_s, y_s, 
                    "Using Logistic Regression to Predict Whether a Nation is Developed or Developing\n Based on Share of Female STEM Graduates and Proportion of Births Attended by Healthcare Professionals", 
                    "Share of Female STEM Graduates (Standardized)",
                    "Proportion of Births Attended by Healthcare Professionals (Standardized)",
                    'tech_vs_births', True)