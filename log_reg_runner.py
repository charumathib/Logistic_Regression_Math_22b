# Don't change these imports. Note that the last two are the
# class implementations that you will implement in
# T2_P3_LogisticRegression.py and T2_P3_GaussianGenerativeModel.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches
from logistic_regression import LogisticRegression
from matplotlib.lines import Line2D

# These are the hyperparameters to the classifiers. You may need to
# adjust these as you try to find the best fit for each classifier.

# Logistic Regression hyperparameters
eta = 0.05 # Learning rate
lam = 0.0 # Lambda for regularization

# Whether or not you want the plots to be displayed
show_charts = True


# DO NOT CHANGE ANYTHING BELOW THIS LINE!
# -----------------------------------------------------------------

# Visualize the decision boundary that a model produces
def visualize_boundary(model, X, y, title, width=2):
    # Create a grid of points
    weights = model.fit(x_s, y_s)
    colors = ["red", "blue"]
    colormap = c.ListedColormap(colors)
    countries = data["Country "].to_numpy()
    
    def map_helper (x):
        if x == 0:
            return "Developed"
        return "Developing"

    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap)
    
    custom = [Line2D([0], [0], marker='o', color='w', label='Developed', markerfacecolor='r', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='Developing', markerfacecolor='b', markersize=10) ]

    legend = ax.legend(handles=custom, loc="lower right")
    ax.add_artist(legend)

    for i in range(len(X)):
        name = countries[i][:8]
        plt.annotate(xy=[X[i][0],X[i][1]], s=name)

    x_min, x_max = min(list(X[:, 0])), max(list((X[:, 0])))
    y_min, y_max = min(list(X[:, 1])), max(list((X[:, 1])))

    print(x_min)
    print(y_min)
    b = -weights[0]/weights[2]
    m = -weights[1]/weights[2]

    xd = np.array([x_min, x_max])
    yd = m * xd + b

    plt.plot(xd, yd, 'k', lw=1, ls='--', label='Log Reg Decision Boundary')
    plt.fill_between(xd, yd, y_min, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, y_max, color='tab:red', alpha=0.2)
    plt.xlabel("Real GDP Growth")
    plt.ylabel("Human Development Index")
    plt.legend()
    plt.title("Using Logistic Regression to Predict Whether a Nation is Developed or Developing\n Based on GDP Growth and Human Development Index")

    # Saving the image to a file, and showing it as well
    
    plt.savefig('developed_vs_developing.png')
    plt.show()


# A mapping from string name to id
class_labels = {
    'Developed': 0,       # also corresponds to 'red' in the graphs
    'Developing': 1,       # also corresponds to 'blue' in the graphs
}

data = pd.read_excel(r"./math_proj.xlsx")
print(data)
x1 = data["Real GDP growth"].to_numpy()
x2 = data["Human Development Index"].to_numpy()
x_s = np.concatenate((x1.reshape(x1.shape[0],1), x2.reshape(x2.shape[0],1)), axis=1)
y_s = data["Class"].to_numpy()
lr = LogisticRegression(eta=eta, lam=lam, runs=1000)
visualize_boundary(lr, x_s, y_s, 'logistic_regression_result')