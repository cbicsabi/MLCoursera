import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d


def load_data(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    # print('Dimensions: ', data.shape)
    # print(data[1:6, :])
    return data


def plot_data(data, label_x, label_y, label_pos, label_neg, axes=None):
    pos = data[:, 2] == 1
    neg = data[:, 2] == 0

    if axes is None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], marker='x', label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend()
    plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost_function(theta, X, y):
    m = len(y)

    h = sigmoid(X.dot(theta))
    J = -1/m*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))

    return J


def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))

    # Un-vectorised form:
    # grad = np.zeros(theta.size).reshape(3, 1)
    # for i in range(m):
    #     grad = grad + (h[i] - y[i]) * X[i, :].T

    grad = (h - y).dot(X)
    grad = 1/m*grad
    return grad


def main():
    data = load_data('ex2data1.txt', ',')

    X = np.c_[np.ones(data.shape[0]), data[:, 0:2]]
    y = data[:, 2]

    # plot_data(data, 'Test 1 Score', 'Test 2 Score', 'Admitted', 'Not Admitted')

    initial_theta = np.zeros(X.shape[1])
    cost = cost_function(initial_theta, X, y)
    grad = gradient(initial_theta, X, y)
    print('Cost: \n', cost)
    print('Grad: \n', grad)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.backends.backend_qt5agg

    gui_env = ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
               'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'Agg']
    for gui in gui_env:
        try:
            # print("testing", gui)
            matplotlib.use(gui, warn=False, force=True)
            break
        except:
            continue
    # print("Using:", matplotlib.get_backend())
    main()
