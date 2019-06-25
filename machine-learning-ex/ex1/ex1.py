import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d


def warm_up_exercise():
    return np.eye(5)


def plot_data(X, y):
    plt.figure()
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.plot(X, y, color='red', marker='x', linestyle='None')
    plt.show()


def compute_cost(X, y, theta):
    m = y.size
    h = X.dot(theta)
    sqrdErrors = np.square(h - y)
    J = 1 / (2 * m) * sum(sqrdErrors)

    return J


def gradient_descent(X, y, alpha, theta, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def main():
    warm_up_exercise()

    data = pd.read_csv("ex1data1.txt", header=None)

    X, y = np.matrix(data.T.values)
    X = np.transpose(X)
    y = np.transpose(y)

    # plot_data(X, y)

    m = len(X)
    X = np.concatenate((np.ones(X.shape), X), axis=1)
    theta = np.zeros((2, 1))
    iterarions = 1500
    alpha = 0.01

    # theta for minimized cost J
    theta, Cost_J = gradient_descent(X, y, alpha, theta, iterarions)
    print('theta: ', theta.ravel())

    plt.plot(Cost_J)
    plt.ylabel('Cost J')
    plt.xlabel('Iterations')

    # Predict profit for a city with population of 35000 and 70000
    print(theta.T.dot([1, 3.5]) * 10000)
    print(theta.T.dot([1, 7]) * 10000)

    theta0_vals = np.linspace(-10, 10, 50)
    theta1_vals = np.linspace(-1, 4, 50)
    xx, yy = np.meshgrid(theta0_vals, theta1_vals, indexing='xy')

    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.matrix([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i, j] = compute_cost(X, y, t)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(xx, yy, J_vals, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
    ax.set_zlabel('Cost')
    ax.set_zlim(J_vals.min(), J_vals.max())

    plt.show()


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
