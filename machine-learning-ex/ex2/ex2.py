import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


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


def cost_function_reg(theta, X, y, lam=1):
    m = len(y)

    h = sigmoid(X.dot(theta))
    reg = lam/(2*m) * sum(theta[1:]**2)
    J = -1/m*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + reg

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


def gradient_reg(theta, X, y, lam=1):
    m = len(y)
    h = sigmoid(X.dot(theta))

    # Un-vectorised form:
    # grad = np.zeros(theta.size).reshape(3, 1)
    # for i in range(m):
    #     grad = grad + (h[i] - y[i]) * X[i, :].T

    grad = (h - y).dot(X)

    reg = np.matrix([i if ind > 0 else 0 for ind, i in enumerate(theta)])

    grad = 1/m*grad[0] + lam/m*reg
    return grad


def predict(theta, X):
    return (sigmoid(X.dot(theta.T)) >= 0.5).astype("int")


def main():
    # LOGISTIC REGRESSION
    data = load_data('ex2data1.txt', ',')

    X = np.c_[np.ones(data.shape[0]), data[:, 0:2]]
    y = data[:, 2]

    # plot_data(data, 'Test 1 Score', 'Test 2 Score', 'Admitted', 'Not Admitted')

    initial_theta = np.zeros(X.shape[1])
    cost = cost_function(initial_theta, X, y)
    grad = gradient(initial_theta, X, y)
    print('Cost: \n', cost)
    print('Grad: \n', grad)

    res = minimize(cost_function, initial_theta, method=None, jac=gradient, args=(X, y), options={'maxiter': 400})
    print('Cost at theta found by fminunc:\n', res.fun)
    print('Theta:\n', res.x.T)

    show_decision_boundary = False
    if show_decision_boundary:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
        x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
        h = h.reshape(xx1.shape)
        ax.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
        plot_data(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted', axes=ax)

    p = predict(res.x, X)
    print(f'Train accuracy: {100*sum(p == y)/p.size}%')

    # REGULARIZED LOGISTIC REGRESSION
    data2 = load_data('ex2data2.txt', ',')

    X = data2[:, 0:2]
    y = np.c_[data2[:, 2]]

    # plot_data(data2, "Microchip Test 1", "Microchip Test 2", "y = 1", "y = 0")

    poly = PolynomialFeatures(6)
    XX = poly.fit_transform(X)
    initial_theta = np.zeros(XX.shape[1])

    J = cost_function_reg(initial_theta, XX, y, 1)
    print(f'Cost at initial theta (zeros): {J[0]}')

    grad = gradient_reg(initial_theta, XX, y, 1)
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
