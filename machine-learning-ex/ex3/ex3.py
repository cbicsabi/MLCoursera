import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

# load MATLAB files
from scipy.io import loadmat


def sigmoid(z):
    return 1/(1+np.exp(-z))


# def cost_function(theta, X, y):
#     m = len(y)
#
#     h = sigmoid(X.dot(theta))
#     J = -1/m*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
#
#     return J
#
#
# def cost_function_reg(theta, X, y, lam=1):
#     m = len(y)
#     h = sigmoid(X.dot(theta))
#
#     reg = lam/(2*m) * np.sum(np.square(theta[1:]))
#
#     J = -1*(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + reg
#
#     return J


def lrcostFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg / (2 * m)) * np.sum(
        np.square(theta[1:]))

    # if np.isnan(J[0]):
    #     return np.inf
    return J #[0]


# def gradient(theta, X, y):
#     m = len(y)
#     h = sigmoid(X.dot(theta))
#
#     # Un-vectorised form:
#     # grad = np.zeros(theta.size).reshape(3, 1)
#     # for i in range(m):
#     #     grad = grad + (h[i] - y[i]) * X[i, :].T
#
#     grad = (h - y).dot(X)
#     grad = 1/m*grad
#     return grad
#
#
# def gradient_reg(theta, X, y, lam=1):
#     m = len(y)
#     h = sigmoid(X.dot((theta.reshape(-1, 1))))
#
#     # Un-vectorised form:
#     # grad = np.zeros(theta.size).reshape(3, 1)
#     # for i in range(m):
#     #     grad = grad + (h[i] - y[i]) * X[i, :].T
#
#     reg = lam/m * np.matrix([i if ind > 0 else 0 for ind, i in enumerate(theta)]).reshape(-1, 1)
#
#     grad = 1/m * X.T.dot(h - y) + reg
#     return grad


def lrgradientReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))

    r = (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    grad = (1 / m) * X.T.dot(h - y) + r

    return grad.flatten()


# def one_vs_all(X, y, num_labels, lam):
#     initial_theta = np.zeros((X.shape[1], 1))
#     all_theta = np.zeros((num_labels, X.shape[1]))
#
#     for i in range(num_labels):
#         theta = minimize(cost_function_reg, initial_theta, method=None, jac=gradient_reg, args=(X, (y == i)*1, lam),
#                          options={'maxiter': 50})
#         all_theta[i] = theta.x
#
#     return all_theta.T


def oneVsAll(features, classes, n_labels, reg):
    initial_theta = np.zeros((features.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, features.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(reg, features, (classes == c)*1), method=None,
                       jac=lrgradientReg, options={'maxiter': 50})
        all_theta[c-1] = res.x
    return all_theta


def predict_one_vs_all(all_theta, X):
    X_theta = X.dot(all_theta.T)
    return np.argmax(X_theta, axis=1)+1


def predict(theta1, theta2, X):
    a1 = X

    z2 = a1.dot(theta1.T)
    a2 = np.c_[np.ones(z2.shape[0]), sigmoid(z2)]

    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3, axis=1)+1


def main():
    data = loadmat('ex3data1.mat')
    # print(data.keys())

    X = np.c_[np.ones(data['X'].shape[0]), data['X']]
    y = data['y']

    weights = loadmat('ex3weights.mat')
    # print(weights.keys())

    theta1, theta2 = weights['Theta1'], weights['Theta2']

    sample = np.random.choice(X.shape[0], 20)
    plt.imshow(X[sample, 1:].reshape(-1, 20).T)
    plt.axis('off')
    # plt.show()

    # theta_t = np.array([-2, -1, 1, 2])
    # X_t = np.c_[np.ones(5), (np.arange(15).reshape(5, 3, order='F') + 1)/10]
    # y_t = np.array([i if i >= 0.5 else 0 for i in [1, 0, 1, 0, 1]])
    # lambda_t = 3
    #
    # J = cost_function_reg(theta_t, X_t, y_t, lambda_t)
    # grad = gradient_reg(theta_t, X_t, y_t, lambda_t)
    # print(J)
    # print(grad)
    #
    # J = lrcostFunctionReg(theta_t, lambda_t, X_t, y_t)
    # grad = lrgradientReg(theta_t, lambda_t, X_t, y_t)
    # print(J)
    # print(grad)

    num_labels = 10
    lam = 0.1
    # theta = oneVsAll(X, y, num_labels, lam)

    # pred = predict_one_vs_all(theta, X).reshape(-1, 1)
    # print(f'Training Set Accuracy: {np.mean(pred == y) * 100}')

    pred = predict(theta1, theta2, X)
    print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel()) * 100))


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
