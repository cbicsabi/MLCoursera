import numpy as np


def normal_equation(feature_matrix, prediction_vector):
    X = feature_matrix
    print("X:\n", X)
    y = prediction_vector
    print("y:\n", y)
    Xt = np.transpose(X)
    print("Xt:\n", Xt)
    XtX = Xt.dot(X)
    print("XtX:\n", XtX)
    XtX_inv = np.linalg.pinv(XtX)
    print("XtX_inv:\n", XtX_inv)
    XtX_inv_Xt = XtX_inv.dot(Xt)
    print("XtX_inv_Xt:\n", XtX_inv_Xt)
    XtX_inv_Xty = XtX_inv_Xt.dot(y)
    print("XtX_inv_Xty:\n", XtX_inv_Xty)

    theta = np.dot(np.dot(np.linalg.inv(Xt.dot(X)), Xt), y)
    print("theta:\n", theta)

    return XtX_inv_Xty


def main():
    X = np.matrix("1, 2104, 5, 1, 45;"
                  "1, 1416, 3, 2, 40;"
                  "1, 1534, 3, 2, 30;"
                  "1, 852, 2, 1, 30")
    y = np.matrix("460; 232; 315; 178")

    print("!!!!!", np.linalg.matrix_rank(X))

    theta = normal_equation(X, y)

    print(X.dot(theta))
    print(y)

    XT = np.transpose(X)

    XTX = XT.dot(X)

    inv = np.linalg.pinv(XTX)

    inv_XT = inv.dot(XT)

    theta = inv_XT.dot(y)

    print(np.linalg.det(XTX))
    print(theta)
    print(X.dot(theta))
    print(y)


if __name__ == "__main__":
    main()
