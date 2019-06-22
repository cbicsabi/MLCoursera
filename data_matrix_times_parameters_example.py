import numpy as np
from numpy.linalg import inv


def main():
    data_matrix = np.matrix('1, 2104;1, 1416;1, 1534;1, 852')
    model = np.matrix('-40; 0.25')

    prediction = data_matrix * model

    print(data_matrix)
    print(data_matrix.shape, "\n")
    print(model)
    print(model.shape, "\n")
    print(prediction)
    print(np.transpose(prediction))

    A = np.matrix('1,3,2;4,0,1')
    B = np.matrix('1,3;0,1;5,2')

    # print(A)
    # print(A.shape, "\n")
    # print(B)
    # print(B.shape, "\n")
    # print(A*B)

    C = np.matrix('1,2;3,4')
    print(C)
    print(inv(C))
    print(inv(C*inv(C)))

    print(np.matrix('4,-4,-3') * np.matrix('4;2;4'))

    D = np.random.rand(3, 3)
    E = np.random.rand(3, 3)
    v = np.random.rand(3, 1)
    print(D)
    print(E)
    print(v)
    DE = np.dot(D, E)
    print(DE)
    print(np.dot(DE, v))
    

if __name__ == "__main__":
    main()
