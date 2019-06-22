def compute_theta(init_theta, iteration_no, alpha):
    """Let it be that F(x) = x^2 so that F'(x) = 2x"""
    theta = init_theta
    for i in range(iteration_no):
        theta -= alpha*(2*theta)
        print(alpha, i, "-", theta)
    return theta


def main():
    compute_theta(3, 100, 0.1)
    compute_theta(3, 100, 0.22)


if __name__ == "__main__":
    main()


