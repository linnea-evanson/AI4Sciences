import numpy as np



def linear_regression(X, y):
    """
    Performs linear regression of X to get y

    :parameter:
    X: dependent variable
    y: independent variable
    :returns:
    prediction of y using X
    """

    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


if __name__ == '__main__': #this only runs if this file is called directly. But will not run if functions in this file
    # are called from elsewhere
    n, p = 10, 3
    X = np.random.randn(n, p)
    true_coefs = np.random.randn(p)
    y = np.dot(X, true_coefs) + 0.01 * np.random.randn(n)
    pred_coefs = linear_regression(X, y)
    print(pred_coefs)
    print(true_coefs)