# Functions implementations for Machine Learning course, project 1

import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        # w error and gradient computation
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / len(y)
        # w update
        w = w - gamma * grad
    # Compute the total loss for the final w
    err = y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        idx = np.random.choice(range(len(y)))
        yn = y[idx]
        xn = tx[idx]
        # w error and stochastic gradient
        err  = yn - xn.dot(w)
        grad = -xn.T.dot(err)
        # w update 
        w = w - gamma * grad
    # Compute the total loss for the final w
    err = y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    return w, loss


def least_squares(y, tx):
    # w is the solution to the system (Xt*X)w = Xt*y 
    w    = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    err  = y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    return w, loss


def ridge_regression(y, tx, lamb):
    aI = lamb * np.identity(tx.shape[1])
    w  = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    err= y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError



