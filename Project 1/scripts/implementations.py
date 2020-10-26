# École polytechnique fédérale de Lausanne, Switzerland
# CS-433 Machine Learning, project 1
# 
# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
# David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch
# Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch
#
# Regresssion and classification functions implementations.

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
    # w is the solution to the system (Xt*X + lambda*I)w = Xt*y
    aI = lamb * np.identity(tx.shape[1])
    w  = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    err= y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    y_ = (y+1)/2
    w = initial_w
    threshold = 1e-8
    losses = []
  
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y_, tx, w, gamma)
        loss = loss/len(y)
                  
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    y_ = (y+1)/2
    w = initial_w
    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y_, tx, w, gamma, lambda_)
        loss/= len(y)
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return w, loss

def sigmoid(t): 
    # Logistic function, equivalent to f(t) = e^t / (1 + e^t) 
    return 1.0 / (1.0 + np.exp(-t))


def calculate_loss(y, tx, w):
    # Logistic regression loss function
    # epsilon for having safe logarithm!
    epsilon = 1e-5  
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + epsilon)) + (1 - y).T.dot(np.log(1 - pred + epsilon)) 
    return np.squeeze(- loss)


def calculate_gradient(y, tx, w):
    # Gradient for logistic regression
    return tx.T.dot(sigmoid(tx.dot(w))-y)


def learning_by_gradient_descent(y, tx, w, gamma):
    # Gradient descet schema
    grad = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    w = w - gamma * grad
    return w, loss


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    # Do one step of gradient descent, using the penalized logistic regression.
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # Return the loss and updated w.
    w = w - gamma * gradient
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


