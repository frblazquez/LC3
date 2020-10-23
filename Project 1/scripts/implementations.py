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

def sigmoid(t): 
    # Logistic function, equivalent to f(t) = e^t / (1 + e^t) 
    return 1.0 / (1.0 + np.exp(-t)) # Reviewed


def calculate_loss(y, tx, w):
    """
    compute the negative log likelihood for the logistic regression.
    INPUTS: y #### Do not put inputs if they don't need to be described!
            X
            w
    OUTPUTS: negative log likelihood
    """
    # When you tested this, where does y come from??
    epsilon = 1e-5  
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + epsilon)) + (1 - y).T.dot(np.log(1 - pred + epsilon)) # Reviewed
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient, in a given point w, of the loss for the logistic regression.
    INPUTS: y
            X
            w
    OUTPUTS: the gradient vector ### Of course calculate_gradient computes the gradient! Not necessary!
    """
    return tx.T.dot(sigmoid(tx.dot(w))-y) # Reviewed


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Return the loss and the updated w.
    INPUTS: y
            X
            w
            gamma := the step size
    OUTPUTS:w
            loss
    """
    grad = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    w = w - gamma * grad
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression via gradient descent
    INPUTS:
            y
            X
            initial_w := the initialization of w for the algorithm
            max_iters := max number of iterations
            gamma := step size gamma
            
    
    OUTPUTS:
            w*, loss
    """
    y_ = (y-1)/2 # IMPORTANT! Shouldn't it be +1 instead of -1??
    w = initial_w
    
    threshold = 1e-8

    # init parameters
    losses = []
  
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y_, tx, w, gamma)
        
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
                   
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression via gradient descent
    INPUTS:
            y
            X
            max_iter := max number of iterations
            gamma := step size gamma
            lambda := penalizing factor lambda
            threshold := threshold for the update of the loss
            
    
    OUTPUTS:
            w*, loss
    """
    y_ = (y-1)/2 # IMPORTANT! Shouldn't it be +1 instead of -1??
    threshold = 1e-8

    # init parameters
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y_, tx, w, gamma, lambda_)
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return w, loss
