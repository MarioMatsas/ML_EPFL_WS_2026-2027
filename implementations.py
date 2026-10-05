import numpy as np
import random

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform stochastic gradient descent."""
    for _ in range(max_iters):
        # Calculate the gradient at each step
        # Calculate the loss at each step
        gradient = -tx.T @ (y - tx @ initial_w) / len(y)

        # Update weights
        initial_w -= gamma * gradient
        
    loss = np.mean((y - tx @ initial_w) ** 2) / 2
    return initial_w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform stochastic gradient descent."""
    for _ in range(max_iters):
        # Get a random index for stochastic gradient descent
        i = random.randint(0, len(y) - 1)
        # Calculate the gradient at each step
        # Calculate the loss at each step
        gradient = -tx[i:i+1].T @ (y[i:i+1] - tx[i:i+1] @ initial_w)
        
        # Update weights
        initial_w -= gamma * gradient

    loss = np.mean((y - tx @ initial_w) ** 2) / 2
    return initial_w, loss

def least_squares(y, tx):
    """Perform least squares regression."""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Perform ridge regression."""
    w = np.linalg.solve(tx.T@tx + 2*len(y)*lambda_*np.eye(tx.shape[1]), tx.T@y)
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Perform logistic regression."""
    for _ in range(max_iters):
        # Calculate the gradient at each step
        # Calculate the loss at each step
        gradient = (tx.T@ (sigmoid(tx@initial_w) - y)) / len(y)

        # Update weights
        initial_w -= gamma * gradient

    loss = -np.mean(y*np.log(sigmoid(tx@initial_w)) + (1-y)*np.log(1-sigmoid(tx@initial_w)))
    return initial_w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Perform regularized logistic regression."""
    for _ in range(max_iters):
        # Calculate the gradient at each step
        # Calculate the loss at each step
        gradient = (tx.T@(sigmoid(tx@initial_w) - y)) / len(y) + 2*lambda_*initial_w

        # Update weights
        initial_w -= gamma * gradient

    loss = -np.mean(y*np.log(sigmoid(tx@initial_w)) + (1-y)*np.log(1-sigmoid(tx@initial_w)))
    return initial_w, loss


def sigmoid(z):
    return 1 / (1 + np.exp(-z))