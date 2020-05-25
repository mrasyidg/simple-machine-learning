import numpy as np

"""
Functions included in this File are Hypothesis function,
Cost function and Stochastic Gradient Descent function.

This file of functions will be used in the implementation.
"""

# Hypothesis function : ℎ(𝑥) = 𝑤 ** 𝑇 * 𝑥
def h(x):
    global w

    # To test this function:
    # w, x = [1,2,3], [2,3,4]
    # assert is_similar(h(x),20)

    predictions= np.dot(x,w)
    
    return predictions


# Cost function : J(𝑤) = 1 / 2 * sum ( ( X.dot(w) - y ) ** 2) 
def cost_function(X,y,w):
    m = len(y)
    predictions = X.dot(w)
    square_error = (predictions - y)**2
    
    return 1/ (2 * m) * np.sum(square_error)


# Stochastic Gradient Descent function
def stochastic_gradient_descent(X,y,w,alpha,num_iters):
    m=len(y)
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(w)
        error = np.dot(X.transpose(),(predictions -y))
        descent = alpha * 1/m * error
        w -= descent
        J_history.append(cost_function(X,y,w))
    
    return w, J_history