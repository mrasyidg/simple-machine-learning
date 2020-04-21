import numpy as np

def cost_fun(X,y,w): # w is theta
    m = len(y)
    predictions = X.dot(w)
    square_error = (predictions - y) ** 2
    
    return 1/( 2 * m) * np.sum(square_error)

def gradient_descent(X, y, w, alpha, num_iters): # w is theta
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        predictions = X.dot(w)
        error = np.dot( X.transpose(),(predictions - y))
        descent = alpha * 1 / m * error
        theta -= descent
        J_history.append(cost_fun(X,y,w))
    
    return theta, J_history

# TODO: finish the program file