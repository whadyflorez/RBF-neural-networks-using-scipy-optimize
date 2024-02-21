import numpy as np
from scipy.optimize import line_search

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    grad[1:-1] = 200 * (x[1:-1] - x[:-2]**2) - 400 * x[1:-1] * (x[2:] - x[1:-1]**2) - 2 * (1 - x[1:-1])
    return grad

def nonlinear_conjugate_gradient(fun, grad, x0, max_iter, tol):
    x = x0
    gradient = grad(x)
    conjugate_direction = -gradient
    iteration = 0

    while np.linalg.norm(gradient) > tol and iteration < max_iter:
        # Perform line search
        linesrch = line_search(fun, grad, x, conjugate_direction, gradient)
        alpha=linesrch[0]
        # Update parameters
        x = x + alpha * conjugate_direction
        new_gradient = grad(x)
        beta = np.dot(new_gradient, new_gradient) / np.dot(gradient, gradient)
        conjugate_direction = -new_gradient + beta * conjugate_direction
        gradient = new_gradient

        iteration += 1
        print('iter',iteration,'loss funtion',fun(x))

    return x

# Example usage
x0 = np.array([-1.5, 1.5, -1.5])
max_iter=1000
tol=1.0e-5
x_min = nonlinear_conjugate_gradient(rosenbrock, rosenbrock_gradient, x0,max_iter,tol)
print("Minimum found at:", x_min)
print("Function value at minimum:", rosenbrock(x_min))

