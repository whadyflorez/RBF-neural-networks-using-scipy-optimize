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

def bfgs(fun, grad, x0, max_iter, tol):
    n = len(x0)
    H = np.eye(n)  # Initial approximation of the inverse Hessian
    x = x0
    gradient = grad(x)
    iteration = 0

    while np.linalg.norm(gradient) > tol and iteration < max_iter:
        conjugate_direction = -np.dot(H, gradient)

        # Perform line search
        lsrch= line_search(fun, grad, x, conjugate_direction, gradient,\
                           amax=1.0)
        alpha=lsrch[0]
        # Update parameters
        x_next = x + alpha * conjugate_direction
        gradient_next = grad(x_next)
        s = x_next - x
        y = gradient_next - gradient

        # BFGS update
        rho = 1.0 / np.dot(y, s)
        A = np.eye(n) - rho * np.outer(s, y)
        B = np.eye(n) - rho * np.outer(y, s)
        H = np.dot(np.dot(A, H), B) + rho * np.outer(s, s)

        x = x_next
        gradient = gradient_next
        iteration += 1

    return x

# Example usage
x0 = np.array([-1.5, 1.5, -1.5])
max_iter=1000
tol=1.0e-5
x_min = bfgs(rosenbrock, rosenbrock_gradient, x0,max_iter,tol)
print("Minimum found at:", x_min)
print("Function value at minimum:", rosenbrock(x_min))
