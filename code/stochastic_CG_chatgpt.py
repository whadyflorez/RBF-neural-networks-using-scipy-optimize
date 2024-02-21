import numpy as np
from scipy.optimize import line_search

""""
This code explores a simple problem of least squares with stochastic CG and BFGS
 (not L-BFGS) this latter should be explored
Inexact linesearch is also helpful 
"""

ndata=1000
xdata=np.array(np.linspace(0.0,10.0,ndata))
ydata=np.zeros(ndata)
for i in range(ndata):
    ydata[i]=xdata[i]**2-3.0*xdata[i]-1.0

def modelo(x,p):
    s=0.0
    for i in range(3):
        s+=p[i]*x**i
    return s    

def loss(p,*args):
    ibatch=args[0]
    ybatch=0.0
    for i in ibatch:
        ybatch+=(modelo(xdata[i],p)-ydata[i])**2
    return ybatch

def grad(p,*args):
    ibatch=args[0]
    gradbatch = np.zeros_like(p)
    for i in ibatch:
        for j in range(3):
            gradbatch[j]+=2.0*(modelo(xdata[i],p)-ydata[i])*xdata[i]**j
    return gradbatch

def nonlinear_conjugate_gradient(fun, grad, x0, max_iter, tol):
    x = x0
    
    lr0=1.0e-4
    vhist=10
    arr=np.arange(ndata)
#    np.random.shuffle(arr)
    batchsize=100
    ibatch=arr[0:batchsize]
    
    gradient = grad(x,ibatch)
    conjugate_direction = -gradient
    iteration = 0

    while np.linalg.norm(gradient) > tol and iteration < max_iter:
        # Perform line search
        linesrch = line_search(fun, grad, x, conjugate_direction, gradient,\
                                c1=0.0001, c2=0.9,amax=lr0,args=(ibatch,),maxiter=2)
        alpha=linesrch[0]
        if alpha==None: alpha=0.0
#        alpha=lr0
        # Update parameters
        x = x + alpha * conjugate_direction
         
        new_gradient = grad(x,ibatch)
        beta = np.dot((new_gradient-gradient), new_gradient) / np.dot(gradient, gradient)
        conjugate_direction = -new_gradient + beta * conjugate_direction
        gradient = new_gradient
 
#the same trick as in BFGS update the batch after a few iterations        
        if np.mod(iteration,vhist)==0: 
            np.random.shuffle(arr)
            ibatch=arr[0:batchsize]        


        iteration += 1
        print(f'{alpha:.3e} iter:{iteration} \
              loss batch:{fun(x,ibatch):.3e} loss:{fun(x,arr):.3e}')

    return x

def bfgs(fun, grad, x0, max_iter, tol):
    n = len(x0)
    H = np.eye(n)  # Initial approximation of the inverse Hessian
    x = x0

    lr0=1.0e-1
    arr=list(range(ndata))
#    np.random.shuffle(arr)
    batchsize=100
    ibatch=arr[0:batchsize]

    gradient = grad(x,ibatch)
    iteration = 0

    while np.linalg.norm(gradient) > tol and iteration < max_iter:
        conjugate_direction = -np.dot(H, gradient)

        # Perform line search
#        lsrch = line_search(fun, grad, x, conjugate_direction, gradient,\
#                        c1=1.0e-4, c2=0.9,amax=lr0,args=(ibatch,),maxiter=3)    
#        alpha=lsrch[0]
#        if alpha==None: alpha=lr0

        alpha=lr0

       # Update parameters
        x_next = x + alpha * conjugate_direction


        gradient_next = grad(x_next,ibatch)
        s = x_next - x
        y = gradient_next - gradient

        # BFGS update
        rho = 1.0 / np.dot(y, s)
        A = np.eye(n) - rho * np.outer(s, y)
        B = np.eye(n) - rho * np.outer(y, s)
        H = np.matmul(np.matmul(A, H), B) + rho * np.outer(s, s)

        x = x_next
        gradient = gradient_next

#inspired by the paper Stochastic L-BFGS: Improved Convergence Rates and Practical Acceleration Strategies        
        if np.mod(iteration,10)==0: 
            np.random.shuffle(arr)
            ibatch=arr[0:batchsize]        
        
        iteration += 1
           
        
        print(f'{alpha:.3e} iter:{iteration} \
              loss batch:{fun(x,ibatch):.3e} loss:{fun(x,arr):.3e}')


    return x


# Example usage
x0 = np.array([-1.5, 1.5, -1.5])
max_iter=1000
tol=1.0e-5
np.random.seed(777)
x_min = nonlinear_conjugate_gradient(loss, grad, x0,max_iter,tol)
#x_min = bfgs(loss, grad, x0,max_iter,tol)
print("Minimum found at:", x_min)
#print("Function value at minimum:", loss(x_min))



