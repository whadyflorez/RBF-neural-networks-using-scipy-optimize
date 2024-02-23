import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# 2d Rosenbrock function
def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def gradf(x):
    dy=torch.zeros(2)
    with torch.no_grad():
        dy[0]=-2*(1.0-x[0])-400*(x[1]-x[0]**2)*x[0]
        dy[1]=200*(x[1]-x[0]**2)
    return dy


# Gradient descent
x_gd = 10*torch.ones(2)
x_gd.requires_grad = True
gd = optim.SGD([x_gd], lr=1e-5)
history_gd = []
for i in range(100):
    gd.zero_grad()
    objective = f(x_gd)
    gf=gradf(x_gd)
    x_gd.grad=gf
    gd.step()
    history_gd.append(objective.item())


# L-BFGS
def closure():
    lbfgs.zero_grad()
    objective = f(x_lbfgs)
    gf=gradf(x_lbfgs)
    x_lbfgs.grad=gf
    return objective

x_lbfgs = 10*torch.ones(2)
x_lbfgs.requires_grad = True

lbfgs = optim.LBFGS([x_lbfgs],
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")
                    
history_lbfgs = []
for i in range(100):
    history_lbfgs.append(f(x_lbfgs).item())
    lbfgs.step(closure)


# Plotting
plt.semilogy(history_gd, label='GD')
plt.semilogy(history_lbfgs, label='L-BFGS')
plt.legend()
plt.show()