#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 19:47:49 2023

@author: whadymacbook2016

https://datascience.stackexchange.com/questions/44324/how-to-get-out-of-local-minimums-on-stochastic-gradient-descent
https://arxiv.org/abs/2208.00441
https://arxiv.org/abs/2105.14694
https://maths-people.anu.edu.au/~brent/pub/pub011.html
https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
https://www.mdpi.com/2076-3417/10/6/2036
https://arxiv.org/pdf/1905.06738.pdf
 
https://doi.org/10.3390/electronics9111809
https://link.springer.com/article/10.1007/s10898-022-01205-4
https://www.sciencedirect.com/science/article/pii/S0893608020303579
In this article, we describe a modern gradient-based RBFN implementation based on the same computational machinery that is used in modern deep learning. While gradient-based methods for training RBFN’s have been criticized (Chen, Cowan, & Grant, 1991) because of local optima, tailored training methods have been used in the past with good results 

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
from scipy.optimize import minimize
from matplotlib import cm
from matplotlib.ticker import LinearLocator

n=10
x=np.linspace(-1.,1.,n)
y=np.linspace(-1.,1.,n)
nd=n**2
xcoord=np.zeros((nd,2))
k=0
for i in range(n):
    for j in range(n):
        xcoord[k,:]=np.array([x[i],y[j]])
        k+=1

def rbf(x,xs):
    c=0.1
    r=np.linalg.norm(x-xs)
    y=np.sqrt(r**2+c**2)
    return y

def paraboloid(x):
    y=x[0]**2/2.+x[1]**2/3.
    return y

z=np.zeros(nd)
for i in range(nd):
    z[i]=paraboloid(xcoord[i])

A=np.zeros((nd,nd))

for i in range(nd):
    for j in range(nd):
        A[i,j]=rbf(xcoord[i],xcoord[j])

AT=np.transpose(A)



np.random.seed(1000)
a,b=-1.0,1.0
xk=(b-a)*np.random.rand(nd)+a
#xk=np.random.normal(size=nd)

def model(x,p):
    fiv=np.zeros(nd)
    for i in range(nd):
        fiv[i]=rbf(x,xcoord[i])
    y=np.dot(p,fiv)
    return y    
        

def loss(p):
    res=np.matmul(A,p)-z
    y=0.5*np.dot(res,res)
    return y

def grad(p):
   ATA=np.matmul(AT,A)
   ATB=np.matmul(AT,z)
   y=2.0*(np.matmul(ATA,p)-ATB)
   return y


# iter=1000000
# lr=1.0e-5
# for i in range(iter):
#     xk+=-lr*grad(xk)
#     print('loss',loss(xk))

def callback(xk):
    print('loss',loss(xk))
    return
    
sol=minimize(loss,xk,method='CG',jac=grad,callback=callback)
print('success',sol.success)
print(sol.message)
print('loss',loss(sol.x))
# # Plot the results
xplot=np.zeros((n,n))
yplot=np.zeros((n,n))
zplot=np.zeros((n,n))
zplotmodel=np.zeros((n,n))
for i in range(n):
  xplot[i,:]=x
  yplot[:,i]=y
for i in range(n):
  for j in range(n):
    xp=np.array([xplot[i,j],yplot[i,j]])
    zplot[i,j]=paraboloid(xp)
    zplotmodel[i,j]=model(xp,sol.x)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xplot, yplot, zplot)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xplot, yplot, zplotmodel)


