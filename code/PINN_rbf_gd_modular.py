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
from scipy.optimize import basinhopping
from matplotlib import cm
from matplotlib.ticker import LinearLocator

n=40
x=np.linspace(-1.,1.,n)
y=np.linspace(-1.,1.,n)
nd=n**2
xcoord=np.zeros((nd,2))
xx, yy = np.meshgrid(x, y)
xcoord = np.column_stack((xx.ravel(), yy.ravel()))


f1=[]
f2=[]
f3=[]
f4=[]
f5=[]
for i in range(nd):
    if xcoord[i,1]==-1.0: #bottom dirichlet
        f1.append(i)
    if xcoord[i,0]==-1.0 and xcoord[i,1]!=-1.0 and xcoord[i,1]!=1.0:
        f2.append(i) #left
    if xcoord[i,1]==1.0:
        f3.append(i)#top
    if xcoord[i,0]==1.0 and xcoord[i,1]!=-1.0 and xcoord[i,1]!=1.0:
        f4.append(i)#right dirichlet
    if xcoord[i,0]<1.0 and xcoord[i,1]>-1.0 and xcoord[i,1]<1.0 and xcoord[i,0]>-1.0:
        f5.append(i)#internal
       

c=0.1
def rbf(x,xs):
    r=np.linalg.norm(x-xs)
    y=np.sqrt(r**2+c**2)
    return y

def laplace_rbf(x,xs):
    r=np.linalg.norm(x-xs)
    y=(2.0*c**2+r**2)/rbf(x,xs)**3
    return y    

def drbf(x,xs):
    r=np.linalg.norm(x-xs)
    y=np.sqrt(r**2+c**2)
    dfdx=(x[0]-xs[0])/rbf(x,xs)
    dfdy=(x[1]-xs[1])/rbf(x,xs)   
    return dfdx,dfdy

def paraboloid(x):
    y=x[0]**2/2.+x[1]**2/3.
    return y

z=np.zeros(nd)


A=np.zeros((nd,nd))


for i in f1:
    for j in range(nd):  
        A[i,j]=rbf(xcoord[i],xcoord[j])
        z[i]=1.0
for i in f4:
    for j in range(nd):  
        A[i,j]=rbf(xcoord[i],xcoord[j])
        z[i]=1.0  
for i in f2:
    for j in range(nd):  
        A[i,j]=drbf(xcoord[i],xcoord[j])[0]
        z[i]=0.0          
for i in f3:
    for j in range(nd):  
        A[i,j]=drbf(xcoord[i],xcoord[j])[1]
        z[i]=0.0    
for i in f5:
    for j in range(nd):  
        A[i,j]=laplace_rbf(xcoord[i],xcoord[j])
        z[i]=-1000.0          
        
    
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
   y=(np.matmul(ATA,p)-ATB)
   return y


# iter=1000000
# lr=1.0e-5
# for i in range(iter):
#     xk+=-lr*grad(xk)
#     print('loss',loss(xk))

def callback(xk):
    print('loss',loss(xk))
    return

def callbackbasin(x,f,accept):
    print('loss',f,'accept:',accept)
    return

kwargsbasin={'tol':1.0e-6}

#options={'maxiter':200000,'maxfun':600000}    
#sol=minimize(loss,xk,method='BFGS',jac=grad,callback=callback,options=options)
sol=basinhopping(loss, xk,niter=5,minimizer_kwargs=None,callback=None,disp=True)
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
    zplotmodel[i,j]=model(xp,sol.x)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xplot, yplot, zplotmodel)


