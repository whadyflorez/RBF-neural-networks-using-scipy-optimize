# -*- coding: utf-8 -*-
"""ejemplo_modelo_torch_chatgpt_con_funciones.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10HGBzMBfqk13ETIDuE1CjtpjeirsLOLh
"""



"""# Como definir un problema en pytorch sin usar classes"""

import torch
import torch.optim as optim
import numpy as np
import pytorch_soo as soo
from pytorch_soo.line_search_spec import LineSearchSpec
import matplotlib.pyplot as plt
import sa


n = 10
nd = n**2
x = torch.linspace(-1., 1., n).double()
y = torch.linspace(-1., 1., n).double()
# Crear una cuadrícula de coordenadas
xx, yy = torch.meshgrid(x, y)
# Apilar las coordenadas en un tensor 2D
xcoord = torch.stack((xx.flatten(), yy.flatten()), dim=1).double()


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
    r=torch.norm(x-xs)
    y=torch.sqrt(r**2+c**2)
#    y = torch.exp(-r**2/c**2)
    return y

def laplace_rbf(x,xs):
    r=torch.norm(x-xs)
    y=(2.0*c**2+r**2)/rbf(x,xs)**3
#    exp_term = torch.exp(-r**2/c**2)
#    y = (4.0*r**2 / c**4 - 4.0/c**2) * exp_term
    return y    

def drbf(x,xs):
    r=torch.norm(x-xs).double()
    dfdx=(x[0]-xs[0])/rbf(x,xs)
    dfdy=(x[1]-xs[1])/rbf(x,xs) 
#    exp_term = torch.exp(-r**2 / c**2)
#    dfdx = -2.0*(x[0] - xs[0]) / c**2 * exp_term
#    dfdy = -2.0*(x[1] - xs[1]) / c**2 * exp_term
    return dfdx,dfdy

z=torch.zeros(nd).double()
A=torch.zeros(nd,nd).double()

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
        
#AT=torch.t(A)
torch.manual_seed(777)  

#lam=1.0
#A=A+torch.eye(nd,nd)*lam

def model(x,p):
    fiv=torch.zeros(nd).double()
    for i in range(nd):
        fiv[i]=rbf(x,xcoord[i])
    y=torch.dot(p,fiv)
    return y    
        
def loss(p,*args):
    ibatch=args[0]
    ybatch=0.0
    for i in ibatch:
        ybatch+=0.5*(torch.dot(A[i,:],p)-z[i])**2
    return ybatch

def grad(p,*args):
   ibatch=args[0]
   with torch.no_grad():
       gradbatch = torch.zeros_like(p)
       for i in ibatch:
           gradbatch+=(torch.dot(A[i,:],p)-z[i])*A[i,:]
   return gradbatch   

# Inicializar los parámetros del modelo
a,b=-1.0,1.0
p = torch.rand(nd).double()
p = (b-a)*p+a
#p=torch.linalg.solve(A,z)+10
#p.requires_grad=True


# Definir el optimizador
optimizer = optim.LBFGS([p],lr=0.01,history_size=100,max_iter=10,\
 max_eval=40,line_search_fn=None )
#optimizer = soo.HFCR_Newton([p],max_cr=50,max_newton=50)
#optimizer = optim.NAdam([p],lr=0.1)



# Función para realizar un paso de optimización con LBFGS
def closure():
    optimizer.zero_grad()
    loss_val=loss(p,arr.tolist())
    p_grad=grad(p,ibatch)
    p.grad = p_grad
    return loss_val

# Ciclo de entrenamiento
max_iter = 100
batch_size =100

arr=torch.randperm(nd)
ibatch=arr[0:batch_size]

for i in range(max_iter):
    optimizer.step(closure)
    if np.mod(i,2)==0:
        arr=torch.randperm(nd)
        ibatch=arr[0:batch_size]
    loss_val=closure()  
    print(f'iteration {i}, Loss: {loss_val.item()}')

#p=torch.linalg.solve(A,z) #para comparar

# # Plot the results
xplot=np.zeros((n,n))
yplot=np.zeros((n,n))
zplotmodel=np.zeros((n,n))
for i in range(n):
  xplot[i,:]=x.numpy()
  yplot[:,i]=y.numpy()
for i in range(n):
  for j in range(n):
    xp=torch.tensor([xplot[i,j],yplot[i,j]])
    zplotmodel[i,j]=model(xp,p)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xplot, yplot, zplotmodel)


