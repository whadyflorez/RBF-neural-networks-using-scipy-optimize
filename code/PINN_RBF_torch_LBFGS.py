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


n=10
x=torch.linspace(-1.,1.,n)
y=torch.linspace(-1.,1.,n)
nd=n**2
xcoord=torch.zeros(nd,2)
xx, yy = torch.meshgrid(x, y)
xcoord = torch.column_stack((xx.ravel(), yy.ravel()))

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

c=0.01
def rbf(x,xs):
    r=torch.norm(x-xs)
    y=torch.sqrt(r**2+c**2)
    return y

def laplace_rbf(x,xs):
    r=torch.norm(x-xs)
    y=(2.0*c**2+r**2)/rbf(x,xs)**3
    return y    

def drbf(x,xs):
    dfdx=(x[0]-xs[0])/rbf(x,xs)
    dfdy=(x[1]-xs[1])/rbf(x,xs)   
    return dfdx,dfdy

z=torch.zeros(nd+1)
A=torch.zeros(nd+1,nd+1)


for i in f1:
    for j in range(nd):  
        A[i,j]=rbf(xcoord[i],xcoord[j])
    A[i,nd]=1.0
    z[i]=1.0
for i in f4:
    for j in range(nd):  
        A[i,j]=rbf(xcoord[i],xcoord[j])
    A[i,nd]=1.0
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
A[nd,0:nd]=1.0    
        
#AT=torch.t(A)
torch.manual_seed(1000)     

def model(x,p):
    fiv=torch.zeros(nd+1)
    for i in range(nd):
        fiv[i]=rbf(x,xcoord[i])
    fiv[nd]=1.0    
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
a,b=0.0,1.0
p = torch.randn(nd+1)
p = (b-a)*p+a
#p.requires_grad=True

# Definir el optimizador
optimizer = optim.LBFGS([p], lr=0.01,history_size=100,max_iter=20,max_eval=200,\
                        line_search_fn='strong_wolfe' )
#optimizer = soo.HFCR_Newton([p],max_cr=50,max_newton=10)
#optimizer = optim.RMSprop([p],lr=0.01)


    # Función para realizar un paso de optimización con LBFGS
def closure():
    optimizer.zero_grad()
    loss_val=loss(p,arr.tolist())
    p_grad=grad(p,ibatch)
    p.grad = p_grad 
    return loss_val

# Ciclo de entrenamiento
max_iter = 10000
batch_size =int((nd+1))

arr=torch.randperm(nd+1)
ibatch=arr[0:batch_size]

for i in range(max_iter):
    optimizer.step(closure)
    arr=torch.randperm(nd+1)
    ibatch=arr[0:batch_size]
    loss_val=closure()    
    print(f'iteration {i}, Loss: {loss_val.item()}')
 

