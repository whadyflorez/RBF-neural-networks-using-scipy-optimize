# -*- coding: utf-8 -*-
"""ejemplo_modelo_torch_chatgpt_con_funciones.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10HGBzMBfqk13ETIDuE1CjtpjeirsLOLh
"""



"""# Como definir un problema en pytorch sin usar classes"""

import torch
import torch.optim as optim

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

c=0.1
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

z=torch.zeros(nd)
A=torch.zeros(nd,nd)


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
        
AT=torch.t(A)
torch.manual_seed(1000)     

def model(x,p):
    fiv=torch.zeros(nd)
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
   gradbatch = torch.zeros_like(p)
   for i in ibatch:
       gradbatch+=(torch.dot(A[i,:],p)-z[i])*A[i,:]
   return gradbatch   
        

# # Función para definir el modelo
# def simple_model(x, weights, bias):
#     return torch.matmul(x, weights) + bias

# # Función de pérdida personalizada
# def custom_loss(y_pred, y_true):
#     return torch.mean((y_pred - y_true) ** 2)

# # Función para calcular los gradientes manualmente de forma analítica
# def calculate_gradients(x, y, weights, bias):
#     with torch.no_grad():
#         gradient = torch.mean(2 * (simple_model(x, weights, bias) - y) * x, dim=0)
#         bias_gradient = torch.mean(2 * (simple_model(x, weights, bias) - y))

#     return gradient, bias_gradient

# # Inicializar los parámetros del modelo
# weights = torch.randn(1, 1, requires_grad=True)
# bias = torch.randn(1, requires_grad=True)

# # Datos de ejemplo
# x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
# y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

# # Definir el optimizador
# #optimizer = optim.SGD([weights, bias], lr=0.1)
# optimizer = optim.SGD([weights, bias], lr=0.01, momentum=0.9)

# # Ciclo de entrenamiento
# epochs = 200
# batch_size = 3
# num_batches = len(x_train) // batch_size

# for epoch in range(epochs):
#     # Iterar sobre los lotes de datos
#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = start_idx + batch_size

#         # Obtener el lote actual
#         x_batch = x_train[start_idx:end_idx]
#         y_batch = y_train[start_idx:end_idx]

#         # Paso de adelante: Calcular la predicción y la pérdida
#         y_pred = simple_model(x_batch, weights, bias)
#         loss = custom_loss(y_pred, y_batch)

#         # Calcular los gradientes manualmente de forma analítica
#         weight_gradient, bias_gradient = calculate_gradients(x_batch, y_batch, weights, bias)

#         # Actualizar los parámetros del modelo utilizando el optimizador
#         optimizer.zero_grad()
#         weights.grad = weight_gradient.reshape_as(weights)
#         bias.grad = bias_gradient.reshape_as(bias)
#         optimizer.step()

#     # Imprimir la pérdida
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# # Evaluar el modelo entrenado
# with torch.no_grad():
#     print("Predicciones después del entrenamiento:")
#     y_pred = simple_model(x_train, weights, bias)
#     for i, (predicted, true) in enumerate(zip(y_pred, y_train)):
#         print(f'Entrada: {x_train[i].item()}, Predicción: {predicted.item()}, Valor real: {true.item()}')
