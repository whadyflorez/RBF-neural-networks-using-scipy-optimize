import torch
import ncg_optimizer as optim


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
   with torch.no_grad():
       gradbatch = torch.zeros_like(p)
       for i in ibatch:
           gradbatch+=(torch.dot(A[i,:],p)-z[i])*A[i,:]
   return gradbatch   

# Inicializar los parámetros del modelo
a,b=-1.0,1.0
p = torch.randn(nd)
p = (b-a)*p+a


# Definir el optimizador y closure para algunos
optimizer = optim.BASIC([p],method='CD',line_search='Strong_Wolfe', lr=1.0e-1)
def closure():
    optimizer.zero_grad()
    loss_val=loss(p,arr.tolist())
    p_grad=grad(p,ibatch)
    p.grad = p_grad 
    return loss_val

# Ciclo de entrenamiento
max_iter = 100000
batch_size =nd

arr=torch.randperm(nd)
ibatch=arr[0:batch_size]

for i in range(max_iter):
    optimizer.step(closure)
    arr=torch.randperm(nd)
    ibatch=arr[0:batch_size]
    loss_val=closure()
    print(f'iteration {i}, Loss: {loss_val}')


