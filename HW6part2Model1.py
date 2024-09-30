'''
Created on Saturday April 6 01:41:26 2024

@author: Frank A. Segui Gonzalez, frank.segui1@upr.edu
@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

INGE 4035
Asignación 6

Parte 2: MODELO #3
Modelo 1 de 'machine learning' con una implementacion de 'gradient descent logistic regression' que 
es capaz de predecir los resultados de una prueba no invasiva basandose en los resultados de las 
pruebas de bajo costo (archivo de txt)
Genera una grafica de la data, grafica de 'cost function' junto a las w , el modelo
donde se visualiza el decision boundry y otro con los falsos positivos y los falsos negativos marcados

Última actualizacion 4/9/2024 
'''
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#load and plot the data

data = np.loadtxt('non_inavive_tests_results.txt', delimiter=',', comments='#') 
x1 = data[:,0] # exam 1
x2 = data[:,1] # exam 2
y = data[:,2]  # decision

# Calculates probability of passing
test1 = int(input("Score of test 1: "))
test2 = int(input("Score of test 2: "))

test1 = (test1-np.mean(x1))/np.std(x1)

test2 = (test2-np.mean(x2))/np.std(x2)

plt.figure(figsize = (20,15))
plt.plot(x1[y==1],x2[y==1],'+',color = 'black', mfc='powderblue',label='pass')
plt.plot(x1[y==0],x2[y==0],'o',color = 'dimgray',label='failed')
plt.xlabel('exam 1', fontsize = 18, weight = "bold"); plt.ylabel('exam 2', fontsize = 18, weight = "bold")
plt.axis('square')
plt.grid(color='silver')
plt.legend(fontsize=25)

# z normalization 
x1 = (x1-np.mean(x1))/np.std(x1) 

x2 = (x2-np.mean(x2))/np.std(x2)

# logistic regression using gradient descent 
# the functon is of the form: f=1/(1+exp(-z)), z=w1x1+w2x2+b
# use a log-loss function for the cost
    
alpha     = 1  # learning rate
niter     = 1000000 # max number of iterations
epsilon   = 1e-5   # acceptable tolerance (relative change of J)
m = np.size(y)       # number of training examples

w1v  = np.zeros(niter,dtype=float)
w2v  = np.zeros(niter,dtype=float)
w3v  = np.zeros(niter,dtype=float)
bv   = np.zeros(niter,dtype=float)
Jv   = np.zeros(niter,dtype=float)    # cost function (log-loss)

x3 = x1*x2

w1,w2,w3,b = 0,0,0,0                      # initial values for w1,w2, w3 and b
w1v[0],w2v[0],w3v[0],bv[0] = w1, w2, w3, b

x1sqr = x1**2 

x2sqr = x2**2

z       = w1*x1 + w2*x2 + w3*x3 + b     # Equation of model

ys      = 1/(1+np.exp(-z))            # sigmoid (obtain probabilities) 
Jv[0] = -1/m*np.sum( y*np.log(ys) + (1-y)*(np.log(1-ys)) )    # cost function

for k in range(1,niter):
    w1     = w1 - alpha/m*np.sum((ys-y)*(x1))    # update w1
    w2     = w2 - alpha/m*np.sum((ys-y)*(x2))    # update w2
    w3     = w3 - alpha/m*np.sum((ys-y)*(x3))    # update w3
    b      = b  - alpha/m*np.sum((ys-y)*(1))     # update b
    z       = w1*x1 + w2*x2 + w3*x3 +b
    ys     = 1/(1+np.exp(-z)) 
    w1v[k], w2v[k] , w3v[k], bv[k]  = w1, w2, w3, b
    Jv[k] = -1/m*np.sum( y*np.log(ys) + (1-y)*(np.log(1-ys)) ) # update cost function
    tolc = (Jv[k-1]-Jv[k])/Jv[k]
    
    if tolc<0:
        print('error is increasing at %i iterations' %k)
        w1v = w1v[:k]
        w2v = w2v[:k]
        w3v = w3[:k]
        bv = bv[:k]
        Jv = Jv[:k]
        break
    elif tolc<epsilon:
        print('target tolerance reached at %i iterations' %k)
        w1v = w1v[:k+1]
        w2v = w2v[:k+1]
        w3v = w3v[:k+1]
        bv = bv[:k+1]
        Jv = Jv[:k+1]
        break
else:
    print('target tolerance was not reached within max. # of iterations')

# the desicion boundary (line where ym=0)
x = np.linspace(np.min(x1),np.max(x1))
v = np.linspace(np.min(x2), np.max(x2))

X, Y = np.meshgrid(x,v)

modelo = lambda x,y: w1*(x) +w2*(y) + w3*(x*y) +b

myFun = modelo(X,Y)


iterv = np.linspace(1,len(Jv),len(Jv))
plt.figure(figsize = (20,15))
plt.subplot(151)
plt.plot(iterv,Jv)
plt.xlabel('iteration #',  fontsize = 18, weight = "bold"); plt.ylabel('J',  fontsize = 18, weight = "bold")
plt.subplot(152)
plt.plot(w1v, Jv)
plt.xlabel('w1',  fontsize = 18, weight = "bold"); plt.ylabel('cost',  fontsize = 18, weight = "bold")
plt.subplot(153)
plt.plot(w2v, Jv)
plt.xlabel('w2',  fontsize = 18, weight = "bold"); plt.ylabel('cost',  fontsize = 18, weight = "bold")
plt.subplot(154)
plt.plot(w3v, Jv)
plt.xlabel('w3',  fontsize = 18, weight = "bold"); plt.ylabel('cost',  fontsize = 18, weight = "bold")
plt.subplot(155)
plt.plot(bv, Jv)
plt.xlabel('b',  fontsize = 18, weight = "bold"); plt.ylabel('cost',  fontsize = 18, weight = "bold")
plt.tight_layout()

plt.figure(figsize = (20,15))
plt.plot(x1[y==1],x2[y==1],'+',color = 'black',label='pass')
plt.plot(x1[y==0],x2[y==0],'o',color = 'dimgray',label='fail')
CS=plt.contour(x,v,myFun,0, colors = 'blue',linestyles='solid',linewidths = 4 ) 
plt.xlabel('exam 1', fontsize = 18, weight = "bold"); plt.ylabel('exam 2', fontsize = 18, weight = "bold")
plt.axis('square')
plt.grid(color='silver')
plt.legend(fontsize=25)
plt.tight_layout()

PF = np.zeros(np.size(x1))

for j in range (np.size(x1)):
    myTest = modelo(x1[j],x2[j])
    Percent  = (1/(1+np.exp(-myTest)))*100
    if(Percent >= 50):
        PF[j] = 1       # Saves what the model predicts as pass
    else:
        PF[j] = 0       # Saves what the model predicts as fail
        
count = 0

failed = np.zeros(np.size(x1), dtype = int)
failedP = np.zeros(np.size(x1), dtype = int)
failedO = np.zeros(np.size(x1), dtype = int)

for l in range(np.size(x1)):
    if(y[l]==PF[l]):
        count = count + 1       # Counts how much data the model predicted correctly
        # Marks False all the other points that are not predicted correctly
        failed[l] = False
        failedP[l] = False
        failedO[l] = False;
    else:
        failed[l] = l           # Saves indexes of data that the model predicted incorrectly
        if(y[l] == 1):
            failedP[l] = l      # Saves false positives 
        else:
            # Saves false negatives
            failedO[l] = l
    
            
#Erases all extra zeros
failed = failed[failed != False]
failedP = failedP[failedP != False]
failedO = failedO[failedO != False]

print(f"\n This model got {(count/np.size(x1))*100}% accuracy")

plt.figure(figsize = (20,15))
plt.plot(x1[y==1],x2[y==1],'+',color = 'black',label='pass')
plt.plot(x1[y==0],x2[y==0],'o',color = 'dimgray',label='fail')
plt.plot(x1[failedO],x2[failedO],'o',color = 'red',label='False Positives')
plt.plot(x1[failedP],x2[failedP],'+',color = 'red',label='False Negatives')
CS=plt.contour(x,v,myFun,0, colors = 'blue',linestyles='solid',linewidths = 4 ) 
plt.xlabel('exam 1', fontsize = 18, weight = "bold"); plt.ylabel('exam 2', fontsize = 18, weight = "bold")
plt.axis('square')
plt.grid(color='silver')
plt.legend(fontsize=25)
plt.tight_layout()


zr = w1*test1 + w2*test2 + w3*(test1*test2) +b

probs = (1/(1+np.exp(-zr)))*100

print(f"\n The test has a {probs}% of passing")




