import numpy as np

1 + 3
a = 7
b = a + 1
print("b = ", b)

#Vectores
v = np.array([1,2,3,-1])
w = np.array([2,3,0,5])
print("v + w = ", v + w)
print("2 * v = ", 2 * v)
print("v ** 2 = ", v ** 2)

#Matrices (ejecutar los comandos uno a uno para ver los resultados)
A = np.array([[1,2,3,4], [0,1,2,3,4], [2,3,4,5,6], [0,0,1,2,3], [0,0,0,0,1]])
print(A)
A[0:2, 3:5]
A[:2, 3:]

# %%

import matplotlib.pyplot as plt #libreria para graficar

# ...
# Aca, crear la matriz y resolver el sistema para calcular a,b y c
a = -3/2
b = 11/2
c = -3
# ...
 
xx = np.array([1,2,3])
yy = np.array([1,2,0]) 
x = np.linspace(0,4,100)  #genera 100 puntow equiespaciado entre 0 y 4
f = lambda t: a*t**2+b*t+c #esto genera una funcion f de t
plt.plot(xx,yy,'*')
plt.plot(x,f(x))
plt.show()
 
