# -*- coding: utf-8 -*-
"""
Labo 6 -- Ejercicios
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %%



#1) Hacer un programa que reciba una matriz A ∈ Rn×n y un entero positivo k y que aplique k iteraciones del método de la potencia con un vector aleatorio inicial v ∈ Rn. El programa debe devolver un vector a ∈ Rk, donde ai sea la aproximación al autovalor obtenida en el paso i.

# Recomendaciones:
#  Recordar normalizar el vector en cada paso.
#  Pueden comparar con np.linalg.eigvals() para verificar resultados.

def ejercicio_1(A, k):
    """
    Calcula el autovector al autovalor asociado de valor máximo

    Devuelve (a, v) con a autovalor, y v autovector de A

    Arguments:
    ----------

    A: np.array
        Matriz de la cual quiero calcular el autovector y autovalor

    niter: int (> 0)
        Cantidad de iteraciones

    eps: Epsilon
        Tolerancia utilizada en el criterio de parada
    """
    v = np.random.rand(A.shape[0]) #A es cuadrada, [0] = [1]
    normaV = np.linalg.norm(v)
    v = v / normaV
    res = []
    
    for i in range(k):
        v = (A @ v) / np.linalg.norm(A @ v)
        a = (v.T @ (A @ v)) / (v.T @ v)
        res.append(a)
    
    res = np.array(res)
    
    return res

#Verificacion
A = np.array([[2,0],[0,1]])
k = 4

res1= ejercicio_1(A, k)
res2 = np.linalg.eigvals(A)


# %%
def ejercicio_2():
    A = np.random.randint(1, 100, size=(100, 100))
    k = 100
    
    resultados = ejercicio_1(A, k)
    
    
    
    #Genera el gráfico de las aproximacioes obtenidas en funcion del numero de iteraciones
    fig, ax = plt.subplots()
    
    #Grafica la regionEste
    ax.plot(range(1,101), resultados, 
            marker = '.',        #Tipo de punto (punto, círculo, estrella, etc)
            linestyle = '-',     #Tipo de línea (sólida, punteada, etc)
            linewidth = 0.5,     #Ancho de línea
            label = 'Autovalores') #Etiqueta que va a mostrarse en la leyenda
    
    #Agrega título, etiquetas a los ejes y limita el rango de valores de los ejes
    ax.set_title('Aproximaciones obtenidas en funcion del numero de iteraciones')
    ax.set_xlabel('Cantidad de iteraciones')
    ax.set_ylabel('Valor del autovalor')
    
    
    #Muestra la leyenda
    ax.legend()
    plt.show()
    
    return resultados

e2 = ejercicio_2()

#El metodo converge rapidamente?
#Luego de haber visualizado el grafico realizado, se puede concluir que el metodo converge rapidamente, esto es, se consigue de forma rapida el autovalor de modulo maximo.
#Mas aun, ya se puede ver que ya en las primeras iteraciones los valores ya se acercan al autovalor que se busca.

# %%

def ejercicio_3():
    C = np.random.randint(1, 100, size=(100, 100))
    A = 1/2 * (C + C.T) #Nos aseguramos una matriz con todos sus autovalores reales
    B = A + 500*np.eye(100, 100)
    k = 100
     
    resultados = ejercicio_1(B, k)
     
    lambdaMax = max(abs(np.linalg.eigvals(B)))
     
    errores = []
    
    for valor in resultados:
        errores.append(abs(lambdaMax - valor))
        
    errores = np.array(errores)
    
    #Genera el gráfico de las aproximacioes obtenidas en funcion del numero de iteraciones
    fig, ax = plt.subplots()
    
    #Grafica los errores
    ax.plot(range(1,101), errores, 
            marker = '.',        #Tipo de punto (punto, círculo, estrella, etc)
            linestyle = '-',     #Tipo de línea (sólida, punteada, etc)
            linewidth = 0.5,     #Ancho de línea
            label = 'Error') #Etiqueta que va a mostrarse en la leyenda
    
    #Agrega título, etiquetas a los ejes y limita el rango de valores de los ejes
    ax.set_title('Errores absolutos en funcion del numero de iteracion')
    ax.set_xlabel('Cantidad de iteraciones')
    ax.set_ylabel('Valor del error')
    
    
    #Muestra la leyenda
    ax.legend()
    plt.show()
    
    #Genera el gráfico de las aproximacioes obtenidas en funcion del numero de iteraciones
    fig, ax = plt.subplots()
    
    #Grafica los errores en base logaritmica
    ax.plot(range(1,101), np.log(errores), 
            marker = '.',        #Tipo de punto (punto, círculo, estrella, etc)
            linestyle = '-',     #Tipo de línea (sólida, punteada, etc)
            linewidth = 0.5,     #Ancho de línea
            label = 'Error Log') #Etiqueta que va a mostrarse en la leyenda
    
    #Agrega título, etiquetas a los ejes y limita el rango de valores de los ejes
    ax.set_title('Errores absolutos en funcion del numero de iteracion')
    ax.set_xlabel('Cantidad de iteraciones')
    ax.set_ylabel('Valor del error')
    
    
    #Muestra la leyenda
    ax.legend()
    plt.show()
    
    #Buscamos el autovalor mas cercano a el autovalor maximo
    lambda2 = sorted(abs(np.linalg.eigvals(B)))[1]
    
    valores_funcion = []
    
    for i in range(1,k+1):
        f_i= 2*np.log(lambda2/lambdaMax)*i + np.log(errores[0])
        valores_funcion.append(f_i)
    
    valores_funcion = np.array(valores_funcion)
    
    #Genera el gráfico de las aproximacioes obtenidas en funcion del numero de iteraciones
    fig, ax = plt.subplots()
    
    #Grafica los errores en base logaritmica
    ax.plot(range(1,101), np.log(errores), 
            marker = '.',        #Tipo de punto (punto, círculo, estrella, etc)
            linestyle = '-',     #Tipo de línea (sólida, punteada, etc)
            linewidth = 0.5,     #Ancho de línea
            label = 'Error Log') #Etiqueta que va a mostrarse en la leyenda
    
    #Grafica los errores en base logaritmica
    ax.plot(range(1,101), valores_funcion, 
            marker = '.',        #Tipo de punto (punto, círculo, estrella, etc)
            linestyle = '-',     #Tipo de línea (sólida, punteada, etc)
            linewidth = 0.5,     #Ancho de línea
            label = 'Funcion') #Etiqueta que va a mostrarse en la leyenda
    #Muestra la leyenda
    ax.legend()
    plt.show()
    
ejercicio_3()   

 


# %%



def power_iteration(A, k):
    """
    Calcula el autovector al autovalor asociado de valor máximo

    Devuelve (a, v) con a autovalor, y v autovector de A

    Arguments:
    ----------

    A: np.array
        Matriz de la cual quiero calcular el autovector y autovalor

    k: int (> 0)
        Cantidad de iteraciones
        
    """
    v = np.random.rand(A.shape[0]) #Vector inicial aleatorio. A es cuadrada, [0] = [1]
    v = v / np.linalg.norm(v)  # Normalización inicial
    
    a = []
    for i in range(k):
        w = A @ v
        v = w / np.linalg.norm(w)
        ai = (v.T @ (A @ v)) / (v.T @ v)
        a.append(ai)
    
    a = np.array(a)

    return a, v

# Prueba con una matriz
A = np.array([[4,1], [2,3]])
k = 10

autovalores_aprox, autovector_final = power_iteration(A, k)
print("Autovalores aproximados en cada iteración:", autovalores_aprox)
print("Autovector final aproximado:", autovector_final)

# Comparación con valores exactos
print("Autovalores exactos:", np.linalg.eigvals(A))

# %%



# 2) (a) Tomar una matriz A ∈ R100×100 de coordenadas aleatorias y utilizar el programa para realizar 100 iteraciones del método.

import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, k):
    """
    Método de la potencia para aproximar el autovalor dominante.
    
    Devuelve:
    - autovalores_aprox: vector con las aproximaciones de cada iteración.
    - autovector_final: autovector obtenido al final del proceso.
    
    Parámetros:
    ----------
    A : np.array
        Matriz cuadrada de la cual queremos estimar el mayor autovalor.
    k : int
        Número de iteraciones del método de la potencia.
    """
    n = A.shape[0]
    v = np.random.rand(n)  # Vector inicial aleatorio
    v = v / np.linalg.norm(v)  # Normalización inicial
    
    autovalores_aprox = np.zeros(k)  # Almacena la aproximación en cada iteración
    
    for i in range(k):
        w = A @ v  # Multiplicación con la matriz
        v = w / np.linalg.norm(w)  # Normalización
        
        autovalores_aprox[i] = np.dot(v, A @ v)  # Aproximación del autovalor dominante
    
    return autovalores_aprox, v

# (a) Generar una matriz aleatoria A ∈ R100×100 y aplicar el método
np.random.seed(42)  # Fijamos una semilla para reproducibilidad
A = np.random.rand(100, 100)  # Matriz aleatoria de valores en [0,1]
k = 100  # Número de iteraciones

autovalores_aprox, autovector_final = power_iteration(A, k)

# Comparación con el autovalor de mayor módulo
autovalores_exactos = np.linalg.eigvals(A)
lambda_max = max(abs(autovalores_exactos))  # Autovalor de mayor módulo
print(f"Autovalor de mayor módulo exacto: {lambda_max:.4f}")


#  (b) Graficar las aproximaciones obtendidas en función del número de iteraciones. ¿Considera que el método converge rápidamente?
 
# (b) Graficar la evolución de las aproximaciones
plt.figure(figsize=(10, 5))
plt.plot(range(1, k+1), autovalores_aprox, marker="o", linestyle="-", color="b", markersize=4, label="Aproximaciones del autovalor")
plt.xlabel("Número de iteraciones")
plt.ylabel("Autovalor dominante aproximado")
plt.title("Convergencia del método de la potencia")
plt.legend()
plt.grid(True)
plt.show()

#  Recomendaciones:
# En este caso comparar con un vector (np.linalg.eigvals()) no es útil. Queremos comparar sólo con el autovalor de módulo máximo de ese vector ¿Qué norma vectorial podemos usar?

# Analíticamente, puede verse que la velocidad de convergencia está dada por la relación entre el segundo autovalor de mayor módulo y el primer autovalor de mayor módulo.
#  Más precisamente, el error en cada paso se multiplica aproximadamente por (λ2/λ1)^2.

#El método de la potencia converge rápidamente cuando el cociente 𝜆2/𝜆1 es pequeño. Para verificar esto:
    #1) Ordenamos los valores en orden de magnitud:
autovalores_sorted = sorted(abs(autovalores_exactos), reverse=True)
lambda_1 = autovalores_sorted[0]
lambda_2 = autovalores_sorted[1]

#2)Calculamos la velocidad de convergencia:
factor_convergencia = (lambda_2 / lambda_1)**2
print(f"Factor de convergencia teórico: {factor_convergencia:.4f}")

#Si (𝜆2/𝜆1)^2 es pequeño (≪1), el método converge rápidamente.
# %%
 

# 3)
# (a) Tomar una matriz C ∈ R100×100 de coordenadas aleatorias y considerar la matriz simétrica A = 1/2 *(C +C^t) (de esta forma nos aseguramos una matriz con todos sus autovalores reales).
#  Definir B = A+500I y aplicar 100 pasos del método de la potencia a B.

import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, k):
    """Método de la potencia para aproximar el autovalor dominante."""
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)  # Normalización inicial

    autovalores_aprox = np.zeros(k)  # Vector con aproximaciones
    for i in range(k):
        w = A @ v
        v = w / np.linalg.norm(w)
        autovalores_aprox[i] = np.dot(v, A @ v)  # Aproximación del autovalor dominante

    return autovalores_aprox, v

# Generar matriz aleatoria C y construir A simétrica
np.random.seed(42)
C = np.random.rand(100, 100)
A = 0.5 * (C + C.T)  # Matriz simétrica

# Definir B = A + 500I
B = A + 500 * np.eye(100)

# Aplicar 100 iteraciones del método de la potencia
k = 100
autovalores_aprox, autovector_final = power_iteration(B, k)




#  (b) Llamamos λmax al autovalor de mayor módulo de B y definimos el vector de errores e ∈ R100:
#  ei = |λmax −ai|.
#  Graficar los errores en función del número de iteración ¿Puede decir que función es?

# Calcular autovalor de mayor módulo
autovalores_exactos = np.linalg.eigh(B)[0]  # np.linalg.eigh() es más preciso para matrices simétricas
lambda_max = max(abs(autovalores_exactos))  # Tomamos el mayor autovalor

# Definir vector de errores
errores = np.abs(lambda_max - autovalores_aprox)

# Graficar el error en función de la iteración
plt.figure(figsize=(10, 5))
plt.plot(range(1, k+1), errores, marker="o", linestyle="-", color="b", markersize=4, label="Error |λmax - ai|")
plt.xlabel("Número de iteraciones")
plt.ylabel("Error")
plt.title("Evolución del error en el método de la potencia")
plt.legend()
plt.grid(True)
plt.show()


#  (c) Graficar log(ei) y volver a pensar el ítem (b).
#  Sabiendo que el factor por el que se multiplica el error es aproximadamente (λ2/λ1)^2 la pendiente de la recta obtenida deber´ ıa ser aproximadamente 2log (λ2/λ1).
# Para comparar los valores obtenidos experimentalmente, en el mismo gráfico representar la función
#  y(x) = 2log (λ2/λ1)x + log(e0).

# Obtener segundo mayor autovalor
autovalores_sorted = sorted(abs(autovalores_exactos), reverse=True)
lambda_1 = autovalores_sorted[0]
lambda_2 = autovalores_sorted[1]

# Calcular pendiente teórica
pendiente_teorica = 2 * np.log(lambda_2 / lambda_1)

# Definir función teórica y(x) = 2 log(λ2/λ1) * x + log(e0)
y_teorico = pendiente_teorica * np.arange(k) + np.log(errores[0])

# Graficar log(ei)
plt.figure(figsize=(10, 5))
plt.plot(range(1, k+1), np.log(errores), marker="o", linestyle="-", color="r", markersize=4, label="log(error)")
plt.plot(range(1, k+1), y_teorico, linestyle="--", color="black", label="Teoría: $2\log(λ_2/λ_1)x + \log(e_0)$")
plt.xlabel("Número de iteraciones")
plt.ylabel("log(Error)")
plt.title("Comparación entre error logarítmico y función teórica")
plt.legend()
plt.grid(True)
plt.show()

# Mostrar valores teóricos
print(f"Pendiente teórica esperada: {pendiente_teorica:.4f}")



#  Recomendaciones:
#  Se puede usar la función sorted() para ordenar los coeficientes de un vector. En este caso que
#  trabajamos con una matriz simétrica también puede ser útil la función np.linalg.eigh().
