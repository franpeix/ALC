#Guia practica 2: Aritmetica de punto flotante. Numero de condicion

#Aritmetica de punto flotante

#Ejercicio 7: algunos experimentos

import numpy as np
# %%
#a)
p = 1**34
q = 1

print(f'Resultado esperado : 1 ; resultado obtenido {p + q - p}' )

# %%
#b)
p = 100
q = 1**-15

r1 = (p + q) + q
r2 = ((p + q) + q) + q

print(f'Resultado 1 esperado : Verdadero ; resultado obtenido {r1 == (p + 2*q)}' )

print(f'Resultado 2 esperado : Verdadero ; resultado obtenido {r2 == (p + 3*q)}' )
# %%
#c)

print(f'Resultado esperado : Verdadero ; resultado obtenido {(0.1 + 0.2) == 0.3}' )
# %%
#d)

print(f'Resultado esperado : Verdadero ; resultado obtenido {(0.1 + 0.3) == 0.4}' )
# %%
#e)

print(f'Resultado esperado : 1 ; resultado obtenido {1**-323}' )
# %%
#f)

print(f'Resultado esperado : 1 ; resultado obtenido {1**-324}' )
# %%
#g)
epsilon = np.finfo(float).eps

print(f'Resultado esperado : numero considerablemente pequeño (epsilon dividido por 2) ; resultado obtenido {epsilon/2}' )
# %%
#h)
epsilon = np.finfo(float).eps

print(f'Resultado esperado : 1 + el error (epsilon) ; resultado obtenido {(1 + epsilon/2) + epsilon/2 }' )
# %%
#i)
epsilon = np.finfo(float).eps

print(f'Resultado esperado : 1 + el error (epsilon) ; resultado obtenido {1 + (epsilon/2 + epsilon/2) }' )
# %%
#j) 
epsilon = np.finfo(float).eps

print(f'Resultado esperado : error (epsilon) ; resultado obtenido {((1 + epsilon/2) + epsilon/2) -1 }' )
# %%
#k)

print(f'Resultado esperado : error (epsilon) ; resultado obtenido {(1 + (epsilon/2 + epsilon/2)) -1}' )
# %%
#l)
resultados =  []

for j in range(1,26):
    res = np.sin((10**j) * np.pi)
    resultados.append(res)

print(f'Resultados esperados : array de 0 ; resultados obtenidos {resultados}' )
# %%
#m) 
resultados =  []

for j in range(1,26):
    res = np.sin((np.pi / 2) + (np.pi * (10**j)))
    resultados.append(res)

print(f'Resultados esperados : array de 1 ; resultados obtenidos {resultados}' )
