#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np
import matplotlib as plt
import seaborn as sns

#%%
def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

    for j in range(n):
        for i in range(j+1, n):
            Ac[i,j] =  Ac[i,j]/Ac[j,j]
            cant_op +=1
            for k in range (j+1, n):
                Ac[i,k] = Ac[i,k] - Ac[i,j] * Ac[j,k]
            
                cant_op += 2


                
    ## hasta aqui
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)
    
    return L, U, cant_op


A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])
c = elim_gaussiana(A)
#%%
def main():
    n=7
    valores_operaciones = []
    dimension_matriz = []
    for i in range (2, n+1):
        B = np.eye(i) - np.tril(np.ones((i,i)),-1) 
        B[:i,i-1] = 1
        
        L,U,cant_oper = elim_gaussiana(B)
        
        valores_operaciones.append(cant_oper)
        dimension_matriz.append(i)
        
    #B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    #B[:n,n-1] = 1
    #print('Matriz B \n', B)
    
    sns.lineplot(x = dimension_matriz, y = valores_operaciones )
    
    #L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
# %%
#Ejercicio 3: escribir funcioes que calculen la solucion de un sistema
#a) 
def resolver_sist_triang_inf(L, b):
    y = np.zeros(b.shape)
    y[0] = b[0]
    for i in range(1, b.shape[0]):
        y[i] = b[i] - (L[i, : i] @ y[:i]) #recorre la fila i, cada columna hasta el i (excluyendo)
        #y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    return y


A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])
c = elim_gaussiana(A)
b = np.array([1,3,3,5])
y = resolver_sist_triang_inf(c[0], b)
# %%
#b)
def resolver_sist_triang_sup(U, y):
    x = np.zeros(y.shape)
    x[y.shape[0]-1] = y[y.shape[0]-1] / U[y.shape[0]-1, y.shape[0]-1]
    for i in range(y.shape[0]-2, -1,-1):
        x[i] = (y[i] - (U[i,y.shape[0]-1:i:-1] @ x[y.shape[0]:i:-1]))/U[i][i]
        
    return x

x = resolver_sist_triang_sup(c[1], y)    

print("Solución x:", x)

# %%
#c)
def resolver_sist(A, b):
    L,U = elim_gaussiana(A)[0:2]
    y = resolver_sist_triang_inf(L, b)
    x = resolver_sist_triang_sup(U, y)
    return x

np.allclose(x, resolver_sist(A, b))
# %%
#d)
def resolver_sistemas():
    n=7
    for i in range (2, n+1):
        B = np.eye(i) - np.tril(np.ones((i,i)),-1) 
        B[:i,i-1] = 1
        b = np.arange(1, i+1)
        print(resolver_sist(B, b))

print(resolver_sistemas())        