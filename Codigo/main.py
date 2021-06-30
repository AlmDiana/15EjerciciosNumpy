import numpy as np
#Ejercicio 6

print("\n1. << Ejercicio 6 >>\n")
print("Create a vector with values ranging from 10 to 49\n")
z = np.arange(10,50)
print(z)
print("\n")

#Ejercicio 12

print("\n2. << Ejercicio 12 >>\n")
print("Create a 10x10 array with random values and find the minimum and maximum values\n")
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
print("\n")


#Ejercicio 18

print("\n3. << Ejercicio 18 >>\n")
print("Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?\n")
print(np.unravel_index(100,(6,7,8)))
print("\n")

#Ejercicio 24

print("\n4. << Ejercicio 24 >>\n")
print("What is the output of the following script? \n")
print("Output 1: ")
print(sum(range(5),-1)) # Salida: 9
from numpy import *
print ("Output 2:")
print(sum(range(5),-1)) # Salida: 10

#Ejercicio 30

print("\n5. << Ejercicio 30 >>\n")
print("Consider a generator function that generates 10 integers and use it to build an array\n")
#Función que genera 10 elementos
def generate():
    for x in range(10):
        yield x
#Construcción del array
K = np.fromiter(generate(),dtype=float,count=-1)
print("ARRAY: ", K)

#Ejercicio 37

print("\n6. << Ejercicio 37 >>\n")
print("Create random vector of size 10 and replace the maximum value by 0\n")
matriz = np.random.randint(0, 100, 10)
print("Arreglo original: ")
print(matriz)
matriz[matriz.argmax()] = 0
print("Arreglo remplazado: ")
print(matriz)

#Ejercicio 42

print("\n7. << Ejercicio 42 >>\n")
print("How to find the closest value (to a given scalar) in an array?\n")
Z = np.random.randint(0, 1000, 100)
print("Arreglo: \n", Z)
valor = input("\n Ingrese un valor aleatorio entre 0 y 1000: ")
index = (np.abs(Z-int(valor))).argmin()
print(Z[index])

#Ejercicio 48

print("\n8. << Ejercicio 48 >>\n")
print("Generate a generic 2D Gaussian­like array\n")
#La funcion Gaussiana o campana de Gaus se utiliza para inferir la probabilidad de un valor
x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
gaussiana = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
print("2D Gaussian como array:")
print(gaussiana)

#Ejercicio 54
print("\n9. << Ejercicio 54 >>\n")
print("Create an array class that has a name attribute \n")
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)

#Ejercicio 60
print("\n10. << Ejercicio 60 >>\n")
print("How to get the diagonal of a dot product?\n")
A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))
# Slow version
print("SLOW VERSION: ",np.diag(np.dot(A, B)))
# Fast version
print("FAST VERSION: ",np.sum(A * B.T, axis=1))
# Faster version
print("FASTER VERSION: ", np.einsum("ij,ji->i", A, B))

#Ejercicio 66

print("\n11. << Ejercicio 66 >>\n")
print("How to compute averages using a sliding window over an array?\n")
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print("ARRAY:", moving_average(Z, n=3))


#Ejercicio 72

print("\n12. << Ejercicio 72 >>\n")
print("Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?\n")
from numpy.lib import stride_tricks
Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(" ARRAY: \n", R)

#Ejercicio 78

print("\n13. << Ejercicio 78 >>\n")
print("Consider a 16x16 array, how to get the blocksum (block size is 4x4)?\n")
Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1)

print(S)

#Ejercicio 84

print("\n14. << Ejercicio 84 >>\n")
print("Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?\n")

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))
C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)

#Ejercicio 90

print("\n15. << Ejercicio 90 >>\n")
print("Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n.\n")

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
[2.0, 0.0, 1.0, 1.0],
[1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
