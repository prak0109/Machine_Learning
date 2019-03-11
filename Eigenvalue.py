import numpy as np
from numpy import array
from numpy.linalg import norm
from numpy import linalg as LA

a = np.array([[1, 3, 2, 4, 6],[3, 2, 7, 8, 7],[2, 7, 3, 7, 8],[4, 8, 7, 4, 9],[6, 7, 8, 9, 5]])

print(a)

w,v = LA.eigh(a)
print(w)
print(v)

u,s,v = LA.svd(a)

print(s)

x = a*a
print(x)

w1,v1 = LA.eigh(x)

print(w1)


