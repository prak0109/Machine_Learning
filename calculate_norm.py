import numpy as np
from numpy import array
from numpy.linalg import norm
from numpy import linalg as LA

x = np.array([ -10, 2, 4, 8, 9])
a= x.T

L2 = LA.norm(a)
print(L2)

L1 = LA.norm(a,1)
print(L1)

L3= LA.norm(a,3)
print(L3)

Linf = LA.norm(a, np.inf)
print(Linf)


