import numpy as np
from numpy import array
from numpy.linalg import norm
from numpy import linalg as LA

w = np.array([[1, 3, 2, 4, 6],[3, 2, 7, 8, 7],[2, 7, 3, 7, 8],[4, 8, 7, 4, 9],[6, 7, 8, 9, 5]])

print(w)
#Norm of w
a=LA.norm(w)
print(a)


y = np.array([1, 0, 0, 0, 0])
b = np.array([0, 1, 0, 0, 0])

y0=y.T
b1=b.T
print(b1)

print(y)
print(y0)

#calculate Norm of y0
Norm_y0 = LA.norm(y0)
print(Norm_y0)

y1 = w*(y0) + (b1)
print(y1)
#Norm of ||y1||2 in problem
X = LA.norm(y1)
print(X)

#Calculate ||y2||2

y2 = w*y1 + b1
print(y2)

Y2= LA.norm(y2)
print(Y2)

# calculate ||y2||2/||y0||2

result = Y2/(Norm_y0)
print(result)





