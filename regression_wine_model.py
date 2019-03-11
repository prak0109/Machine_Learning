import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model

wine = datasets.load_wine()

#print(wine.keys())

#print(wine.DESCR)

wine_X = wine.data[:,np.newaxis,3]

print(wine_X)

wine_X_train = wine_X[:-20]
wine_X_test = wine_X[-20:]


wine_y_train = wine.target[:-20]
wine_y_test = wine.target[-20:]

model = linear_model.LinearRegression()

model.fit(wine_X_train,wine_y_train)

wine_y_predicted = model.predict(wine_X_test)

msi = mean_squared_error(wine_y_predicted,wine_y_test)

print(msi)

plt.scatter(wine_X_test,wine_y_test)
plt.plot(wine_X_test,wine_y_predicted)

plt.show()










