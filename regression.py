import pandas as pd
import quandl,math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('CHRIS/MGEX_IH1')

df = df[['Open','High','Low','Last','Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Last']*100
df['PCT_change'] = (df['Last'] - df['Open']) / df['Open']*100

df = df[['Open','Last','HL_PCT','PCT_change']]

forecast_col = 'Last'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)
df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy =clf.score(X_test,y_test)
print(accuracy)

