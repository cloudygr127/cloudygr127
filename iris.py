import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = neighbors.KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train.ravel())
prediction = model.predict(x_test)
precision = accuracy_score(y_test,prediction)
print(prediction,y_test)
print(precision)