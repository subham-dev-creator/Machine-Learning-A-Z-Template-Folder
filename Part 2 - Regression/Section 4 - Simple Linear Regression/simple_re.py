import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

"""from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color = "orange")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("SALARY VS EXPERICENCE ")
plt.xlabel("YEARS OF EXPERIENCE")
plt.ylabel("SALARY")
plt.show()

plt.scatter(X_test,y_test,color = "orange")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("SALARY VS EXPERICENCE ")
plt.xlabel("YEARS OF EXPERIENCE")
plt.ylabel("SALARY")
plt.show()