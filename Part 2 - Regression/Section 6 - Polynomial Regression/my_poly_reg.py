# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualize Linear Reg Model
plt.scatter(X,y,color ="red")
plt.plot(X,lin_reg.predict(X),color ="black")
plt.title("LEVEL VS SALARY (LINEAR MODEL)")
plt.xlabel("LEVEL")
plt.ylabel("SALARY")
plt.show()

#visualize Poly Reg Model
X_grid=np.arange(min(X),max(X),0.05)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color = "red")
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color ="black")
plt.title("LEVEL VS SALARY (Poly MODEL)")
plt.xlabel("LEVEL")
plt.ylabel("SALARY")
plt.show()

#predicting salary by Linear Reg
lin_reg.predict(6.5)

#predicting salary by Poly Reg
lin_reg_2.predict(poly_reg.fit_transform(6.5))
