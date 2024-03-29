import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Mall_Customers.csv')

X=dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("No. of cluters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)


plt.scatter()