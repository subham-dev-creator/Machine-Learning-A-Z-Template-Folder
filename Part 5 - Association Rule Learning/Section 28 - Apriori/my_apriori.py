import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transaction=[]

for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])


from apyori import apriori
