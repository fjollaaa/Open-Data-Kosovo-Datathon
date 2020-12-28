from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_excel("COVID-19.xlsx")
dataset.rename(columns = {'Decision label':'Covid-19'}, inplace = True)
dataset=dataset.loc[dataset['Covid-19'] == 'COVID-19']
dataset=dataset[['Age','Covid-19']]
dataset = dataset.drop(dataset[dataset['Age'] == "*"].index)
dataset['Age'] = dataset['Age'].astype(int)
dataset=dataset.groupby('Age').count()
plt.style.use('ggplot')
dataset.plot.bar( fontsize=8)
print(dataset)
plt.show()


