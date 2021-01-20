import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest

df = pd.read_csv("titanic_train.csv")
print(df.head())
print(df[df['Survived'] == 1].head())

x1 = df[df['Survived'] == 1]['Fare'].dropna().to_numpy()
x2 = df[df['Survived'] == 0]['Fare'].dropna().to_numpy()

sns.kdeplot(x1, label='Survived')
sns.kdeplot(x2, label='Did Not Survive')
plt.legend;
plt.show()

print(x1.mean(), x2.mean())
print(ztest(x1, x2))

