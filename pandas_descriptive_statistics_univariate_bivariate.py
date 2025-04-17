import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('datasets/weight-height.csv')

height = data['Height']
print(height)

'''plt.figure(figsize=(12, 8))
height.plot(kind = 'hist',
            title = 'Height Histogram')
plt.show()

plt.figure(figsize=(12, 8))

height.plot(kind = 'box',
            title = 'Height Box-plot')
plt.show()

height.plot(kind = 'kde',
            title = 'Height KDE', figsize=(12, 8))
plt.show()

weight = data['Weight']
plt.figure(figsize=(12, 8))
weight.plot(kind = 'hist',
            title = 'Weight Histogram')
plt.show()

weight = data['Weight']
plt.figure(figsize=(12, 8))
weight.plot(kind = 'box',
            title = 'Weight Box-plot')
plt.show()'''

## Bivariate Analysis
'''plt.figure(figsize=(12, 8))
sns.scatterplot(x = "Height", y = "Weight", data=data)
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x = "Height", y = "Weight", hue='Gender', data=data)
plt.show()

# Distribution plots
sns.FacetGrid(data, hue = 'Gender', height = 5)\
              .map(sns.distplot, 'Height')\
              .add_legend()
plt.show()

sns.FacetGrid(data, hue = 'Gender', height = 5)\
    .map(sns.distplot, 'Weight').add_legend()
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Gender', y ='Height', data = data)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Gender', y ='Weight', data = data)
plt.show()'''

# Violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x = 'Gender', y ='Height', data = data)
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x = 'Gender', y ='Weight', data = data)
plt.show()
