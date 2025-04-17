import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

automobile_data = pd.read_csv('datasets/auto-mpg.csv')

# Investigate some trash data
automobile_data = automobile_data.replace('?', np.nan)
automobile_data = automobile_data.dropna()
# Some columns have no use for us
automobile_data.drop(['origin', 'car name'], axis=1, inplace=True)

# Date and Time interval preprocessing
automobile_data['model year'] = '19' + automobile_data['model year'].astype(str)
import datetime
automobile_data['age'] = datetime.datetime.now().year - \
    pd.to_numeric(automobile_data['model year'])
automobile_data.drop(['model year'], axis=1, inplace=True)

print(automobile_data.dtypes)
automobile_data['horsepower'] = pd.to_numeric(automobile_data['horsepower'], errors='coerce')

automobile_data.to_csv('datasets/automobile_data_processed.csv', index=False)

'''automobile_data.plot.scatter(x='displacement', y='mpg', figsize=(12, 8))
plt.show()

automobile_data.plot.scatter(x='horsepower', y='mpg', figsize=(12, 8))
plt.show()

automobile_data.plot.hexbin(x='acceleration', y='mpg', gridsize=20, figsize=(12, 8))
plt.show()

automobile_grouped = automobile_data.groupby(['cylinders']).mean()[['mpg', 'horsepower',
                                                                    'acceleration', 'displacement']]
print(automobile_grouped)

automobile_grouped.plot.line(figsize=(12, 8))
plt.show()'''

### Multivariate data analysis
fig, ax = plt.subplots()

'''automobile_data.plot(x='horsepower', y='mpg',
                     kind='scatter', s=60, c='cylinders',
                     cmap='magma_r', title='Automobile Data',
                     figsize=(12, 8), ax=ax)
plt.show()

fig, ax = plt.subplots()

automobile_data.plot(x='acceleration', y='mpg',
                     kind='scatter', s=60, c='cylinders',
                     cmap='magma_r', title='Automobile Data',
                     figsize=(12, 8), ax=ax)
plt.show()

fig, ax = plt.subplots()

automobile_data.plot(x='displacement', y='mpg',
                     kind='scatter', s=60, c='cylinders',
                     cmap='viridis', title='Automobile Data',
                     figsize=(12, 8), ax=ax)

plt.show()'''

print(automobile_data['acceleration'].cov(automobile_data['mpg']))
print(automobile_data['acceleration'].corr(automobile_data['mpg']))
print(automobile_data['horsepower'].cov(automobile_data['mpg']))
print(automobile_data['horsepower'].corr(automobile_data['mpg']))
print(automobile_data['horsepower'].cov(automobile_data['displacement']))
print(automobile_data['horsepower'].corr(automobile_data['displacement']))

### Covariance
automobile_data_cov = automobile_data.cov()
print(automobile_data_cov)

automobile_data_corr = automobile_data.corr()
print(automobile_data_corr)

plt.figure(figsize=(12, 8))
sns.heatmap(automobile_data_corr, annot=True)
plt.show()

## Linear Regression
mpg_mean = automobile_data['mpg'].mean()
horsepower_mean = automobile_data['horsepower'].mean()

automobile_data['horsepower_mpg_cov'] = (automobile_data['horsepower'] - horsepower_mean) * \
                                        (automobile_data['mpg'] - mpg_mean)
automobile_data['horsepower_var'] = (automobile_data['horsepower'] - horsepower_mean)**2

beta = automobile_data['horsepower_mpg_cov'].sum() / automobile_data['horsepower_var'].sum()

print(f'beta = {beta}')
alpha = mpg_mean - (beta * horsepower_mean)

print(f'alpha = {alpha}')
y_pred = alpha + beta * automobile_data['horsepower']

automobile_data.plot(x='horsepower', y='mpg',
                     kind='scatter', s=50, figsize=(12, 8))

plt.plot(automobile_data['horsepower'], y_pred, color='red')

plt.show()
