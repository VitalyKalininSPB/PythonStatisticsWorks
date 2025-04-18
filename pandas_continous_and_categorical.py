import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tips_data = pd.read_csv('datasets/tips.csv')
print(tips_data.shape)

print(tips_data.dtypes)
print(tips_data.head())

continuous_tips = tips_data[["time", "total_bill", "tip", "size"]]
print(continuous_tips.groupby(by='time').mean())

plt.figure(figsize=(12, 8))
sns.barplot(x='time', y='total_bill', data=continuous_tips)
plt.title('Bill and Tips')
plt.xticks(rotation=90)
plt.show()

mean = tips_data['tip'].mean()
print(mean)

tips_data['above average'] = (tips_data['tip'] - mean) > 0

plt.figure(figsize=(12, 8))

sns.countplot(x='time',
              hue='above average',
              data=tips_data,
              order = tips_data['time'].value_counts().index)
plt.title('Bill and Tips')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='time',
              hue='sex',
              data=tips_data,
              order = tips_data['time'].value_counts().index)
plt.title('Bill and Tips')
plt.show()

plt.figure(figsize=(12, 8))

sns.boxplot(x='sex',
            y='total_bill',
            data=tips_data,
            palette='nipy_spectral')
plt.title('Bill and Tips')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='size',
            y='total_bill',
            data=tips_data,
            palette='nipy_spectral')
plt.title('Bill and Tips')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='size',
              data=tips_data,
              order = tips_data['size'].value_counts().index )
plt.xticks(rotation=70)
plt.title('Bill and Tips')
plt.show()
