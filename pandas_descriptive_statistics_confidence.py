import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

automobile_data = pd.read_csv('datasets/automobile_data_processed.csv')
automobile_data.boxplot('mpg', figsize=(12, 8))
plt.title('Miles per gallon')
plt.show()

'''automobile_data['mpg'].plot.kde(figsize=(12, 8))
plt.xlabel('mpg')
plt.title('Density plot for MPG')
plt.show()'''

plt.figure(figsize=(12, 8))
sns.boxplot(x='cylinders', y='mpg', data=automobile_data)
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x='cylinders', y='mpg', data=automobile_data, inner=None)
sns.swarmplot(x='cylinders', y='mpg', data=automobile_data, color='w')
plt.show()

cylinder_stats = automobile_data.groupby(['cylinders'])['mpg'].agg(['mean', 'count', 'std'])

ci95_high = []
ci95_low = []

for i in cylinder_stats.index:
    mean, count, std = cylinder_stats.loc[i]
    ci95_high.append(mean + 1.96 * (std / math.sqrt(count)))
    ci95_low.append(mean - 1.96 * (std / math.sqrt(count)))

cylinder_stats['ci95_HIGH'] = ci95_high
cylinder_stats['ci95_LOW'] = ci95_low

print(cylinder_stats)

# 4-cylinders statistics
cylinders = 4
cylinders4_df = automobile_data.loc[automobile_data['cylinders'] == cylinders]

plt.figure(figsize=(12, 8))
sns.distplot(cylinders4_df['mpg'], rug=True, kde=True, hist=False)
plt.show()

plt.figure(figsize=(12, 8))

sns.distplot(cylinders4_df['mpg'], rug=True, kde=True, hist=False)

# Add more experiments here
plt.stem([cylinder_stats.loc[cylinders]['mean']],
         [0.07], linefmt = 'C1',
         markerfmt = 'C1', label = 'mean')

plt.stem([cylinder_stats.loc[cylinders]['ci95_LOW']],
         [0.07], linefmt = 'C2',
         markerfmt = 'C2', label = '95% CI High')

plt.stem([cylinder_stats.loc[cylinders]['ci95_HIGH']],
         [0.07], linefmt = 'C3',
         markerfmt = 'C3', label = '95% CI Low')

plt.xlabel('mpg')
plt.legend()
plt.show()
