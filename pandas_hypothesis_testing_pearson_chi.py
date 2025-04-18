import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import chi2_contingency

e_comm_data = pd.read_csv('datasets/E-commerce.csv', index_col=0)

e_comm_data = e_comm_data[['Recommended IND', 'Rating']]
e_comm_data[['Rating']].hist(figsize=(10, 8))
plt.show()

df_for_obs = pd.crosstab(e_comm_data['Recommended IND'], e_comm_data['Rating'])
print(df_for_obs)

chi2, p_value, degrees_of_freedom, expected_values = chi2_contingency(df_for_obs.values)
print('Chi2 stats: {}'. format(round(chi2, 3)))
print('The p-values: {}'.format(p_value))
print('The degree of freedom: {}'.format(degrees_of_freedom))
print(expected_values)

expected_df = pd.DataFrame({
    '0': expected_values[0],
    '1': expected_values[1],
})

plt.figure(figsize=(10, 8))

plt.bar(expected_df.index, expected_df['1'], label="Recommended")
plt.bar(expected_df.index, expected_df['0'], label="Not recommended")
plt.legend()
plt.show()

ratings_recommended = e_comm_data[e_comm_data['Recommended IND'] == 1]
ratings_not_recommended = e_comm_data[e_comm_data['Recommended IND'] == 0]
'''ratings_recommended[['Rating']].hist(figsize=(10, 8))
plt.show()'''

plt.figure(figsize=(10, 8))
plt.hist(ratings_recommended['Rating'], label="Recommended")
plt.hist(ratings_not_recommended['Rating'], label="Not recommended")
plt.legend()
plt.show()
