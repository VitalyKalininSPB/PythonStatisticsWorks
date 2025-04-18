import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import researchpy as rp

from statsmodels.formula.api import ols

bike_sharing_data = pd.read_csv('datasets/bike_sharing_data_processed.csv')
bike_sharing_data['weathersit'].unique()

bike_sharing_data.boxplot(column=['cnt'], by='weathersit', figsize=(12, 8))
plt.show()

print(stats.f_oneway(bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 1],
               bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 2],
               bike_sharing_data['cnt'][bike_sharing_data['weathersit'] == 3],))

result = ols('cnt ~ C(weathersit)', data = bike_sharing_data).fit()
print(result.summary())

# Tukey's method
from statsmodels.stats.multicomp import MultiComparison
mul_com = MultiComparison(bike_sharing_data['cnt'], bike_sharing_data['weathersit'])
mul_result = mul_com.tukeyhsd()
print(mul_result)
