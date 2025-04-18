import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

import researchpy as rp
from scipy import stats

bike_sharing_data = pd.read_csv('datasets/day.csv')
print(bike_sharing_data.shape)

bike_sharing_data = bike_sharing_data[['season',
                                       'mnth',
                                       'holiday',
                                       'workingday',
                                       'weathersit',
                                       'temp',
                                       'cnt']]

bike_sharing_data.to_csv('datasets/bike_sharing_data_processed.csv', index=False)

# Detects value of freedom in categorical values
print(bike_sharing_data['season'].unique())
print(bike_sharing_data['workingday'].unique())
print(bike_sharing_data['holiday'].unique())
print(bike_sharing_data['weathersit'].unique())

print(bike_sharing_data.groupby('workingday')['cnt'].describe())
bike_sharing_data.boxplot(column=['cnt'], by='workingday', figsize=(12, 8))
plt.show()

# Sampling
sample_01 = bike_sharing_data[(bike_sharing_data['workingday'] == 1)]
sample_02 = bike_sharing_data[(bike_sharing_data['workingday'] == 0)]

# Different parts should be equal in size
print(sample_01.shape)
print(sample_02.shape)
sample_01 = sample_01.sample(231)
print(sample_01.shape)

#The hypothesis being tested:
#
#    Null hypothesis (H0): u1 = u2, which translates to the mean of sample_01 is equal to the mean of sample 02
#    Alternative hypothesis (H1): u1 ? u2, which translates to the means of sample01 is not equal to sample 02

#Homogeneity of variance
#Of these tests, the most common assessment for homogeneity of variance is Levene's test. The Levene's test uses an F-test to test the null hypothesis that the variance is equal across groups. A p value less than .05 indicates a violation of the assumption.
#
#https://en.wikipedia.org/wiki/Levene%27s_test
print(stats.levene(sample_01['cnt'], sample_02['cnt']))

## Normal distribution  of residuals
### Checking difference between two pair points
diff = scale(np.array(sample_01['cnt']) - np.array(sample_02['cnt'], dtype=float))
plt.hist(diff)
plt.show()

### Checking for normality by Q-Q plot graph
plt.figure(figsize=(12, 8))
stats.probplot(diff, plot=plt, dist='norm')
plt.show()

### Shapiro method
print(stats.shapiro(diff))

# T-test (scipy)
print(stats.ttest_ind(sample_01['cnt'], sample_02['cnt']))

# T-test (researchpy)
descriptives, results = rp.ttest(sample_01['cnt'], sample_02['cnt'])
print(descriptives)
print(results)

### Temperature
bike_sharing_data[['temp']].boxplot(figsize=(12, 8))
# Divide normalized temp to 2 categories (above and below the mean)
bike_sharing_data['temp_category'] = \
    bike_sharing_data['temp'] > bike_sharing_data['temp'].mean()

print(bike_sharing_data.groupby('temp_category')['cnt'].describe())
bike_sharing_data.boxplot(column=['cnt'], by='temp_category', figsize=(12, 8))

sample_01 = bike_sharing_data[(bike_sharing_data['temp_category'] == True)]
sample_02 = bike_sharing_data[(bike_sharing_data['temp_category'] == False)]
print(sample_01.shape)
print(sample_02.shape)
sample_01 = sample_01.sample(364)
print(stats.levene(sample_01['cnt'], sample_02['cnt']))
diff = scale(np.array(sample_01['cnt']) - np.array(sample_02['cnt']))
plt.hist(diff)

plt.figure(figsize=(12, 8))
stats.probplot(diff, plot=plt)
plt.show()

print(stats.shapiro(diff))
# T-test
print(stats.ttest_ind(sample_01['cnt'], sample_02['cnt']))
descriptives, results = rp.ttest(sample_01['cnt'], sample_02['cnt'], equal_variances=False)
print(descriptives)
print(results)
