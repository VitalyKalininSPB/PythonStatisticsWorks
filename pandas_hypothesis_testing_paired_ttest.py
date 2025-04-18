import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from scipy import stats
import researchpy as rp

bp_reading = pd.read_csv('datasets/blood_pressure.csv')

bp_reading[['bp_before', 'bp_after']].boxplot(figsize=(12, 8))
plt.show()

print(stats.levene(bp_reading['bp_after'], bp_reading['bp_before']))
bp_reading['bp_diff'] = scale(bp_reading['bp_after'] - bp_reading['bp_before'])

bp_reading[['bp_diff']].hist(figsize=(12, 8))

plt.figure(figsize=(15, 8))
stats.probplot(bp_reading['bp_diff'], plot=plt)
plt.title('Blood pressure difference Q-Q plot')
plt.show()

stats.shapiro(bp_reading['bp_diff'])
print(stats.ttest_rel(bp_reading['bp_after'], bp_reading['bp_before']))
### **Note:-** __Here, `t-test = -3.337` and `p-value = 0.0011` since p-value is less than the significant value hence null-hypothesis is rejected`(Alpha = 0.05)`__


