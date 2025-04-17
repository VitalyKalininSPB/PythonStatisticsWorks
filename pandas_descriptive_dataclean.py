import pandas as pd
import numpy as np

data = pd.read_csv('datasets/mobile_price_classification.csv')
print(data.dtypes)

data = data.rename(columns = {"blue" : "bluetooth",
                              "fc" : "fc_megapixel",
                              "pc" : "pc_megapixel",
                              "m_dep" : "m_depth"})
dupes = data.duplicated()
print(len(data))
print(sum(dupes))

data = data.drop_duplicates()
print(data.isnull().sum())
data['fc_megapixel'] = data['fc_megapixel'].fillna(0)
print(data.isnull().sum())

print(len(data['ram'].unique()))
data['ram'] = data['ram'].fillna(method='backfill')
# NA value is erased
print(len(data['ram'].unique()))
# Replacing nan with median of the column
data['mobile_wt'] = data['mobile_wt'].fillna(data['mobile_wt'].median())

data = data.dropna()
data.to_csv('mobile_data_cleaned.csv', index = False)

# Numeric and categorical data
numeric_data = data.drop(['bluetooth', 'dual_sim', 'four_g', 'three_g',
                          'touch_screen', 'wifi', 'price_range'], axis=1)

categorical_data = data[['bluetooth', 'dual_sim', 'four_g', 'three_g',
                       'touch_screen', 'wifi', 'price_range']]

# Handling outliers
from matplotlib import pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize = (10, 8))
sns.boxplot(numeric_data['ram'],
            orient = 'v')
plt.show()

fig, ax = plt.subplots(figsize = (10, 8))
sns.boxplot(numeric_data['mobile_wt'],
            orient = 'v')
plt.show()

# All properties
fig, ax = plt.subplots(figsize = (10, 8))
bp = sns.boxplot(data = numeric_data)
bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
plt.show()

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_array = scaler.fit_transform(numeric_data)
scaled_data = pd.DataFrame(scaled_array, columns = numeric_data.columns)
fig, ax = plt.subplots(figsize = (10, 8))
bp = sns.boxplot(data = scaled_data)
bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
plt.show()

from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
robust_scaled_array = robust_scaler.fit_transform(numeric_data)
robust_scaled_data = pd.DataFrame(robust_scaled_array, columns = numeric_data.columns)
fig, ax = plt.subplots(figsize = (10, 8))
bp = sns.boxplot(data = robust_scaled_data)
bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
plt.show()

Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

outliers_removed_data = numeric_data[~ ((numeric_data < (Q1 - 1.5 * IQR)) \
                                     | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]

print(outliers_removed_data.shape)
print(numeric_data.shape)

# Graph without outliers
fig, ax = plt.subplots(figsize = (10, 8))
bp = sns.boxplot(data = outliers_removed_data)
bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
plt.show()

from sklearn.model_selection import train_test_split
scaled_data = scaled_data.reset_index()
categorical_data = categorical_data.reset_index()
final_df = pd.concat([scaled_data, categorical_data], axis=1)

fig, ax = plt.subplots(figsize = (10, 8))
bp = sns.boxplot(data = final_df)
bp.set_xticklabels(bp.get_xticklabels(), rotation=90)
plt.show()

X = final_df.drop('price_range', axis=1)
y = final_df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    random_state = 101)

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(solver='lbfgs',
                                    multi_class='multinomial',
                                    max_iter=10000)
logistic_model.fit(X_train, y_train)
print(logistic_model.score(X_test, y_test))
