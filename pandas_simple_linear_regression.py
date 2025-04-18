import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

house_data = pd.read_csv('datasets/kc_house_data.csv')

# Clean
house_data.drop(['id', 'lat', 'long', 'zipcode'], inplace=True, axis=1)
house_data['date'] = pd.to_datetime(house_data['date'])
house_data['house_age'] = house_data['date'].dt.year - house_data['yr_built']

house_data.drop('date', inplace=True, axis=1)
house_data = house_data.drop('yr_built', axis=1)
house_data['renovated'] = house_data['yr_renovated'].apply(lambda x:0 if x == 0 else 1)
house_data.drop('yr_renovated', inplace=True, axis=1)

house_data.to_csv('datasets/house_data_processed.csv', index=False)

sns.lmplot(x='sqft_living', y='price', data=house_data);
plt.show()

sns.lmplot(x='house_age', y='price', data=house_data)
plt.show()

sns.lmplot(x='floors', y='price', data=house_data)
plt.show()

### Scaling dataset and one feature for simple linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = house_data[['sqft_living']]
y = house_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_regression = LinearRegression()
model = linear_regression.fit(X_train, y_train)
y_pred = model.predict(X_test)

### Regression line
plt.figure(figsize=(10, 8))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, c='r')
plt.show()

# Metrics
print("Training score : ", linear_regression.score(X_train, y_train))
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("Testing score : ", score)
theta_0 = linear_regression.coef_
print(theta_0)
intercept = linear_regression.intercept_
print(intercept)


