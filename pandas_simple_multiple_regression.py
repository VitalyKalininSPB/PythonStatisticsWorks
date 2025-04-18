import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

house_data = pd.read_csv('datasets/house_data_processed.csv')

target = house_data['price']
features = house_data.drop('price', axis=1)

from yellowbrick.target import FeatureCorrelation
feature_names = list(features.columns)
visualizer = FeatureCorrelation(labels = feature_names)
visualizer.fit(features, target)
visualizer.poof()
plt.show()

### Select K-Best features to predict price of houses
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
select_univariate = SelectKBest(f_regression, k=5).fit(features, target)
features_mask = select_univariate.get_support()
print(features_mask)

selected_columns = features.columns[features_mask]
print(selected_columns)
selected_features = features[selected_columns]

from sklearn.preprocessing import scale
X = pd.DataFrame(data=scale(selected_features), columns=selected_features.columns)
y = target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)
df = pd.DataFrame({'test': y_test, 'Predicted': y_pred})

from sklearn.metrics import r2_score
score = linear_regression.score(X_train, y_train)
r2score = r2_score(y_test, y_pred)
print('Score: {}'.format(score))
print('r2_score: {}'.format(r2score))

print(linear_regression.coef_)
print(linear_regression.intercept_)

