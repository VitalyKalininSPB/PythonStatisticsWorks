import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

diabetes_data = pd.read_csv('datasets/diabetes.csv')

diabetes_data.hist(figsize=(20, 10));
plt.show()

### Check for imbalance
diagnosis_count = diabetes_data.diagnosis.value_counts()
print('Class 0:', diagnosis_count[0])
print('Class 1:', diagnosis_count[1])
print('Proportion:', round(diagnosis_count[0] / diagnosis_count[1], 2), ': 1')
diagnosis_count.plot(kind='bar', title='Count (target)', figsize = (12, 10));
plt.show()

### Separate input features and target
y = diabetes_data.diagnosis
X = diabetes_data.drop('diagnosis', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.25,
                                                    random_state = 27)
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(solver='liblinear')
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(pd.crosstab (y_pred, y_test))

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))

## Balancing data using `Sklear.utils.resample`
X = pd.concat([X_train, y_train], axis=1)
X = X.reset_index(drop=True)
non_diabetic = diabetes_data[diabetes_data.diagnosis==0]
diabetic = diabetes_data[diabetes_data.diagnosis==1]

## Oversampling
from sklearn.utils import resample
over_sampled = resample(diabetic,
                        replace = True,
                        n_samples = len(non_diabetic),
                        random_state = 27)
over_sampled = pd.concat([non_diabetic, over_sampled])
over_sampled = over_sampled.reset_index(drop=True)
diagnosis_count = over_sampled.diagnosis.value_counts()
print('Class 0:', diagnosis_count[0])
print('Class 1:', diagnosis_count[1])
print('Proportion:', round(diagnosis_count[0] / diagnosis_count[1], 2), ': 1')

diagnosis_count.plot(kind='bar', title='Count (target)', figsize = (12, 10));
plt.show()

y_train = over_sampled.diagnosis
X_train = over_sampled.drop('diagnosis', axis=1)
logistic_model = LogisticRegression(solver='liblinear')
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
print(pd.crosstab (y_pred, y_test))

print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))

## Undersampling
under_sampled = resample(non_diabetic,
                         replace=True,
                         n_samples=len(diabetic),
                         random_state=27) # reproducible results

under_sampled = pd.concat([diabetic, under_sampled])
under_sampled = under_sampled.reset_index(drop=True)

print('Class 0:', diagnosis_count[0])
print('Class 1:', diagnosis_count[1])
print('Proportion:', round(diagnosis_count[0] / diagnosis_count[1], 2), ': 1')
diagnosis_count.plot(kind='bar', title='Count (target)', figsize= (12, 10));
plt.show()

y_train = under_sampled.diagnosis
X_train = under_sampled.drop('diagnosis', axis=1)
logistic_model = LogisticRegression(solver='liblinear')
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print(pd.crosstab (y_pred, y_test))
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))

