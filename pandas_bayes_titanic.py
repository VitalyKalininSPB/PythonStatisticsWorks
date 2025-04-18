import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv("dataset/titanic.csv")
print(titanic_df.shape)

titanic_df = titanic_df[['Sex', 'Survived']]
titanic_df['Sex'] = titanic_df['Sex'].astype('category', copy = False).cat.codes
titanic_df.isnull().any()
titanic_df = titanic_df.dropna()
print(titanic_df.shape)

### Splitting into train and test
features = titanic_df[['Sex', 'Survived']]
label = titanic_df['Survived']

from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(features,
                                                    label,
                                                    test_size=0.2)

### Computing Probabilities Manually
survival_num_train = Y_train.value_counts()
print(survival_num_train)
survival_prob_train = survival_num_train[1] / len(Y_train) * 100
print(survival_prob_train)

survival_num_test = y_test.value_counts()
print(survival_num_test)
survival_prob_test = survival_num_test[1] / len(y_test) * 100
print(survival_prob_test)
# Result: Test and train probabilites are practically equal

# Probability tests with additional knowing about of sex of survivals
x_test_men = x_test.loc[x_test['Sex'] == 1]
x_test_women = x_test.loc[x_test['Sex'] == 0]

survival_num_men_test = x_test_men['Survived'].value_counts()
print(survival_num_men_test)
survival_prob_men_test = survival_num_men_test[1] / len(x_test_men['Survived']) * 100
print(survival_prob_men_test)

survival_num_women_test = x_test_women['Survived'].value_counts()
print(survival_num_women_test)
survival_prob_women_test = survival_num_women_test[1] / len(x_test_women['Survived']) * 100
print(survival_prob_women_test)
# Women and childrens went first to the boats
# Probabilities aren't equal

## Gaussian Naive Bayes model when sex is known
X_train = X_train.drop('Survived', axis=1)
x_test = x_test.drop('Survived', axis=1)

model = GaussianNB()
model.fit(X_train, Y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
