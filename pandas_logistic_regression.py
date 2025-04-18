import pandas as pd

data = pd.read_csv('datasets/voice.csv')

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
data['label'] = labelEncoder.fit_transform(data['label'].astype(str))
data.boxplot(by ='label', column =['meanfreq'], grid = False, figsize=(10, 8))
import matplotlib.pyplot as plt
plt.show()

### Spilting the data into train and test data
from sklearn.model_selection import train_test_split
features = data.drop('label', axis=1)
target = data['label']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
logistic_model = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("Training score : ", logistic_model.score(x_train, y_train))

### Accuracy, precision, recall scores
from sklearn.metrics import accuracy_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy : ' , acc)
print('Precision Score : ', pre)
print('Recall Score : ', recall)

from yellowbrick.target import FeatureCorrelation
feature_names = list(features.columns)
visualizer = FeatureCorrelation(labels = feature_names)
visualizer.fit(features, target)
visualizer.poof()
plt.show()

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest

# Chi2
print('Chi2')
select_univariate = SelectKBest(chi2, k=4).fit(features, target)
features_mask = select_univariate.get_support()
selected_columns = features.columns[features_mask]
print(selected_columns)
selected_features = features[selected_columns]

x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size =.2)
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy : ' , acc)
print('Precision Score : ', pre)
print('Recall Score : ', recall)

print('f_classif')
select_univariate = SelectKBest(f_classif, k=4).fit(features, target)
features_mask = select_univariate.get_support()
selected_columns = features.columns[features_mask]
print(selected_columns)

x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size =.2)
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy : ' , acc)
print('Precision Score : ', pre)
print('Recall Score : ', recall)


print('mutual_info_classif')
select_univariate = SelectKBest(mutual_info_classif, k=4).fit(features, target)
features_mask = select_univariate.get_support()
selected_columns = features.columns[features_mask]
print(selected_columns)

x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size =.2)
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy : ' , acc)
print('Precision Score : ', pre)
print('Recall Score : ', recall)
# There is was better in tutorial :)
