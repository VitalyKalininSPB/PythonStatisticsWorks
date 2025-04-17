import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

height_weight_data = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')
print(height_weight_data.head())
# Check if index in shape also
print(height_weight_data.shape)
height_weight_data.drop('Index', inplace=True, axis=1)
print(height_weight_data.shape)
print(height_weight_data.isnull().sum())

min_height = height_weight_data['Height'].min()
max_height = height_weight_data['Height'].max()
min_weight = height_weight_data['Weight'].min()
max_weight = height_weight_data['Weight'].max()
range_of_height = max_height - min_height
range_of_weight = max_weight - min_weight

weight = height_weight_data['Weight']
sorted_weight = weight.sort_values().reset_index(drop=True)

def mean(data):
    num_elements = len(data)
    print('Number of elements: ', num_elements)

    weight_sum = data.sum()
    print('Sum: ', weight_sum)

    return weight_sum / num_elements

def median(data):
    num_elements = len(data)

    if(num_elements % 2 == 0):
        return  (data[(num_elements / 2) - 1] + data[(num_elements / 2)]) / 2

    else:
        return (data[((num_elements + 1) / 2) - 1])

print(mean(height_weight_data['Weight']))
weight_mean = height_weight_data['Weight'].mean()
print(weight_mean)

print(median(height_weight_data['Weight']))
print(median(sorted_weight))
weight_median = height_weight_data['Weight'].median()
print(weight_median)

# Mean and median (are equal now)
'''plt.figure(figsize=(12, 8))
height_weight_data['Weight'].hist(bins=30)
plt.axvline(weight_mean, color='r', label='mean')
plt.axvline(weight_median, color='g', label='median')
plt.legend()
plt.show()'''


# Add outliers to our datalistOfSeries = [pd.Series(['Male', 205, 460], index=height_weight_data.columns ),
listOfSeries = [pd.Series(['Male', 205, 460], index=height_weight_data.columns ),
                pd.Series(['Female', 202, 390], index=height_weight_data.columns ),
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 202, 390], index=height_weight_data.columns ),
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 200, 490], index=height_weight_data.columns )]

height_weight_updated = height_weight_data._append(listOfSeries , ignore_index=True)

# Outliers affected mean value strongly
updated_weight_mean = height_weight_updated['Weight'].mean()
print(updated_weight_mean)
# But for median is only slightly change
updated_weight_median = height_weight_updated['Weight'].median()
print(updated_weight_median)

'''
plt.figure(figsize=(12, 8))
plt.bar(height_weight_updated['Height'], height_weight_updated['Weight'])
plt.axhline(updated_weight_mean, color='r', label='mean')
plt.axhline(updated_weight_median, color='b', label='median')
plt.show()

plt.figure(figsize=(12, 8))
height_weight_data['Weight'].hist(bins=20)
plt.axvline(updated_weight_mean, color='r', label='mean')
plt.axvline(updated_weight_median, color='g', label='median')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
height_weight_updated['Weight'].hist(bins=100)
plt.axvline(updated_weight_mean, color='r', label='mean')
plt.axvline(updated_weight_median, color='g', label='median')
plt.legend()
plt.show()'''


# Mode
height_counts = {}
for p in height_weight_data['Height']:
    if p not in height_counts:
        height_counts[p] = 1
    else:
        height_counts[p] += 1

print(height_counts)

'''plt.figure(figsize=(25, 10))
x_range = range(len(height_counts))
plt.bar(x_range, list(height_counts.values()), align='center')
plt.xticks(x_range, list(height_counts.keys()))
plt.xlabel('Height')
plt.ylabel('Count')
plt.show()'''

height_mean = height_weight_data['Height'].mean()
height_median = height_weight_data['Height'].median()
height_mode = height_weight_data['Height'].mode().values[0]
plt.figure(figsize=(12, 8))
height_weight_data['Height'].hist()
plt.axvline(height_mean, color='r', label='mean')
plt.axvline(height_median, color='g', label='median')
plt.axvline(height_mode, color='y', label='mode')
plt.legend()
plt.show()
