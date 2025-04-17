import pandas as pd
import matplotlib.pyplot as plt

height_weight_data = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')
height_weight_data.drop('Index', inplace=True, axis=1)
num_records = height_weight_data.shape[0]
print(num_records)

height_data = height_weight_data[['Height']].copy()
weight_data = height_weight_data[['Weight']].copy()

counts = [1] * num_records
height_data['counts_height'] = counts
weight_data['counts_weight'] = counts
print(height_data['counts_height'])

weight_data = weight_data.sort_values('Weight')
height_data = height_data.sort_values('Height')
height_data = height_data.groupby('Height', as_index=False).count()
print(height_data.head(10))

weight_data = weight_data.groupby('Weight', as_index=False).count()

# Cumulative sum
height_data['cumcounts_height'] = height_data['counts_height'].cumsum()
print(height_data.head(10))
weight_data['cumcounts_weight'] = weight_data['counts_weight'].cumsum()

# Quantiles
print('Q1&Q3')
q1_height = height_weight_data['Height'].quantile(.25)
print(q1_height)
q3_height = height_weight_data['Height'].quantile(.75)
print(q3_height)
iqr_height = q3_height - q1_height
print(iqr_height)

'''plt.figure(figsize=(12, 8))
height_weight_data['Height'].hist(bins=30)
plt.axvline(q1_height, color='r', label='Q1')
plt.axvline(q3_height, color='g', label='Q2')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
height_weight_data['Weight'].hist(bins=30)
plt.axvline(height_weight_data['Weight'].quantile(.25), color='r', label='Q1')
plt.axvline(height_weight_data['Weight'].quantile(.75), color='g', label='Q2')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(height_weight_data['Weight'], height_weight_data['Height'], s=100)
plt.axvline(height_weight_data['Weight'].quantile(.25), color='r', label='Q1 Weight')
plt.axvline(height_weight_data['Weight'].quantile(.75), color='g', label='Q2 Weight')
plt.axhline(height_weight_data['Height'].quantile(.25), color='y', label='Q1 Height')
plt.axhline(height_weight_data['Height'].quantile(.75), color='m', label='Q2 Height')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.bar(height_data['Height'], height_data['cumcounts_height'])
plt.axvline(height_weight_data['Height'].quantile(.25), color='y', label='25%')
plt.axvline(height_weight_data['Height'].quantile(.50), color='m', label='50%')
plt.axvline(height_weight_data['Height'].quantile(.75), color='r', label='75%')
plt.axhline(.25 * num_records, color='y', label='25%')
plt.axhline(.5 * num_records, color='m', label='50%')
plt.axhline(.75 * num_records, color='r', label='75%')
plt.show()'''

# Variation (Дисперсия)
def variance(data):

    diffs = 0
    avg = sum(data) / len(data)
    for n in data:
        diffs += (n - avg)**2
    return (diffs/(len(data)-1))

print('Variance')
print(variance(height_weight_data['Height']))
print(variance(height_weight_data['Weight']))
print(height_weight_data['Height'].var())
print(height_weight_data['Weight'].var())

# Deviance
print('Deviance')
std_height = (variance(height_weight_data['Height'])) ** 0.5
std_weight = (variance(height_weight_data['Weight'])) ** 0.5
print(std_height)
print(std_weight)
print(height_weight_data['Height'].std())
print(height_weight_data['Weight'].std())

weight_mean = height_weight_data['Weight'].mean()
weight_std = height_weight_data['Weight'].std()

plt.figure(figsize=(12, 8))
height_weight_data['Weight'].hist(bins=20)
plt.axvline(weight_mean, color='r', label='mean')
plt.axvline(weight_mean - weight_std, color='g', label='1 standard deviation')
plt.axvline(weight_mean + weight_std, color='g', label='1 standard deviation')
plt.legend()
plt.show()

listOfSeries = [pd.Series(['Male', 40, 30], index=height_weight_data.columns ),
                pd.Series(['Female', 66, 37], index=height_weight_data.columns ),
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 202, 390], index=height_weight_data.columns ),
                pd.Series(['Female', 77, 210], index=height_weight_data.columns ),
                pd.Series(['Male', 88, 203], index=height_weight_data.columns )]

height_weight_updated = height_weight_data._append(listOfSeries , ignore_index=True)

plt.figure(figsize=(12, 8))
height_weight_updated['Weight'].hist(bins=100)
plt.show()

plt.figure(figsize=(12, 8))
height_weight_updated['Height'].hist(bins=100)
plt.show()

print('Q1&Q3 updated')
print(height_weight_updated['Height'].quantile(.25))
print(height_weight_updated['Height'].quantile(.75))
