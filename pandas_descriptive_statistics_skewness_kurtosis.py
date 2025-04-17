import pandas as pd
import matplotlib.pyplot as plt

height_weight_data = pd.read_csv('datasets/500_Person_Gender_Height_Weight_Index.csv')
height_weight_data.drop('Index', inplace=True, axis=1)

height_weight_data[['Height']].plot(kind = 'kde',
                                    title = 'Height', figsize=(12, 8))

height_weight_data[['Weight']].plot(kind = 'kde',
                                    title = 'Height', figsize=(12, 8))
plt.show()

print(height_weight_data['Height'].skew())
print(height_weight_data['Weight'].skew())

listOfSeries = [pd.Series(['Male', 400, 300], index=height_weight_data.columns ),
                pd.Series(['Female', 660, 370], index=height_weight_data.columns ),
                pd.Series(['Female', 199, 410], index=height_weight_data.columns ),
                pd.Series(['Male', 202, 390], index=height_weight_data.columns ),
                pd.Series(['Female', 770, 210], index=height_weight_data.columns ),
                pd.Series(['Male', 880, 203], index=height_weight_data.columns )]

height_weight_updated = height_weight_data._append(listOfSeries , ignore_index=True)
print(height_weight_updated['Height'].skew())
print(height_weight_updated['Weight'].skew())

height_weight_updated[['Weight']].plot(kind = 'hist', bins=100,
                                       title = 'weight', figsize=(12, 8))
plt.show()

height_weight_updated[['Height']].plot(kind = 'kde',
                                       title = 'Height', figsize=(12, 8))
plt.show()

### Kurtosis
print("Kurtosis")
print(height_weight_data['Height'].kurtosis())
print(height_weight_data['Weight'].kurtosis())
print(height_weight_updated['Height'].kurtosis())
print(height_weight_updated['Weight'].kurtosis())
