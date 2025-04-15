import pandas as pd
import os

# Where our data lives
CSV_PATH = os.path.join('.', 'collection-master',
                        'artwork_data.csv')

# Specify an Index
df = pd.read_csv(CSV_PATH, index_col='id')

# Save for later
df.to_pickle(os.path.join('.', 'data_frame.pickle'))

# Let's load the data for the first time
df = pd.read_pickle(os.path.join('.', 'data_frame.pickle'))

# How many unique artists in collection?
artists = df['artist']
pd.unique(artists)
print(len(pd.unique(artists)))
print(df.shape)

# How many artworks by Francis Bacon?
artworks = df[df['artist'] == 'Bacon, Francis']
print(len(artworks))
# Another way
s = df['artist'] == 'Bacon, Francis'
print(s.value_counts())
print(type(artworks))
print(type(s))
# Another way
artist_counts = df['artist'].value_counts()
artist_counts['Bacon, Francis']

# What is artwork with biggest dimensions?
# Try multiplication
# df['height'] * df['width']
# df['width'].sort_values().head()
# df['width'].sort_values().tail()
# Try to convert
# pd.to_numeric(df['width'])
# Force NaNs
df.loc[:, 'width'] = pd.to_numeric(df['width'], errors='coerce')
df.loc[:, 'height'] = pd.to_numeric(df['height'], errors='coerce')
df['height'] * df['width']
# Check that all records measured in one unit
print(df['units'].value_counts())

# Assign - create new columns with size
area = df['height'] * df['width']
df = df.assign(area=area)

df['area'].max()
df['area'].idxmax()
print(df.loc[df['area'].idxmax(), :])

# Save for later
df.to_pickle(os.path.join('.', 'data_frame.pickle'))
