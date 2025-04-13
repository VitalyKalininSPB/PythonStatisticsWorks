import pandas as pd
import os

# CSV to Pickles example
# Where our data lives
CSV_PATH = os.path.join('collection-master',
                        'artwork_data.csv')

# Read just 5 rows to see what's there
df = pd.read_csv(CSV_PATH, nrows=5)

# Specify an Index
df = pd.read_csv(CSV_PATH, nrows=5,
                 index_col='id')
# Limit columns
df = pd.read_csv(CSV_PATH, nrows=5,
                 index_col='id',
                 usecols=['id', 'artist'])

# All columns that we need
COLS_TO_USE = ['id', 'artist',
               'title', 'medium', 'year',
               'acquisitionYear', 'height',
               'width', 'units']

# Proper data loading
df = pd.read_csv(CSV_PATH,
                 usecols=COLS_TO_USE,
                 index_col='id')

# Save for later
df.to_pickle(os.path.join('.', 'data_frame.pickle'))

pd.set_option('display.max_columns', 17)

# Let's load the data for the first time
df = pd.read_pickle('data_frame.pickle')
#print(df.head())
#print(len(df))
#print('Unique artists: ' + str(len(df['artist'].unique())))
#print(df['artist'].value_counts())

df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['height'] = pd.to_numeric(df['height'], errors='coerce')

# Assign - create new columns with size
df['area'] = df['height'] * df['width']


# ITERATION
#small_df = df.iloc[49980:50019, :].copy()
df = df.iloc[49080:50019, :].copy()
#grouped = small_df.groupby('artist')
#print(type(grouped))

#for name, group_df in grouped:
#    print(name)
#    print(group_df[['artist', 'title']])
#    break

# Aggregate
# Mins
#for name, group_df in small_df.groupby('artist'):
#    min_year = group_df['acquisitionYear'].min()
#    print("{}: {}".format(name, min_year))

# Frequent filling
def fill_values(series):
    #print(f"fill value {series}")
    values_counted = series.value_counts()
    if values_counted.empty:
        print(f"Empty fill value {series}")
        return series
    most_frequent = values_counted.index[0]
    #print(f"Most frequent {most_frequent}")
    new_medium = series.fillna(most_frequent)
    return new_medium

def transform_df(source_df):
    group_dfs = []
    for name, group_df in source_df.groupby('artist'):
        filled_df = group_df.copy()
        filled_df.loc[:, 'medium'] = fill_values(group_df['medium'])
        group_dfs.append(filled_df)

    new_df = pd.concat(group_dfs)
    return new_df

filled_df = df.copy()
grouped_mediums = df.groupby('artist')['medium']
print(f"Before: {filled_df['medium'].isna().sum()}")
filled_df.loc[:, 'medium'] = grouped_mediums.transform(fill_values)
print(f"After: {filled_df['medium'].isna().sum()}")

#filled_df = transform_df(small_df)
#print(filled_small_df[['artist','title','medium']])

#print(df[['artist','title','medium']])

# Compare differences
# !!! DON'T ITERATE OVER PANDAS DATAFRAME - JUST FOR TUTORIAL
for index, row in filled_df.iterrows():
    if (row['medium'] != df.loc[index]['medium']):
        print(row[['artist','medium']])

# Filter
grouped_titles = df.groupby('title')

condition = lambda x: len(x.index) > 1
dup_titles_df = grouped_titles.filter(condition)
dup_titles_df.sort_values('title', inplace=True)
print(len(grouped_titles))
print(len(dup_titles_df))
print(f"percent: {len(dup_titles_df)/len(grouped_titles)*100:.3f}")


print(df.loc[15852])
