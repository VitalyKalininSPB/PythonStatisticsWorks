import pandas as pd
import os

master = pd.read_pickle(os.path.join("nhl", "master.pickle"))
scoring = pd.read_pickle(os.path.join("nhl", "scoring.pickle"))
team_splits = pd.read_pickle(os.path.join("nhl", "team_splits.pickle"))

mi = scoring.set_index(['playerID', 'year'])
print(type(mi))
print(mi.index)
print(len(mi.index.levels))
print(mi.index.levels[0])
print(mi.index.levels[1])

print(mi.groupby(level="year")['G'].max().head())
new_table = mi.loc[mi.groupby(level="year")['G'].idxmax()]
print(new_table["G"])
print(mi.loc[('gretzwa01', 1982), :])

# Slices
idx = pd.IndexSlice
print(mi.index._is_lexsorted())
sliced = mi.loc[idx[:, 1999:2000], :]
print(sliced.head())

locs = mi.index.get_locs((
    idx['aaltoan01':'adamscr01', 1998:2000]
))
print(locs)
sliced = mi.iloc[locs, :]
print(sliced.head())

import numpy as np

def get_many_locs(df, slices):
    arr = np.empty(0, dtype="int")
    for s in slices:
        locs = df.index.get_locs((s))
        arr = np.concatenate((arr, locs))
    return arr

locs = get_many_locs(
           mi,
           (
               idx['aaltoan01':'adamscr01', 1997:2000],
               idx['aaltoan01':'adamscr01', 2004:2006]
           ))

sliced = mi.iloc[locs, :]
print(sliced.head(10))

# Resort
mi = mi.sort_index(level='year')
print(mi.head())

swapped = mi.swaplevel()
print("Swapped Data")
print(swapped.head())
