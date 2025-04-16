import pandas as pd
import os

scoring = pd.read_pickle(os.path.join("nhl", "scoring.pickle"))
team_splits = pd.read_pickle(os.path.join("nhl", "team_splits.pickle"))

three_years = scoring.loc[(scoring.year > 2000) &
                          (scoring.year < 2004)]

# One record per player/year
three_years = three_years.drop_duplicates(subset=["playerID", "year"])

# Get three random players with complete history between 2001-2003
counts = three_years["playerID"].value_counts()
# print(counts.head(18))
ids = counts[counts == 3].sample(3).index

# Get actual records corresponding for those players
df3 = three_years.loc[three_years['playerID'].isin(ids)]
basic_df3 = df3[["playerID", "year", "G"]]

print(basic_df3)

pivot = basic_df3.pivot(index="playerID", columns="year", values="G")
pivot.index.name = "playerID"
pivot.columns.name = "year"

print(pivot)
print(pivot.columns)

# Reverse operation - melt
pivot = pivot.reset_index()
pivot.columns.name = None
print(pivot.melt(id_vars="playerID", var_name="year", value_vars=[2001, 2002], value_name="goals"))

# Two or more values
larger_df3 = df3[["playerID", "year", "G", "A"]]
print(larger_df3)

# Multiindex
test = larger_df3.pivot(index="playerID", columns="year",
                        values=["G", "A"])
print(test)
print(test.columns)

import matplotlib.pyplot as plt
plt.show()
