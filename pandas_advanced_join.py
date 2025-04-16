import pandas as pd
import os

master = pd.read_pickle(os.path.join("nhl", "master.pickle"))
scoring = pd.read_pickle(os.path.join("nhl", "scoring.pickle"))
teams = pd.read_pickle(os.path.join("nhl", "teams.pickle"))
team_splits = pd.read_pickle(os.path.join("nhl", "team_splits.pickle"))

'''
# Left right join column drops as index, pandas takes right index as default for merged table
scoring.index = scoring.index + 3
pd.merge(master, scoring, left_index=True, right_on="playerID").head()

# Index is playerID now
pd.merge(master, scoring.set_index("playerID", drop=True),
                                   left_index=True, right_index=True)

scoring = scoring.reset_index(drop=True)
pd.merge(master, scoring, left_index=True, right_on="playerID")

print(master)

# Drop random records
master2 = master.drop(master.sample(5).index)

# Left and right / both analysis
merged = pd.merge(master2, scoring, left_index=True,
                  right_on="playerID", how="right", indicator=True)
# Look at Indicator column
print(merged["_merge"].value_counts())

merged = merged.filter(regex="^(?!(birth)).*")
print(merged)
merged.to_pickle(os.path.join("nhl", "scoring.pickle"))'''


# Different names for one ID
teams2 = teams[["tmID", "name"]]
teams2 = teams2.drop_duplicates()
print(teams2["tmID"].value_counts())
# Attention: CHI == 2
print(teams2.loc[teams2["tmID"] == "CHI"])
# Different names

# Experiment: 2 records from TeamSplits and join with CHI
teams2 = teams2[teams2["tmID"] == "CHI"]
team_splits2 = team_splits[team_splits["tmID"] == "CHI"].sample(2)
print(team_splits2)
# Bad results!
print(pd.merge(teams2, team_splits2))

# Multiple indexes join
print('Multiple indexes join')
merged = pd.merge(team_splits, teams,
                        left_on=["tmID", "year"],
                        right_on=["tmID", "year"])
print(merged)

merged.to_pickle(os.path.join("nhl", "team_splits.pickle"))
