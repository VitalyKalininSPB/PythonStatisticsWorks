import os    # For file paths
import pandas as pd
from pandas import isnull

# Master.csv
master = pd.read_csv(os.path.join(".", "nhl", "Master.csv"))
print(master.head())
print(master.columns)

# How many null elements in column?
print(isnull(master["playerID"]).value_counts())

# Drop N/A
master_orig = master.copy()
master = master.dropna(subset=["playerID"])
print(master_orig.shape)
print(master.shape)
master = master.dropna(subset=["firstNHL", "lastNHL"], how="all")
print(master.shape)
master = master.loc[master["lastNHL"] >= 1980]
print(master.shape)

# Columns filtering
master = master.filter(regex="(playerID|pos|^birth)|(Name$)")
print(master.columns)

def mem_mib(df):
    print("{0:.2f} MiB".format(
        df.memory_usage().sum() / (1024 * 1024)
    ))

# Print memory usage
mem_mib(master)
mem_mib(master_orig)

def make_categorical(df, col_name):
    df.loc[:, col_name] = pd.Categorical(df[col_name])

# Make some columns categorical
make_categorical(master, "pos")
make_categorical(master, "birthCountry")
make_categorical(master, "birthState")

# Set new index
master = master.set_index("playerID")

master.to_pickle(os.path.join(".", "nhl", "master.pickle"))




# Scoring
scoring = pd.read_csv(os.path.join(".", "nhl", "Scoring.csv"))

def recent_nhl_only(df):
    return df[(df["lgID"] == "NHL") & (df["year"] >= 1980)]

scoring = recent_nhl_only(scoring)

# Remove some columns
scoring = scoring.filter(regex="^(?!(Post|PP|SH)).*")
scoring = scoring.iloc[:, [0, 1, 3, 6, 7, 8, 9, 14]]
print(scoring.columns)

make_categorical(scoring, "tmID")

scoring.reset_index(drop=True, inplace=True)
scoring.to_pickle(os.path.join(".", "nhl", "scoring.pickle"))


# Teams.csv
teams = pd.read_csv(os.path.join(".", "nhl", "Teams.csv"))
teams = recent_nhl_only(teams)
teams = teams[["year", "tmID", "name"]]
make_categorical(teams, "tmID")
teams.to_pickle(os.path.join(".", "nhl", "teams.pickle"))

# TeamSplits.csv
team_splits = pd.read_csv(os.path.join(".", "nhl", "TeamSplits.csv"))
team_splits = recent_nhl_only(team_splits)
cols_to_drop = team_splits.columns[3:11]
team_splits = team_splits.drop(columns=cols_to_drop)
team_splits = team_splits.drop(columns="lgID")
make_categorical(team_splits, "tmID")

team_splits.to_pickle(os.path.join(".", "nhl", "team_splits.pickle"))
