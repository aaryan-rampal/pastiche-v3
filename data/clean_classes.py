# %%
import os
import pandas as pd

# %%
DATA_DIR = "/Volumes/Extreme SSD/wikiart"
# %%
df = pd.read_csv("classes.csv")

# %%
exists = []
for row in df.itertuples():
    filename = row.filename
    full_path = os.path.join(DATA_DIR, filename)
    exists.append(os.path.exists(full_path))
df["exists"] = exists

# %%
# sample 4000 rows where exists is True
df_filtered = df[df["exists"] == True]
df_sampled = df_filtered.sample(n=4000, random_state=42)
print("Sampled number of rows:", len(df_sampled))

# %%
# checking unique phash in sampled data
print("Unique phash in sampled data:", df_sampled["phash"].nunique())
print("Sampled number of rows:", len(df_sampled))
# %%
# %%
# make start of artist names uppercase
df_sampled["artist"] = df_sampled["artist"].str.title()
df_sampled["artist"].value_counts()
# %%
df_sampled["genre"] = df_sampled["genre"].str.title()
df_sampled["genre"].value_counts()

# %%
df_sampled.to_csv("classes_truncated.csv", index=False)

# %%
