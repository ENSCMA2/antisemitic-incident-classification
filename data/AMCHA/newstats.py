import pandas as pd
from datetime import datetime
df = pd.read_csv("searchincidents.csv")

rn = datetime.now()
day = rn.day
month = rn.month
year = rn.year

df = df[~(df["Classification"].str.contains("BDS ACTIVITY"))]
print("Without BDS Activity", df.shape)
df = df[~(df["Classification"].str.contains("DEMONIZATION"))]
print("Without BDS Activity or Demonization", df.shape)
df = df[~(df["Classification"].str.contains("DENYING JEWS SELF-DETERMINATION"))]
print("Without BDS Activity, Demonization, or Denying Jews Self-Determination", df.shape)

df.to_csv(f"search_filtered_stringently_{month}_{day}_{year}.csv")

df = pd.read_csv("searchincidents.csv")

df = df[(~(df["Classification"] == "BDS ACTIVITY"))
		& (~(df["Classification"] == "BDS ACTIVITY, BDS ACTIVITY"))]
print("Without solely BDS Activity", df.shape)
df = df[(~(df["Classification"] == "DEMONIZATION"))
        & (~(df["Classification"] == "BDS ACTIVITY, DEMONIZATION"))
        & (~(df["Classification"] == "DEMONIZATION, BDS ACTIVITY"))]
print("Without solely BDS Activity or Demonization", df.shape)
df = df[(~(df["Classification"] == "DENYING JEWS SELF-DETERMINATION"))
		& (~(df["Classification"] == "BDS ACTIVITY, DEMONIZATION, DENYING JEWS SELF-DETERMINATION"))
		& (~(df["Classification"] == "BDS ACTIVITY, DENYING JEWS SELF-DETERMINATION, DEMONIZATION"))
		& (~(df["Classification"] == "DENYING JEWS SELF-DETERMINATION, BDS ACTIVITY, DEMONIZATION"))
		& (~(df["Classification"] == "DENYING JEWS SELF-DETERMINATION, DEMONIZATION, BDS ACTIVITY"))
		& (~(df["Classification"] == "DEMONIZATION, DENYING JEWS SELF-DETERMINATION, BDS ACTIVITY"))
		& (~(df["Classification"] == "DEMONIZATION, BDS ACTIVITY, DENYING JEWS SELF-DETERMINATION"))
		& (~(df["Classification"] == "DEMONIZATION, DENYING JEWS SELF-DETERMINATION"))
		& (~(df["Classification"] == "DENYING JEWS SELF-DETERMINATION, DEMONIZATION"))
		& (~(df["Classification"] == "BDS ACTIVITY, DENYING JEWS SELF-DETERMINATION"))
		& (~(df["Classification"] == "DENYING JEWS SELF-DETERMINATION, BDS ACTIVITY"))]
print("Without solely BDS Activity, Demonization, or Denying Jews Self-Determination", df.shape)

df.to_csv(f"search_filtered_leniently_{month}_{day}_{year}.csv")

