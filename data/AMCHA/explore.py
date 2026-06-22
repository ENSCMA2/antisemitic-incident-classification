import pandas as pd

sheet = pd.read_csv("anonymized_checked.csv")

data = []
cats = ["ANTISEMITIC EXPRESSION", "TARGETING JEWISH STUDENTS AND STAFF",
		"BDS ACTIVITY"]
for year in range(2015, 2025):
	df = sheet[sheet["Date"].str.contains(f"/{year}")]
	row = [year]
	for cat in cats:
		c = df[df["Category"] == cat].shape[0] / df.shape[0]
		row.append(c)
	data.append(row)

stats = pd.DataFrame(data = data, columns = ["Year"] + cats)
stats.to_csv("stats.csv")
