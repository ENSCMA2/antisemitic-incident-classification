import pandas as pd
import json
import os

src_path = "/Users/karinahalevy/Documents/Git/scraping-college-newspapers/res/"
for paper in os.listdir(src_path):
	if ".DS_Store" in paper:
		continue
	src_df = pd.read_csv(f"{src_path}/{paper}/anon.csv")
	tgt_df = pd.read_csv(f"amcha_results/news-gpt-4o-{paper}/0_false_True.csv")
	anon = pd.concat([src_df, tgt_df], axis = 1)
	anon = anon[(anon["anonymized_description"].str.contains("Israel"))
				| (anon["anonymized_description"].str.contains("Palestine"))
				| (anon["anonymized_description"].str.contains("Zionism"))
				| (anon["anonymized_description"].str.contains("Zionist"))
				| (anon["anonymized_description"].str.contains("Gaza"))
				| (anon["anonymized_description"].str.contains("Jew"))
				| (anon["anonymized_description"].str.contains("Hillel"))
				| (anon["anonymized_description"].str.contains("Swastika"))
				| (anon["anonymized_description"].str.contains("Nazi"))
				| (anon["anonymized_description"].str.contains("semitic"))
				| (anon["anonymized_description"].str.contains("semitism"))]
	anon.to_csv(f"amcha_results/news-gpt-4o-{paper}/0_false_True_filtered.csv")