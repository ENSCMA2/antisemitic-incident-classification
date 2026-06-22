from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import pandas as pd
import os

base_path = "/Users/karinahalevy/Documents/Git/scraping-college-newspapers/res/"
for paper in os.listdir(base_path):
	print(paper)
	if paper == ".DS_Store":
		continue
	if os.path.exists(f"{base_path}/{paper}/articles_scraped.csv"):
		df = pd.read_csv(f"{base_path}/{paper}/articles_scraped.csv")
	else:
		df = pd.read_csv(f"{base_path}/{paper}/articles.csv")
	analyzer = AnalyzerEngine()
	engine = AnonymizerEngine()
	anonymized_sd = []
	anonymized_d = []
	l = df.shape[0]

	def anon(txt):
		return engine.anonymize(txt, [i for i in analyzer.analyze(text=txt, language="en") if i.entity_type == "PERSON"]).text

	for i, tem in df.iterrows():
		print(i, l)
		d = f'{tem["title"]}\n{tem["article_text"]}'
		try:
			ad = anon(d)
		except Exception as e:
			print(e)
			ad = d
		anonymized_d.append(ad)

	df["anonymized_description"] = anonymized_d
	df.to_csv(f"{base_path}/{paper}/anon.csv")