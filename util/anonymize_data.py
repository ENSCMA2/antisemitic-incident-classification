from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import pandas as pd
import os

df = pd.read_csv("../data/AMCHA/search_filtered_leniently_10_10_2024.csv")
analyzer = AnalyzerEngine()
engine = AnonymizerEngine()
anonymized_sd = []
anonymized_d = []
l = df.shape[0]

def anon(txt):
	return engine.anonymize(txt, [i for i in analyzer.analyze(text=txt, language="en") if i.entity_type == "PERSON"]).text

for i, tem in df.iterrows():
	print(i, l)
	sd = tem["Short Description"]
	d = tem["Description"]
	try:
		asd = anon(sd)
		ad = anon(d)
	except Exception as e:
		print(e)
		asd = sd
		ad = d
	anonymized_sd.append(asd)
	anonymized_d.append(ad)

df["Anonymized_Short_Description"] = anonymized_sd
df["Anonymized_Description"] = anonymized_d
df.to_csv("../data/AMCHA/anonymized_10_10_24.csv")