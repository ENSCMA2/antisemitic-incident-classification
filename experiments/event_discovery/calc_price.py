import pandas as pd
import os

total = 0
for d in os.listdir("amcha_results"):
	if "." not in d:
		for sd in os.listdir(f"amcha_results/{d}"):
			if ".csv" in sd and "sample" not in sd and "filtered" not in sd:
				print(d, sd)
				df = pd.read_csv(f"amcha_results/{d}/{sd}")
				in_sum = sum(df["in_tokens"].to_list())
				out_sum = sum(df["out_tokens"].to_list())
				price = in_sum * 2.5 / 1000000 + out_sum * 10 / 1000000
				total += price
				print(price)
for d in os.listdir("adl_results"):
	if "." not in d:
		for sd in os.listdir(f"adl_results/{d}"):
			if ".csv" in sd and "sample" not in sd and "filtered" not in sd:
				print(d, sd)
				df = pd.read_csv(f"adl_results/{d}/{sd}")
				in_sum = sum(df["in_tokens"].to_list())
				out_sum = sum(df["out_tokens"].to_list())
				price = in_sum * 2.5 / 1000000 + out_sum * 10 / 1000000
				total += price
				print(price)

for f in ["../../data/TOXIGEN/control_adl.csv", 
		  "../../data/TOXIGEN/prompts/controlprompts.csv"]:
	df = pd.read_csv(f)
	in_sum = sum(df["in_tokens"].to_list())
	out_sum = sum(df["out_tokens"].to_list())
	price = in_sum * 2.5 / 1000000 + out_sum * 10 / 1000000
	print(f, price)
	total += price

print(total)
