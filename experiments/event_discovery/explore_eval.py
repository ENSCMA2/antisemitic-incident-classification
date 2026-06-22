import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import os
import sys
import ast

def report(s, b, d, model):
	print(s, b, d)
	cols = ["Class/Category", "Percent Antisemitic", "n"]
	rows = []
	pred_sheet = pd.read_csv(f"amcha_results/{model}/{s}_{b}_{d}.csv")["response"].tolist()

	true_cats = {'antisemitic expression'.upper(): "B",
	            'targeting Jewish students and staff'.upper(): "C"}
	true_cats_rev = {true_cats[k]: k for k in true_cats}

	true_classes_rev = {1: 'physical assault'.upper(),
				        2: 'discrimination'.upper(),
				        3: 'destruction of Jewish property'.upper(),
				        4: 'genocidal expression'.upper(),
				        5: 'suppression of speech/movement/assembly'.upper(),
				        6: 'bullying'.upper(),
				        7: 'denigration'.upper(),
				        8: 'historical'.upper(),
				        9: 'condoning terrorism'.upper()}
	true_classes = {true_classes_rev[k]: k for k in true_classes_rev}

	l2n = {"A": 0, "B": 1, "C": 2}
	n2l = {0: "A", 1: "B", 2: "C"}
	preds = []
	cat_preds = []
	class_preds = []
	dupe_class_preds = []

	for (i, pred) in enumerate(pred_sheet):
		try:
			trunc = pred[pred.index("{"):pred.index("}") + 1]
			response = json.loads(trunc)
			preds.append(1 if response["category"] != "A" else 0)
			cat_preds.append(l2n[response["category"]])
			rt = response["type_of_expression"]
			class_preds.append(rt)
		except Exception as e:
			try:
				mp = pred.replace("]", "]}")
				trunc = mp[mp.index("{"):mp.index("}") + 1].replace("'", '"').replace("C,", '"C",').replace("D,", '"D",').replace("B,", '"B",').replace("A,", '"A",').replace("False", "false").replace("True", "true")
				response = json.loads(trunc)
				preds.append(1 if response["category"] != "A" else 0)
				try:
					cat_preds.append(l2n[response["category"]])
				except:
					cat_preds.append(l2n[true_cats[response["category"].upper()]])
				rt = response["type_of_expression"]
				class_preds.append(rt)
				dupe_class_preds.append(rt)
			except Exception as e2:
				preds.append(0)
				cat_preds.append(l2n["A"])
				rt = [0]
				class_preds.append(rt)
				dupe_class_preds.append(rt)
	print(set(preds))
	print(set(cat_preds))
	if b == "false":
		rows.append(["Overall", np.mean(preds), "N/A"])
	else:
		rows.append(["Overall", "N/A", "N/A"])

	for cat in [1, 2]:
		inds = [i for i in range(len(preds)) if cat_preds[i] == cat]
		subpreds = np.array(preds)[inds]
		print(cat, len(inds), Counter(np.array(cat_preds)[inds]))
		p = np.array(cat_preds) == cat
		rows.append([true_cats_rev[n2l[cat]], np.mean(subpreds), 
					 np.sum(p)])
	def lilmean(n):
		lst = [r[n] for r in rows[1:]]
		return np.mean(lst)
	rows.append(["Category Mean", 
				 lilmean(1),
				 "N/A"])
	rl = len(rows)
	ccmp = np.zeros((len(true_classes), len(true_classes))).astype(int)
	for classif in true_classes:
		inds = [i for i in range(len(preds)) if true_classes[classif] in class_preds[i]]
		subpreds = np.array(preds)[inds]
		number = true_classes[classif]
		for ind in inds:
			for other in class_preds[ind]:
				ccmp[number - 1][other - 1] += 1
		p = np.array([number in pred for pred in class_preds]).astype(int)
		rows.append([classif, np.mean(subpreds), np.sum(p)])
	def tm(n):
		return np.mean([r[n] for r in rows[rl:]])
	rows.append(["Type Mean",
				 tm(1), "N/A"])
	df = pd.DataFrame(data = rows, columns = cols)
	if not os.path.exists(f"amcha_results/{model}"):
		os.mkdir(f"amcha_results/{model}")
	df.to_csv(f"amcha_results/{model}/sample_report_{s}_{b}_{d}.csv")


# for model in sys.argv[1:]:
# 	report(0, "true", True, model)
# 	report(0, "true", False, model)
# 	report(0, "false", True, model)
# 	report(0, "false", False, model)
# 	report(1, "true", False, model)
for paper in os.listdir("/Users/karinahalevy/Documents/Git/scraping-college-newspapers/res"):
	if paper != ".DS_Store":
		report(0, "false", True, f"news-gpt-4o-{paper}")
