import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import os
import sys
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

def report(s, b, d, paras, model):
	print(s, b, d)
	cols = ["Class/Category", "Percent Antisemitic", "Accuracy", "Precision", "Recall", "F1", "WF1", "n", "wn"]
	true_sheet = pd.read_csv("../../data/AMCHA/anonymized_sample.csv")
	true_cats = {'antisemitic expression'.upper(): "B",
	            'targeting Jewish students and staff'.upper(): "C",
	            'BDS activity'.upper(): "D"}
	true_cats_rev = {true_cats[k]: k for k in true_cats}
	true_classes = {'SUPPRESSION OF SPEECH/MOVEMENT/ASSEMBLY': 5, 
					'CALLS FOR BDS': 12, 
					'BDS EVENT': 13, 
					'BDS VOTE': 14, 
					'DENIGRATION': 7, 
					'DENYING JEWS SELF-DETERMINATION': 10, 
					'CONDONING TERRORISM': 9, 
					'HISTORICAL': 8, 
					'BULLYING': 6, 
					'DESTRUCTION OF JEWISH PROPERTY': 3, 
					'PHYSICAL ASSAULT': 1, 
					'GENOCIDAL EXPRESSION': 4, 
					'DEMONIZATION': 11, 
					'DISCRIMINATION': 2}
	true_classes_rev = {true_classes[k]: k for k in true_classes}

	l2n = {"A": 0, "B": 1, "C": 2, "D": 3}
	n2l = {0: "A", 1: "B", 2: "C", 3: "D"}

	cat_reldata = []
	class_reldata = {}
	reldata = []
	for para in paras:
		rows = []
		pred_sheet = pd.read_csv(f"results/amcha_{model}_{s}_{b}_{d}_{para}_vanilla.csv")["response"].tolist()
		preds = []
		cat_preds = []
		class_preds = []
		dupe_class_preds = []
		trues = []
		cat_trues = []
		class_trues = []
		dupe_class_trues = []
		for (i, pred), (j, tem) in zip(enumerate(pred_sheet), true_sheet.iterrows()):
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
				except Exception as e2:
					preds.append(0)
					cat_preds.append(l2n["A"])
					rt = [0]
					class_preds.append(rt)
			trues.append(1)
			cat_trues.append(l2n[true_cats[tem["Category"]]])
			tc = [true_classes[c] for c in tem["Classification"].split(", ")]
			class_trues.append(tc)
			for it in tc:
				dupe_class_trues.append(it)
				dupe_class_preds.append(rt)
		assert(len(dupe_class_preds) == len(dupe_class_trues))
		cat_reldata.append(cat_preds)
		reldata.append(preds)
		if b == "false":
			rows.append(["Overall", np.mean(preds), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
		else:
			rows.append(["Overall", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
		for cat in [1, 2, 3]:
			inds = [i for i in range(len(trues)) if cat_trues[i] == cat]
			subpreds = np.array(preds)[inds]
			p = np.array(cat_preds) == cat
			t = np.array(cat_trues) == cat
			rows.append([true_cats_rev[n2l[cat]], np.mean(subpreds), 
						 accuracy_score(t, p), precision_score(t, p),
						 recall_score(t, p), f1_score(t, p), "N/A", np.mean(t), "N/A"])
		def lilmean(n):
			lst = [r[n] for r in rows[1:]]
			return np.mean(lst)
		rows.append(["Category Mean", 
					 lilmean(1), lilmean(2), lilmean(3), lilmean(4), lilmean(5),
					 np.sum([r[5] * r[7] for r in rows[1:]]),
					 "N/A",
					 "N/A"])
		rl = len(rows)
		ccmt = np.zeros((len(true_classes), len(true_classes))).astype(int)
		ccmp = np.zeros((len(true_classes), len(true_classes))).astype(int)
		for classif in true_classes:
			inds = [i for i in range(len(trues)) if true_classes[classif] in class_trues[i]]
			subpreds = np.array(preds)[inds]
			number = true_classes[classif]
			for ind in inds:
				for other in class_trues[ind]:
					ccmt[number - 1][other - 1] += 1
				for other in class_preds[ind]:
					ccmp[number - 1][other - 1] += 1
			p = np.array([number in pred for pred in class_preds]).astype(int)
			try:
				class_reldata[classif].append(p.tolist())
			except:
				class_reldata[classif] = [p.tolist()]
			t = np.array([number in true for true in class_trues]).astype(int)
			pdd = np.array([number in pred for pred in dupe_class_preds]).astype(int)
			td = np.array([number == true for true in dupe_class_trues]).astype(int)
			rows.append([classif, np.mean(subpreds), accuracy_score(t, p), precision_score(t, p),
						 recall_score(t, p), f1_score(t, p), f1_score(td, pdd), np.mean(t), np.sum(td)])
		def tm(n):
			return np.mean([r[n] for r in rows[rl:]])
		rows.append(["Type Mean",
					 tm(1), tm(2), tm(3), tm(4), tm(5), 
					 np.sum([r[6] * r[-1] for r in rows[rl:]]) / np.sum([r[-1] for r in rows[rl:]]), "N/A", tm(-1)])
		df = pd.DataFrame(data = rows, columns = cols)
		if not os.path.exists(f"results/{model}"):
			os.mkdir(f"results/{model}")
		df.to_csv(f"results/{model}/sample_report_{s}_{b}_{d}_{para}.csv")
	def ka(rd):
		try:
			return krippendorff.alpha(rd)
		except:
			return 100
	cat_reldata = np.array(cat_reldata).astype(int)
	reldata = np.array(reldata).astype(int)
	jsn = {"krippendorff_alpha": {"binary": ka(reldata) if b == "false" else "N/A",
								  "categories": ka(cat_reldata)}}
	def pct_ag(arr):
		tarr = arr.T
		print("tarr shape", tarr.shape)
		points = 0
		for sarr in tarr:
			yeah = sarr[0]
			getpt = True
			for element in sarr[1:]:
				if element != yeah:
					getpt = False
			points += getpt
		return points / tarr.shape[0]
	print("sum", np.sum(reldata))
	jsn["percent_agreement"] = {"binary": pct_ag(reldata) if b == "false" else "N/A",
								"categories": pct_ag(cat_reldata)}
	def fk(arr):
		narr = np.array(arr).T
		agg, useless = aggregate_raters(narr)
		f = fleiss_kappa(agg)
		if np.isnan(f):
			return 100
		return f
	jsn["fleiss_kappa"] = {"binary": fk(reldata) if b == "false" else "N/A",
						   "categories": fk(cat_reldata)}
	for key in class_reldata:
		class_reldata[key] = np.array(class_reldata[key]).astype(int)
		jsn["krippendorff_alpha"][f"class_{key}"] = krippendorff.alpha(class_reldata[key])
		jsn["percent_agreement"][f"class_{key}"] = pct_ag(class_reldata[key])
		jsn["fleiss_kappa"][f"class_{key}"] = fk(class_reldata[key])

	with open(f"results/agreement_{s}_{b}_{d}.json", "w") as o:
		json.dump(jsn, o)

	cols = ["", "fleiss_kappa", "percent_agreement", "krippendorff_alpha"]
	rows = []
	for key in jsn["fleiss_kappa"]:
		rows.append([key, jsn["fleiss_kappa"][key], jsn["percent_agreement"][key], jsn["krippendorff_alpha"][key]])
	adf = pd.DataFrame(columns = cols, data = rows)
	adf.to_csv(f"results/agreement_{s}_{b}_{d}.csv")

paras = [0, 1, 2, 3, 4]
for model in sys.argv[1:]:
	report(0, "true", True, paras, model)
	report(0, "true", False, paras, model)
	report(0, "false", True, paras, model)
	report(0, "false", False, paras, model)
