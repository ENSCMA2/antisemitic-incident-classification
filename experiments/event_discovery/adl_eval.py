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
	cols = ["Type", "Percent Antisemitic", "Accuracy", "Precision", "Recall", "F1", "WF1", "n", "wn"]
	rows = []
	true_sheet = pd.read_csv("../../data/HeatMapAntisemitic.csv")
	pred_sheet = pd.read_csv(f"adl_results/{model}/{s}_{b}_{d}.csv")["response"].tolist()

	true_classes_rev = {1: 'Assault',
				        2: 'Harassment',
				        3: 'Vandalism'}
	true_classes = {true_classes_rev[k]: k for k in true_classes_rev}

	l2n = {"A": 0, "B": 1, "C": 2}
	n2l = {0: "A", 1: "B", 2: "C"}
	preds = []
	class_preds = []
	dupe_class_preds = []
	trues = []
	class_trues = []
	dupe_class_trues = []
	for (i, pred), (j, tem) in zip(enumerate(pred_sheet), true_sheet.iterrows()):
		if "llama" in model:
			if "dreams ideals virtues" in pred:
				response = {"type_of_incident": [0]}
			else:
				try:
					response = ast.literal_eval(ast.literal_eval(pred)[-1]["content"])
				except Exception as e:
					inner = ast.literal_eval(pred)[-1]["content"]
					print("unfixed")
					print(inner)
					if "}" not in inner:
						inner = inner + "}"
					inner = inner[inner.index("{"):inner.index("}") + 1]
					inner = inner.replace("'':", "''").replace(": '}", ": ''}").replace(",]", "]").replace("true", "True").replace("false", "False")
					inner = inner.replace(".}", ".'}").replace("\n", "").replace("\t", "").replace("     ", "")
					if "'reasoning':" in inner:
						inner = inner[:inner.index(",'reasoning'") + 1] + "}"
					if "'rationale':" in inner:
						inner = inner[:inner.index("'rationale'")] + "}"
					if "'justification':" in inner:
						inner = inner[:inner.index("'justification'")] + "}"
					if "'other_type_of_incident':" in inner:
						inner = inner[:inner.index("'other_type_of_incident'")] + "}"
					inner = inner.replace("],'}", "]}").replace("0, }", "0]}")
					print("fixed")
					print(inner)
					response = ast.literal_eval(inner)
			preds.append(1 if 0 not in response["type_of_incident"] else 0)
			rt = response["type_of_incident"]
			class_preds.append(rt)
		else:
			try:
				trunc = pred[pred.index("{"):pred.index("}") + 1]
				response = json.loads(trunc)
				preds.append(1 if 0 not in response["type_of_incident"] else 0)
				rt = response["type_of_incident"]
				class_preds.append(rt)
			except Exception as e:
				try:
					mp = pred.replace("]", "]}")
					trunc = mp[mp.index("{"):mp.index("}") + 1].replace("'", '"').replace("False", "false").replace("True", "true")
					response = json.loads(trunc)
					preds.append(1 if 0 not in response["type_of_incident"] else 0)
					rt = response["type_of_incident"]
					class_preds.append(rt)
				except Exception as e2:
					preds.append(0)
					rt = [0]
					class_preds.append(rt)
		trues.append(1)
		cs = tem["type"].split(";")
		tc = [true_classes[c.split(":")[-1].strip()] for c in cs if c.split(":")[-1].strip() in true_classes]
		class_trues.append(tc)
		for it in tc:
			dupe_class_trues.append(it)
			dupe_class_preds.append(rt)
	assert(len(dupe_class_preds) == len(dupe_class_trues))
	print(len(preds), len(trues))
	if b == "false":
		rows.append(["Overall", np.mean(preds), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
	else:
		rows.append(["Overall", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
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
				if other <= ccmp.shape[0]:
					ccmp[number - 1][other - 1] += 1
		p = np.array([number in pred for pred in class_preds]).astype(int)
		t = np.array([number in true for true in class_trues]).astype(int)
		true_pos = np.sum((p == 1) * (t == 1))
		false_pos = np.sum((p == 1) * (t == 0))
		true_neg = np.sum((p == 0) * (t == 0))
		false_neg = np.sum((p == 0) * (t == 1))
		prec = true_pos / (true_pos + false_pos)
		rec = true_pos / (true_pos + false_neg)
		pdd = np.array([number in pred for pred in dupe_class_preds]).astype(int)
		td = np.array([number == true for true in dupe_class_trues]).astype(int)
		this_row = [classif, np.mean(subpreds), accuracy_score(t, p), precision_score(t, p, pos_label = 1),
					 recall_score(t, p, pos_label = 1), f1_score(t, p, pos_label = 1), f1_score(td, pdd, pos_label = 1), np.mean(t), np.sum(td)]
		rows.append(this_row)
	def tm(n):
		return np.mean([r[n] for r in rows[rl:]])
	rows.append(["Type Mean",
				 tm(1), tm(2), tm(3), tm(4), tm(5), 
				 np.sum([r[6] * r[-1] for r in rows[rl:]]) / np.sum([r[-1] for r in rows[rl:]]), "N/A", tm(-1)])
	df = pd.DataFrame(data = rows, columns = cols)
	if not os.path.exists(f"adl_results/{model}"):
		os.mkdir(f"adl_results/{model}")
	df.to_csv(f"adl_results/{model}/sample_report_{s}_{b}_{d}.csv")
	if not os.path.exists(f"ccs/{model}"):
		os.mkdir(f"ccs/{model}")
	np.save(f"ccs/{model}/TRUE_{s}_{b}_{d}", ccmt)
	np.save(f"ccs/{model}/PRED_{s}_{b}_{d}", ccmp)
	plt.matshow(ccmt)
	plt.close()
	for (x, y), value in np.ndenumerate(ccmt):
		plt.text(x, y, f"{value}", va="center", ha="center", size = 7, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	plt.savefig(f"ccs/{model}/TRUE_{s}_{b}_{d}.png")
	plt.clf()
	plt.matshow(ccmp)
	plt.close()
	for (x, y), value in np.ndenumerate(ccmp):
		plt.text(x, y, f"{value}", va="center", ha="center", size = 7, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	plt.savefig(f"ccs/{model}/PRED_{s}_{b}_{d}.png")
	plt.clf()
	plt.close()

for model in sys.argv[1:]:
	report(0, "true", True, model)
	report(0, "true", False, model)
	report(0, "false", True, model)
	report(0, "false", False, model)
	report(1, "true", False, model)
