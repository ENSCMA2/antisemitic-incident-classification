import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import os
import sys
import ast

def report(uni):
	print(uni)
	cols = ["Class/Category", "Percent Antisemitic", "Accuracy", "Precision", "Recall", "F1", "WF1", "n", "wn"]
	rows = []
	sheet = pd.read_csv(f"wild_results/{uni}_res.csv")
	print(sheet.shape)
	sheet = sheet[sheet["classifications"].str.len() > 0]
	print(sheet.shape)

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
	trues = []
	cat_trues = []
	class_trues = []
	dupe_class_trues = []
	for i, tem in sheet.iterrows():
		bin_response = tem["binary response"]
		category = "A"
		for ltr, cat in [("B", "cat_AS"), ("C", "cat_TRG")]:
			if tem[cat]:
				category = ltr
				continue

		preds.append(bin_response)
		cat_preds.append(l2n[category])
		rt = []
		for i, cls_ in enumerate([c for c in sheet.columns if "cls" in c]):
			if tem[cls_]:
				rt.append(i + 1)

		class_preds.append(rt)
		print(tem["category"])
		trues.append(tem["category"] is not None and type(tem["category"]) == str and tem["category"].lower() != "none")
		if type(tem["category"]) == str:
			cat_trues.append(l2n[true_cats[tem["category"].upper()]])
		else:
			cat_trues.append(l2n["A"])
		cs = tem["classifications"].split(", ")
		tc = [true_classes[c.upper()] for c in cs if c.upper() in true_classes]
		class_trues.append(tc)
		for it in tc:
			dupe_class_trues.append(it)
			dupe_class_preds.append(rt)

	assert(len(dupe_class_preds) == len(dupe_class_trues))
	rows.append(["Overall", np.mean(preds), accuracy_score(trues, preds), 
				 precision_score(trues, preds), recall_score(trues, preds), 
				 f1_score(trues, preds), "N/A", "N/A", "N/A"])
	cm = confusion_matrix(cat_trues, cat_preds, labels = [0, 1, 2, 3])
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
	disp.plot()
	if not os.path.exists(f"figs/wild"):
		os.mkdir(f"figs/wild")
	plt.savefig(f"figs/wild/{uni}.png")
	plt.clf()
	for cat in [1, 2]:
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
	df.to_csv(f"wild_results/report_{uni}.csv")
	np.save(f"ccs/TRUE_{uni}", ccmt)
	np.save(f"ccs/PRED_{uni}", ccmp)
	plt.matshow(ccmt)
	plt.close()
	for (x, y), value in np.ndenumerate(ccmt):
		plt.text(x, y, f"{value}", va="center", ha="center", size = 7, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	plt.savefig(f"ccs/TRUE_{uni}.png")
	plt.clf()
	plt.matshow(ccmp)
	plt.close()
	for (x, y), value in np.ndenumerate(ccmp):
		plt.text(x, y, f"{value}", va="center", ha="center", size = 7, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
	plt.savefig(f"ccs/PRED_{uni}.png")
	plt.clf()
	plt.close()

for uni in ["harvard", "columbia", "uiuc", "stanford", "umich"]:
	report(uni)
