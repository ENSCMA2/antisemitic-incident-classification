import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import os
import sys
import ast

def control_report(s, b, d, model):
	print(s, b, d)
	cols = ["Class/Category", "Percent Antisemitic"]
	rows = []
	true_sheet = pd.read_csv("../../data/TOXIGEN/prompts/controlprompts_4410.csv")
	pred_sheet = pd.read_csv(f"amcha_results/control-{model}/{s}_{b}_{d}.csv")["response"].tolist()
	raw_preds = []

	true_cats = {'antisemitic expression'.upper(): "B",
	            'targeting Jewish students and staff'.upper(): "C"}
	true_cats_rev = {true_cats[k]: k for k in true_cats}

	l2n = {"A": 0, "B": 1, "C": 2}
	n2l = {0: "A", 1: "B", 2: "C"}
	preds = []
	trues = []
	for (i, pred), (j, tem) in zip(enumerate(pred_sheet), true_sheet.iterrows()):
		try:
			response = ast.literal_eval(ast.literal_eval(pred)[-1]["content"])
		except:
			fixed = ast.literal_eval(pred)[-1]["content"].replace("'s", "s").replace("s' ", "s ").replace("'''", "''")
			try:
				fixed = fixed[fixed.index("{"):fixed.index("}") + 1]
				response = ast.literal_eval(fixed)
			except Exception as e:
				try:
					fixed = fixed[fixed.index("{"):fixed.index("other_category")] + "}"
					fixed = fixed.replace("'}", "}").replace('"}', '}')
				except Exception as e:
					fixed = fixed.replace("'}", "}").replace('"}', '}')
					# fixed = {"category": "A", "type_of_expression": []}
				if "I can't help you with that" in fixed:
					response = {"category": "A", "type_of_expression": []}
				elif "Can I help you with something else" in fixed or "I can" in fixed:
					response = {"category": "A", "type_of_expression": []}
				else:
					f = fixed.replace("\'", '\"').replace("\n", "").replace(",}", "}").replace(", }", "}").replace("A,", '"A",').replace(",]", "]").replace("B,", '"B",').replace("C,", '"C",')
					f = f.replace("8,}", "8]}")
					if "}" not in f:
						f = f + "}"
					# print("??")
					f = f.replace("8,}", "8]}").replace(" ", "").replace(",}", "}").replace("0}", "0]}").replace(":0]}", ":[0]}").replace('[8,"', '[8],"')
					f = f.replace('traditions"]', 'traditions"')
					# print(f)
					response = json.loads(f)
		if b == "true":
			preds.append(1 if  "A" not in response["category"].strip() else 0)
		else:
			preds.append(1 if "describes_antisemitic_event" in response.keys() and response["describes_antisemitic_event"] else 0)
		raw_preds.append(response["category"].strip())
		trues.append(0)
	print(Counter(raw_preds))
	print(Counter(preds))
	rows.append(["Overall", np.mean(preds)])
	df = pd.DataFrame(data = rows, columns = cols)
	if not os.path.exists(f"amcha_results/{model}"):
		os.mkdir(f"amcha_results/control-{model}")
	df.to_csv(f"amcha_results/control-{model}/sample_report_{s}_{b}_{d}.csv")

def report(s, b, d, model):
	print(s, b, d)
	cols = ["Class/Category", "Percent Antisemitic", "Accuracy", "Precision", "Recall", "F1", "WF1", "n", "wn"]
	rows = []
	true_sheet = pd.read_csv("../../data/AMCHA/anonymized_10_10_24.csv")
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
	trues = []
	cat_trues = []
	class_trues = []
	dupe_class_trues = []
	for (i, pred), (j, tem) in zip(enumerate(pred_sheet), true_sheet.iterrows()):
		if "llama" in model:
			try:
				response = ast.literal_eval(ast.literal_eval(pred)[-1]["content"])
			except:
				fixed = ast.literal_eval(pred)[-1]["content"].replace("'s", "s").replace("s' ", "s ").replace("'''", "''")
				try:
					fixed = fixed[fixed.index("{"):fixed.index("}") + 1]
					response = ast.literal_eval(fixed)
				except Exception as e:
					try:
						fixed = fixed[fixed.index("{"):fixed.index("other_category")] + "}"
						fixed = fixed.replace("'}", "}").replace('"}', '}')
					except Exception as e:
						fixed = fixed.replace("'}", "}").replace('"}', '}')
						# fixed = {"category": "A", "type_of_expression": []}
					if "I can't help you with that" in fixed:
						response = {"category": "A", "type_of_expression": []}
					elif "Can I help you with something else" in fixed:
						response = {"category": "A", "type_of_expression": []}
					else:
						f = fixed.replace("\'", '\"').replace("\n", "").replace(",}", "}").replace(", }", "}").replace("A,", '"A",').replace(",]", "]").replace("B,", '"B",').replace("C,", '"C",')
						f = f.replace("8,}", "8]}")
						if "}" not in f:
							f = f + "}"
						print("??")
						print(f)
						f = f.replace("8,}", "8]}")
						response = json.loads(f)
			if s == 0 and b == "false" and not d:
				print(response)
			preds.append(1 if response["category"].strip() != "A" else 0)
			cat_preds.append(l2n[response["category"].strip()])
			rt = response["type_of_expression"]
			if type(rt) == int:
				rt = [rt]
			class_preds.append(rt)
		else:
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
		cs = tem["Classification"].split(", ")
		tc = [true_classes[c] for c in cs if c in true_classes]
		class_trues.append(tc)
		for it in tc:
			dupe_class_trues.append(it)
			dupe_class_preds.append(rt)
	print(set(preds))
	print(set(cat_preds))
	assert(len(dupe_class_preds) == len(dupe_class_trues))
	if b == "false":
		rows.append(["Overall", np.mean(preds), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
	else:
		rows.append(["Overall", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
	cm = confusion_matrix(cat_trues, cat_preds, labels = [0, 1, 2, 3])
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
	disp.plot()
	if not os.path.exists(f"figs/{model}"):
		os.mkdir(f"figs/{model}")
	plt.savefig(f"figs/{model}/{s}_{b}_{d}.png")
	plt.clf()
	for cat in [1, 2]:
		inds = [i for i in range(len(trues)) if cat_trues[i] == cat]
		subpreds = np.array(preds)[inds]
		print(cat, len(inds), Counter(np.array(cat_preds)[inds]))
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
	if not os.path.exists(f"amcha_results/{model}"):
		os.mkdir(f"amcha_results/{model}")
	df.to_csv(f"amcha_results/{model}/sample_report_{s}_{b}_{d}.csv")
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

# for model in sys.argv[1:]:
# 	report(0, "true", True, model)
# 	report(0, "true", False, model)
# 	report(0, "false", True, model)
# 	report(0, "false", False, model)
# 	report(1, "true", False, model)
for model in sys.argv[1:]:
	control_report(0, "true", True, model)
	control_report(0, "true", False, model)
	control_report(0, "false", True, model)
	control_report(0, "false", False, model)
	control_report(1, "true", False, model)
