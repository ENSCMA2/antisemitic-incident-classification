import transformers
import os
import openai
import pandas as pd
import time
from itertools import chain
import json
import sys
import time
from adl_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 'Antisemitic Incident:Assault'
# 'Antisemitic Incident:Harassment'
# 'Antisemitic Incident:Vandalism'
df = pd.read_csv("../../data/HEATMapData.csv")
df = df[(df["type"].str.contains("Antisemitic Incident"))]
df.to_csv("../../data/HEATMapAntisemitic.csv")
tuples = [(tem["description"], tem["date"], tem["city"], tem["state"]) for i, tem in df.iterrows()]

responses = []
in_toks = []
out_toks = []

model_id = "meta-llama/Llama-3.2-3B-Instruct"

def log(msg):
    with open("log.txt", "a") as o:
        o.write(f"{msg}\n")

def procedure(tuples, mid):
    pipeline = transformers.pipeline("text-generation",
                                     model = model_id,
                                     model_kwargs = {"torch_dtype": 
                                                     torch.bfloat16},
                                     device_map = "auto",)
    answers = {}

    def get_response(prompt, pipeline):
        messages= [{"role": "assistant", 
                   "content": sysprompt},
                  {"role": "user", "content": prompt}]
        outputs = pipeline(messages, max_new_tokens = 256, 
                           num_return_sequences = 1,
                           temperature = 1.5)
        answer = outputs[0]["generated_text"]
        return answer

    for shot in [0, 
                 1
                 ]:
        answers[shot] = {}
        for forced in [True, 
                       False
                       ]:
            answers[shot][forced] = {}
            for definition in [True, 
                               False
                               ]:
                if (definition and shot == 1) or (shot == 1 and (not forced)):
                    continue
                pth = f"adl_results/{mid.replace('/', '-')}/{shot}_{str(forced).lower()}_{definition}.csv"
                if os.path.exists(pth):
                    existing = pd.read_csv(pth)
                    answers[shot][forced][definition] = {"vanilla": existing["response"].tolist()}
                    shape = existing.shape
                else:
                    answers[shot][forced][definition] = {"vanilla": []}
                    shape = (0, 0)
                for i, (text, date, city, state) in enumerate(tuples):
                    if i < shape[0]:
                        continue
                    log(f"sfdi: {shot}, {forced}, {definition}, {i}")
                    uprompt = template(text, date, city, state, shot, forced, verbose = i == 0, define = definition)
                    ans = get_response(uprompt, pipeline)
                    try:
                        answers[shot][forced][definition]["vanilla"].append(ans)
                    except:
                        answers[shot][forced][definition]["vanilla"] = [ans]
                    if i % 5 == 0:
                        df = pd.DataFrame({"response": answers[shot][forced][definition]["vanilla"]})
                        df.to_csv(pth)
                df = pd.DataFrame({"response": answers[shot][forced][definition]["vanilla"]})
                df.to_csv(pth)

# procedure(tuples, model_id)
df2 = pd.read_csv(f"../../data/control_adl_{df.shape[0]}.csv")
tuples2 = [(tem["description"], tem["date"], tem["city"], tem["state"]) for i, tem in df2.iterrows()]

procedure(tuples2, f"control-{model_id}")

