import os
import openai
from openai import AzureOpenAI
import pandas as pd
import time
from itertools import chain
import json
from utils import *

api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = "https://rtp2-shared.openai.azure.com/"

client = AzureOpenAI(
    # This is the default and can be omitted
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint=azure_endpoint,
)

df = pd.read_csv("../../data/AMCHA/anonymized_checked.csv")
n = int(df.shape[0] * 0.02)
exp = df[df["Category"] == "ANTISEMITIC EXPRESSION"].sample(n, random_state = 0)
targ = df[df["Category"] == "TARGETING JEWISH STUDENTS AND STAFF"].sample(n, random_state = 0)
bds = df[df["Category"] == "BDS ACTIVITY"].sample(n, random_state = 0)
sampled = pd.concat([exp, targ, bds])
sampled.to_csv("../../data/AMCHA/anonymized_sample.csv")
tuples = [(tem["Anonymized_Description"], tem["Date"], tem["University"], tem["Anonymized_Short_Description"]) for i, tem in sampled.iterrows()]

responses = []
in_toks = []
out_toks = []

cat_strings, type_strings = get_sample_strings(df)

def get_response(prompt, sprompt = None):
    try:
        response = client.chat.completions.create(
                  model="gpt-4-turbo-1106-kh",
                  messages=[
                    {"role": "system", 
                     "content": sysprompt},
                    {"role": "user", "content": prompt},
                  ],
                  temperature = 0
                )
    except:
        print("retrying in 60")
        time.sleep(60)
        try:
            response = client.chat.completions.create(
                      model="gpt-4-turbo-1106-kh",
                      messages=[
                        {"role": "system", 
                         "content": sysprompt},
                        {"role": "user", "content": prompt},
                      ],
                      temperature = 0
            )
        except:
            print("too many tokens")
            response = client.chat.completions.create(
                      model="gpt-4-turbo-1106-kh",
                      messages=[
                        {"role": "system", 
                         "content": sysprompt},
                        {"role": "user", "content": sprompt},
                      ],
                      temperature = 0
            )
    answer = response.choices[0].message.content
    out_tok = response.usage.completion_tokens
    in_tok = response.usage.prompt_tokens
    time.sleep(0.5)
    return answer, (in_tok, out_tok)

in_lengths = {}
out_lengths = {}
answers = {}

paras = ["Does this quoted text describe an antisemitic event?",
         "Does this quoted text describe an event that is harmful to Jews?",
         "Does this quoted text describe an event that harms Jews?",
         "Does this quoted text describe an instance of antisemitism?",
         "Does this quoted text describe an event that is harmful to Jewish people?"]

for shot in [0, 
             1
             ]:
    in_lengths[shot] = {}
    out_lengths[shot] = {}
    answers[shot] = {}
    for forced in [True, 
                   False
                   ]:
        in_lengths[shot][forced] = {}
        out_lengths[shot][forced] = {}
        answers[shot][forced] = {}
        for definition in [True, 
                           False
                           ]:
            if (definition and shot == 1) or ((not forced) and shot == 1):
                continue
            for pnum, para in enumerate(paras):
                print(para, shot, forced, definition)
                pth = f"results/amcha_gpt4_1106_{shot}_{str(forced).lower()}_{definition}_{pnum}_vanilla.csv"
                if os.path.exists(pth):
                    existing = pd.read_csv(pth)
                    in_lengths[shot][forced][definition] = {"vanilla": existing["out_tokens"].tolist()}
                    out_lengths[shot][forced][definition] = {"vanilla": existing["in_tokens"].tolist()}
                    answers[shot][forced][definition] = {"vanilla": existing["response"].tolist()}
                    shape = existing.shape
                else:
                    in_lengths[shot][forced][definition] = {"vanilla": []}
                    out_lengths[shot][forced][definition] = {"vanilla": []}
                    answers[shot][forced][definition] = {"vanilla": []}
                    shape = (0, 0)
                for i, (text, date, uni, stext) in enumerate(tuples):
                    if i < shape[0]:
                        continue
                    print(i)
                    uprompt = template(text, date, uni, shot, forced, para, verbose = i == 0, define = definition)
                    sprompt = template(stext, date, uni, shot, forced, para, verbose = i == 0, define = definition)
                    ans, (inn, out) = get_response(uprompt, sprompt)
                    try:
                        in_lengths[shot][forced][definition]["vanilla"].append(inn)
                        out_lengths[shot][forced][definition]["vanilla"].append(out)
                        answers[shot][forced][definition]["vanilla"].append(ans)
                    except:
                        in_lengths[shot][forced][definition]["vanilla"] = [inn]
                        out_lengths[shot][forced][definition]["vanilla"] = [out]
                        answers[shot][forced][definition]["vanilla"] = [ans]
                    if i % 5 == 0:
                        df = pd.DataFrame({"response": answers[shot][forced][definition]["vanilla"],
                                           "in_tokens": in_lengths[shot][forced][definition]["vanilla"],
                                           "out_tokens": out_lengths[shot][forced][definition]["vanilla"]})
                        df.to_csv(pth)
                df = pd.DataFrame({"response": answers[shot][forced][definition]["vanilla"],
                				   "in_tokens": in_lengths[shot][forced][definition]["vanilla"],
                				   "out_tokens": out_lengths[shot][forced][definition]["vanilla"]})
                df.to_csv(pth)

