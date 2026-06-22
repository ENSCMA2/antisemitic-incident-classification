import os
import openai
from openai import AzureOpenAI
import pandas as pd
import time
from itertools import chain
import json
from utils import *

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

df = pd.read_csv("../../data/AMCHA/anonymized_10_10_24.csv")
tuples = [(tem["Anonymized_Description"], tem["Anonymized_Short_Description"], tem["Date"], tem["University"]) for i, tem in df.iterrows()]

responses = []
in_toks = []
out_toks = []

def get_response(prompt):
    response = client.chat.completions.create(
              model="gpt-4o",
              messages=[
                {"role": "system", 
                 "content": sysprompt},
                {"role": "user", "content": prompt},
              ],
              temperature = 0
            )
    answer = response.choices[0].message.content
    out_tok = response.usage.completion_tokens
    in_tok = response.usage.prompt_tokens
    time.sleep(0.6)
    return answer, (in_tok, out_tok)

def procedure(tups, shorthand):
    in_lengths = {}
    out_lengths = {}
    answers = {}

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
                if (definition and shot == 1) or (shot == 1 and (not forced)):
                    continue
                pth = f"amcha_results/{shorthand}/{shot}_{str(forced).lower()}_{definition}.csv"
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
                for i, (text, date, uni) in enumerate(tups):
                    if i < shape[0]:
                        continue
                    log(f"SHOTS: {shot}, FORCED: {forced}, DEFINITION: {definition}")
                    print(shot, forced, i)
                    uprompt = template(text, date, uni, shot, forced, verbose = i == 0, define = definition)
                    ans, (inn, out) = get_response(uprompt)
                    print(ans)
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


# procedure(tuples, "gpt-4o")
df2 = pd.read_csv("../../data/TOXIGEN/prompts/controlprompts.csv").sample(df.shape[0], random_state = 0)
df2.to_csv(f"../../data/TOXIGEN/prompts/controlprompts_{df.shape[0]}.csv")
tuples2 = [(tem["Description"], tem["Date"], tem["University"]) for i, tem in df2.iterrows()]

procedure(tuples2, "control-gpt4o")

