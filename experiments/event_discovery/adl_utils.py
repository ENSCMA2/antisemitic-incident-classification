import os
import openai
import pandas as pd
import time
from itertools import chain
import json
from openai import OpenAI
import sys
from groq import Groq
import time

types = {1: 'Assault',
         2: 'Harassment',
         3: 'Vandalism',}

definitions = {'Assault': 
               "cases where Jewish people (or people perceived to be Jewish) were targeted with physical violence accompanied by evidence of antisemitic animus",
               'Harassment': 
               "cases in which one or more Jewish people (or people who were perceived to be Jewish) were harassed verbally or in writing with antisemitic slurs, stereotypes or conspiracy theories",
               'Vandalism': 
               "cases where property was damaged in a manner that incorporated evidence of antisemitic intent or which had an antisemitic impact on Jews.",
}
cats = {'B': {"cat": 'Antisemitic Incident',
              "types": [1, 2, 3]},}

def log(txt):
    with open("log.txt", "a") as o:
        o.write(f"{txt}\n")

def get_sample_strings(df):
    icl_samples = {}
    for c in cats:
        icl_samples[c] = {}
        subcat = df[(df["type"].str.contains(cats[c]["cat"]))]
        already = []
        for t in cats[c]["types"]:
            subtype = subcat[(subcat["type"].str.contains(types[t]))]
            if len(already) > 0:
                subtype = subtype[~(subtype["type"].str.contains("|".join(already)))]
            samp = subtype.sample(n = 1, random_state = 0)
            already.append(types[t])
            icl_samples[c][t] = samp.iloc[0, :]
    type_strings = {}
    for c in cats:
        samples = icl_samples[c]
        for t in samples:
            string = f'''Text: {samples[t]['description']}\nAntisemitic Incident: true\nType of incident: {types[t]}'''
            type_strings[t] = string
    return type_strings

type_strings = get_sample_strings(pd.read_csv("../../data/HEATMapData.csv"))

def get_lesson(shots, example):
    if shots > 0:
        return f'''Here are some examples of texts that describe antisemitic events, along with fine-grained labels of what types of antisemitism they describe:\n{example}\nNow, it's your turn.\n'''
    return ""

def intro(txt, dt, city, state, forced):
    ff = "antisemitic " if forced else ""
    q = "W" if forced else '''Does this quoted text describe an antisemitic event? If so, w'''
    return f'''The following text, enclosed in single quotes, describes an {ff}event that happened on {dt} in {city}, {state}.
'{txt}'
{q}hat type of incident does it fall under?
Give an answer in the following format:\n'''

def template(txt, dt, city, state, shots, forced, verbose = False, define = False):
    if verbose:
        print(f"SHOTS: {shots}\nFORCED: {forced}")
    cattpe = ''''type_of_incident': int list,'''
    unforced_ans = f"{{'describes_antisemitic_event': bool,"
    other = ''''other_type_of_incident': str}.'''
    type_0 = "'Does not describe antisemitic event'" if not forced else "'Other type of incident'"

    def get_type_exp(num):
        return f'''\nIf the best answer is {num}, write a type of incident that you think would be the best fit in the 'other_type_of_incident' field. Otherwise, leave 'other_type_of_incident' as an empty string.'''
    if not forced:
        form = f'''{unforced_ans}\n{cattpe}}}.'''
    else:
        form = f'''{{{cattpe}\n{other}'''
    itypes = [f"{i}: '{types[i]}'" for i in types]
    zero = "'Does not describe an antisemitic incident'" if not forced else "'Other type of incident'"
    if not define:
        rest = "\n".join([f"0: {zero}"] + itypes)  
    else: 
        defd = [f"{itypes[i - 1]} - {definitions[types[i]]}" for i in types]
        rest = "\n".join([f"0: {zero}"] + defd) 
    if forced:
        rest += f'''{get_type_exp(0)}'''
    example = "\n".join([type_strings[t] for t in type_strings])
    lesson = get_lesson(shots, example)
    if not define:
        bigdef = ""
    else:
        bigdef = "Antisemitism is defined as hostility to, prejudice towards, or discrimination against Jews. "
    ret = f'''{lesson}{bigdef}{intro(txt, dt, city, state, forced)}{form}\nFor 'type_of_incident', list all numbers corresponding to the the following options that apply to the quoted text:\n{rest}'''
    if verbose:
        log(f"PROMPT:\n{ret}")
    return ret

sysprompt = "You are evaluating and analyzing characteristics of textual descriptions of events. Give all answers in JSON format."