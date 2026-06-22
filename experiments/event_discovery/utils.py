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

types = {1: 'physical assault',
         2: 'discrimination',
         3: 'destruction of Jewish property',
         4: 'genocidal expression',
         5: 'suppression of speech/movement/assembly',
         6: 'bullying',
         7: 'denigration',
         8: 'historical',
         9: 'condoning terrorism',}

definitions = {'physical assault': 
               "Physically attacking Jewish students or staff because of their Jewishness or perceived association with Israel",
               'discrimination': 
               "Unfair treatment or exclusion of Jewish students or staff because of their Jewishness or perceived association with Israel",
               'destruction of Jewish property': 
               "Inflicting damage or destroying property owned by Jews or related to Jews",
               'genocidal expression':
               "Using imagery (e.g. swastika) or language that expresses a desire or will to kill Jews or exterminate the Jewish people",
               'suppression of speech/movement/assembly': 
               "Preventing or impeding the expression of Jewish students, such as by removing or defacing Jewish students’ flyers, attempting to disrupt or shut down speakers at Jewish or pro-Israel events, or blocking access to Jewish or pro-Israel student events",
               'bullying': 
               "Tormenting Jewish students or staff because of their Jewishness or perceived association with Israel",
               "denigration":
               "Unfairly ostracizing, vilifying or defaming Jewish students or staff because of their Jewishness or perceived association with Israel",
               "historical":
               'Using symbols, images and tropes associated with historical antisemitism, including by making "mendacious, dehumanizing, demonizing, or stereotypical allegations about Jews as such, or the power of Jews as a collective-especially but not exclusively, the myth about a world Jewish conspiracy or of Jews controlling the media, economy, governments, or other societal institutions"',
               "condoning terrorism":
               "Calling for, aiding or justifying the killing or harming of Jews",
}
cats = {'C': {"cat": 'targeting Jewish students and staff',
              "types": [1, 2, 3, 4, 5, 6, 7]},
        'B': {"cat": 'antisemitic expression',
              "types": [8, 9]}, }

def log(txt):
    with open("log.txt", "a") as o:
        o.write(f"{txt}\n")

def remove_html(txt):
    open_tag = txt.find("<p>")
    close_tag = txt.find("<a")
    trimmed = txt[open_tag + len("<p>"):close_tag].replace("<p>", "").replace("</p>", "").replace("<ul>", "").replace("</ul>", "").replace("<li>", "").replace("</li>", "")
    return trimmed.replace("<em>", "").replace("</em>", "").replace("[1]", "").replace("[2]", "").replace("[1-2]", "").replace("[3]", "").strip()

def get_sample_strings(df):
    icl_samples = {}
    for c in cats:
        icl_samples[c] = {}
        subcat = df[(df["Category"].str.contains(cats[c]["cat"].upper()))]
        already = []
        for t in cats[c]["types"]:
            subtype = subcat[(subcat["Classification"].str.contains(types[t].upper()))]
            if len(already) > 0:
                subtype = subtype[~(subtype["Classification"].str.contains("|".join(already)))]
            samp = subtype.sample(n = 1, random_state = 0)
            already.append(types[t].upper())
            icl_samples[c][t] = samp.iloc[0, :]
    cat_strings = {}
    type_strings = {}
    for c in cats:
        samples = icl_samples[c]
        cstring = []
        for t in samples:
            string = f'''Text: {samples[t]['Description']}\nCategory: {cats[c]['cat']}\nType of expression: {types[t]}'''
            cstring.append(remove_html(string))
            type_strings[t] = remove_html(string)
        cat_strings[c] = "\n".join(cstring)
    return cat_strings, type_strings

cat_strings, type_strings = get_sample_strings(pd.read_csv("../../data/AMCHA/anonymized_10_10_24.csv"))

def get_lesson(shots, example):
    if shots > 0:
        return f'''Here are some examples of texts that describe antisemitic events, along with fine-grained labels of what types of antisemitism they describe:\n{example}\nNow, it's your turn.\n'''
    return ""

def intro(txt, dt, uni, forced):
    ff = "antisemitic " if forced else ""
    q = "W" if forced else '''Does this quoted text describe an antisemitic event? If so, w'''
    return f'''The following text, enclosed in single quotes, describes an {ff}event that happened on {dt} at {uni}.
'{remove_html(txt)}'
{q}hat category of antisemitism does the event fall under, and what type of expression does it fall under?
Give an answer in the following format:\n'''

def template(txt, dt, uni, shots, forced, verbose = False, define = False):
    if verbose:
        print(f"SHOTS: {shots}\nFORCED: {forced}")
    cattpe = ''''category': char,\n'type_of_expression': int list,'''
    unforced_ans = f"{{'describes_antisemitic_event': bool,"
    other = ''''other_category': str,\n'other_type_of_expression': str}.'''
    cat_a = "'Does not describe an antisemitic incident'" if not forced else "'Other category'"
    type_0 = cat_a if not forced else "'Other type of expression'"
    default_cat_list = "\n".join([f"A: {cat_a}", 
                                   "B: 'antisemitic expression'",
                                   "C: 'targeting Jewish students and staff'"])
    if define:
        default_cat_list = "\n".join([f"A: {cat_a}", 
                                       "B: 'antisemitic expression' - Language, imagery or behavior deemed antisemitic by the U.S. State Department definition of antisemitism, or wholly consistent with that definition",
                                       "C: 'targeting Jewish students and staff' - Incidents that directly target Jewish students on campus or other Jewish members of the campus community for harmful or hateful action based on their Jewishness or perceived support for Israel"])
    def get_cat_exp(ltr):
        return f'''\nIf the best answer is {ltr}, write a category that you think would be the best fit in the 'other_category' field. Otherwise, leave 'other_category' as an empty string.''' if forced else ""
    def get_type_exp(num):
        return f'''\nIf the best answer is {num}, write a type of expression that you think would be the best fit in the 'other_type_of_expression' field. Otherwise, leave 'other_type_of_expression' as an empty string.'''
    example = "\n".join([cat_strings[c] for c in cat_strings])
    if not forced:
        form = f'''{unforced_ans}\n{cattpe}}}.'''
    else:
        form = f'''{{{cattpe}\n{other}'''
    cat_list = default_cat_list
    itypes = [f"{i}: '{types[i]}'" for i in types]
    zero = "'Does not describe an antisemitic incident'" if not forced else "'Other type of expression'"
    if not define:
        rest = "\n".join([f"0: {zero}"] + itypes)  
    else: 
        defd = [f"{itypes[i - 1]} - {definitions[types[i]]}" for i in types]
        rest = "\n".join([f"0: {zero}"] + defd) 
    if forced:
        rest += f'''{get_type_exp(0)}'''
    lesson = get_lesson(shots, example)
    if not define:
        bigdef = ""
    else:
        bigdef = "Antisemitism is defined as hostility to, prejudice towards, or discrimination against Jews. "
    ret = f'''{lesson}{bigdef}{intro(txt, dt, uni, forced)}{form}\nFor 'category', put the letter corresponding to the strongest option below for the quoted text:\n{cat_list}\nFor 'type_of_expression', list all numbers corresponding to the the following options that apply to the quoted text:\n{rest}'''
    if verbose:
        log(f"PROMPT:\n{ret}")
    return ret

sysprompt = "You are evaluating and analyzing characteristics of textual descriptions of events. Give all answers in JSON format."