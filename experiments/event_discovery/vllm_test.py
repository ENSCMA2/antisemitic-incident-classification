import os
import openai
import pandas as pd
import time
from itertools import chain
import json
from openai import OpenAI
import sys
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_KEY"),
)

df = pd.read_csv("../../data/AMCHA/anonymized_checked.csv")
subset = df.sample(5)
tuples = [(tem["Anonymized_Description"], tem["Date"], tem["University"]) for i, tem in subset.iterrows()]

types = {1: 'physical assault',
         2: 'discrimination',
         3: 'destruction of Jewish property',
         4: 'genocidal expression',
         5: 'suppression of speech/movement/assembly',
         6: 'bullying',
         7: 'denigration',
         8: 'historical',
         9: 'condoning terrorism',
         10: 'denying Jews self-determination',
         11: 'demonization',
         12: 'Calls for BDS',
         13: 'BDS event',
         14: 'BDS vote'}

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
               "denying Jews self-determination":
               "Denying Israel the right to exist or promoting the elimination of Israel as a Jewish state",
               "demonization":
               "Using symbols, images and tropes associated with classic antisemitism to characterize Israel, Israelis, Zionism or Zionists, such as claiming that Israelis are evil or blood-thirsty and deliberately murder children or that Zionism is white supremacy, or delegitimizing Israel by insinuating that Israel is an illegitimate state and does not belong in the family of nations",
               "Calls for BDS":
               "Promoting BDS verbally or by writing, signing or publicizing resolutions, petitions, statements or op-eds calling for BDS",
               "BDS vote":
               "Considering, discussing or voting on resolutions calling for BDS",
               "BDS event":
               "Holding events which promote BDS"
}
cats = {'C': {"cat": 'targeting Jewish students and staff',
              "types": [1, 2, 3, 4, 5, 6, 7]},
        'B': {"cat": 'antisemitic expression',
              "types": [8, 9, 10, 11]}, 
        'D': {"cat": 'BDS activity',
              "types": [12, 13, 14]}}

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

alphabet = "BCDEFGHIJKLMNOPQRSTUVWXYZ"

responses = []
in_toks = []
out_toks = []

'''
for resolutions: 
    General Antisemitism (General A/S), 
    State Department Definition (SD Definition), 
    International Holocaust Remembrance Alliance Definition (IHRA Definition), 
    Anti-BDS and Study Abroad in Israel
for presidential statements: 
    Disruption, 
    General Antisemitism, 
    Israel, 
    BDS, 
    Harassment
'''

cat_strings = {}
type_strings = {}
for c in cats:
    samples = icl_samples[c]
    cstring = []
    for t in samples:
        string = f'''Text: {samples[t]['Description']}\nCategory: {cats[c]['cat']}\nType of expression: {types[t]}'''
        cstring.append(string)
        type_strings[t] = string
    cat_strings[c] = "\n".join(cstring)

def get_lesson(shots, example):
    if shots > 0:
        return f'''Here are some examples of texts that describe antisemitic events, along with fine-grained labels of what types of antisemitism they describe:\n{example}\nNow, it's your turn.\n'''
    return ""

def intro(txt, dt, uni, forced):
    ff = "antisemitic " if forced else ""
    q = "W" if forced else '''Does this quoted text describe an antisemitic event? If so, w'''
    return f'''The following text, enclosed in single quotes, describes an {ff}event that happened on {dt} at {uni}.
'{txt}'
{q}hat category of antisemitism does the event fall under, and what type of expression does it fall under?
Give an answer in the following format:\n'''

def template(txt, dt, uni, shots, forced, cat = None, tpe = None, verbose = False, define = False):
    if verbose:
        print(f"SHOTS: {shots}\nFORCED: {forced}\n CAT: {cat}\n TYPE: {tpe}")
    cattpe = ''''category': char,\n'type_of_expression': int list,'''
    unforced_ans = f"{{'describes_antisemitic_event': bool,"
    other = ''''other_category': str,\n'other_type_of_expression': str}.'''
    cat_a = "'Does not describe an antisemitic incident'" if not forced else "'Other category'"
    type_0 = cat_a if not forced else "'Other type of expression'"
    default_cat_list = "\n".join([f"A: {cat_a}", 
                                   "B: 'antisemitic expression'",
                                   "C: 'targeting Jewish students and staff'",
                                   "D: 'BDS activity'"])
    if define:
        default_cat_list = "\n".join([f"A: {cat_a}", 
                                       "B: 'antisemitic expression' - Incidents that directly target Jewish students on campus or other Jewish members of the campus community for harmful or hateful action based on their Jewishness or perceived support for Israel",
                                       "C: 'targeting Jewish students and staff' - Language, imagery or behavior deemed antisemitic by the U.S. State Department definition of antisemitism, or wholly consistent with that definition",
                                       "D: 'BDS activity' - Incidents that promote BDS verbally or by writing, signing or publicizing resolutions, petitions, statements or op-eds calling for BDS; consider, discuss or vote on resolutions calling for BDS; or hold events which promote BDS."])
    def get_cat_exp(ltr):
        return f'''\nIf the best answer is {ltr}, write a category that you think would be the best fit in the 'other_category' field. Otherwise, leave 'other_category' as an empty string.''' if forced else ""
    def get_type_exp(num):
        return f'''\nIf the best answer is {num}, write a type of expression that you think would be the best fit in the 'other_type_of_expression' field. Otherwise, leave 'other_type_of_expression' as an empty string.'''
    if cat is None and tpe is None:
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
            # print(defd, type(defd))
            rest = "\n".join([f"0: {zero}"] + defd) 
        if forced:
            rest += f'''{get_type_exp(0)}'''
    else:
        cat_keys = [key for key in cats if key != cat] if cat is not None else cats.keys()
        cat_values = [cats[key]["cat"] for key in cat_keys]
        cat_string = "\n".join([f"{alphabet[i]}: '{cat_values[i]}'" for i in range(len(cat_values))])
        type_keys = list(chain(*[cats[key]["types"] for key in cat_keys]))
        def tstring_and_example(type_values, lookup, dont):
            type_string = "\n".join([f"{i + 1}: '{type_values[i]}'" for i in range(len(type_values))])
            example = "\n".join([lookup[c] for c in lookup if c != dont])
            return type_string, example
        if cat is not None:
            type_values = [types[t] for t in type_keys]
            type_string, example = tstring_and_example(type_values, cat_strings, cat)
            cat_list = f'''A: {cat_a}\n{cat_string}'''
            if not forced:
                cat_list += f'''\n{alphabet[len(cat_keys)]}: 'Other category'{get_cat_exp(alphabet[len(cat_keys)])}'''
        elif tpe is not None:
            type_values = [types[t] for t in type_keys if t != tpe]
            type_string, example = tstring_and_example(type_values, type_strings, tpe)
            if not forced:
                cat_list = f'''{default_cat_list}\nE: 'Other category'{get_cat_exp('E')}'''
            else:
                cat_list = f'''{default_cat_list}{get_cat_exp('A')}'''
        if not forced:
            form = f'''{unforced_ans}\n{cattpe}\n{other}'''
        else:
            form = f'''{{{cattpe}\n{other}'''
        unforced_exp = f'''\n{len(type_values) + 1}: 'Other type of expression'{get_type_exp(len(type_values) + 1)}'''
        forced_exp = get_type_exp(0)
        rest = f'''0: {type_0}\n{type_string}{unforced_exp if not forced else forced_exp}'''
    lesson = get_lesson(shots, example)
    ret = f'''{lesson}{intro(txt, dt, uni, forced)}{form}\nFor 'category', put the letter corresponding to the strongest option below for the quoted text:\n{cat_list}\nFor 'type_of_expression', list all numbers corresponding to the the following options that apply to the quoted text:\n{rest}'''
    if verbose:
        print("PROMPT:\n", ret)
    return ret
sysprompt = "You are evaluating and analyzing texts on whether they describe events that are harmful toward Jewish people. Give all answers in JSON format."

model = sys.argv[1]

def get_response(prompt):
    response = client.chat.completions.create(
              model=model,
              messages=[
                {"role": "assistant", 
                 "content": sysprompt},
                {"role": "user", "content": prompt},
              ],
              temperature = 0
            )
    answer = response.choices[0].message.content
    out_tok = response.usage.completion_tokens
    in_tok = response.usage.prompt_tokens
    time.sleep(2)
    print(sysprompt)
    print(prompt)
    return answer, (in_tok, out_tok)

answers = {}
in_lengths = {}
out_lengths = {}

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
            pth = f"results/amcha_toy_{model.replace('/', '-')}_{shot}_{str(forced).lower()}_{definition}_vanilla.csv"
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
            print(pth, shape)
            for i, (text, date, uni) in enumerate(tuples):
                if i < shape[0]:
                    continue
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


