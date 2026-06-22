import os
import openai
from openai import OpenAI
import pandas as pd
import time
from itertools import chain
import json

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

infodf = pd.read_csv("../chatgpt_mini_batch/AMCHA_test.csv")
df = pd.read_csv("../chatgpt_mini_batch/amcha_test_tweets_1106.csv")

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

cats = {'C': {"cat": 'targeting Jewish students and staff',
              "types": [1, 2, 3, 4, 5, 6, 7]},
        'B': {"cat": 'antisemitic expression',
              "types": [8, 9, 10, 11]}, 
        'D': {"cat": 'BDS activity',
              "types": [12, 13, 14]}}

alphabet = "BCDEFGHIJKLMNOPQRSTUVWXYZ"

responses = []
in_toks = []
out_toks = []

def get_lesson(shots, example):
    if shots > 0:
        return f'''Here are some examples of texts that describe antisemitic events, along with fine-grained labels of what types of antisemitism they describe:\n{example}\nNow, it's your turn.\n'''
    return ""

def intro(txt, forced):
    ff = "antisemitic " if forced else ""
    q = "W" if forced else '''Is this quoted tweet harmful toward Jewish people? If so, w'''
    return f'''The following tweet, enclosed in single quotes, is harmful to Jewish people.
'{txt}'
{q}hat category of antisemitism does the tweet fall under, and what type of expression does it fall under?
Give an answer in the following format:\n'''

def template(txt, info, shots, forced, cat = None, tpe = None, verbose = False):
    if verbose:
        print(f"SHOTS: {shots}\nFORCED: {forced}\n CAT: {cat}\n TYPE: {tpe}")
    cattpe = ''''category': char,\n'type_of_expression': int list,'''
    unforced_ans = f"{{'antisemitic': bool,"
    other = ''''other_category': str,\n'other_type_of_expression': str}.'''
    cat_a = "'Not antisemitic'" if not forced else "'Other category'"
    type_0 = cat_a if not forced else "'Other type of expression'"
    default_cat_list = "\n".join([f"A: {cat_a}", 
                                   "B: 'antisemitic expression'",
                                   "C: 'targeting Jewish students and staff'",
                                   "D: 'BDS activity'"])
    def get_cat_exp(ltr):
        return f'''\nIf the best answer is {ltr}, write a category that you think would be the best fit in the 'other_category' field. Otherwise, leave 'other_category' as an empty string.''' if forced else ""
    def get_type_exp(num):
        return f'''\nIf the best answer is {num}, write a type of expression that you think would be the best fit in the 'other_type_of_expression' field. Otherwise, leave 'other_type_of_expression' as an empty string.'''
    if cat is None and tpe is None:
        if not forced:
            form = f'''{unforced_ans}\n{cattpe}}}.'''
        else:
            form = f'''{{{cattpe}\n{other}'''
        cat_list = default_cat_list
        zero = "'Not antisemitic'" if not forced else "'Other type of expression'"
        rest = "\n".join([f"0: {zero}",
                           "1: 'physical assault'",
                           "2: 'discrimination'",
                           "3: 'destruction of Jewish property'",
                           "4: 'genocidal expression' ",
                           "5: 'suppression of speech/movement/assembly'",
                           "6: 'bullying'",
                           "7: 'denigration'",
                           "8: 'historical'",
                           "9: 'condoning terrorism'",
                           "10: 'denying Jews self-determination'",
                           "11: 'demonization'",
                           "12: 'Calls for BDS'",
                           "13: 'BDS event'" ,
                           "14: 'BDS vote'"])
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
    ret = f'''{intro(txt, forced)}{form}\nFor 'category', put the letter corresponding to the strongest option below for the quoted text:\n{cat_list}\nFor 'type_of_expression', list all numbers corresponding to the the following options that apply to the quoted text:\n{rest}'''
    if verbose:
        print("PROMPT:\n", ret)
    return ret
sysprompt = "You are evaluating and analyzing tweets on whether they are harmful toward Jewish people. Give all answers in JSON format."

def get_response(prompt):
    response = client.chat.completions.create(
              model="gpt-4-1106-preview",
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

in_lengths = {}
out_lengths = {}
answers = {}

for shot in [0, 
             # 1
             ]:
    in_lengths[shot] = {}
    out_lengths[shot] = {}
    answers[shot] = {}
    for forced in [True, 
                   False
                   ]:
        pth = f"amcha_gpt4_tweet_out_1106_{shot}_{str(forced).lower()}_vanilla.csv"
        if os.path.exists(pth):
            existing = pd.read_csv(pth)
            in_lengths[shot][forced] = {"vanilla": existing["out_tokens"].tolist()}
            out_lengths[shot][forced] = {"vanilla": existing["in_tokens"].tolist()}
            answers[shot][forced] = {"vanilla": existing["response"].tolist()}
            shape = existing.shape
        else:
            in_lengths[shot][forced] = {"vanilla": []}
            out_lengths[shot][forced] = {"vanilla": []}
            answers[shot][forced] = {"vanilla": []}
            shape = (0, 0)
        print(pth, shape)
        for (i, text), (j, info) in zip(df.iterrows(), infodf.iterrows()):
            if i < shape[0]:
                continue
            print(shot, forced, i)
            uprompt = template(text["response"], info, shot, forced, verbose = i == 0)
            ans, (inn, out) = get_response(uprompt)
            print(ans)
            try:
                in_lengths[shot][forced]["vanilla"].append(inn)
                out_lengths[shot][forced]["vanilla"].append(out)
                answers[shot][forced]["vanilla"].append(ans)
            except:
                in_lengths[shot][forced]["vanilla"] = [inn]
                out_lengths[shot][forced]["vanilla"] = [out]
                answers[shot][forced]["vanilla"] = [ans]
            if i % 10 == 0:
                df = pd.DataFrame({"response": answers[shot][forced]["vanilla"],
                                   "in_tokens": in_lengths[shot][forced]["vanilla"],
                                   "out_tokens": out_lengths[shot][forced]["vanilla"]})
                df.to_csv(pth)
            # cat_total_in = 0
            # cat_total_out = 0
            # for c in cats:
            #     catprompt = template(text, date, uni, shot, forced, cat = c, verbose = i == 0)
            #     if i % 2000 == 0:
            #         out = get_response(catprompt)
            #         cat_total_out += out
            #     cat_total_in += len(word_tokenize(catprompt)) + systoks

            # try:
            #     in_lengths[shot][forced]["category_ablation"] += cat_total_in
            #     if i % 2000 == 0:
            #         out_lengths[shot][forced]["category_ablation"].append(cat_total_out)
            # except:
            #     in_lengths[shot][forced]["category_ablation"] = cat_total_in
            #     if i % 2000 == 0:
            #         out_lengths[shot][forced]["category_ablation"] = [cat_total_out]
            # type_total_in = 0
            # type_total_out = 0
            # for t in types:
            #     typeprompt = template(text, date, uni, shot, forced, tpe = t, verbose = i == 0)
            #     if i % 2000 == 0:
            #         out = get_response(typeprompt)
            #         type_total_out += out
            #     type_total_in += len(word_tokenize(typeprompt)) + systoks
            # try:
            #     in_lengths[shot][forced]["type_ablation"] += type_total_in
            #     if i % 2000 == 0:
            #         out_lengths[shot][forced]["type_ablation"].append(type_total_out)
            # except:
            #     in_lengths[shot][forced]["type_ablation"] = type_total_in
            #     if i % 2000 == 0:
            #         out_lengths[shot][forced]["type_ablation"] = [type_total_out]

        df = pd.DataFrame({"response": answers[shot][forced]["vanilla"],
        				   "in_tokens": in_lengths[shot][forced]["vanilla"],
        				   "out_tokens": out_lengths[shot][forced]["vanilla"]})
        df.to_csv(pth)

