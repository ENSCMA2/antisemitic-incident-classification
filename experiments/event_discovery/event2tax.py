import os
import openai
from openai import OpenAI
import pandas as pd
import time
from itertools import chain
import json
from itertools import chain
api_key = os.environ.get("OPENAI_API_KEY")
azure_endpoint = "https://rtp2-shared.openai.azure.com/"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

df = pd.read_csv("../../data/AMCHA/anonymized_checked.csv")
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

icl_samples = []
for c in cats:
    subcat = df[(df["Category"].str.contains(cats[c]["cat"].upper()))]
    already = []
    for t in cats[c]["types"]:
        subtype = subcat[(subcat["Classification"].str.contains(types[t].upper()))]
        if len(already) > 0:
            subtype = subtype[~(subtype["Classification"].str.contains("|".join(already)))]
        samp = subtype.sample(n = 1, random_state = 0)
        already.append(types[t].upper())
        icl_samples.append(samp["Anonymized_Description"].tolist()[0])
cat_strings = "\n".join([f"Example {i + 1}: {icl_samples[i]}" for i in range(len(icl_samples))])
alphabet = "BCDEFGHIJKLMNOPQRSTUVWXYZ"

responses = []
in_toks = []
out_toks = []

def get_lesson(example):
    return f'''Here are some examples of descriptions of antisemitic events:\n{example}'''

def template(examples):
    if len(examples) == 0:
        return '''Provide a definition of antisemitism and a comprehensive taxonomy of different categories of antisemitism and their definitions.
Your answer should be formatted as follows:
{{'Definition': str,
'Taxonomy': dict}}.
For 'Taxonomy', the keys should be names of categories of antisemitism, and the values should be the definitions of those categories.'''
    lesson = get_lesson(examples)
    return f'''{lesson}\nGiven these examples, provide a definition of antisemitism and a comprehensive taxonomy of different categories of antisemitism and their definitions.
Your answer should be formatted as follows:
{{'Definition': str,
'Taxonomy': dict}}.
For 'Taxonomy', the keys should be names of categories of antisemitism, and the values should be the definitions of those categories.'''
sysprompt = "You are analyzing texts that describe events that are harmful toward Jewish people and attempting to devise a definition and taxonomy of antisemitism. Give your answer in JSON format."

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
    time.sleep(0.5)
    print(sysprompt)
    print(prompt)
    return answer, (in_tok, out_tok)

# print(get_response(template(cat_strings)))

hmd = pd.read_csv("../../data/HEATMapData.csv")
ts = hmd["type"].tolist()
items = [item.split(";") for item in ts]
adl_types = list(set(chain(*items)))
print("TYPES:", adl_types)
icl_samples = []
already = []
incident_numbers = []
for t in adl_types:
    subcat = hmd[hmd["type"].str.contains(t)]
    if len(already) > 0:
        subcat = subcat[~(subcat["type"].str.contains("|".join(already)))]
    if subcat.shape[0] > 0:
        samp = subcat.sample(n = 1, random_state = 0)
    else:
        print(t)
        sub = hmd[hmd["type"] == t]
        print(sub.shape)
        samp = sub.sample(n = 1, random_state = 0)
        while samp["id"].tolist()[0] in incident_numbers:
            samp = sub.sample(n = 1, random_state = 0)
    already.append(t)
    s = samp["description"].tolist()[0]
    incident_numbers.append(samp["id"].tolist()[0])
    icl_samples.append(s)

cat_strings = "\n".join([f"Example {i + 1}: {icl_samples[i]}" for i in range(len(icl_samples))])
# print(get_response(template(cat_strings)))

print(get_response(template([])))
