import os
import json
import time
import requests
import argparse
import pandas as pd
import random

MAX_ATTEMPTS = 10
API_KEY = os.getenv("OPENAI_API_CMU")
API_ORG = os.getenv("OPENAI_API_ORG_CMU")

def retry_request(url, payload, headers):
    for i in range(MAX_ATTEMPTS):
        try:
            response = requests.post(url, data=json.dumps(
                payload), headers=headers, timeout=90)
            json_response = json.loads(response.content)
            if "error" in json_response:
                print(json_response)
                print(f"> Sleeping for {2 ** i}")
                time.sleep(2 ** i) 
            else:
                return json_response
        except:
            print(f"> Sleeping for {2 ** i}")
            time.sleep(2 ** i)  # exponential back off
    raise TimeoutError()

def query_model(
    dataset,
    model: str = 'gpt-3.5-turbo-0613',
    temperature: float = 0.7,
    n_gen: int = 1,
    max_tokens: int = 256
):
 
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-type': 'application/json',
        'Accept': 'application/json',\
        'Authorization': f'Bearer {API_KEY}', \
        'OpenAI-Organization': API_ORG
    }

    print(f"> Prompting {model}")
    prompt = "Does this comment target Jewish people?"
    payload_data = {"role": "user", "content": prompt}
    payload = {"messages": [payload_data], 
               "temperature": temperature, 
               "model": model, 
               "n": n_gen,
               "max_tokens": max_tokens}
    response = retry_request(url, payload, headers)

    usage = []
    if "choices" in response:
        answers = [choice["message"]["content"].strip() for choice in response["choices"]]
        usage += [response["usage"]]
        res += [{"answers": answers, "usage": usage}]
    else:
        print("> Error!")
    
