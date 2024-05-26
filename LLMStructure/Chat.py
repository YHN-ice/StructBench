#!/usr/bin/env python3
import openai

# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai


# gpt-4
def api_wrapper_gpt44(messages=None):
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    # openai.api_base = "https://conmmunity-openai-3.openai.azure.com/"
    openai.api_base = "https://conmmunity-openai-990.openai.azure.com/"
    openai.api_key = '<SECRET>'  # key2

    if messages is None:
        messages = [{"role": "system", "content": "You are Dylan Thomas now."},
                    {"role": "user", "content": "Do not go gentle into"}]
    response = openai.ChatCompletion.create(
        # engine=f"gpt-4-4",
        # engine=f"gpt-4-bb",
        # engine=f"gpt-4-990",
        engine=f"gpt-4-990-32k",
        messages=messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response


# gpt-35-33/34
def api_wrapper(messages=None):
    ver = 33
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    openai.api_base = f"https://community-openai-{ver}.openai.azure.com/"
    openai.api_key = '<SECRET>' if ver == 33 else '<SECRET>'  # gpt35-33

    if messages is None:
        messages = [{"role": "system", "content": "you are Shakespear now"},
                    {"role": "user", "content": "To live or die, this is a"}]
    response = openai.ChatCompletion.create(
        engine=f"gpt35-{ver}",
        messages=messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response


# gpt3
def api_wrapper_custom(messages=None):
    openai.api_key = "<SECRET>"
    if messages is None:
        messages = [{"role": "system", "content": "you are Shakespeare now"},
                    {"role": "user", "content": "I think Twitter is "}]
    response = openai.Completion.create(
        # model="gpt-3.5-turbo",
        # model="text-davinci-003",
        model="davinci",
        # messages=messages
        prompt="Say this is a Hello."
    )

    # print(response.choices[0].message)
    return response


# gpt4
def api_wrapper_xhs(message_or_prompt):
    import json, requests
    gpt_param = {"max_tokens": 500,
                 "temperature": 0.8, "top_p": 1, "stream": False,
                 "frequency_penalty": 0.1, "presence_penalty": 0.1, "stop": None}
    gpt4_turbo_key = "<SECRET>"
    gpt4_turbo_url = 'https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2023-05-15'
    key, url = gpt4_turbo_key, gpt4_turbo_url

    if isinstance(message_or_prompt, str):
        messages = [{"role": "user", "content": message_or_prompt}]
    else:
        messages = message_or_prompt
    headers = {
        'api-key': key,
        'Content-Type': 'application/json'
    }
    payload = {
        "messages": messages,
    }
    payload.update(gpt_param)
    payload_json = json.dumps(payload)
    response_json = requests.request("POST", url, headers=headers, data=payload_json).json()
    return response_json


# openai gpt4
def api_wrapper_opanai(message_or_prompt):
    if isinstance(message_or_prompt, str):
        messages = [{"role": "user", "content": message_or_prompt}]
    else:
        messages = message_or_prompt
    from openai import ChatCompletion

    openai.api_key = "<SECRET>"
    client = ChatCompletion()

    completion = client.create(
        model="gpt-4",
        max_tokens=500,
        temperature=0.8,
        top_p=1,
        stream=False,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=None,
        messages=messages
    )

    #print(completion.choices[0].message)
    return completion


# xhs
def api_wrapper_new():
    import requests
    import json
    key = '<SECRET>'
    url = 'https://runway.devops.xiaohongshu.com/openai/resource_name/gpt4-ptu/deployments/gpt4-PTU/chat/completions?api-version=2023-05-15'

    headers = {
        'api-key': key,
        'Content-Type': 'application/json'
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Does Azure OpenAI support customer managed keys?"
            },
            {
                "role": "assistant",
                "content": "Yes, customer managed keys are supported by Azure OpenAI."
            },
            {
                "role": "user",
                "content": "Do other Azure AI services support this too?"
            }
        ]
    }
    payload_json = json.dumps(payload)
    response = requests.request("POST", url, headers=headers, data=payload_json, timeout=30).json()
    print("Here is the content", response)
    return response

