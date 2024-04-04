# 处理法院观点数据
from tqdm import tqdm
import random
import json
import openai
# -*- coding: utf-8 -*-

import openai
import os
import json
import threading
import time
import torch
from tqdm import tqdm
import argparse
import re
# 调试使用
import sys 
sys.path.insert(0, sys.path[0]+"/../")
from luwen import luwen_response, search
import tiktoken

# 设置代理
# os.environ["http_proxy"] = "http://10.162.19.200:6789"
# os.environ["https_proxy"] = "http://10.162.19.200:6789"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# key_pool = [
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
#     "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",
# ]
# openai.api_key = key_pool[0]
openai.api_base = "https://api.xiamoai.top/v1"
key_pool = [
    "sk-JsDmtzIuCup9tENA55Bb3755D0Ba4839B3D3D4Da0576D183",
]
openai.api_key = key_pool[0]

prompt_templates = {
    # "drop":("{question}\n下面是参考知识。要上述问题，下面哪一个知识是没有用的？请告诉我知识的序号。\n{context}"),
    "drop":("问题：{question}\n参考知识：\n{context}\n请问哪一个知识与问题无关？请告诉我知识的序号。"),
    "combine":("{question}\n参考知识：{context}\n请把参考知识中与问题相关的知识提取出来。"),
    "extra":("{question}\n{context}\n要回答这个问题，除了上述知识，你还需要我提供什么知识？请给出两个所需的额外知识。"),
    "answer":("{question}下面提供可能与对话相关的资料，你可以参考以下资料如果它们与上面的文本内容有关。\n{context}")
}

def dav_response(text_list, model_name):
    # enc = tiktoken.get_encoding("p50k_base")
    # max_token = [len(enc.encode(t)) for t in text_list]
    # max_token = 4096 - max(max_token)
    max_gen_token = 500
    max_token = 2000 - max_gen_token
    text_list = [el[:max_token] for el in text_list]
    response = openai.Completion.create(
        model=model_name, prompt=text_list, max_tokens=max_gen_token)  # greedy
    return response

def dav2_response(text_list):
    return dav_response(text_list, "text-davinci-002")

def dav3_response(text_list):
    return dav_response(text_list, "text-davinci-003")

def turbo_response(text_list):
    assert len(text_list) == 1, "gpt-3.5-turbo使用batch必须为1"
    text = text_list[0]
    enc = tiktoken.get_encoding("cl100k_base")
    max_token = 4096 - len(enc.encode(text)) - 500
    # (1608 in the messages, 2496 in the completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0)
    return response

llm_response_function = {
    "text-davinci-002": dav2_response,
    "text-davinci-003": dav3_response,
    "gpt-3.5-turbo": turbo_response,
    "luwen": luwen_response,
}

def add_knowledge(query, in_context_contents, combine_way="extra"):
    prompt_list = []
    for k in range(len(query)):
        context = []
        for i in range(len(in_context_contents[k])):
            context.append("知识{}".format(i+1) + in_context_contents[k][i])
        context = "\n".join(context)
        prompt_template = prompt_templates[combine_way]
        prompt = prompt_template.replace("{question}", query[k]).replace("{context}", context)
        prompt_list.append(prompt)
    return prompt_list

def get_knowledge_combination(query, feedback, in_context_contents,count_dic):
    data_all=[]
    for i in range(len(feedback)):
        if feedback[i] == "can not combine!":
            count_dic["can_not_combine"]+=1
            response_function = llm_response_function["luwen"]
            prompt = add_knowledge([query[i]], [in_context_contents[i]], "extra")
            response = response_function(prompt[0])
            if "不需要再提供" not in response:
                count_dic["need_extra"]+=1
                new_query = query[i]+response
                new_knowledge = search(input=new_query,kg_names=["融合知识"],top_k=1)[0]
                if new_knowledge not in in_context_contents[i]:
                    in_context_contents[i].append(new_knowledge)
        data_all.append({"input":query[i],"context":in_context_contents[i]})
    return in_context_contents, data_all

def get_answer(query, combined_knowledge):
    responses = []
    response_function = llm_response_function["luwen"]
    prompt = add_knowledge(query, combined_knowledge, "answer")
    for el in tqdm(prompt[:50]):
        responses.append(response_function(el))
    return responses

def get_openai_response(query, in_context_contents, combine_way="extra"):
    response_function = llm_response_function["text-davinci-003"]
    # if combine_way=="drop":
    #     query = [el.replace("请问下面案件中被告人犯的是什么罪？\n","") for el in query]
    prompt = add_knowledge(query, in_context_contents, combine_way)
    combine_labels, droped_knowledge, combined_knowledge=[], [], []
    i = 0
    batch = 4
    data_all=[]
    while i < len(query):
        prompt_list = prompt[i:i+batch]
        in_context_batch = in_context_contents[i:i+batch]
        try:
            response = response_function(prompt_list)
            resp_temp = response.copy()
            responses = []
            for run in range(len(prompt_list)):
                data_all.append({"input":prompt_list[run],"context":in_context_batch[run],"resp":resp_temp["choices"][run]["text"]})
                responses.append(resp_temp["choices"][run]["text"])
                print(resp_temp["choices"][run]["text"])
                # responses.append(resp_temp["choices"][run]["message"]["content"])
        except openai.error.RateLimitError as e:
            e = repr(e)
            print(e)
            if "limit" in e:
                time.sleep(60)
            elif "quota" in e:
                if len(key_pool) == 0:
                    print("用光了key！")
                    exit()
                print(f"=== 当前key: {key_pool[0]} ===")
                openai.api_key = key_pool[0]
                key_pool = key_pool[1:]
                time.sleep(1)
            continue
        except openai.error.APIError as e:
            e = repr(e)
            print(e)
            time.sleep(60)
        except Exception as e:
            e = repr(e)
            print(e)
            exit()  
        time.sleep(1)
        i += batch

        if combine_way=="drop":
            for k in range(len(responses)):
                matches = re.findall(r'知识([1-9])', responses[k])
                new_context_lst = in_context_batch[k].copy()
                if matches:
                    for el in set(matches):
                        tmp = new_context_lst.pop(int(el)-1)
                    combine_labels.append("can not combine!")
                else:
                    combine_labels.append("can combine!")
                droped_knowledge.append(new_context_lst)
            if i>=len(query):
                return combine_labels, droped_knowledge, data_all
        else:
            for k in range(len(responses)):
                combined_knowledge.append([responses[k]])
            if i>=len(query):
                return combined_knowledge, data_all

path = "/root/data1/liang/self-correct-retriever/result/chatgpt_combine.json"
write_path = "/root/data1/liang/self-correct-retriever/test_result/combined.json"

with open(path,"r") as f:
    lines = f.readlines()
    result = []
    for line in tqdm(lines):
        data = json.loads(line)
        input = data["input"]
        input = input.split("\n参考知识：")[0]
        context = data["context"]
        combined_knowledge, combine_data = get_openai_response([input], [context], combine_way="combine")
        responses = get_answer([input], combined_knowledge)
        print(combined_knowledge)
        print(responses)
