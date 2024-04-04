# -*- coding: utf-8 -*-

import openai
import os
import json
import threading
import time
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel
import re
from clc.langchain_application import LangChainApplication, torch_gc
from luwen import luwen_response, search
from utils.few_shot import cut_sentences, extra_knowledge_handler
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
openai.api_base = "https://fast.xeduapi.com/v1"
key_pool = [
    "sk-8vR1Vy0oiYWH751RF08a305fFa2c47Ba84579149Ae4a0fB5",
]
openai.api_key = key_pool[0]

prompt_templates = {
    "abstract":("问题：{question}\n请摘要上述问题，不超过450字。"),

    # "drop":("问题：{question}\n参考知识：\n{context}\n请给每一个知识与问题之间的相关度打分（满分10分），请告诉我分数最低的知识的编号，并给出打分的理由。"),
    "drop":("问题：{question}\n参考知识：\n{context}\n请问哪一个知识与问题无关？请告诉我知识的数字编号(注意不要回答问题)和你的理由。"),

    # "combine":("案件：{question}\n参考知识：{context}\n请保留参考知识中和案件相关的句子，删除其他无用信息。"),
    # "combine":("知识：{context}\n如果知识是法院观点，请摘要重要部分；如果知识是法条定义或罪名定义，请摘要重要部分。"),#cvg
    # "combine":("问题：{question}\n参考知识：{context}\n请保留参考知识中和问题相关的句子，删除其他无用信息(注意不要回答问题)。"),#artrecite
    # "combine":("问题：{question}\n参考知识：{context}\n请删去参考知识中和问题无关的句子。(有一点相关也要保留)"),#consult
    "combine":("问题：{question}\n下面提供可能与问题相关的资料，你的任务是删除和参考知识中和问题无关的部分。参考知识：{context}"),#qa

    # "extra":("问题：{question}\n参考知识：{context}\n要回答这个问题，除了上述知识，还需要了解什么额外知识(注意不要回答问题)？"),#consult
    "extra":("问题：{question}\n参考知识：{context}\n除了上述知识，还需要了解什么额外知识？(注意不要回答问题)"),#fk
    # "extra":("问题：{question}\n参考知识：{context}\n要回答这个问题，提供的法条知识正确吗？如果不正确，应该提供什么法条知识？"),
    
    "answer":("{question}\n下面提供可能与问题相关的资料，你可以参考以下资料如果它们与上面的文本内容有关。\n{context}"),
    # "answer":("{question}\n下面提供可能与问题相关的案例的法院观点，你可以参考以下法院观点的内容和格式回答问题。\n{context}"),
    # "answer":("{question}\n下面提供可能与问题相关的参考知识:\n{context}"),
}

def dav_response(text_list, model_name):
    enc = tiktoken.get_encoding("p50k_base")
    max_token = [len(enc.encode(t)) for t in text_list]
    max_token = 4096 - max(max_token)-500
    # max_gen_token = 600
    # max_token = 1600 - max_gen_token
    # text_list = [el[:max_token] for el in text_list]
    response = openai.Completion.create(
        model=model_name, prompt=text_list, max_tokens=max_token, temperature = 0.1)  # greedy
    return response

def dav2_response(text_list):
    return dav_response(text_list, "text-davinci-002")

def dav3_response(text_list):
    return dav_response(text_list, "text-davinci-003")

def turbo_response(text_list):
    assert len(text_list) == 1, "gpt-3.5-turbo使用batch必须为1"
    text = text_list[0]
    enc = tiktoken.get_encoding("cl100k_base")
    max_token = 16385 - len(enc.encode(text)) - 500
    # (1608 in the messages, 2496 in the completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0.1)
    return response

def chatglm_response(text_list):
    tokenizer = AutoTokenizer.from_pretrained("/root/data1/liang/self-correct-retriever/baseline/ChatGLM-6B-main/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/root/data1/liang/self-correct-retriever/baseline/ChatGLM-6B-main/chatglm-6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, text_list[0], history=[])
    return response

llm_response_function = {
    "text-davinci-002": dav2_response,
    "text-davinci-003": dav3_response,
    "gpt-3.5-turbo": turbo_response,
    "luwen": luwen_response,
    "chatglm": chatglm_response,
}

def add_knowledge(query, in_context_contents, combine_way="extra", config=None):
    #如果是openai处理，缩减句子长度
    if combine_way=="drop":
        query = [cut_sentences(el,1024) for el in query]
    if combine_way=="combine" or combine_way=="abstract" or combine_way=="answer":
        query = [cut_sentences(el,512) for el in query]
    # if combine_way=="combine":
    #     new_query = []
    #     for i in range(len(query)): 
    #         context = "" 
    #         for j in range(len(in_context_contents[i])):
    #             context += in_context_contents[i][j]
    #         context = context+query[i]+"\n解析为"
    #         new_query.append(context)
    #     return new_query

    if in_context_contents == None:
        return query
    
    prompt_list = []
    for k in range(len(query)):
        context = []
        for i in range(len(in_context_contents[k])):
            context.append("知识{}".format(i+1) + in_context_contents[k][i])
        context = "\n".join(context)
        prompt_template = prompt_templates[combine_way]
        if combine_way == "answer" and config.self_correct:
            context = in_context_contents[k][0]
        prompt = prompt_template.replace("{question}", query[k]).replace("{context}", context)
        prompt_list.append(prompt)
    return prompt_list

def get_knowledge_combination(query, feedback, in_context_contents, config):
    data_all=[] 
    for i in range(len(feedback)):
        if feedback[i] == "can not combine!":
            config.count_dic["can_not_combine"]+=1
            response_function = llm_response_function[config.mlm_api]
            prompt = add_knowledge([query[i]], [in_context_contents[i]], "extra")
            response = response_function(prompt[0])
            print(response)
            if "不需要再提供" not in response:
                config.count_dic["need_extra"]+=1
                response = extra_knowledge_handler(response)
                # new_query = query[i]+response
                new_query = response
                new_knowledge = search(input=new_query,kg_names=["融合知识"],top_k=1)[0]
                if new_knowledge not in in_context_contents[i]:
                    in_context_contents[i].append(new_knowledge)
                in_context_contents[i].append(cut_sentences(new_query, 512))
        else:
            response = "no extra!"
        data_all.append({"input":query[i],"extra":response, "context":in_context_contents[i]})
    return in_context_contents, data_all

def chinese2num(text):
    chinese2num_dic = {"一":"1","二":"2","三":"3"}
    if text not in chinese2num_dic.keys():
        return int(text)
    else:
        return int(chinese2num_dic[text])

def few_shot(query, in_context_contents):
    new_query = []
    for i in range(len(query)):
        new_query.append("".join(in_context_contents[i])+query[i])
    return new_query

def get_llm_response(query, in_context_contents=None, combine_way="answer", config = None):
    print("*"*10+combine_way+"*"*10)
    combine_labels, new_knowledge, data_all = [],[],[]
    if config.mlm_api == "chatglm":
        tokenizer = AutoTokenizer.from_pretrained("/root/data1/liang/self-correct-retriever/baseline/ChatGLM-6B-main/chatglm-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("/root/data1/liang/self-correct-retriever/baseline/ChatGLM-6B-main/chatglm-6b", trust_remote_code=True).half().cuda()
        model = model.eval()
    else:
        response_function = llm_response_function[config.mlm_api]
    
    if in_context_contents:
        # query = add_knowledge(query, in_context_contents, combine_way, config)
        query = few_shot(query, in_context_contents)
    for i in tqdm(range(len(query))):
        if config.mlm_api == "chatglm":
            answer = model.chat(tokenizer, query[i], history=[])[0]
        else:
            answer = response_function(query[i])
        print(answer)
        if in_context_contents:
            data_all.append({"input":query[i],"context":in_context_contents[i],"resp":answer})
        if combine_way=="drop":
            flag = False
            sents = answer.split("。")
            new_context_lst = in_context_contents[i].copy()
            for sent in sents:
                match = re.search(r'(知识([一二三0-9]+))|(第([一二三0-9]+)个知识)', sent)
                # if match and "无关" in sent:
                if match:
                    if match.group(2):
                        index = chinese2num(match.group(2)) - 1
                        if 0 <= index < len(new_context_lst):
                            tmp = new_context_lst.pop(index)
                        else:
                            flag = False
                    elif match.group(4):
                        index = chinese2num(match.group(4)) - 1
                        if 0 <= index < len(new_context_lst):
                            tmp = new_context_lst.pop(index)
                        else:
                            flag = False
                    combine_labels.append("can not combine!")
                    flag = True
                    break
            if not flag:
                combine_labels.append("can combine!")
            new_knowledge.append(new_context_lst)
        if combine_way=="combine":
            new_knowledge.append([answer])
        if combine_way=="answer":
            new_knowledge.append(answer)
    if combine_way=="drop":
        return combine_labels, new_knowledge, data_all
    if combine_way=="combine":
        return new_knowledge, data_all
    if combine_way=="answer":
        return new_knowledge

def get_openai_response(query, in_context_contents=None, combine_way=None, config = None):
    print("*"*10+combine_way+"*"*10)
    response_function = llm_response_function[config.openai_api]
    # if combine_way=="drop":
    #     query = [el.replace("请问下面案件中被告人犯的是什么罪？\n","") for el in query]
    prompt = add_knowledge(query, in_context_contents, combine_way, config)
    combine_labels, droped_knowledge, combined_knowledge=[], [], []
    i = 0
    batch = 1 if config.openai_api=="gpt-3.5-turbo" else 4
    data_all = []
    while i < len(query):
        responses = []
        prompt_list = prompt[i:i+batch]
        if in_context_contents:
            in_context_batch = in_context_contents[i:i+batch]
        try:
            response = response_function(prompt_list)
            resp_temp = response.copy()
            for run in range(len(prompt_list)):
                if config.openai_api=="gpt-3.5-turbo":
                    resp = resp_temp["choices"][run]["message"]["content"]#3.5
                if config.openai_api=="text-davinci-003" or config.openai_api=="text-davinci-002":
                    resp = resp_temp["choices"][run]["text"]#3
                if in_context_contents:
                    data_all.append({"input":prompt_list[run],"context":in_context_batch[run],"resp":resp})
                else:
                    data_all.append({"input":prompt_list[run],"resp":resp})
                resp = resp.replace("\n答案","")
                responses.append(resp)
                print(resp)
        except openai.error.RateLimitError as e:
            e = repr(e)
            print(e)
            for run in range(len(prompt_list)):
                responses.append(prompt_list[run])
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
            for run in range(len(prompt_list)):
                responses.append(prompt_list[run])
            e = repr(e)
            print(e)
            time.sleep(60)
        except Exception as e:
            for run in range(len(prompt_list)):
                responses.append(prompt_list[run])
            e = repr(e)
            print(e)
            exit()  
        time.sleep(1)
        i += batch
        
        if combine_way=="drop":
            for k in range(len(responses)):
                flag = False
                sents = responses[k].split("。")
                new_context_lst = in_context_contents[k].copy()
                for sent in sents:
                    match = re.search(r'(知识([一二三0-9]+))|(第([一二三0-9]+)个知识)', sent)
                    if match:
                        if match.group(2):
                            index = chinese2num(match.group(2)) - 1
                            if 0 <= index < len(new_context_lst):
                                tmp = new_context_lst.pop(index)
                            else:
                                flag = False
                        elif match.group(4):
                            index = chinese2num(match.group(4)) - 1
                            if 0 <= index < len(new_context_lst):
                                tmp = new_context_lst.pop(index)
                            else:
                                flag = False
                        combine_labels.append("can not combine!")
                        flag = True
                        break
                if not flag:
                    combine_labels.append("can combine!")
                droped_knowledge.append(new_context_lst)
            if i>=len(query):
                return combine_labels, droped_knowledge, data_all
        # if combine_way=="drop":
        #     for k in range(len(responses)):
        #         matches = re.findall(r'知识([1-9])', responses[k])
        #         new_context_lst = in_context_batch[k].copy()
        #         if matches:
        #             for el in set(matches):
        #                 tmp = new_context_lst.pop(int(el)-1)
        #             combine_labels.append("can not combine!")
        #         else:
        #             combine_labels.append("can combine!")
        #         droped_knowledge.append(new_context_lst)
        #     if i>=len(query):
        #         return combine_labels, droped_knowledge, data_all
        if combine_way=="combine":
            for k in range(len(responses)):
                combined_knowledge.append([responses[k]])
            if i>=len(query):
                return combined_knowledge, data_all
        if combine_way=="answer":
            for k in range(len(responses)):
                combined_knowledge.append(responses[k])
            if i>=len(query):
                return combined_knowledge