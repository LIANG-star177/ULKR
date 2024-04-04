import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
import torch
import gradio as gr
from clc.langchain_application import LangChainApplication, torch_gc
from transformers import StoppingCriteriaList, StoppingCriteriaList
import random
from tqdm import tqdm
import json
from clc.callbacks import Iteratorize, Stream
from clc.matching import key_words_match_intention, key_words_match_knowledge
from langchain.schema import Document
from llm_api import get_openai_response
# 调试使用
# os.chdir("../../../")
random.seed(2023)

from luwen import search, luwen_response, pred_with_knowledge
from llm_api import get_openai_response, get_knowledge_combination, get_llm_response

class testCFG:
    prompt_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"],
              "cvg":["请写出下面案件的法院观点。"]}
    count_dic = {"can_not_combine":0,
                "need_extra":0}
    task = "cvg"
    test_path = "/root/data1/liang/self-correct-retriever/data/new_cvg_test.json"
    self_correct = True  

config = testCFG()

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + input
    return input

def write_log(data_all,combine_way):
    with open("/root/data1/liang/self-correct-retriever/result/chatgpt_{}.json".format(combine_way), "w", encoding="utf-8") as f:
        for resp in data_all:
            line = json.dumps(resp, ensure_ascii=False)
            f.write(line + "\n")

def test(config):
    inputs, in_context_contents, results, labels=[],[],[],[]
    with open(config.test_path,"r") as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in tqdm(lines[:4]):
            data = json.loads(line) 
            if config.task=="cvg":
                fact = data["input"]
                if not config.self_correct:
                    fact = get_prompt(fact, config.prompt_dic["cvg"])
                labels.append(data["label"])
            if config.task=="ljp":
                fact = data["fact"]
                fact = get_prompt(fact, config.prompt_dic["charge"])
                labels.append(data["meta"]["accusation"])
            in_context_content = search(input=fact, kg_names=["融合知识"], top_k=3)
            inputs.append(fact)
            in_context_contents.append(in_context_content)
    if config.self_correct:
        tiny_inputs, tiny_query_log = get_openai_response(query=inputs,combine_way="abstract")
        tiny_inputs = [get_prompt(el[0], config.prompt_dic["cvg"]) for el in tiny_inputs]   
        inputs = [get_prompt(el, config.prompt_dic["cvg"]) for el in inputs]      
        write_log(tiny_query_log, "abstract")
        combine_labels, droped_knowledge,drop_data = get_openai_response(tiny_inputs, in_context_contents, combine_way="drop")
        write_log(drop_data, "drop")
        # responses = get_llm_response(inputs, droped_knowledge)
        optimized_knowledge, optimized_data = get_knowledge_combination(inputs, combine_labels, droped_knowledge, config.count_dic)
        write_log(optimized_data, "optimize")
        combined_knowledge, combine_data = get_openai_response(tiny_inputs, optimized_knowledge, combine_way="combine")
        write_log(combine_data, "combine")
        responses = get_llm_response(inputs, combined_knowledge)
    else:
        responses = get_llm_response(inputs, in_context_contents)
        # responses = get_llm_response(inputs)
    for i in range(len(inputs)):
        results.append({"fact":inputs[i],"label":labels[i],"pred":responses[i]})

    name_add = "self_corr_" if config.self_correct else ""
    with open("/root/data1/liang/self-correct-retriever/test_result/"+name_add+"re_generation_{}.txt".format(config.task), "w") as f:
        for dic in results:
            json.dump(dic,f,ensure_ascii=False)
            f.write("\n") 
    print(config.count_dic)
test(config)