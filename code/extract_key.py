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
    prompt_dic = {"case":["你是一个法律专家，请提取下面文本中的具有重要法律意义的n_gram（不包括人名和地点），只需告诉我你的答案，并用换行隔开每个n_gram。"],
                  "article":["你是一个法律专家，请提取下面文本中的具有重要法律意义的n_gram（如人物和行为等），只告诉我你的答案，并用换行隔开每个n_gram。"]}
    test_path = "/root/data1/liang/self-correct-retriever/data/hera_knowbase/法条/刑法"
    openai_api = "text-davinci-003"
config = testCFG()

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + "\n" + input
    return input

def test(config):
    inputs, in_context_contents, results, labels=[],[],[],[]
    if "法条" in config.test_path:  
        output_directory = "/root/data1/liang/self-correct-retriever/data/hera_knowbase/法条/新刑法"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory) 
        directory = config.test_path
        for filename in os.listdir(directory):
            if filename.endswith('.json'): 
                knowledge = []
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for line in lines:
                        data = json.loads(line) 
                        fact = data["value"]
                        extract_key_fact = get_prompt(fact, config.prompt_dic["article"])
                        key_el = get_openai_response([extract_key_fact], combine_way="answer",config=config)[0]
                        key_el_lst = key_el.split("\n")
                        new_key_el_lst = []
                        for el in key_el_lst:
                            if len(el)>0:
                                new_key_el_lst.append(re.sub("[^，、0-9\u4e00-\u9fff]+", "", el))
                        knowledge.append({"key":fact,"key2":new_key_el_lst,"value":fact})
            output_file_path = os.path.join(output_directory, filename)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for dic in knowledge:
                    json.dump(dic,output_file,ensure_ascii=False)
                    output_file.write("\n")

    else:
        with open(config.test_path,"r") as f: 
            lines = f.readlines()
            random.shuffle(lines)
            knowledge = []
            i = 0
            for line in tqdm(lines[:1000]):
                data = json.loads(line) 
                fact = data["fact"]
                fact_judge = fact + "根据法条第"+str(data["article"])+"条,被告人犯了"+data["charge"]+"罪,被判处有期徒刑"+str(data["penalty"])+"个月。"
                extract_key_fact = get_prompt(fact, config.prompt_dic["case"])
                key_el = get_openai_response([extract_key_fact], combine_way="answer",config=config)[0]
                key_el_lst = key_el.split("\n")
                new_key_el_lst = []
                for el in key_el_lst:
                    if len(el)>0:
                        new_key_el_lst.append(re.sub("[^，、0-9\u4e00-\u9fff]+", "", el))
                knowledge.append({"key":fact,"key2":new_key_el_lst,"value":fact_judge})
                i+=1
        write_path = "/root/data1/liang/self-correct-retriever/data/knowledge_base/cases.json"
        with open(write_path, "w") as f:
            random.shuffle(knowledge)
            for dic in knowledge:
                json.dump(dic,f,ensure_ascii=False)
                f.write("\n")
        print(len(knowledge))   

test(config)