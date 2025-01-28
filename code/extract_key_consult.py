import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
    test_path = "/data/apps/lacode/self-correct-retriever/data/knowledge_base/train_consult.json"
    openai_api = "text-davinci-003"
config = testCFG()

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + "\n" + input
    return input

def cut_sentences(text, max_length):
    sentences = text.split('。')
    sentences = [el for el in sentences if len(el)>0]
    summary = ''
    char_count = 0
    for i, sentence in enumerate(sentences):
        if char_count + len(sentence)+1 <= max_length:
            summary += sentence + '。'
            char_count += len(sentence)+1
        else:
            if i==0:
                summary = text[:max_length]
            break
    return summary

def test(config):
    with open(config.test_path,"r") as f: 
        lines = f.readlines()
        random.shuffle(lines)
        knowledge = []
        i = 0
        for line in tqdm(lines):
            data = json.loads(line) 
            fact = cut_sentences(data["question"],512)
            view = cut_sentences(data["answer"],512)
            fact_judge = "问题："+fact + "答案："+view
            extract_key_fact = get_prompt(fact_judge, config.prompt_dic["case"])
            key_el = get_openai_response([extract_key_fact], combine_way="answer",config=config)[0]
            key_el_lst = key_el.split("\n")
            new_key_el_lst = []
            for el in key_el_lst:
                if len(el)>0:
                    new_key_el_lst.append(re.sub("[^，、0-9\u4e00-\u9fff]+", "", el))
            knowledge.append({"key":fact,"key2":new_key_el_lst,"value":fact_judge, "art_detail":data["art_detail"],"path":[data["art"]]})
            i+=1
        write_path = "/data/apps/lacode/self-correct-retriever/data/knowledge_base/train_key_consult.json.json"
        with open(write_path, "w") as f:
            random.shuffle(knowledge)
            for dic in knowledge:
                json.dump(dic,f,ensure_ascii=False)
                f.write("\n")
        print(len(knowledge))   

test(config)