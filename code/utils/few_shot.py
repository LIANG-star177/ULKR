import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import torch
import gradio as gr
from transformers import StoppingCriteriaList, StoppingCriteriaList
import random
from tqdm import tqdm
import json
from langchain.schema import Document
# 调试使用
# os.chdir("../../../")
random.seed(2023)


def get_cases_from_test(k):
    with open("/root/data1/luwen/app/langchain_multi_stage/data/exercise_contest/opinion_test.json", "r", encoding="utf-8") as f:
        q,a=[],[]
        text = f.readlines()
        random_lines = random.sample(text, k)
        for line in random_lines:
            data = json.loads(line)
            q.append(data["input"])
            a.append(data["label"])
        return q,a

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

def extra_knowledge_handler(text):
    index = text.find('：')
    if index != -1:
        text = text[index + 1:].strip()
    for el in ["综上所述","总之"]:
        index = text.find(el)
        if index != -1:
            text = text[:index]
            break
    return text