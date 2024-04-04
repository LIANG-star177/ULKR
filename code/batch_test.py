import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

class LangChainCFG:
    llm_model_name = '/data/apps/luwen/luwen_baichuan/output/zju_model_0917_110k'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = '/root/data1/liang/self-correct-retriever/checkpoint/m3e-base'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = '/root/data1/liang/self-correct-retriever/data/legal_articles'
    kg_vector_stores = {
        '法律法条': '/root/data1/liang/self-correct-retriever/data/legal_articles',
        # '法律书籍': '/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_books',
        # '法律文书模板':'/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_templates',
        # '法律案例': '/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_cases',
        # '法律考试': '/data/apps/lacode/wisdomInterrogatory-main/data/cache/judicialExamination',
        # '日常法律问答': '/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_QA',
    }  

config = LangChainCFG()
application = LangChainApplication(config)

def clear_session():
    return '', None, ""

def predict(input,
            kg_names=None,
            history=None,
            max_length=1024,
            top_k = 1,
            **kwargs):
    application.llm_service.max_token = max_length
    # print(input)
    if history == None:
        history = []
    search_text = ''

    now_input = input
    eos_token_ids = [application.llm_service.tokenizer.eos_token_id]
    application.llm_service.history = history[-5:]
    max_memory = 4096 - max_length

    kb_based = True if len(kg_names) != 0 else False

    if len(history) != 0:
        input = "".join(["</s>Human:" + i[0] + " </s>Assistant: " + i[1] for i in application.llm_service.history]) + \
        "</s>Human:" + input
        input = input[len("</s>Human:"):]
    if len(input) > max_memory:
        input = input[-max_memory:]

    if kb_based:
        related_docs_with_score_seq = []
        for kg_name in kg_names:
            # if kg_name=="法律法条":
            #     related_article = key_words_match_knowledge(application.all_articles, application.choices, now_input)
            #     if related_article:
            #         kg_matches = [(Document(page_content=related_article[0], metadata={"value": related_article[1]}),0)]
            #     else:
            #         application.source_service.load_vector_store(application.config.kg_vector_stores[kg_name])
            #         kg_matches = application.source_service.vector_store.similarity_search_with_score(input, k=top_k)
            # else:
            application.source_service.load_vector_store(application.config.kg_vector_stores[kg_name])
            kg_matches = application.source_service.vector_store.similarity_search_with_score(input, k=top_k)
            related_docs_with_score_seq.append(kg_matches)
        related_docs_with_score = related_docs_with_score_seq
        
        if len(related_docs_with_score) > 0:
            input, context_with_score = application.generate_prompt(related_docs_with_score, input,kg_names)
        search_text += context_with_score
    torch_gc()

    print("histroy in call: ", history)
    prompt = f'</s>Human:{input} </s>Assistant: ' 
    # prompt = input
    print("prompt: ",prompt)
    inputs = application.llm_service.tokenizer(prompt, return_tensors="pt").to('cuda')
    inputs = inputs.to('cuda')
    with torch.no_grad():
        pred = application.llm_service.model.generate(**inputs, max_new_tokens=max_length, repetition_penalty=1.2)
    response = application.llm_service.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return response.split("Assistant: ")[1]

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + "\n"+ input
    return input

kg_names = ["法律法条"]
max_length=800
top_k=3
prompt_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"]}
test_path = "/root/data1/liang/self-correct-retriever/data/exercise_contest/data_test.json"
results=[] 

with open(test_path,"r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data = json.loads(line)
        fact_origin = data["fact"]
        fact = get_prompt(fact_origin, prompt_dic["charge"])
        result = predict(input=fact,kg_names=kg_names,max_length=max_length,top_k=top_k)
        results.append({"fact":fact_origin,"label":data["label"],"pred":result})
with open("/root/data1/liang/self-correct-retriever/result/generation.txt", "w") as f:
    for dic in results:
        json.dump(dic,f,ensure_ascii=False)
        f.write("\n")