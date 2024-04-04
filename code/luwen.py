import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import torch
from clc.langchain_application import LangChainApplication, torch_gc
from utils.few_shot import cut_sentences
import random
from tqdm import tqdm
import json
# 调试使用
# os.chdir("../../../")
random.seed(2023)

class LangChainCFG:
    llm_model_name = '/data/apps/luwen/luwen_baichuan/output/zju_model_0917_110k'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = '/root/data1/liang/self-correct-retriever/checkpoint/m3e-base'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = '/root/data1/liang/self-correct-retriever/data/cache/combine_base'
    kg_vector_stores = {
        '法律法条': '/root/data1/liang/self-correct-retriever/data/articles_base',
        # '法律书籍': '/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_books',
        # '法律文书模板':'/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_templates',
        # '法律案例': '/root/data1/liang/self-correct-retriever/data/cache/legal_cases',
        # '法律考试': '/root/data1/liang/self-correct-retriever/data/exams_base',
        # '日常法律问答': '/data/apps/lacode/wisdomInterrogatory-main/data/cache/legal_QA',
    }  
config = LangChainCFG()

def combine_knowledge_dir(source_folders,target_folder):
    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for source_folder in source_folders:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith(".json"):
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_folder, file)
                    shutil.copy2(source_file, target_file)

if len(config.kg_vector_stores)>=1:
    target_folder = "/root/data1/liang/self-correct-retriever/data/cache/combine_base"
    combine_knowledge_dir([val for key,val in config.kg_vector_stores.items()],target_folder)
    config.kg_vector_stores = {'融合知识':target_folder}

application = LangChainApplication(config)
#更换知识库时更新向量库。
# application.source_service.init_source_vector()


def luwen_response(input, max_length=800):
    max_memory = 4096 - max_length
    prompt = f'</s>Human:{input} </s>Assistant: ' 
    # print("prompt: ",prompt)
    if len(prompt) > max_memory:
        prompt = prompt[:max_memory]
    inputs = application.llm_service.tokenizer(prompt, return_tensors="pt").to('cuda')
    inputs = inputs.to('cuda')
    with torch.no_grad():
        pred = application.llm_service.model.generate(**inputs, max_new_tokens=max_length, repetition_penalty=1.2, temperature=0.1)
    response = application.llm_service.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    result = response.split("Assistant: ")
    if len(result)>1:
        return result[1]
    else:
        return result[0]

def search(input,
            kg_names=None,
            top_k = 1,):
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
            related_docs_with_score_seq.append([cut_sentences(el[0].metadata["value"],512) for el in kg_matches])
        related_docs_with_score = related_docs_with_score_seq[0]
        return related_docs_with_score

def pred_with_knowledge(input,related_docs_with_score,kg_names, max_length=1024):
    if len(related_docs_with_score) > 0:
        input, context_with_score = application.generate_prompt(related_docs_with_score, input,kg_names)
    response = luwen_response(input, max_length=max_length)
    return response

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + "\n"+ input
    return input