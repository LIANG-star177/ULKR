import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from tqdm import tqdm
import json
import torch
import jsonlines
import jieba
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from clc.callbacks import Iteratorize, Stream
from clc.matching import key_words_match_intention, key_words_match_knowledge
from langchain.schema import Document
from utils.llm_api2 import get_openai_response, get_llm_response
from utils.BM25 import BM25, get_tokens
from gensim.summarization import bm25
from simcse_train_wgnn import SimcseModel
import numpy as np
import re
from transformers import BertConfig, BertModel, BertTokenizer
from utils.gnn_utils import init_knowledge_base, LocalDocQA, compute_law_metric, gnn_fusion, GGATModel, graph_data, compute_case_metric
# 调试使用
# os.chdir("../../../")
random.seed(2023)

from luwen import search, luwen_response, pred_with_knowledge

# 法考题的知识长度需要512,在2个知识下做实验
class testCFG:
    prompt_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"],
              "cvg":["你是一个法律专家，请根据下面违法事实写出法院观点。\n"],
              "Article Recitation":["回答以下问题，只需直接给出法条内容:"],
              "Question Answering":["请你运用法律知识从A,B,C,D中选出一个正确的答案，并写在[正确答案]和<eoa>之间。例如[正确答案]A<eoa>。请你严格按照这个格式回答。"],
              "Consult":["你是一个法律专家，请依据法律回答下面问题。\n"]}
    count_dic = {"can_not_combine":0,
                "need_extra":0}
    task = "Consult"
    # openai_api = "text-davinci-002"
    # openai_api = "text-davinci-003"
    mlm_api = "chatglm"
    retriever = "sailer"
    top_k = 3
    test_path_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"],
              "cvg":"/data/apps/lacode/self-correct-retriever/data/knowledge_base/test_cvg.json",
              "ljp":"/data/apps/lacode/self-correct-retriever/data/knowledge_base/single_test_ljp.json",
              "Question Answering":"/data/apps/lacode/self-correct-retriever/baseline/LawBench-main/data/zero_shot/1-2.json",
              "Consult":"/data/apps/lacode/self-correct-retriever/data/knowledge_base/test_consult.json"}

    know_path_dic = {"ljp":"/data/apps/lacode/self-correct-retriever/data/hera_knowbase3/类案",
                     "cvg":"/data/apps/lacode/self-correct-retriever/data/hera_knowbase3/法院观点",
                     "Consult":"/data/apps/lacode/self-correct-retriever/data/hera_knowbase3/法律咨询",

    }
    retrieval_model_dic = {"contriever":'/data/apps/lacode/wisdomInterrogatory-main/model/m3e-base',
                           "ulr":"/data/apps/lacode/self-correct-retriever/saved_model/simcse_consult_49_50.pt",
                           "bm25":None,
                           "sailer":"/data/apps/lacode/self-correct-retriever/baseline/SAILER/checkpoint-60000"
    }
    vector_store_dic = {"contriever":None,
                        "ulr":"/data/apps/lacode/self-correct-retriever/data/hera_knowbase3/法律咨询",
                        "bm25":None,
                        "sailer":None,
    }
    test_num = 512
    curr_prompt = ""

config = testCFG()

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    return curr_prompt

def compute_metric(ctxs, labels,know_labels, config):
    score_dict = {}
    if config.task=="ljp":
        is_chars_matchs, is_arts_matchs = [], []
        is_chars_matchs2, is_arts_matchs2 = [], []
        for i in range(len(labels)):
            ctxs_chars, ctxs_arts = [], []
            cnt_char, cnt_art = 0,0
            for el in ctxs[i]:
                pattern = r"根据法条第(.+?)条,被告人犯了(.+?)罪"
                matches = re.search(pattern, el)
                if matches:
                    article = matches.group(1)
                    charge = matches.group(2)
                    if article==labels[i][0]:
                        cnt_art+=1
                    if charge==labels[i][1]:
                        cnt_char+=1
                    ctxs_arts.append(article)
                    ctxs_chars.append(charge)
            is_arts_matchs2.append(cnt_art/len(ctxs[i]))
            is_chars_matchs2.append(cnt_char/len(ctxs[i]))
            if labels[i][0] in ctxs_arts:
                is_arts_matchs.append(1)
            else:
                is_arts_matchs.append(0)
            if labels[i][1] in ctxs_chars:
                is_chars_matchs.append(1)
            else:
                is_chars_matchs.append(0)
        score_dict['char_match_acc'] = float(np.mean(is_chars_matchs))
        score_dict['art_match_acc'] = float(np.mean(is_arts_matchs))
        score_dict['char_match_r_pre'] = float(np.mean(is_chars_matchs2))
        score_dict['art_match_r_pre'] = float(np.mean(is_arts_matchs2))
    if config.task in ["cvg","Consult"]:
        is_chars_matchs,is_chars_matchs2 = [], []
        for i in range(len(labels)):
            cnt_char = 0
            if labels[i][1] in know_labels[i]:
                is_chars_matchs.append(1)
            else:
                is_chars_matchs.append(0)
            for el in know_labels[i]:
                if el==labels[i][1]:
                    cnt_char+=1
            is_chars_matchs2.append(cnt_char/len(know_labels[i]))
        score_dict['case_match_acc'] = float(np.mean(is_chars_matchs))
        score_dict['case_match_r_pre'] = float(np.mean(is_chars_matchs2))
    return score_dict

def write_log(data_all,config):
    with open("/data/apps/lacode/self-correct-retriever/ulr_result/"+config.retriever+"_"+config.task+"+retrieval_{}.json", "w", encoding="utf-8") as f:
        labels, contexts, know_labels = [], [], []
        for resp in data_all:
            contexts.append(resp["cxts"])
            labels.append(resp["answer"])
            if config.task in ["cvg","Consult"]:
                know_labels.append(resp["know_label"])
            line = json.dumps(resp, ensure_ascii=False)
            f.write(line + "\n")
        print(compute_metric(contexts,labels,know_labels,config))

def search(query, vector_store, local_doc_qa=None, config = None):
    top_k = config.top_k
    retriever = config.retriever
    know_label = None
    if retriever in ["contriever","sailer"]:
        kg_matches = vector_store.similarity_search_with_score(query, k=top_k)
        know = [el[0].metadata["value"] for el in kg_matches]
        if config.task in ["cvg","Consult"]:
            know_label = [el[0].metadata["char"] for el in kg_matches]
    if retriever=="ulr":
        related_docs_with_score = local_doc_qa.get_knowledge_based_answer(query, vector_store)
        know = [related_docs_with_score[i][0].page_content for i in range(top_k)]
        if config.task=="cvg":
            know_label = [related_docs_with_score[i][0].metadata["path"].split("[SEP]")[-1] for i in range(top_k)]
        if config.task=="Consult":
            know_label = [related_docs_with_score[i][0].metadata["path"] for i in range(top_k)]
    if retriever=="bm25":
        query = list(jieba.cut(query))
        scores = local_doc_qa.get_scores(query)
        indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        know = [vector_store["know"][i] for i in indexes]
        if config.task in ["cvg","Consult"]:
            know_label = [vector_store["know_label"][i] for i in indexes]
    return know, know_label

def init_vector_store(file_path, model_path, vector_path = None, retriever= "bm25", top_k=None, device= None):
    local_doc_qa = None
    docs = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        data = json.loads(line)
                        key = data["key"]
                        val = data["value"]
                        try:
                            docs.append(Document(page_content=key, metadata={"value": val,"char": data["char"]}))
                        except:
                            docs.append(Document(page_content=key, metadata={"value": val,"char": data["art"]}))
    if retriever in ["contriever","sailer"]:
        model = HuggingFaceEmbeddings(model_name=model_path)
        vector_store = FAISS.from_documents(docs, model)
    if retriever=="ulr":
        init_path = "/data/apps/lacode/wisdomInterrogatory-main/model/m3e-base"
        tokenizer = BertTokenizer.from_pretrained(init_path)
        model = SimcseModel(pretrained_model=init_path, pooling="cls", tokenizer=tokenizer, device=device)
        model.to(device)
        state_dict = torch.load(model_path)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()
        local_doc_qa = LocalDocQA()
        local_doc_qa.init_cfg(model,top_k)
        try:
            vector_store = local_doc_qa.load_vector_store(vector_path)
        except:
            docs = init_knowledge_base(filepath=file_path, sentence_size=1000)
            graph = graph_data(file_path).to(device)
            vector_store = local_doc_qa.init_knowledge_vector_store(docs, graph, vector_path, device)
    if retriever=="bm25":
        try:
            vector_store = {"know":[el.metadata["value"] for el in docs], "know_label":[el.metadata["char"] for el in docs]}
        except:
            vector_store = [el.metadata["value"] for el in docs]
        tokens_docs = get_tokens(vector_store["know"])
        local_doc_qa = bm25.BM25(tokens_docs)
    return vector_store, local_doc_qa

def test(config):
    inputs, in_context_contents, results, labels=[],[],[],[]
    retrieval_data = []
    device = "cuda:0"
    file_path = config.know_path_dic[config.task]
    model_path = config.retrieval_model_dic[config.retriever]
    vector_path = config.vector_store_dic[config.retriever]
    vector_store, local_doc_qa = init_vector_store(file_path, model_path, vector_path, config.retriever, config.top_k, device)
    with open(config.test_path_dic[config.task],"r",encoding="utf-8") as f:
        if config.task=="Consult":
            config.curr_prompt = config.prompt_dic["Consult"][0]
            lines = f.readlines()
            random.shuffle(lines)
            for line in tqdm(lines[:config.test_num]):
                data = json.loads(line) 
                fact = data["question"]
                inputs.append(data["question"])
                labels.append(data["answer"])
                in_context_content, know_label = search(data["question"], vector_store, local_doc_qa, config)
                in_context_contents.append(in_context_content)
                retrieval_data.append({"question":fact, "answer":[data["answer"],data["art"]], "cxts":in_context_content, "know_label":know_label})

        elif config.task=="ljp":
            lines = f.readlines()
            random.shuffle(lines)
            for line in tqdm(lines[:config.test_num]):
                data = json.loads(line)
                fact = data["fact"]
                # fact = get_prompt(fact, config.prompt_dic["charge"])
                labels.append([data["article"], data["charge"]])
                in_context_content,_ = search(fact, vector_store, local_doc_qa, config)
                inputs.append(fact)
                in_context_contents.append(in_context_content)
                retrieval_data.append({"question":fact, "answer":[data["article"], data["charge"]], "cxts":in_context_content})

        elif config.task=="cvg":
            lines = f.readlines()
            random.shuffle(lines)
            for line in tqdm(lines[:config.test_num]):
                data = json.loads(line) 
                fact = data["fact"]
                config.curr_prompt = get_prompt(fact, config.prompt_dic["cvg"])
                labels.append(data["origin_view"])
                in_context_content, know_label = search(fact, vector_store, local_doc_qa, config)
                inputs.append(fact)
                in_context_contents.append(in_context_content)
                retrieval_data.append({"question":fact, "answer":[data["origin_view"], data["char"]], "cxts":in_context_content, "know_label":know_label})

    write_log(retrieval_data, config)
    # inputs = [get_prompt(el, config.prompt_dic[config.task]) for el in inputs] 
    if config.mlm_api=="text-davinci-003":
        responses = get_openai_response(inputs, in_context_contents, config = config)
    else:
        responses = get_llm_response(inputs, in_context_contents, config = config)
    for i in range(len(inputs)):
        results.append({"fact":inputs[i],"label":labels[i],"pred":str(responses[i])})

    with open("/data/apps/lacode/self-correct-retriever/ulr_result/"+config.retriever+"_"+config.mlm_api+"_{}.txt".format(config.task), "w", encoding='utf-8') as f:
        for dic in results:
            json.dump(dic,f,ensure_ascii=False)
            f.write("\n") 
    print(config.count_dic)
test(config)