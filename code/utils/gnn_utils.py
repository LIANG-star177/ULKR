import numpy as np
import jieba
from nltk.translate.bleu_score import sentence_bleu
import json
import re
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Tuple, Dict
from langchain.docstore.document import Document
from pypinyin import lazy_pinyin
from tqdm import tqdm
import datetime
from loguru import logger
from transformers import BertModel
# from langchain.vectorstores import FAISS
from utils.my_faiss import FAISS
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch.conv import GATConv
import pandas as pd
import json
import random
import re
import os

def init_knowledge_base(filepath, sentence_size):
    docs = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    # index = file_path.index("/hera_knowbase/") + len("/hera_knowbase/")
                    # remaining_path = file_path[index:]
                    # path_list = remaining_path.split('/')
                    # path_list = [part.replace(".json","") for part in path_list if part]
                    lines = f.readlines()
                    for line in lines:
                        line = json.loads(line)
                        # line["path_list"] = path_list
                        docs.append(
                            Document(page_content=line["value"],
                                     metadata={"key": "[SEP]".join(line["key2"]),
                                               "path":"[SEP]".join(line["path_list"]),
                                               "node_id":line["node_id"]}))
    return docs


def seperate_tensor(tensor, batch_size):
    grouped_tensors = []
    for i in range(7):
        indices = torch.arange(i, tensor.shape[0], 7)  # 每7个为一组的索引
        grouped_tensor = tensor[indices].view(batch_size, tensor.shape[1])  # 按照索引取出对应行并重排成 [4, 768]
        grouped_tensors.append(grouped_tensor)
    return grouped_tensors[0],grouped_tensors[1],grouped_tensors[2],grouped_tensors[3],grouped_tensors[4],grouped_tensors[5],grouped_tensors[6]

def graph_data(folder_path):
    def read_json_files(folder_path):
        articles,charges,keys,facts,paths = [],[],[],[],[]
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = json.loads(line)
                            # pattern = r"根据法条第(.+?)条,被告人犯了(.+?)罪"
                            # matches = re.search(pattern, line["value"])
                            # if matches:
                            #     article = matches.group(1)
                            #     charge = matches.group(2)
                            article = line["art"]
                            articles.append(article)
                            try:
                                charge = line["char"]
                                charges.append(charge)
                            except:
                                charges.append("无")
                            keys.append(line["key2"])
                            facts.append(line["key"])
                            paths.append(line["path_list"])
        return facts,keys,charges,articles,paths
    facts,keys,charges,articles,paths = read_json_files(folder_path)
    know_df = pd.DataFrame()
    know_df["fact"],know_df["key"],know_df["charge"],know_df["article"],know_df["path"]=facts,keys,charges,articles,paths
    know_df["path"] = ["".join(path) for path in paths]
    unique_path_classes = know_df['path'].unique()
    path_id_dict = {path_class: idx for idx, path_class in enumerate(unique_path_classes)}

    graph = dgl.DGLGraph()  # 创建一个空图
    nodes_type_A = [i for i in range(len(know_df))]
    nodes_type_B = [i for i in range(len(know_df))]
    # nodes_type_C = [i for i in range(len(path_id_dict))]
    nodes_type_C = [i for i in range(len(know_df))]

    node_types = [0] * len(nodes_type_A) + [1] * len(nodes_type_B) + [2]*len(nodes_type_C) # 节点类型标签

    # 创建图，并添加节点以及节点类型信息
    graph.add_nodes(len(nodes_type_A) + len(nodes_type_B)+len(nodes_type_C))
    graph.ndata['node_type'] = torch.tensor(node_types)

    # 假设有不同类型节点之间的连接关系，这里是随机连接的示例
    edges = []
    edges_set = set()
    for i in range(len(know_df)):
        article_value = know_df['article'][i]
        path_value = know_df['path'][i]
        node_C_id = path_id_dict[path_value]
        edges.append((i,i+len(know_df)))
        edges.append((i+len(know_df),i))
        # edges.append((i,len(know_df)*2+node_C_id))
        # edges.append((len(know_df)*2+node_C_id,i))
        edges.append((i,i+len(know_df)*2))
        edges.append((i+len(know_df)*2,i))
        indices = know_df[know_df['article'] == article_value].index.tolist()
        for j in indices:
            if j!=i:
                edges_set.add((i, j))
                edges.append((i,j+len(know_df)))
                edges.append((j+len(know_df),i))
                edges.append((i,j+len(know_df)*2))
                edges.append((j+len(know_df)*2,i))
    edges.extend(list(edges_set))
    src, dst = zip(*edges)
    graph.add_edges(src, dst)
    graph = dgl.add_self_loop(graph)
    return graph

def combine_tensor(origin,entailment_fusion,contradiction_fusion,batch_size):
    concatenated_tensor = torch.zeros(batch_size*3, origin.shape[1])
    # 按照交替顺序将三个张量合并成一个张量
    for i in range(batch_size):
        concatenated_tensor[i * 3] = origin[i]
        concatenated_tensor[i * 3 + 1] = entailment_fusion[i]
        concatenated_tensor[i * 3 + 2] = contradiction_fusion[i]
    return concatenated_tensor

class GGATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, node_types):
        super(GGATModel, self).__init__()
        self.node_types = node_types
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.gat_conv = GATConv(in_dim, hidden_dim, num_heads, allow_zero_in_degree=True)
        self.fc = nn.Linear(hidden_dim * num_heads * node_types, in_dim)
    
    def get_node_features(self, graph, input ,node_type):

        node_feats = self.gat_conv(graph, input)
        graph.ndata['h'] = node_feats
        type_nodes = graph.ndata['h'][graph.ndata['node_type'] == node_type]
        output = type_nodes.view(-1, self.num_heads * self.hidden_dim)
        # graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        # graph.send_and_recv(graph.edges(), fn.copy_e('score', 'm'), fn.sum('m', 'h'))
        # output = graph.ndata.pop('h')
        return output

    def forward(self, graph, know, key, path):

        combined_tensor = torch.cat((know, key, path), dim=0)
        cat_tensor = []
        for node_type in range(self.node_types):
            cat_tensor.append(self.get_node_features(graph, combined_tensor, node_type))
        # 在这里可以对不同类型节点进行不同的操作或注意力机制
        # 例如，可以计算不同类型节点之间的注意力权重或进行不同类型节点特征的融合等
        # 最后可以将不同类型节点的表示进行整合
        # ...
        final_representation = torch.cat(cat_tensor, dim=1)
        output = self.fc(final_representation)
        return output

class GATFusionWithGraph(nn.Module):
    def __init__(self, in_dims, hidden_dim, num_heads, num_types):
        super(GATFusionWithGraph, self).__init__()
        self.num_types = num_types
        self.gat_convs = nn.ModuleList([
            GATConv(in_dims[type_id], hidden_dim, num_heads) for type_id in range(num_types)
        ])

    def forward(self, graph, know, key, path):
        fused_node_feats = []
        node_embeddings = [know, key, path]
        for type_id in range(self.num_types):
            # 对每种节点类型进行 GAT 注意力聚合
            gat_out = self.gat_convs[type_id](graph, node_embeddings[type_id])
            fused_node_feats.append(gat_out)

        # 根据图中的边进行不同类型节点之间的信息传递
        for src_type in range(self.num_types):
            for dst_type in range(self.num_types):
                # 判断图中是否存在连接两种节点类型的边
                if graph.has_edges_between(src_type, dst_type):
                    edge_ids = graph.edge_ids(src_type, dst_type)
                    src_feats = fused_node_feats[src_type]
                    dst_feats = fused_node_feats[dst_type]
                    graph.edges[edge_ids].data['src'] = src_feats
                    graph.edges[edge_ids].data['dst'] = dst_feats
                    graph.apply_edges(lambda edges: {'e': edges.src['src'] * edges.dst['dst']},
                                      etype=edge_ids)
                    interaction_out = graph.edata['e']
                    fused_node_feats[src_type] = interaction_out
                    fused_node_feats[dst_type] = interaction_out

        return fused_node_feats
    
def gnn_fusion(model, data, entailment_sub_g, contradiction_sub_g, fc=None):
    batch_size = int(data.shape[0]/7)
    origin,entailment,entailment_key,entailment_path,contradiction,contradiction_key,contradiction_path=seperate_tensor(data,batch_size)
    # gnn
    entailment_fusion = model(entailment_sub_g,entailment,entailment_key,entailment_path)
    entailment_fusion = torch.cat([entailment,entailment_key,entailment_path,entailment_fusion],dim=1)
    entailment_fusion = fc(entailment_fusion)
    contradiction_fusion = model(contradiction_sub_g,contradiction,contradiction_key,contradiction_path)
    contradiction_fusion = torch.cat([contradiction,contradiction_key,contradiction_path,contradiction_fusion],dim=1)
    contradiction_fusion = fc(contradiction_fusion)

    # fc
    # entailment_fusion = torch.cat([entailment,entailment_key,entailment_path],dim=1)
    # entailment_fusion = fc(entailment_fusion)
    # contradiction_fusion = torch.cat([contradiction,contradiction_key,contradiction_path],dim=1)
    # contradiction_fusion = fc(contradiction_fusion)

    final_tensor = combine_tensor(origin,entailment_fusion,contradiction_fusion,batch_size)
    return final_tensor
    
def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")

class LocalDocQA:
    chunk_size: int = 250
    chunk_conent: bool = False
    score_threshold: int = 0

    def init_cfg(self,
                 embedding_model, 
                 top_k, 
                 ):
        self.top_k = top_k
        self.embeddings = embedding_model

    def init_knowledge_vector_store(self,docs, graph, save_path, device):
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            vector_store = FAISS.from_documents(docs, self.embeddings, graph, device)  # docs 为Document列表
            torch_gc()
            vector_store.save_local(save_path)
            vector_store = FAISS.load_local(save_path, self.embeddings)
            return vector_store
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None

    def load_vector_store(self, path):
        vector_store = FAISS.load_local(path, self.embeddings)
        return vector_store

    def get_knowledge_based_answer(self, query, vector_store):
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        torch_gc()
        return related_docs_with_score


def law_retrival_label(text):
    law_list = re.findall(r'(《.*?》)', text, re.M)
    articles = text.split('》')

    # 条目对不上
    if len(law_list) > 0 and len(articles) != len(law_list) + 1:
        return None

    cnt = 1
    valid_law = {}

    for law_name in law_list:
        law_name = law_name.replace("《","").replace("》","")
        article_list = re.findall(r'(第[\u4e00-\u9fa5]*?条)', articles[cnt], re.M)
        if law_name in valid_law:
            valid_law[law_name].update(article_list)
        else:
            valid_law[law_name] = set(article_list)
        cnt += 1

    return valid_law

def compute_case_metric(preds, labels):
    score_dict = {
        "case_match": 0, #法律名称预测正确的案件比例
    }
    preds = [el["path"] for el in preds]
    labels = [el["path"] for el in labels]
    is_case_matchs = []
    for pred, label in zip(preds, labels):
        label = "[SEP]".join(label)
        if label in pred:
            is_case_matchs.append(1)
        else:
            is_case_matchs.append(0)
    score_dict['case_match'] = float(np.mean(is_case_matchs))
    return score_dict
   
# Metric2
def compute_law_metric(preds, labels):

    score_dict = {
        "law_match": 0, #法律名称预测正确的案件比例
        "law_article_match": 0, #法律和法条都预测正确的案件比例
        "punishment_match": 0
    }

    is_law_matchs = []
    is_law_article_matchs = []

    for pred, label in zip(preds, labels):
        label = "《刑法》"+label
        law_preds = {}
        keys = []
        for source in pred:
            valid_law_pred = law_retrival_label(source)
            if len(valid_law_pred)==0:
                continue
            key_temp = list(valid_law_pred.keys())[0]
            values_tmp = list(valid_law_pred.values())[0]
            if key_temp not in keys:
                keys.append(key_temp)
                law_preds[key_temp]=values_tmp
            else:
                law_preds[key_temp].update(values_tmp)
        valid_law_label = law_retrival_label(label)
        if not valid_law_label:
            is_law_matchs.append(0)
            is_law_article_matchs.append(0)
            continue

        if len(law_preds)!=0 and set(valid_law_label.keys()).issubset(set(law_preds.keys())):
            is_law_matchs.append(1)
            law_article_match = True
            for k, v in valid_law_label.items():
                # if top_k==1:
                #     if valid_law_pred[k] != v:
                #         law_article_match = False
                # else:
                if not v.issubset(law_preds[k]):
                    law_article_match = False
                break
            if law_article_match:
                is_law_article_matchs.append(1)
            else:
                is_law_article_matchs.append(0)
        else:
            is_law_matchs.append(0)
            is_law_article_matchs.append(0)

    score_dict['law_match'] = float(np.mean(is_law_matchs))
    score_dict['law_article_match'] = float(np.mean(is_law_article_matchs))

    return score_dict, is_law_matchs, is_law_article_matchs