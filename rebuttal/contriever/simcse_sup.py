# -*- encoding: utf-8 -*-

import random
import time
from typing import List
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, AutoModel, AutoTokenizer
import json
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sentence_transformers
from sentence_transformers.util import cos_sim

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
# os.environ['RANK'] = '0,1,2,3'
# dist.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
DEVICE = torch.device("cuda:0")
# random.seed(2023)

# 基本参数
EPOCHS = 2
BATCH_SIZE = 4
LR = 1e-6
MAXLEN = 512
POOLING = 'cls'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 

# 预训练模型目录
BERT = "/home/u12321044/share/liang_52/wisdomInterrogatory-main/model/text2vec_large"
AUTO = "/home/u12321044/share/liang_52/art-rag/pretrained_model/SAILER"
model_path = AUTO

# 微调后参数存放位置
SAVE_PATH = '/home/u12321044/share/liang_52/self-correct-retriever/rebuttal/sailer/simcse_sup_random_large.pt'
EMBEDDING_PATH = '/home/u12321044/share/liang_52/self-correct-retriever/rebuttal/sailer/saved_model/'

# 数据位置
SNIL_TRAIN = '/home/u12321044/share/liang_52/self-correct-retriever/rebuttal/lecard_data/query_positive_negative_tuples.json'
STS_DEV = "/home/u12321044/share/liang_52/self-correct-retriever/rebuttal/lecard_data/test_queries.json"
# STS_TEST = 'data/processed_random_large/test.txt'
STS_TEST = "/home/u12321044/share/liang_52/self-correct-retriever/rebuttal/lecard_data/test_queries.json"

def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集
    """
    #TODO: 把lqcmc的数据生成正负样本, 拿来做测试
    def load_lecard_data(path):   
        # 初始化一个列表来存储三元组
        train_tuples = []

        # 逐行读取文件并加载到列表中
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tuple_dict = json.loads(line.strip())
                train_tuples.append((tuple_dict["query"],tuple_dict["positive"]["document"],tuple_dict["negative"]["document"]))     
        return train_tuples
        
    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f] 
        
    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:            
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]   
        
    assert name in ["snli", "lqcmc", "sts"]
    if name == 'snli':
        return load_lecard_data(path)    
    return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path) 
    

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    
    
class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])
    
    
class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        # self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
                  
            
def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss
        
def train(model, train_dl, optimizer) -> None:
    """模型训练函数 
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 50 == 0:
            model.eval()
            logger.info(f'loss: {loss.item():.4f}')
            output_dir = save_model(model, batch_idx)
            topk_evaluate(output_dir, batch_idx)
            model.train()

def save_model(model, batch_idx):
    output_dir = EMBEDDING_PATH+"{}".format(batch_idx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        model_to_save = model.module.bert if hasattr(model, 'module') else model.bert
        # model.module.bert.save_pretrained(output_dir)
        #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)
    return output_dir

def topk_evaluate(model_path, batch_idx):
    query_file_path = '/home/u12321044/share/liang_52/art_case/data/LeCaRD-main/updated_query.json'
    with open(query_file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f) 
    candidate_file_path = '/home/u12321044/share/liang_52/art_case/data/LeCaRD-main/candidate_base.json'
    with open(candidate_file_path, 'r', encoding='utf-8') as f:
        candidate_library = json.load(f) 
    model = sentence_transformers.SentenceTransformer(model_path).to(DEVICE)

    results = []

    for query in tqdm(queries):
        sentences1_list = []
        query_id = query['ridx']
        query_text = query['q']
        candidate_scores = query.get('candidate_scores', {})

        for candidate_id, score in candidate_scores.items():
            if candidate_id in candidate_library:
                candidate_data = candidate_library[candidate_id]
                document_text = candidate_data.get('ajjbqk', '')  # 使用 'ajjbqk' 作为文档内容
                sentences1_list.append((candidate_id, document_text))

        # 对每个候选文档进行编码
        encoded_sentences1 = [(candidate_id, model.encode(document_text)) for candidate_id, document_text in tqdm(sentences1_list)]
        # 对 query 进行编码
        encoded_sentence2 = model.encode(query_text)

        # 计算相似度
        similarities = [
            (candidate_id, cos_sim(encoded_sentence2, encoded_sentence).item())
            for candidate_id, encoded_sentence in encoded_sentences1
        ]

        # 按相似度降序排序
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        # 将结果按照指定格式保存
        for rank, (candidate_id, similarity) in enumerate(sorted_similarities, start=1):
            results.append(f"{query_id} Q0 {candidate_id} {rank} {similarity} LHT")

    # 将结果写入文件
    with open('/home/u12321044/share/liang_52/self-correct-retriever/rebuttal/sailer/result/output_trec_{}.txt'.format(batch_idx), 'w') as f:
        for line in results:
            f.write(line + '\n')

        # for j in range(len(top_k)):
        #     docs_lsts[j].append([docs_dic[sentences1_list[i]] for i in top_indices[:top_k[j]]])

    # score_dict, corr_index = pred_law_metric(docs_lsts, labels[:eva_length], top_k)
    # print(score_dict)
    return 

if __name__ == '__main__':
    
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data('snli', SNIL_TRAIN)
    random.shuffle(train_data)                        
    # dev_data = load_data('snli', STS_DEV)
    # test_data = load_data('snli', STS_TEST)    
    train_data = TrainDataset(train_data)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    # dev_dataloader = DataLoader(dev_data, batch_size=int(BATCH_SIZE/4), sampler=DistributedSampler(dev_data))
    # test_dataloader = DataLoader(test_data, batch_size=int(BATCH_SIZE/4), sampler=DistributedSampler(test_data))
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    best = 0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, optimizer)
    # logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval