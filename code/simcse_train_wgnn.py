import json
import random
import os
import re
import torch
import random
from tqdm import tqdm
import json
from clc.callbacks import Iteratorize, Stream
from langchain.schema import Document
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
from transformers import BertConfig, BertModel, BertTokenizer
from utils.gnn_utils import init_knowledge_base, LocalDocQA, compute_law_metric, gnn_fusion, GGATModel, graph_data, compute_case_metric
import json
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

random.seed(2023)

# 基本参数
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-6
MAXLEN = 512
POOLING = 'cls'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
USE_MY = True
# 预训练模型目录
BERT = '/data/apps/lacode/wisdomInterrogatory-main/model/m3e-base'
model_path = BERT

# 微调后参数存放位置
SAVE_PATH = '/data/apps/lacode/self-correct-retriever/saved_model'
EMBEDDING_PATH = '/data/apps/lacode/self-correct-retriever/saved_model/simcse_consult/'

# 数据位置
SNIL_TRAIN = "/data/apps/lacode/self-correct-retriever/data/knowledge_base/simcse_train_consult.json"
STS_DEV = "/data/apps/lacode/self-correct-retriever/data/knowledge_base/simcse_test_consult.json"
# STS_TEST = 'data/processed_random_large/test.txt'
STS_TEST = "/data/apps/lacode/self-correct-retriever/data/knowledge_base/simcse_test_consult.json"

filepath="/data/apps/lacode/self-correct-retriever/data/hera_knowbase3/法律咨询"
docs = init_knowledge_base(filepath=filepath, sentence_size=1000)

def load_data(name: str, path: str) -> List:

    def load_snli_data(path):        
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], 
                     line['entailment'], 
                     line['entailment_key'], 
                     line['entailment_path'], 
                     line['entailment_node_id'],
                     line['contradiction'], 
                     line['contradiction_key'], 
                     line['contradiction_path'],
                     line['contradiction_node_id']) for line in f]
        
    assert name in ["snli", "lqcmc", "sts"]
    if name == 'snli':
        return load_snli_data(path)    
    

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return self.tokenizer([text[0], text[1], "[SEP]".join(text[2]),"[SEP]".join(text[3]),
                               text[5], "[SEP]".join(text[6]),"[SEP]".join(text[7])], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt'),text[4],text[8]
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    
    
class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str, tokenizer, device):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.gnn = GGATModel(in_dim=768, hidden_dim=768, num_heads=2, node_types=3)
        self.fc = nn.Linear(768*4, 768)
        self.tokenizer = tokenizer
        self.device = device
        self.pooling = pooling
    
    def embed_query(self, text: str) -> List[float]:
        texts = [text]
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        texts = self.tokenizer(texts, max_length=512,truncation=True, padding='max_length', return_tensors='pt').to(self.device)
        embeddings = self.bert(**texts, output_hidden_states=True).last_hidden_state[:, 0]
        return embeddings.tolist()[0]
        
    def forward(self, input_ids, attention_mask, token_type_ids, entailment_sub_g, contradiction_sub_g):
        
        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0]
            out = gnn_fusion(self.gnn, out, entailment_sub_g, contradiction_sub_g, self.fc)
            return out  # [batch, 768]
        
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
                  
            
def simcse_sup_loss(y_pred: 'tensor', device) -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=device)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1).to(device)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

def save_model(model, batch_idx, tokenizer):
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

def topk_evaluate(path, embedding_model, graph, top_k, eva_ratio=1.0, device=None):
    querys, labels = [], []
    with jsonlines.open(path, 'r') as f:
        for line in f:
            querys.append(line['origin'])
            labels.append({"know":line['entailment'],"key":line["entailment_key"],"path":line["entailment_path"]})
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(embedding_model,top_k)
    vector_store = local_doc_qa.init_knowledge_vector_store(docs, graph, filepath, device)
    # local_doc_qa.get_knowledge_based_answer(querys, vector_store)
    dics_lst, similarity_lsts = [], []
    eva_length=int(len(labels)*eva_ratio)
    for i in tqdm(range(eva_length)):
        related_docs_with_score = local_doc_qa.get_knowledge_based_answer(querys[i], vector_store)
        dics_lst.append(
            {"know":[related_docs_with_score[i][0].page_content for i in range(top_k)],
             "key":[related_docs_with_score[i][0].metadata["key"] for i in range(top_k)],
             "path":[related_docs_with_score[i][0].metadata["path"] for i in range(top_k)]}
        )
        similarity_lst = [related_docs_with_score[i][1] for i in range(top_k)]
        similarity_lsts.append(similarity_lst)
    score_dict = compute_case_metric(dics_lst, labels[:eva_length])
    return score_dict, [el["know"] for el in dics_lst], similarity_lsts
        

def train(model, train_dl, dev_dl, optimizer, tokenizer, graph, device, epoch) -> None:
    """模型训练函数 
    """
    model.train()
    best = 0
    early_stop_batch = 0
    tq = tqdm(train_dl)
    for batch_idx, source in enumerate(tq, start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        entailment_nodes = torch.cat(source[1]).tolist()
        entailment_sub_g = graph.subgraph(entailment_nodes).to(device)

        contradiction_nodes =  torch.cat(source[2]).tolist()
        contradiction_sub_g = graph.subgraph(contradiction_nodes).to(device)

        source = source[0]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 7, -1).to(device)
        attention_mask = source.get('attention_mask').view(real_batch_num * 7, -1).to(device)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 7, -1).to(device)
        
        # 训练
        out = model(input_ids, attention_mask, token_type_ids, entailment_sub_g, contradiction_sub_g)
        loss = simcse_sup_loss(out, device)
        tq.set_postfix(loss=np.around(loss.cpu().detach().numpy(),4))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 50 == 0 and torch.cuda.current_device()==0:
            model.eval()
            logger.info(f'loss: {loss.item():.4f}')
            # if not USE_MY:
            #     output_dir = save_model(model, batch_idx, tokenizer)
            # else:
            #     output_dir = model
            save_path = SAVE_PATH + "/simcse_consult_"+str(epoch)+"_"+str(batch_idx)+".pt" 
            # torch.save(model.state_dict(), save_path)
            score_dict, _, _ = topk_evaluate(STS_TEST, model, graph, top_k=3, eva_ratio=1.0, device=device)
            score = score_dict["case_match"]
            logger.info(f"case_match: {score:.4f} in batch: {batch_idx}")
            # corrcoef = eval(model, dev_dl)
            model.train()
            if best < score_dict["case_match"]:
                early_stop_batch = 0
                best = score_dict["case_match"]
                torch.save(model.state_dict(), save_path)
                logger.info(f"case_match: {best:.4f} in batch: {batch_idx}, save model")
                continue
            early_stop_batch += 1
            if early_stop_batch == 60:
                logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {batch_idx* BATCH_SIZE}")
                return 
            
def test(model, graph, top_k, best_path, device):
    model.load_state_dict(torch.load(best_path))
    model.eval()
    score_dict, doc_lsts, similarity_lsts = topk_evaluate(STS_TEST, model, graph, top_k=top_k, eva_ratio=0.1, device=device)
    print(score_dict)
    inputs, labels = [], []
    with open(STS_TEST, 'r') as file:
        for line in file.readlines():
            line = json.loads(line.strip('\n'))
            inputs.append(line['origin'])
            labels.append(line['entailment'])
    pd.DataFrame({'input': inputs,
                  'label': labels,
                  'doc_lsts': doc_lsts,
                  'similarity_lsts': similarity_lsts
                  }
                 ).to_csv(SAVE_PATH+'/preds_detail_{}.csv'.format(top_k), index=False, encoding="utf_8_sig")
    with open(SAVE_PATH+'/preds_metric_{}.txt'.format(top_k), 'w') as o_file:
        o_file.write(json.dumps(score_dict))

def initialize_process(rank, world_size, master_addr, master_port):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda", local_rank)
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')

    graph = graph_data(filepath).to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data('snli', SNIL_TRAIN)
    random.shuffle(train_data)                        
    dev_data = load_data('snli', STS_DEV)
    test_data = load_data('snli', STS_TEST)    
    train_data, dev_data, test_data = TrainDataset(train_data,tokenizer), TrainDataset(dev_data,tokenizer), TrainDataset(test_data,tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=DistributedSampler(train_data))
    dev_dataloader = DataLoader(dev_data, batch_size=int(BATCH_SIZE/4), sampler=DistributedSampler(dev_data))
    test_dataloader = DataLoader(test_data, batch_size=int(BATCH_SIZE/4), sampler=DistributedSampler(test_data))
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING, tokenizer=tokenizer, device=DEVICE)
    model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer, tokenizer, graph, DEVICE, epoch)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')

    # test(model, graph, top_k=3, best_path="/data/apps/lacode/self-correct-retriever/saved_model/simcse_cvg_50.pt", device = DEVICE)
    
if __name__ == '__main__':
    # world_size = 4  # 总进程数，等于 GPU 数量
    # mp.spawn(setup, args=(world_size,), nprocs=world_size)
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # DEVICE = torch.device("cuda", local_rank)

    # logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    # # load data
    # train_data = load_data('snli', SNIL_TRAIN)
    # random.shuffle(train_data)                        
    # dev_data = load_data('snli', STS_DEV)
    # test_data = load_data('snli', STS_TEST)    
    # train_data, dev_data, test_data = TrainDataset(train_data), TrainDataset(dev_data), TrainDataset(test_data)
    # train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=DistributedSampler(train_data))
    # dev_dataloader = DataLoader(dev_data, batch_size=int(BATCH_SIZE/4), sampler=DistributedSampler(dev_data))
    # test_dataloader = DataLoader(test_data, batch_size=int(BATCH_SIZE/4), sampler=DistributedSampler(test_data))
    # # load model    
    # assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    # model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    # model.to(DEVICE)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                     device_ids=[local_rank],
    #                                                     output_device=local_rank,
    #                                                     find_unused_parameters=True)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # # train
    # best = 0
    # for epoch in range(EPOCHS):
    #     logger.info(f'epoch: {epoch}')
    #     train(model, train_dataloader, dev_dataloader, optimizer)
    # logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    # if local_rank==0:
    #     test(top_k=3, path="/root/data1/liang/simcse_training/saved_model/new_embedding_random_large_4/11000")
    
    
    # dev_corrcoef = eval(model, dev_dataloader)
    # test_corrcoef = eval(model, test_dataloader)
    # logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    # logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
    world_size = 6
    master_addr = '127.0.0.1'
    master_port = 8888
    mp.set_start_method('spawn')
    # 创建多个子进程并运行训练代码
    processes = []
    for rank in range(world_size):
        # rank+=1
        p = mp.Process(target=initialize_process, args=(rank, world_size, master_addr, master_port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()