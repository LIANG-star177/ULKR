import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns
import json
import random
import pandas as pd

from luwen import search, luwen_response, pred_with_knowledge

device = "cuda:0"
model_type = "led"
retrieval_model_dic = {"contriever":'/data/apps/lacode/wisdomInterrogatory-main/model/m3e-base',
                       "sailer":"/data/apps/lacode/self-correct-retriever/baseline/SAILER/checkpoint-60000",
                        "ulr":"/data/apps/lacode/self-correct-retriever/saved_model/simcse_ljp_2_3900.pt",
                        "bm25":None,
                        "led":"/data/apps/lacode/self-correct-retriever/saved_model/simcse_ljp_0_1000.pt"}
# 加载预训练的BERT模型和tokenizer
model_path = retrieval_model_dic[model_type]  # 替换为你的BERT模型名字
tokenizer = BertTokenizer.from_pretrained("/data/apps/lacode/wisdomInterrogatory-main/model/m3e-base")

if model_type!="ulr" and model_type!="led":
    bert = BertModel.from_pretrained(model_path)
    bert.to(device)
else:
    model = SimcseModel(pretrained_model="/data/apps/lacode/wisdomInterrogatory-main/model/m3e-base", pooling="cls", tokenizer=tokenizer, device=device)
    model.to(device)
    state_dict = torch.load(model_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    try:
        bert = model.module.bert
        gnn = model.module.gnn
        fc = model.module.fc
    except:
        bert = model.bert
        gnn = model.gnn
        fc = model.fc    

with open("/data/apps/lacode/self-correct-retriever/data/knowledge_base/single_train_ljp.json","r",encoding="utf-8") as f:
    lines = f.readlines()
    random.shuffle(lines)
    facts, charges = [],[]
    for line in lines:
        data = json.loads(line)
        facts.append(data["fact"])
        # fact = get_prompt(fact, config.prompt_dic["charge"])
        charges.append(data["charge"])
know_df = pd.DataFrame()
know_df["fact"],know_df["charge"]=facts,charges
print(len(know_df))
know_df.head(2)

# 假设 df 是你的 DataFrame，charge 列包含标签信息
# 获取每个标签的出现频率
my_label = ["抢劫","盗窃","诈骗","故意杀人","故意伤害"]
label_counts = know_df['charge'].value_counts()
for key,val in label_counts.items():
    if key in my_label:
        print(key,val)
# 选择出现频率最高的四个标签
top_labels = label_counts.head(4).index.tolist()
# print(top_4_labels)
top_labels = my_label

# 从每个选定的标签中随机选择 100 个样本
selected_data = pd.DataFrame()
for label in top_labels:
    label_data = know_df[know_df['charge'] == label]
    if len(label_data) > 200:
        label_sample = label_data.sample(n=200, random_state=42)  # 从标签数据中随机选择 100 个样本
    else:
        label_sample = label_data  # 如果标签数据不足 100 个，选择全部样本
    selected_data = pd.concat([selected_data, label_sample])

# 现在 selected_data 包含了从四个出现频率最高的标签中随机选择的 100 个样本
selected_data.head(2)
print(len(selected_data))

data = list(selected_data["fact"])
encoded_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt') for text in data]
encoded_texts = torch.cat(encoded_texts, dim=0).to(device)
with torch.no_grad():
    entailment = bert(encoded_texts, output_hidden_states=True).last_hidden_state[:, 0].cpu()
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
bert_embeddings_2d = tsne.fit_transform(entailment)

# 创建一个DataFrame来存储数据
df = pd.DataFrame(bert_embeddings_2d, columns=['x', 'y'])
df['text'] = data

my_label_dic = {"抢劫":"Robbery","盗窃":"Theft","诈骗":"Fraud","故意杀人":"Intentional homicide","故意伤害":"Intentional injury"}
# 根据罪名将数据分类
df['crime'] = [my_label_dic[el] for el in list(selected_data["charge"])]

# 绘制t-SNE可视化图
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'D',"p"]  # 可能需要更多标记，取决于分类数量
# colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
colors =sns.color_palette("muted", 5)
plt.xlim(-40, 50)
plt.ylim(-25, 30)
for i, crime in enumerate(df['crime'].unique()):
    subset = df[df['crime'] == crime]
    plt.scatter(subset['x'], subset['y'], marker=markers[i], color=colors[i], s=10, label=crime)
plt.subplots_adjust(right=0.8, top=0.8)
plt.axis('off')
plt.title(model_type)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
plt.savefig('tsne_{}.pdf'.format(model_type), format='pdf')
plt.show()