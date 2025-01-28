import math
import jieba
import numpy as np
import logging
import pandas as pd
from collections import Counter
jieba.setLogLevel(logging.INFO)

# 测试文本
text = [
"自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。",
"自然语言处理是一门融语言学、计算机科学、数学于一体的科学。因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。",
"自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。因而它是计算机科学的一部分。"
]

class BM25(object):
  def __init__(self,docs):
    self.docs = docs   # 传入的docs要求是已经分好词的list 
    self.doc_num = len(docs) # 文档数
    self.vocab = set([word for doc in self.docs for word in doc]) # 文档中所包含的所有词语
    self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num # 所有文档的平均长度
    self.k1 = 1.5
    self.b = 0.75

  def idf(self,word):
    if word not in self.vocab:
      word_idf = 0
    else:
      qn = {}
      for doc in self.docs:
        if word in doc:
          if word in qn:
            qn[word] += 1
          else:
            qn[word] = 1
        else:
          continue
      word_idf = np.log((self.doc_num - qn[word] + 0.5) / (qn[word] + 0.5))
    return word_idf

  def score(self,word):
    score_list = []
    for index,doc in enumerate(self.docs):
      word_count = Counter(doc)
      if word in word_count.keys():
        f = (word_count[word]+0.0) / len(doc)
      else:
        f = 0.0
      r_score = (f*(self.k1+1)) / (f+self.k1*(1-self.b+self.b*len(doc)/self.avgdl))
      score_list.append(self.idf(word) * r_score)
    return score_list 

  def score_all(self,sequence):
    sum_score = []
    for word in sequence:
      sum_score.append(self.score(word))
    sim = np.sum(sum_score,axis=0) 
    return sim
  
def get_tokens(text):
    stopwords = open('/data/apps/lacode/self-correct-retriever/code/utils/stopwords.txt').read().split('\n')
    doc_list = text
    docs = []
    for sentence in doc_list:
        sentence_words = jieba.lcut(sentence)
        tokens = []
        for word in sentence_words:
            if word in stopwords:
                continue
            else:
                tokens.append(word)
        docs.append(tokens)
    return docs

if __name__ == "__main__":
        
  # 获取停用词;
        docs = get_tokens(text)
        bm = BM25(docs)

        score = bm.score_all(['自然语言', '计算机科学', '领域', '人工智能', '领域'])

        print(score)
