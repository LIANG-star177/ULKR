import numpy as np

def ndcg(golden, current, n = -1):
    log2_table = np.log2(np.arange(2, 102)) # 第一个log2(i + 1) = log2(1 + 1) = 1

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n] # 转numpy float数组，并取前n个值
  # dcg由cg加权求和而来
  #  此处分子对rel_i进行了指数放大
  #  权重为log2_i，使用log2_table预先计算再截取的形式进行加速
        dcg = np.sum(np.divide(np.power(2, rel) - 1, 
          log2_table[:rel.shape[0]]))
        return dcg
 
 # 共len(current)个搜索结果用来评估搜索引擎
 # 最后给出搜索引擎的ndcg值，是各个搜索结果的ndcg的平均
    ndcgs = []
    for i in range(len(current)):
        k = len(current[i]) if n == -1 else n # 如果规定了n，就计算ndcg@n；如果没有，就计算ndcg@len(current[i])
        idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k) # 计算idcg@k
        dcg = dcg_at_n(current[i], n=k) # 计算dcg@k
        tmp_ndcg = 0 if idcg == 0 else dcg / idcg # 计算当前搜索结果的ndcg@k
        ndcgs.append(tmp_ndcg)
 # 计算所有搜索结果的ndcg的平均值
    return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))