import re
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import cn2an
import warnings 
import jieba
warnings.filterwarnings("ignore")
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from charge_dic import get_charge_labels, get_similar_charge

def digit_to_chinese(number):
    chinese_digits = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九',
    }

    chinese_units = ['', '十', '百', '千', '万']

    number_str = str(number)
    result = ""
    length = len(number_str)
    zero_flag = False

    for i, digit in enumerate(number_str):
        if digit == '0':
            zero_flag = True
            if i == length - 1:
                result += chinese_digits[digit]
            continue
        else:
            if zero_flag:
                result += chinese_digits['0']
                zero_flag = False
            result += chinese_digits[digit] + chinese_units[length - i - 1]

    return result

def evaluate_art(path):
    preds, labels=[],[]
    cnt = 0
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            fact = line["fact"]
            label = line["label"]
            # if isinstance(label,list):
            #     label = label[1].strip("零")
            label = [digit_to_chinese(el) for el in label]
            pred = line["pred"]
            pred_art = re.findall(r'第([一二三四五六七八九十百零]+)条',pred)
            pred_digit_art = re.findall(r"第(\d+)条",pred)
            pred_art = [el for el in pred_art] + [digit_to_chinese(el) for el in pred_digit_art]
            if label in pred_art:
                pred_art = label
                cnt+=1
            else:
                if pred_art:
                    pred_art = pred_art[0]
                else:
                    pred_art = "无"
            labels.append(label)
            preds.append(pred_art)
        y_true = np.array(labels)
        y_pred = np.array(preds)
        result = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(result)
        result = accuracy_score(y_true, y_pred)
        print(result)

def evaluate_art_in_opinion(path):
    preds, labels=[],[]
    cnt = 0
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            label = line["label"]
            label_art = re.findall(r'第([一二三四五六七八九十百零]+)条',label)
            label_digit_art = re.findall(r"第(\d+)条",label)
            label_art = [el for el in label_art] + [digit_to_chinese(el) for el in label_digit_art]
            if len(label_art)>1:
                label_art = label_art[0]
            else:
                cnt+=1
                continue
            
            pred = line["pred"]
            pred_art = re.findall(r'第([一二三四五六七八九十百零]+)条',pred)
            pred_digit_art = re.findall(r"第(\d+)条",pred)
            pred_art = [el for el in pred_art] + [digit_to_chinese(el) for el in pred_digit_art]
            if label_art in pred_art:
                pred_art = label_art
            else:
                if pred_art:
                    pred_art = pred_art[0]
                else:
                    pred_art = "无"
            labels.append(label_art)
            preds.append(pred_art)
        y_true = np.array(labels)
        y_pred = np.array(preds)
        result = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(result)
        result = accuracy_score(y_true, y_pred)
        print(result, cnt)

def evaluate_char(path):
    preds, labels=[],[]
    c_set = get_charge_labels()
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            label = line["label"]
            if isinstance(label,list):
                label = label[0]
            label = label.replace("[","").replace("]","")
            pred = line["pred"]
            labels.append(label)
            cur_c = "#"
            contain_charge_set = set()
            for c in c_set:
                if c in pred:
                    contain_charge_set.add(c)
            if len(contain_charge_set) == 1:
                cur_c = list(contain_charge_set)[0]
            if cur_c != "#":
                preds.append(cur_c)
                continue
            flag=0
            label_c = get_similar_charge(label)
            pred_c = get_similar_charge(pred)
            if label_c == pred_c:
                flag=1
                preds.append(label)
            if flag==0:
                preds.append("无")
        y_true = np.array(labels)
        y_pred = np.array(preds)
        result = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(result)
        result = accuracy_score(y_true, y_pred)
        print(result)

def evaluate_char_in_opinion(path):
    cnt=0
    preds, labels=[],[]
    c_set = get_charge_labels()
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            label = line["label"]
            # cur_c = "#"
            # contain_charge_set = set()
            # for c in c_set:
            #     similar_c = get_similar_charge(c)
            #     if c in label or similar_c in label:
            #         contain_charge_set.add(c)
            # if len(contain_charge_set) == 1:
            #     cur_c = list(contain_charge_set)[0]
            # if cur_c != "#":
            #     label = cur_c
            #     labels.append(cur_c)
            # else:
            #     cnt+=1
            #     continue

            pred = line["pred"]
            # cur_c = "#"
            # contain_charge_set = set()
            # for c in c_set:
            #     similar_c = get_similar_charge(c)
            #     if c in label or similar_c in label:
            #         contain_charge_set.add(c)
            # if len(contain_charge_set) == 1:
            #     cur_c = list(contain_charge_set)[0]
            # if cur_c != "#":
            #     preds.append(cur_c)
            #     continue

            flag=0
            label_c = get_similar_charge(label)
            pred_c = get_similar_charge(pred)
            if set(label_c).issubset(set(pred_c)):
                flag=1
                preds.append(label_c[0])
            if flag==0:
                preds.append("无")
            labels.append(label_c[0])
        y_true = np.array(labels)
        y_pred = np.array(preds)
        result = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(result)
        result = accuracy_score(y_true, y_pred)
        print(result, cnt)

def get_pt_cls(pt):
    if pt > 10 * 12:
        ret_penalty = 9
    elif pt > 7 * 12:
        ret_penalty = 8
    elif pt > 5 * 12:
        ret_penalty = 7
    elif pt > 3 * 12:
        ret_penalty = 6
    elif pt > 2 * 12:
        ret_penalty = 5
    elif pt > 1 * 12:
        ret_penalty = 4
    elif pt > 9:
        ret_penalty = 3
    elif pt > 6:
        ret_penalty = 2
    elif pt > 0:
        ret_penalty = 1
    else:
        ret_penalty = 0
    return ret_penalty

def trans(text):
    text = re.sub("[^0-9一二两三四五六七八九十百年月]", "", text)
    year = 0
    month = 0
    if "年" in text:
        year, text = text.split("年")
        year = cn2an.cn2an(year, "smart")
    if "月" in text:
        month = text.replace("月", "")
        month = cn2an.cn2an(month, "smart")
    return int(year * 12 + month)
        
def text2penalty(text):
    if text == '#':
        return -1
    if "至" in text:
        t1,  t2 = text.split("至")
        p1 = trans(t1)
        p2 = trans(t2)
        if p1 > p2:
            p1, p2 = p2, p1
        if p1 == -1 and p2 == -1:
            return -1
        if p1 == -1:
            return p2
        return (p1 + p2) // 2
    return trans(text)

def evaluate_pen(path):
    preds, labels=[],[]
    with open(path,"r") as f:
        lines = f.readlines()
        cnt=0
        for line in lines:
            line = json.loads(line)
            label = line["label"]
            if isinstance(label,list):
                label = label[2]
            text = line["pred"]
            pt = "#"
            for span in re.sub("，|、|，|。", " ", text).split():
                re_res = re.findall("有期徒刑(.*[年|月])|拘役(.*[年|月])", span)
                if len(re_res) > 0:
                    pt = [s for s in re_res[0] if s != ""][0]
                    break
            digit_p = 0
            try:
                digit_p = text2penalty(pt)
            except Exception as e:
                cnt+=1
            labels.append(label)
            preds.append(digit_p)
        y_true = [get_pt_cls(y) for y in labels]
        y_pred = [get_pt_cls(y) for y in preds]
        result = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(result)
        result = accuracy_score(y_true, y_pred)
        print(result)
        print(cnt) 

def evaluate_pen_in_opinion(path):
    preds, labels=[],[]
    with open(path,"r") as f:
        lines = f.readlines()
        cnt=0
        for line in lines:
            line = json.loads(line)
            text = line["label"]
            pt = "#"
            for span in re.sub("，|、|，|。", " ", text).split():
                re_res = re.findall("有期徒刑(.*[年|月])|拘役(.*[年|月])|(.*[年|月])有期徒刑", span)
                if len(re_res) > 0:
                    pt = [s for s in re_res[0] if s != ""][0]
                    break
            digit_p = 0
            try:
                digit_p = text2penalty(pt)
                if digit_p==-1:
                    result = re.search(r'判处被告人有期徒刑(\d+)个月', text)
                    digit_p = result.group(1)
            except Exception as e:
                cnt+=1
            labels.append(digit_p)

            text = line["pred"]
            pt = "#"
            for span in re.sub("，|、|，|。", " ", text).split():
                re_res = re.findall("有期徒刑(.*[年|月])|拘役(.*[年|月])|(.*[年|月])有期徒刑", span)
                if len(re_res) > 0:
                    pt = [s for s in re_res[0] if s != ""][0]
                    break
            digit_p = 0
            try:
                digit_p = text2penalty(pt)
            except Exception as e:
                cnt+=1
            preds.append(digit_p)
        y_true = [get_pt_cls(y) for y in labels]
        y_pred = [get_pt_cls(y) for y in preds]
        result = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(result)
        result = accuracy_score(y_true, y_pred)
        print(result, cnt)


def calculate(pred, trg):
    rouge = Rouge()
    bleuscore1, bleuscore2, bleuscoren = 0,0,0
    rougescore1, rougescore2, rougescorel = 0,0,0
    num = len(pred)
    for i in tqdm(range(num)):
        trg[i]=" ".join([w for w in trg[i].replace(" ","")])
        pred[i]=" ".join([w for w in pred[i].replace(" ","")])
        reference = [trg[i]]
        candidate = pred[i]
        try:
            rouge_score = rouge.get_scores(pred[i], trg[i])
        except:
            continue
        r1 = rouge_score[0]["rouge-1"]['r']
        r2 = rouge_score[0]["rouge-2"]['r']
        rl = rouge_score[0]["rouge-l"]['r']
        b1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        b2 = sentence_bleu(reference, candidate,weights=(0, 1, 0, 0))
        bn = sentence_bleu(reference, candidate,weights=(0.25, 0.25, 0.25, 0.25))
        rougescore1 += r1
        rougescore2 += r2
        rougescorel += rl
        bleuscore1 += b1
        bleuscore2 += b2
        bleuscoren += bn
    bleuscore1, bleuscore2, bleuscoren =  bleuscore1/num, bleuscore2/num, bleuscoren/num
    rougescore1, rougescore2, rougescorel = rougescore1/num, rougescore2/num, rougescorel/num
    return {"b1": bleuscore1, "b2": bleuscore2, "bn": bleuscoren,
            "r1": rougescore1, "r2": rougescore2, "rl": rougescorel}


def evaluate_opinion(path):
    preds, labels=[],[]
    with open(path,"r") as f:
        lines = f.readlines()
        cnt=0
        for line in lines:
            line = json.loads(line)
            labels.append(line["label"])
            preds.append(line["pred"])
    metric_sum_lst = calculate(preds, labels)
    print("==="*20+"相似度指标")
    print("*"*10+"bleu"+"*"*10)
    print("bleu1:{0:.4f},bleu2:{1:.4f},bleun:{2:.4f}".format(
        metric_sum_lst["b1"], metric_sum_lst["b2"], metric_sum_lst["bn"])) 
    print("*"*10+"rouge"+"*"*10)
    print("rouge1:{0:.4f},rouge2:{1:.4f},rougel:{2:.4f}".format(
        metric_sum_lst["r1"], metric_sum_lst["r2"], metric_sum_lst["rl"]))

#法条记忆问答
def compute_rouge(hyps, refs):
    assert(len(hyps) == len(refs))
    hyps = [' '.join(jieba.cut(h)) for h in hyps]
    hyps = [h if h.strip() != "" else "无内容" for h in hyps]
    refs = [' '.join(jieba.cut(r)) for r in refs]
    return Rouge().get_scores(hyps, refs)

def compute_ftcs(path):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    preds, labels=[],[]
    with open(path,"r") as f:
        lines = f.readlines()
        cnt=0
        for line in lines[:100]:
            line = json.loads(line)
            labels.append(line["label"].replace("答案：", "").replace("回答：", "").replace("回答:", "").replace("答案：", ""))
            preds.append(line["pred"].replace("答案：", "").replace("回答：", "").replace("回答:", "").replace("答案：", ""))
    metric_sum_lst = calculate(preds, labels)
    print("==="*20+"相似度指标")
    print("*"*10+"bleu"+"*"*10)
    print("bleu1:{0:.4f},bleu2:{1:.4f},bleun:{2:.4f}".format(
        metric_sum_lst["b1"], metric_sum_lst["b2"], metric_sum_lst["bn"])) 
    print("*"*10+"rouge"+"*"*10)
    print("rouge1:{0:.4f},rouge2:{1:.4f},rougel:{2:.4f}".format(
        metric_sum_lst["r1"], metric_sum_lst["r2"], metric_sum_lst["rl"]))

def multi_choice_judge(prediction, option_list, answer_token):
    # a dict, key: letters in the option list, value: count of the letter in the prediction
    count_dict, abstention, accuracy = {}, 0, 0
    for option in option_list:
        option_count = prediction.count(option)
        count_dict[option] = 1 if option_count > 0 else 0  # multiple occurrence of the same letter is counted as 1

    if sum(count_dict.values()) == 0:
        abstention = 1
    # if the answer token is the only predicted token, the prediction is correct 
    elif count_dict[answer_token] == 1 and sum(count_dict.values()) == 1:
        accuracy = 1
    return {"score": accuracy, "abstention": abstention}

def compute_jec_kd(path):
    """
    Compute the Accuracy
    The JEC_KD dataset has 4 options for each question: A, B, C, D
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """
    preds, labels, questions=[],[],[]
    with open(path,"r") as f:
        lines = f.readlines()
        cnt=0
        for line in lines[:100]:
            line = json.loads(line)
            labels.append(line["label"])
            preds.append(line["pred"])
            questions.append(line["fact"].replace("请你运用法律知识从A,B,C,D中选出一个正确的答案，并写在[正确答案]和<eoa>之间。例如[正确答案]A<eoa>。请你严格按照这个格式回答。",""))
    score_list, abstentions = [], 0
    option_list = ["A", "B", "C", "D"]
    for i in range(len(labels)):
        question, prediction, answer = questions[i], preds[i], labels[i]
        assert answer.startswith("正确答案：") and answer[5] in option_list, f"answer[5]: {answer}, question: {question}"

        answer_letter = answer[5]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    print("score:",accuracy)
    print("abstention_rate",abstentions / len(labels))

evaluate_art("/data/apps/lacode/self-correct-retriever/test_result/generation_ljp.txt")
evaluate_char("/data/apps/lacode/self-correct-retriever/test_result/generation_ljp.txt")
# evaluate_pen("/data/apps/lacode/self-correct-retriever/test_result/generation_ljp.txt")

#CVG
# evaluate_art_in_opinion("/data/apps/lacode/self-correct-retriever/test_result/self_corr_generation_cvg.txt")
# evaluate_char_in_opinion("/data/apps/lacode/self-correct-retriever/test_result/self_corr_generation_cvg.txt")
# evaluate_pen_in_opinion("/data/apps/lacode/self-correct-retriever/test_result/self_corr_generation_cvg.txt")
# evaluate_opinion("/data/apps/lacode/self-correct-retriever/test_result/self_corr_generation_cvg.txt")

#法条背诵
# compute_ftcs("/data/apps/lacode/self-correct-retriever/test_result/self_corr_llm_generation_Article Recitation.txt")

#法考问题
# compute_jec_kd("/data/apps/lacode/self-correct-retriever/test_result/self_corr_generation_Question Answering.txt")

#法律问答
compute_ftcs("/data/apps/lacode/self-correct-retriever/test_result/_chatglm_generation_Consult.txt")