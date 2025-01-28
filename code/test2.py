import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from luwen import search, luwen_response, pred_with_knowledge
from llm_api import get_openai_response, get_knowledge_combination, get_llm_response

# 法考题的知识长度需要512,在2个知识下做实验
class testCFG:
    prompt_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"],
              "cvg":["请写出下面案件的法院观点(包含罪名，法条和刑期)。"],
              "Article Recitation":["回答以下问题，只需直接给出法条内容:"],
              "Question Answering":["请你运用法律知识从A,B,C,D中选出一个正确的答案，并写在[正确答案]和<eoa>之间。例如[正确答案]A<eoa>。请你严格按照这个格式回答。"],
              "Consult":["请回答下列问题，首先给出回答，然后给出对应的法律依据(法律依据需要给出法条具体内容)。\n"]}
    count_dic = {"can_not_combine":0,
                "need_extra":0}
    task = "Consult"
    # openai_api = "text-davinci-002"
    openai_api = "gpt-3.5-turbo"
    mlm_api = "chatglm"
    test_path_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"],
              "cvg":"/data/apps/lacode/self-correct-retriever/data/new_cvg_test.json",
              "Article Recitation":"/data/apps/lacode/self-correct-retriever/baseline/LawBench-main/data/zero_shot/1-1.json",
              "Question Answering":"/data/apps/lacode/self-correct-retriever/baseline/LawBench-main/data/zero_shot/1-2.json",
              "Consult":"/data/apps/lacode/self-correct-retriever/baseline/LawBench-main/data/zero_shot/3-8.json"}
    test_num = 500
    self_correct = False  

config = testCFG()

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + input
    return input

def write_log(data_all,combine_way):
    with open("/data/apps/lacode/self-correct-retriever/result/chatgpt_{}.json".format(combine_way), "w", encoding="utf-8") as f:
        for resp in data_all:
            line = json.dumps(resp, ensure_ascii=False)
            f.write(line + "\n")

def test(config):
    inputs, in_context_contents, results, labels=[],[],[],[]
    retrieval_data = []
    with open(config.test_path_dic[config.task],"r",encoding="utf-8") as f:
        if config.task in ["Article Recitation","Question Answering","Consult"]:
            items = json.load(f)
            print(len(items))
            random.shuffle(items)
            for item in tqdm(items[:config.test_num]):
                inputs.append(item["question"])
                labels.append(item["answer"])
                in_context_content = search(input=item["question"], kg_names=["融合知识"], top_k=3)
                in_context_contents.append(in_context_content)
                retrieval_data.append({"question":item["question"], "answer":item["answer"], "cxts":in_context_content})

        elif config.task=="ljp":
            lines = f.readlines()
            random.shuffle(lines)
            for line in tqdm(lines[:config.test_num]):
                fact = data["fact"]
                fact = get_prompt(fact, config.prompt_dic["charge"])
                labels.append(data["meta"]["accusation"])
                in_context_content = search(input=fact, kg_names=["融合知识"], top_k=2)
                inputs.append(fact)
                in_context_contents.append(in_context_content)

        elif config.task=="cvg":
            lines = f.readlines()
            random.shuffle(lines)
            for line in tqdm(lines[:config.test_num]):
                data = json.loads(line) 
                fact = data["input"]
                if not config.self_correct:
                    fact = get_prompt(fact, config.prompt_dic["cvg"])
                labels.append(data["label"])
                in_context_content = search(input=fact, kg_names=["融合知识"], top_k=2)
                inputs.append(fact)
                in_context_contents.append(in_context_content)

    write_log(retrieval_data, "retrieval")

    if config.self_correct:
        # tiny_inputs, tiny_query_log = get_openai_response(query=inputs,combine_way="abstract")
        # tiny_inputs = [get_prompt(el[0], config.prompt_dic["cvg"]) for el in tiny_inputs]  
        if config.task == "cvg": 
            inputs = [get_prompt(el, config.prompt_dic[config.task]) for el in inputs]
        # write_log(tiny_query_log, "abstract")
        combine_labels, droped_knowledge,drop_data = get_openai_response(inputs, in_context_contents, combine_way="drop", config = config)
        write_log(drop_data, "drop")
        optimized_knowledge, optimized_data = get_knowledge_combination(inputs, combine_labels, droped_knowledge, config)
        write_log(optimized_data, "optimize")
        combined_knowledge, combine_data = get_openai_response(inputs, optimized_knowledge, combine_way="combine", config = config)
        write_log(combine_data, "combine")
        if config.task in ["Article Recitation","Question Answering","Consult"]:
            inputs = [get_prompt(el, config.prompt_dic[config.task]) for el in inputs] 
        responses = get_openai_response(inputs, combined_knowledge, combine_way="answer", config = config)
    else:
        if config.task in ["Article Recitation","Question Answering","Consult"]:
            inputs = [get_prompt(el, config.prompt_dic[config.task]) for el in inputs] 
        # responses = get_openai_response(inputs, in_context_contents, combine_way="answer", config = config)
        responses = get_llm_response(inputs, combine_way="answer", config = config)
    for i in range(len(inputs)):
        results.append({"fact":inputs[i],"label":labels[i],"pred":str(responses[i])})

    name_add = "self_corr_" if config.self_correct else ""
    with open("/data/apps/lacode/self-correct-retriever/test_result/"+name_add+"_"+config.mlm_api+"_generation_{}.txt".format(config.task), "w", encoding='utf-8') as f:
        for dic in results:
            json.dump(dic,f,ensure_ascii=False)
            f.write("\n") 
    print(config.count_dic)
test(config)