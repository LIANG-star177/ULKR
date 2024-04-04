# -*- coding: utf-8 -*-

import openai
import os
import json
import threading
import time
from tqdm import tqdm
import argparse
from mycode.utils import loader
from mycode.prompt import wyq
import tiktoken

# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

key_pool = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]
openai.api_key = key_pool[0]


def dav_response(text_list, model_name):
    enc = tiktoken.get_encoding("p50k_base")
    max_token = [len(enc.encode(t)) for t in text_list]
    max_token = 4096 - max(max_token)
    response = openai.Completion.create(
        model=model_name, prompt=text_list, max_tokens=max_token)  # greedy
    return response

def dav2_response(text_list):
    return dav_response(text_list, "text-davinci-002")

def dav3_response(text_list):
    return dav_response(text_list, "text-davinci-003")

def turbo_response(text_list):
    assert len(text_list) == 1, "gpt-3.5-turbo使用batch必须为1"
    text = text_list[0]
    enc = tiktoken.get_encoding("cl100k_base")
    max_token = 4096 - len(enc.encode(text)) - 500
    # (1608 in the messages, 2496 in the completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0)
    return response


def get_openai_response(data, in_context_contents, key_pool, args, meta):
    assert len(data) == len(in_context_contents)
    llm_response_function = {
        "text-davinci-002": dav2_response,
        "text-davinci-003": dav3_response,
        "gpt-3.5-turbo": turbo_response,
    }
    response_function = llm_response_function[args.model]

    truncated_data = []
    for i in tqdm(range(len(data)), desc="pre-process prompt"):
        case = data[i]
        fact = case["fact"].replace(" ", "") # raw fact
        if args.use_split_fact: # reframed fact
            fact = [case["fact_split"]["zhuguan"], case["fact_split"]["keguan"], case["fact_split"]["shiwai"]]
            fact = " ".join(fact)

        max_len = 512
        fact = loader.truncate_text(fact, max_len=max_len)
        # wyq
        prompt = wyq.retrieved_label_option_fewshot(fact, in_context_contents[i], args)

        obj = {}
        obj["prompt"] = prompt
        obj["caseID"] = case["caseID"]
        truncated_data.append(obj)
        # break

    print("Starting")
    i = 0
    batch = args.batch
    while i < len(truncated_data):
        print(f"=== {i} / {len(data)} ===")
        prompt_list = truncated_data[i:i+batch]
        text_list = [t["prompt"] for t in prompt_list]
        print(text_list[0])
        try:
            response = response_function(text_list)
            resp_temp = response.copy()
            responses = []
            for run in range(len(prompt_list)):
                resp_temp["choices"] = [response["choices"][run]]
                resp_temp["caseID"] = prompt_list[run]["caseID"]
                responses.append(resp_temp.copy())
        except openai.error.RateLimitError as e:
            e = repr(e)
            print(e)
            if "limit" in e:
                time.sleep(60)
            elif "quota" in e:
                if len(key_pool) == 0:
                    print("用光了key！")
                    exit()
                print(f"=== 当前key: {key_pool[0]} ===")
                openai.api_key = key_pool[0]
                key_pool = key_pool[1:]
                time.sleep(1)
            continue
        except openai.error.APIError as e:
            e = repr(e)
            print(e)
            time.sleep(60)
        except Exception as e:
            # https://platform.openai.com/docs/guides/error-codes/api-errors
            e = repr(e)
            print(e)
            exit()

        time.sleep(1)
        with open(args.output_path, "a+", encoding="utf-8") as f:
            for resp in responses:
                line = json.dumps(resp, ensure_ascii=False)
                f.write(line + "\n")
        i += batch
        # break


def run_llm(args):
    data = []
    with open(args.input_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            case = json.loads(line)
            data.append(case)

    dumped_data = []
    with open(args.output_path, encoding="utf-8") as f:
        for line in f.readlines():
            dumped_data.append(line)

    data = data[len(dumped_data):]
    print("to run data count: ", len(data))

    # load similar cases
    similar_cases = loader.load_similar_case(args.sc_pool_path, args.sc_idx_path)
    similar_cases = similar_cases[len(dumped_data):]

    # load [charge, article, penalty] topk prediction
    topk_label_option = loader.load_topk_option(args.topk_label_option_path)
    topk_label_option = topk_label_option[len(dumped_data):]

    # load relevant article definition
    if args.task == "article":
        article_definition = loader.load_retrieved_articles(args.topk_label_option_path, index_type="num")
        article_definition = article_definition[len(dumped_data):]
    else:
        article_definition = [["#"] for _ in range(len(data))]
    
    in_context_contents = [{
        "similar_cases": similar_cases[i],
        "topk_label_option": topk_label_option[i],
        "article_definition": article_definition[i], 
    }
        for i in range(len(similar_cases))]
    
    # meta knowledge
    meta = {}
    get_openai_response(data, in_context_contents, key_pool, args, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument(
        '--model', default='text-davinci-003', help='text-davinci-002/3 & gpt-3.5-turbo')
    parser.add_argument(
        '--dataset', default="cjo22_sample400", help='[cail18, cjo22, cjo22_sample400]')
    parser.add_argument(
        '--small_model', default="CNN", help='[CNN, TopJudge, ELE]')
    parser.add_argument(
        '--task', default="charge", help='[charge, article, penalty]')
    parser.add_argument(
        '--shot', default=3, help='few shots')
    parser.add_argument(
        '--batch', default=4, help='batch, reduce throughput cost')
    parser.add_argument(
        '--retriever', default="contriever_unsup", help="[bm25, contriever_unsup]")
    parser.add_argument(
        '--use_split_fact', default=False, help="[fact / fact_split]")
    
    # input: testset path, output: llm response path
    parser.add_argument(
        '--input_path', default="", help='testset path')
    parser.add_argument(
        '--output_path', default="", help='llm response path')
    
    # in-context contents path
    parser.add_argument(
        '--sc_pool_path', default="", help='pool of similar case')
    parser.add_argument(
        '--sc_idx_path', default="", help='index list of similar case')
    parser.add_argument(
        '--topk_label_option_path', default="", help='index list of topk label option')
    
    args = parser.parse_args()

    if args.input_path == "":
        prefix = f"data/0527/{args.dataset}/"
        if not args.use_split_fact:
            args.input_path = prefix + "testset.json"
        else:
            args.input_path = prefix + "testset_fact_split.json" 
        
    if args.output_path == "":
        args.output_path = f"data/0603/llm/{args.small_model}/{args.dataset}/{args.task}/{args.shot}shot/{args.model}.json" # sample400
        if not os.path.exists(args.output_path):
            dirname = os.path.dirname(args.output_path)
            os.makedirs(dirname, exist_ok=True)
            with open(args.output_path, "w") as f:
                pass

    if args.sc_pool_path == "":
        prefix = f"data/0529/"
        if not args.use_split_fact:
            args.sc_pool_path = prefix + "lak_cail.json"
        else:
            args.sc_pool_path = prefix + "lak_cail_fact_split.json" 
        
    if args.sc_idx_path == "":
        prefix = f"data/0529/sc_idx/{args.dataset}/{args.retriever}/"
        if args.task in ["charge", "article"]: # sub * 0.5  + obj * 0.5 
            args.sc_idx_path = prefix + "subobj_sc_idxs.json"
        if args.task in ["penalty"]: # sub * 0.25 + obj * 0.25 + ex * 0.5 
            args.sc_idx_path = prefix + "subobjex_sc_idxs.json"

    if args.topk_label_option_path == "":
        args.topk_label_option_path = f"data/0603/small_model_out/{args.small_model}/{args.dataset}/{args.task}_topk.json"

    run_llm(args)
