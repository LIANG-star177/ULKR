from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_path = "/data/apps/luwen/luwen_baichuan/models/baichuan2-7b"
# model_path = "fine-tune/output/baichuan2_1005_110k"
model_path = "/data/apps/lyfcode/Baichuan2-main/fine-tune/output/baichuan2_1021_118k"

def generate_response_simple(prompt):
    torch.cuda.empty_cache()
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to('cuda')
    with torch.no_grad():
        pred = model.generate(**inputs, max_new_tokens=800, repetition_penalty=1.2)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return response

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).half()
# text = "你好，你叫什么名字？\n"

texts = ["你好",
    "工人工地深夜上厕所摔倒死亡赔偿法?",
    "如果喝了两斤白酒后开车，会有什么后果？",
    "你好，我老公在工地干活的时候从架子上摔下来，导致腰椎骨折。现在他已经出院了，但是老板不肯赔偿医药费和误工费。我们该怎么办？"
]

for text in texts:
    resp =generate_response_simple(text)
    print("=== input ===")
    print(text)
    print("=== resp ===")
    print(resp)