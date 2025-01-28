import os
from openai import AsyncOpenAI
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import asyncio
import platform


client = AsyncOpenAI(
    api_key="sk-dc0ae534d44043bcb52a9f5bcc394b9d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


SEMAPHORE_LIMIT = 10
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


REQUEST_DELAY = 1

def truncate_text(text, max_tokens=500):

    return " ".join(text.split()[:max_tokens])

async def compare_cases_async(case_a, case_b, retries=3):

    async with semaphore:

        case_a_truncated = truncate_text(case_a)
        case_b_truncated = truncate_text(case_b)

        prompt = f"""
        你是一个法律专家，负责进行类案检索任务。你将收到两个案例描述（case_a 和 case_b），你需要判断它们的相关性程度。

        案例A描述如下：
        {case_a_truncated}

        案例B描述如下：
        {case_b_truncated}

        请执行以下步骤：
        1. 比较案例A和案例B的相关性。
        2. 判断它们的相关性程度，并返回对应的 `label`：
           - `0`：不相关
           - `1`：弱相关
           - `2`：较强相关
           - `3`：完全相关

        请只返回一个整数（`0`、`1`、`2` 或 `3`），不要返回其他内容。如果返回的内容不是 `0`、`1`、`2` 或 `3`，将被视为无效。
        """

        for attempt in range(retries):
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="qwen-turbo",
                        messages=[
                            {"role": "system", "content": "你是一个法律专家，负责进行类案检索任务。"},
                            {"role": "user", "content": prompt}
                        ]
                    ),
                    timeout=10
                )


                response = completion.choices[0].message.content
                try:
                    predicted_label = int(response.strip())
                    if predicted_label not in [0, 1, 2, 3]:
                        predicted_label = -1
                except ValueError:
                    predicted_label = -1
                return predicted_label
            except asyncio.TimeoutError:
                print(f"Timeout while processing case_a: {case_a_truncated[:50]}... (Attempt {attempt + 1}/{retries})")
            except Exception as e:
                print(f"Error while processing case_a: {case_a_truncated[:50]}..., error: {e} (Attempt {attempt + 1}/{retries})")
            await asyncio.sleep(REQUEST_DELAY)

        print(f"Failed to process case_a: {case_a_truncated[:50]}... after {retries} attempts")
        return -1

async def main():
    true_labels = []
    predicted_labels = []

    with open(r'C:\Users\84065\Desktop\data\home\zhongxiang_sun\code\explanation_project\explanation_model\models_for_paper\data\Lecard_1000.jsonl', 'r', encoding='utf-8') as f, \
         open(r'C:\Users\84065\Desktop\data\home\zhongxiang_sun\code\explanation_project\explanation_model\models_for_paper\data\Lecard_1000_qwenresults.txt', 'w', encoding='utf-8') as result_file:

        tasks = []
        for line in f:
            case = json.loads(line)
            case_a = case['case_a']
            case_b = case['case_b']
            true_label = case['label']


            task = compare_cases_async(case_a, case_b)
            tasks.append((true_label, task))


        results = await asyncio.gather(*[task for _, task in tasks])


        for (true_label, _), predicted_label in zip(tasks, results):
            if predicted_label != -1:
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                result_file.write(f"真实标签: {true_label}, 预测标签: {predicted_label}\n")
                print(f"真实标签: {true_label}, 预测标签: {predicted_label}")


    if len(true_labels) == 0:
        print("没有有效的预测标签，无法计算指标。")
        return


    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')


    with open(r'C:\Users\84065\Desktop\data\home\zhongxiang_sun\code\explanation_project\explanation_model\models_for_paper\data\Lecard_1000_qwenresults.txt', 'a', encoding='utf-8') as result_file:
        result_file.write("\n指标计算结果：\n")
        result_file.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        result_file.write(f"精确率 (Precision): {precision:.4f}\n")
        result_file.write(f"召回率 (Recall): {recall:.4f}\n")
        result_file.write(f"F1分数 (F1 Score): {f1:.4f}\n")

    print("\n指标计算结果：")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")

if __name__ == '__main__':

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


    asyncio.run(main())