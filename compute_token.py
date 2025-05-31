import tiktoken
import jsonlines, json
import re

file_path = "/data/share/data/llama-factory/TokenSkip/outputs/DeepSeek-R1-Distill-Qwen-7B-llmlingua2/gsm8k/7b/TokenSkip/0.5/samples/predictions.jsonl"
results = []


def compute_token_count(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    return len(tokens)

def compute_token_count_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            # response = item['model_output']
            input = item['input']
            # 获取 <|eot_id|>0.5<|eot_id|> 中的 0.5 等
            match = re.search(r'<\|eot_id\|>([\d.]+)<\|eot_id\|>', input)
            if match:
                eot_id = match.group(1)
                input = input.replace(f'<|eot_id|>{eot_id}<|eot_id|>', '')
            else:
                eot_id = 1
            eot_id = float(eot_id)
            # print(f"eot_id: {eot_id}")

            response = item['output']
            token_count = compute_token_count(response)
            results.append({
                'eot_id': int(eot_id * 10),
                'token_count': token_count
            })
            
    print(len(results))
            
    for i in range(5, 11):
        # id = i / 10
        id = i
        count = sum(1 for result in results if result['eot_id'] == id)
        print(f"eot_id: {id}, count: {count}")
        print(f"Mean Token Count: {sum(result['token_count'] for result in results if result['eot_id'] == id) / count if count > 0 else 0}")

def compute_token_count_from_jsonlines(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            if line['accuracy']:
                continue
            response = line['model_output']
            # response = line['compressed_cot']
            token_count = compute_token_count(response)
            results.append(token_count)
            
    print(len(results))
    print(f"Mean Token Count: {sum(results) / len(results) if results else 0}")

def main():
    if file_path.endswith('.json'):
        compute_token_count_from_json(file_path)
    elif file_path.endswith('.jsonl'):
        compute_token_count_from_jsonlines(file_path)
    else:
        print("Unsupported file format. Please provide a .json or .jsonl file.")
        return
    
        
if __name__ == "__main__":
    main()