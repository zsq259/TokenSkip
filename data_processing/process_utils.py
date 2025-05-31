import regex

from data_processing.answer_extraction import extract_math_answer, strip_string

def process_gsm8k_test(item):
    sample = {
        'dataset': 'gsm8k-cot',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': item['question']},
            {'role': 'assistant', 'content': regex.sub(r"<<[^<>]*>>", "", item['cot']) + "\nSo the answer is $\\boxed{" + item['answer'].strip() + "}$."}
        ],
        'answer': item['answer'].replace(',', '')
    }
    yield sample

def process_math_test(item):
    question = item["problem"]
    try:
        answer = extract_math_answer(question, item['solution'], task="cot")
    except:
        return
    sample = {
        "dataset": "math-cot",
        "id": item['id'],
        "level": item["level"],
        "type": item["type"],
        # "category": item["category"],
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n".join(regex.split(r"(?<=\.) (?=[A-Z])", item["solution"]))}
        ],
        "answer": answer
    }
    yield sample

def process_mmlu_pro_test(item):
    """
    处理选择题格式的数据集
    数据格式示例：
    {
        "question_id": 800, 
        "question": "A dress sells for $50.00...", 
        "options": ["$12.50", "$17.50", ...], 
        "answer": "G", 
        "answer_index": 6,
        "cot_content": "", 
        "category": "business", 
        "src": "stemez-Business"
    }
    """
    options_text = ""
    for i, opt in enumerate(item["options"]):
        option_letter = chr(65 + i)
        options_text += f"{option_letter}. {opt}\n"
    
    # 构建完整的问题文本
    question_text = f"{item['question']}\n\n{options_text.strip()}"
    sample = {
        "dataset": item["src"],
        "id": item["question_id"],
        "messages": [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": regex.sub(r"<<[^<>]*>>", "", item["cot_content"]) + "\nSo the answer is $\\boxed{" + item["answer"] + "}$."}
        ],
        "answer": item["answer"]
    }
    yield sample
    