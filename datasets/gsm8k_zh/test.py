from datasets import load_dataset, DatasetDict
import jsonlines

test_file = "test.jsonl"

new_test_file = "new_test.jsonl"

with jsonlines.open(test_file, 'r') as reader:
    data = [item for item in reader]
new_data = []
for item in data:
    item['cot'] = ''
    new_data.append(item)
    
with jsonlines.open(new_test_file, 'w') as writer:
    for item in new_data:
        writer.write(item)