BENCHMARK="mmlu-pro" # "gsm8k", "math"
# OUPTUT_DIR="outputs/DeepSeek-R1-Distill-Qwen-7B-llmlingua2/${BENCHMARK}/"
# MODEL_PATH="../model/DeepSeek-R1-Distill-Qwen-7B"

# OUPTUT_DIR="outputs/Qwen2.5-7B-Instruct/${BENCHMARK}/"
# MODEL_PATH="../model/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

OUPTUT_DIR="outputs/DeepSeek-R1-Distill-Llama-8B/${BENCHMARK}/"
MODEL_PATH="../model/DeepSeek-R1-Distill-Llama-8B"

MODEL_SIZE="7b"
MODEL_TYPE="qwen" # "llama3", "qwen"
DATA_TYPE="train" # "train", "test"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=4096 # 512 for gsm8k, 1024 for math
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

# TokenSkip Settings
ADAPTER_PATH="1"
COMPRESSION_RATIO=0.0


CUDA_VISIBLE_DEVICES=5 python evaluation.py --output-dir ${OUPTUT_DIR} --model-path ${MODEL_PATH} --tokenizer-path ${MODEL_PATH} \
    --model-size ${MODEL_SIZE} --model-type ${MODEL_TYPE} --data-type ${DATA_TYPE}  --max_num_examples ${MAX_NUM_EXAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} --eval_batch_size ${EVAL_BATCH_SIZE} --temperature ${TEMPERATURE} --seed ${SEED} --benchmark ${BENCHMARK} \
    --adapter-path ${ADAPTER_PATH} --compression_ratio ${COMPRESSION_RATIO}