start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

# set the keyword of under-evaluate dimension
# evaluate_step:800
# KEY=helpfulness
# KEY=correctness
# KEY=complexity
# evaluate_step:600
KEY=coherence
# KEY=verbosity
# evaluate_step:500
# KEY=harmlessness

model_names=(
name_of_reward_model
)

BASE_OUTPUT_PATH=../output/hard/${KEY}
BASE_MODEL_PATH=../models

EX_NAME=0219

Trained_checkpoint=""
ZERO_STAGE=2

Train_datasets=../data/mrm_hard/${KEY}/train.json
Test_datasets=../data/mrm_hard/${KEY}/test.json

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

mkdir -p $OUTPUT

export Num_Padding_at_Beginning=0 # this is model related

array_length=${#model_names[@]}

for (( i=0; i<${array_length}; i++ )); do

# mkdir ${BASE_OUTPUT_PATH}/${model_names[i]}
mkdir -p ${BASE_OUTPUT_PATH}/${model_names[i]}/${model_names[i]}_${EX_NAME}


nohup deepspeed --include localhost:0,1 --master_port 1235 ../src/task1_discriminant_rm_deepspeed_cate_3.py.py \
    --model_name_or_path ${BASE_MODEL_PATH}/${model_names[i]} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --train_datasets $Train_datasets \
    --test_datasets $Test_datasets \
    --max_seq_len 2048 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 1 \
    --disable_dropout \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --save_steps 99999 \
    --eval_steps 800 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --dtype bf16 \
    --gradient_checkpointing \
    --output_dir ${BASE_OUTPUT_PATH}/${model_names[i]}/${model_names[i]}_${EX_NAME} > ${BASE_OUTPUT_PATH}/${model_names[i]}/${model_names[i]}_${EX_NAME}/training.log
    # --trained_checkpoint $Trained_checkpoint \
wait

done

wait