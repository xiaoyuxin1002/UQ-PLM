#!/bin/bash

cuda=0
model_name="bert_base-ce"
    
for task_id in "Task1" 
do
    for seed in 0 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=${cuda} python3 train.py --task_id=${task_id} --model_name=${model_name} --version="det" --seed=${seed}
    done
    for seed in 0 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=${cuda} python3 train.py --task_id=${task_id} --model_name=${model_name} --version="sto" --seed=${seed}
    done
    CUDA_VISIBLE_DEVICES=${cuda} python3 test.py --task_id=${task_id} --model_name=${model_name}
done