import os
import math
import time
import random
import argparse
import dill as pk
import numpy as np

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import Model


def parse_args_train():
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--task_id', type=str, default='Task1')
    parser.add_argument('--model_name', type=str, default='bert_base-model_ce')
    parser.add_argument('--stage', type=str, default='train')
    parser.add_argument('--version', type=str, default='det')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    return args


def parse_args_test():
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--task_id', type=str, default='Task1')
    parser.add_argument('--model_name', type=str, default='bert_base-model_ce')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    return args


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
        
        
def myprint(text, file):
    
    file = open(file, 'a')
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, file=file, flush=True)
    file.close()
    
    
def load(args, info, stage):
    
    if stage == info.STAGE_TRAIN: 
        name_list = [info.TYPE_TRAIN, info.TYPE_DEV]
        ori_list = [info.FILE_ORI_TRAIN, info.FILE_ORI_DEV]
        input_list = [info.FILE_INPUT_TRAIN, info.FILE_INPUT_DEV]
    elif stage == info.STAGE_TEST:
        name_list = [info.TYPE_DEV, info.TYPE_TEST_IN, info.TYPE_TEST_OUT]
        ori_list = [info.FILE_ORI_DEV, info.FILE_ORI_TEST_IN, info.FILE_ORI_TEST_OUT]
        input_list = [info.FILE_INPUT_DEV, info.FILE_INPUT_TEST_IN, info.FILE_INPUT_TEST_OUT]
        
    data_list = []
    tokenizer = AutoTokenizer.from_pretrained(info.PLM_NAME)
    for name, ori_file, input_file in zip(name_list, ori_list, input_list):
        myprint(f'Load Data from {name}', info.FILE_STDOUT)
        
        if os.path.isfile(input_file):
            all_inputs, all_labels = pk.load(open(input_file, 'rb'))
            
        else:
            ori_data = pk.load(open(ori_file, 'rb'))
            text1s, text2s, labels = [], [], []
            for row in ori_data:
                
                if args.task_id in ['Task1']:
                    text1s.append(row[0])
                    labels.append(row[1])
                    
                elif args.task_id in ['Task2']:
                    text1s.append(row[0])
                    text2s.append(row[1])
                    labels.append(row[2])
                    
                elif args.task_id in ['Task3']:
                    text1s += [row[0]] * info.NUM_CLASS[1]
                    text2s += row[1:1+info.NUM_CLASS[1]]
                    labels.append(row[-1])
                    
            if len(text2s) == 0: text2s = None
            all_inputs = tokenizer(text1s, text2s, return_tensors='pt', padding=True, truncation=True, max_length=512)
            all_labels = torch.Tensor(labels).long()
            pk.dump((all_inputs, all_labels), open(input_file, 'wb'), -1)
            
        data_list.append((all_inputs.to(info.DEVICE_GPU), all_labels.to(info.DEVICE_GPU)))
    myprint('-'*20, info.FILE_STDOUT)
        
    return data_list


def prepare(info, inputs_train):

    config = AutoConfig.from_pretrained(info.PLM_NAME, num_labels=info.NUM_CLASS[0])
    transformer = AutoModel.from_pretrained(info.PLM_NAME)
    
    model = Model(info, config, transformer).to(info.DEVICE_GPU)
    parameters = list(model.parameters())
    optimizer = AdamW(parameters, lr=info.HP_LR)

    num_updates = math.ceil(inputs_train[1].shape[0] / info.HP_BATCH_SIZE) * info.HP_NUM_EPOCH
    num_warmups = int(num_updates * info.HP_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmups, num_training_steps=num_updates)
    
    return model, optimizer, scheduler


def iter_batch(info, inputs, if_shuffle=False):
    
    all_inputs, all_labels = inputs
    batch_seq = np.arange(all_labels.shape[0])
    if if_shuffle: np.random.shuffle(batch_seq)
    num_batch = math.ceil(batch_seq.shape[0] / info.HP_BATCH_SIZE)
    
    for idx_batch in range(num_batch):
        batch_indices = batch_seq[idx_batch*info.HP_BATCH_SIZE : (idx_batch+1)*info.HP_BATCH_SIZE]
        
        batch_labels = all_labels[batch_indices]
        if info.NUM_CLASS[1] != 1: batch_indices = np.repeat(batch_indices*info.NUM_CLASS[1], info.NUM_CLASS[1]) + np.tile(np.arange(info.NUM_CLASS[1]), batch_indices.shape[0])
        batch_inputs = {k:v[batch_indices] for k,v in all_inputs.items()}
        
        yield batch_inputs, batch_labels