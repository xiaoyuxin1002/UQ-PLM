import dill as pk
import numpy as np
from scipy import stats
from scipy.special import softmax
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from info import Info
from util import parse_args_test, set_seed, myprint, load, iter_batch


def process(info, method, inputs, model, if_eval=True, num_mc=1):
    
    if if_eval: model.eval()
    else: model.train()
        
    with torch.no_grad():
        all_probs, all_labels = [], []
        for idx_batch, (batch_inputs, batch_labels) in enumerate(iter_batch(info, inputs, if_shuffle=False)):
            
            batch_probs = []
            for _ in range(num_mc):
                round_probs = model.infer(batch_inputs, if_prob=method!=info.METHOD_VANILLA)
                batch_probs.append(round_probs)
            all_probs.append(torch.stack(batch_probs).mean(0))
            all_labels.append(batch_labels)
            
    all_probs, all_labels = torch.cat(all_probs), torch.cat(all_labels)
    return all_probs, all_labels


def recalibrate(info, all_logits, all_labels):

    temperature = nn.Parameter(1.5 * torch.ones(1).to(info.DEVICE_GPU))
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=1000)
    
    def eval():        
        loss = F.cross_entropy(all_logits / temperature, all_labels)
        optimizer.zero_grad()
        loss.backward()
        return loss
    optimizer.step(eval)

    return temperature.item()


def get_ece(accuracy, confidence):

    ece = 0
    bins = np.linspace(0, 1, 11)
    for bin_index in range(bins.shape[0]-1):
        bin_lower, bin_upper = bins[bin_index], bins[bin_index+1]
        bin_in = (bin_lower<confidence) * (confidence<bin_upper)
        bin_ratio = bin_in.mean()

        if bin_ratio > 0:
            bin_accs = accuracy[bin_in].mean()
            bin_confs = confidence[bin_in].mean()
            ece += np.abs(bin_accs - bin_confs) * bin_ratio

    return ece


def selective_prediction(all_accs, all_confs):
    
    conf_accs = all_accs[np.argsort(all_confs)]
    conf_rpp = np.cumsum(conf_accs)[~conf_accs].sum() / (all_accs.shape[0]**2)
    
    return conf_rpp


def out_detection(in_confs, out_confs):
    
    conf_far95 = (in_confs < np.percentile(out_confs, 95)).mean()
    
    return conf_far95


def evaluate(args, info, type, all_probs, all_labels, in_confs=None):

    all_confs, all_preds = all_probs.max(-1)
    all_confs = all_confs.to(info.DEVICE_CPU).numpy()
    all_accs = (all_preds == all_labels).to(info.DEVICE_CPU).numpy()
    
    all_scores = {}
    all_scores[info.METRIC_ERR] = 1 - all_accs.mean()
    all_scores[info.METRIC_ECE] = get_ece(all_accs, all_confs)
    all_scores[info.METRIC_RPP] = selective_prediction(all_accs, all_confs)
    if type != info.TYPE_TEST_IN: all_scores[info.METRIC_FAR] = out_detection(in_confs, all_confs)
    
    return all_confs, all_scores


def feed(info, type, method, all_labels, all_probs, all_scores, each_labels, each_probs, each_scores):

    all_labels[type] = each_labels.to(info.DEVICE_CPU).numpy()
    all_probs[(type, method)].append(each_probs.to(info.DEVICE_CPU).numpy())
    for metric, value in each_scores.items():
        all_scores[(type, method, metric)].append(value)


def main():
    
    args = parse_args_test()
    info = Info(args)
    
    myprint('='*20, info.FILE_STDOUT)
    myprint(f'Start {args.stage}ing {args.model_name} for {args.task_id}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    set_seed(args.seed)
    inputs_list = load(args, info, args.stage)
    all_labels, all_probs, all_scores = {}, defaultdict(list), defaultdict(list)
    
    for model_id, model_file in enumerate(info.FILE_MODELS[info.VERSION_DET]):
        myprint(f'Load {info.VERSION_DET} Model {model_id}', info.FILE_STDOUT)
        model = pk.load(open(model_file, 'rb')).to(info.DEVICE_GPU)
        myprint('-'*20, info.FILE_STDOUT)
        
        myprint(f'Calculate Temperature for {info.VERSION_DET} Model {model_id}', info.FILE_STDOUT)
        dev_logits, dev_labels = process(info, info.METHOD_VANILLA, inputs_list[0], model, if_eval=True, num_mc=1)
        temperature = recalibrate(info, dev_logits, dev_labels)
        myprint(f'Temperature = {temperature:.4f}', info.FILE_STDOUT)
        myprint('-'*20, info.FILE_STDOUT)
        
        in_confs = None; vanilla_logits = {}
        for type, inputs in zip(info.TYPE_TESTS, inputs_list[1:]):
            myprint(f'Uncertainty for {type} Data via {info.VERSION_DET} Model {model_id} and {info.METHOD_VANILLA}', info.FILE_STDOUT)
            each_logits, each_labels = process(info, info.METHOD_VANILLA, inputs, model, if_eval=True, num_mc=1)
            each_probs = F.softmax(each_logits, dim=-1); vanilla_logits[type] = each_logits
            each_confs, each_scores = evaluate(args, info, type, each_probs, each_labels, in_confs=in_confs)
            feed(info, type, info.METHOD_VANILLA, all_labels, all_probs, all_scores, each_labels, each_probs, each_scores)
            if type == info.TYPE_TEST_IN: in_confs = each_confs
        myprint('-'*20, info.FILE_STDOUT)
        
        in_confs = None
        for type, inputs in zip(info.TYPE_TESTS, inputs_list[1:]):
            myprint(f'Uncertainty for {type} Data via {info.VERSION_DET} Model {model_id} and {info.METHOD_TEMP_SCALING}', info.FILE_STDOUT)
            each_probs = F.softmax(vanilla_logits[type] / temperature, dim=-1)
            each_labels = torch.from_numpy(all_labels[type]).long().to(info.DEVICE_GPU)
            each_confs, each_scores = evaluate(args, info, type, each_probs, each_labels, in_confs=in_confs)
            feed(info, type, info.METHOD_TEMP_SCALING, all_labels, all_probs, all_scores, each_labels, each_probs, each_scores)
            if type == info.TYPE_TEST_IN: in_confs = each_confs
        myprint('-'*20, info.FILE_STDOUT)
        
        in_confs = None
        for type, inputs in zip(info.TYPE_TESTS, inputs_list[1:]):
            myprint(f'Uncertainty for {type} Data via {info.VERSION_DET} Model {model_id} and {info.METHOD_MC_DROPOUT}', info.FILE_STDOUT)
            each_probs, each_labels = process(info, info.METHOD_MC_DROPOUT, inputs, model, if_eval=False, num_mc=info.HP_NUM_DROPOUT_MC)
            each_confs, each_scores = evaluate(args, info, type, each_probs, each_labels, in_confs=in_confs)
            feed(info, type, info.METHOD_MC_DROPOUT, all_labels, all_probs, all_scores, each_labels, each_probs, each_scores)
            if type == info.TYPE_TEST_IN: in_confs = each_confs
        myprint('-'*20, info.FILE_STDOUT)
        
    for ensemble_id, model_ids in enumerate(combinations(np.arange(len(info.FILE_MODELS[info.VERSION_DET])), info.HP_NUM_ENSEMBLE)):
        in_confs = None
        for type, inputs in zip(info.TYPE_TESTS, inputs_list[1:]):
            myprint(f'Uncertainty for {type} Data via Ensemble {ensemble_id}', info.FILE_STDOUT)
            each_probs = torch.from_numpy(np.mean([all_probs[(type, info.METHOD_VANILLA)][model_id] for model_id in model_ids], axis=0)).float().to(info.DEVICE_GPU)
            each_labels = torch.from_numpy(all_labels[type]).long().to(info.DEVICE_GPU)
            each_confs, each_scores = evaluate(args, info, type, each_probs, each_labels, in_confs=in_confs)
            feed(info, type, info.METHOD_ENSEMBLE, all_labels, all_probs, all_scores, each_labels, each_probs, each_scores)
            if type == info.TYPE_TEST_IN: in_confs = each_confs
        myprint('-'*20, info.FILE_STDOUT)

    for model_id, model_file in enumerate(info.FILE_MODELS[info.VERSION_STO]):
        myprint(f'Load {info.VERSION_STO} Model {model_id}', info.FILE_STDOUT)
        model = pk.load(open(model_file, 'rb')).to(info.DEVICE_GPU)
        myprint('-'*20, info.FILE_STDOUT)

        in_confs = None
        for type, inputs in zip(info.TYPE_TESTS, inputs_list[1:]):
            myprint(f'Uncertainty for {type} Data via {info.VERSION_STO} Model {model_id} and {info.METHOD_LL_SVI}', info.FILE_STDOUT)
            each_probs, each_labels = process(info, info.METHOD_LL_SVI, inputs, model, if_eval=True, num_mc=1)
            each_confs, each_scores = evaluate(args, info, type, each_probs, each_labels, in_confs=in_confs)
            feed(info, type, info.METHOD_LL_SVI, all_labels, all_probs, all_scores, each_labels, each_probs, each_scores)
            if type == info.TYPE_TEST_IN: in_confs = each_confs
        myprint('-'*20, info.FILE_STDOUT)
            
    for type in info.TYPE_TESTS:
        for method in info.METHODS:
            myprint(f'Data Type: {type} & Uncertainty Method: {method}', info.FILE_STDOUT)
            for metrics in info.METRICS:
                result = []
                for metric in metrics:
                    if (type, method, metric) not in all_scores: continue
                    scores = all_scores[(type, method, metric)]
                    mean, sem = np.mean(scores), stats.sem(scores)
                    result.append(f'{metric}: {mean:.4f}±{sem:.4f}')
                if len(result) != 0: myprint(' | '.join(result), info.FILE_STDOUT)
            myprint('-'*20, info.FILE_STDOUT)
        
    pk.dump(all_scores, open(info.FILE_SCORE, 'wb'), -1)
    pk.dump((all_labels, all_probs), open(info.FILE_PROB, 'wb'), -1)
    
    myprint(f'Finish {args.stage}ing {args.model_name} for {args.task_id}', info.FILE_STDOUT)
    myprint('='*20, info.FILE_STDOUT)

    
if __name__=='__main__':
    main()