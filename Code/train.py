import math
import dill as pk

import torch
from torch.nn.utils import clip_grad_norm_

from info import Info
from util import parse_args_train, set_seed, myprint, load, prepare, iter_batch


def test(info, idx_epoch, inputs_dev, model):
    
    model.eval()
    with torch.no_grad():
        
        all_preds, all_labels = [], []
        for idx_batch, (batch_inputs, batch_labels) in enumerate(iter_batch(info, inputs_dev, if_shuffle=False)):
            
            batch_probs = model.infer(batch_inputs)
            all_preds.append(batch_probs.argmax(dim=1))
            all_labels.append(batch_labels)
            
    all_accuracy = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
    myprint(f'Finish Testing Epoch {idx_epoch} | Accuracy {all_accuracy:.4f}', info.FILE_STDOUT)
    
    return all_accuracy


def train(info, idx_epoch, inputs_train, model, optimizer, scheduler):

    model.train()
    num_batch = math.ceil(inputs_train[1].shape[0] / info.HP_BATCH_SIZE)
    report_batch = num_batch // 5
    for idx_batch, (batch_inputs, batch_labels) in enumerate(iter_batch(info, inputs_train, if_shuffle=True)):

        batch_loss = model.learn(batch_inputs, batch_labels)
        optimizer.zero_grad()
        batch_loss.backward()
        clip_grad_norm_(model.parameters(), info.HP_MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        if idx_batch % report_batch == 0:
            myprint(f'Finish Training Epoch {idx_epoch} | Batch {idx_batch} | Loss {batch_loss.item():.4f}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    
def main():
    
    args = parse_args_train()
    info = Info(args)
    
    myprint('='*20, info.FILE_STDOUT)
    myprint(f'Start {args.stage}ing {args.version}-{args.model_name} for {args.task_id}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    set_seed(args.seed)
    inputs_train, inputs_dev = load(args, info, args.stage)
    model, optimizer, scheduler = prepare(info, inputs_train)

    best_accuracy, best_epoch = 0, 0
    for idx_epoch in range(info.HP_NUM_EPOCH):
        
        train(info, idx_epoch, inputs_train, model, optimizer, scheduler)
        epoch_accuracy = test(info, idx_epoch, inputs_dev, model)
        
        if epoch_accuracy >= best_accuracy:
            best_accuracy, best_epoch = epoch_accuracy, idx_epoch
            pk.dump(model, open(info.FILE_MODEL, 'wb'), -1)
            myprint(f'This is the Best Performing Epoch by far - Epoch {idx_epoch} Accuracy {epoch_accuracy:.4f}', info.FILE_STDOUT)
        else:
            myprint(f'Not the Best Performing Epoch by far - Epoch {idx_epoch} Accuracy {epoch_accuracy:.4f} vs Best Accuracy {best_accuracy:.4f}', info.FILE_STDOUT)
        myprint('-'*20, info.FILE_STDOUT)

    myprint(f'Finish {args.stage}ing {args.version}-{args.model_name} for {args.task_id}', info.FILE_STDOUT)
    myprint('='*20, info.FILE_STDOUT)
    
    
if __name__=='__main__':
    main()