import os
from pathlib import Path
from collections import defaultdict


class Info:
    
    def __init__(self, args):
        
        self.metadata()
        self.individual(args)
        
        
    def metadata(self):
        
        self.DEVICE_CPU = 'cpu'
        self.DEVICE_GPU = 'cuda'
        
        self.STAGE_TRAIN = 'train'
        self.STAGE_TEST = 'test'
        
        self.VERSION_DET = 'det'
        self.VERSION_STO = 'sto'

        self.LOSS_BR = 'br' # Brier Loss
        self.LOSS_CE = 'ce' # Cross Entropy
        self.LOSS_FL = 'fl' # Focal Loss
        self.LOSS_LS = 'ls' # Label Smoothing
        self.LOSS_MM = 'mm' # Max Mean Calibration Error
        
        self.METHOD_VANILLA = 'Vanilla'
        self.METHOD_TEMP_SCALING = 'Temp Scaling'
        self.METHOD_MC_DROPOUT = 'MC Dropout'
        self.METHOD_ENSEMBLE = 'Ensemble'
        self.METHOD_LL_SVI = 'LL SVI'
        self.METHODS = [self.METHOD_VANILLA, self.METHOD_TEMP_SCALING, self.METHOD_MC_DROPOUT, self.METHOD_ENSEMBLE, self.METHOD_LL_SVI]
        
        self.METRIC_ERR = 'ERR'
        self.METRIC_ECE = 'ECE'
        self.METRICS_EXPLICIT = [self.METRIC_ERR, self.METRIC_ECE]
        self.METRIC_RPP = 'RPP'
        self.METRIC_FAR = 'FAR95'
        self.METRICS_IMPLICIT = [self.METRIC_RPP, self.METRIC_FAR]
        self.METRICS = [self.METRICS_EXPLICIT, self.METRICS_IMPLICIT]
        
        self.TYPE_TRAIN = 'train'
        self.TYPE_DEV = 'dev'
        self.TYPE_TEST_IN = 'test_in'
        self.TYPE_TEST_OUT = 'test_out'
        self.TYPE_TESTS = [self.TYPE_TEST_IN, self.TYPE_TEST_OUT]
        
        self.TASK2NCLASS = {'Task1':(2,1), 'Task2':(3,1), 'Task3':(1,4)}
        
        self.MODEL2NAME = {'bert_base':'bert-base-cased', 'bert_large':'bert-large-cased',
                     'xlnet_base':'xlnet-base-cased', 'xlnet_large':'xlnet-large-cased',
                     'electra_base':'google/electra-base-discriminator', 'electra_large':'google/electra-large-discriminator',
                     'roberta_base':'roberta-base', 'roberta_large':'roberta-large',
                     'deberta_base':'microsoft/deberta-base', 'deberta_large':'microsoft/deberta-large'}
        
        self.HP_LOSS_FL = (0.2, 5, 3)
        self.HP_LOSS_LS = 0.1
        self.HP_LOSS_MM = 1
        
        self.HP_NUM_DROPOUT_MC = 10
        self.HP_NUM_ENSEMBLE = 5
        self.HP_NUM_SVI_MC = 50
        
        self.HP_WARMUP_RATIO = 0.1
        self.HP_MAX_GRAD_NORM = 1.0
        self.HP_BATCH_SIZE = 16
        self.HP_NUM_EPOCH = 5
        self.HP_MODEL2LR = {'bert_base':2e-5, 'xlnet_base':2e-5, 'electra_base':2e-5, 'roberta_base':2e-5, 'deberta_base':2e-5,
                     'bert_large':5e-6, 'xlnet_large':5e-6, 'electra_large':5e-6, 'roberta_large':5e-6, 'deberta_large':5e-6}
        
        self.DIR_CURR = os.getcwd()
        self.DIR_DATA = os.path.join(self.DIR_CURR, '../Data')
        self.DIR_RESULT = os.path.join(self.DIR_CURR, '../Result')
        
    
    def individual(self, args):
        
        self.DIR_ORI = os.path.join(self.DIR_DATA, args.task_id, 'Original')
        self.FILE_ORI_TRAIN = os.path.join(self.DIR_ORI, f'{self.TYPE_TRAIN}.pkl')
        self.FILE_ORI_DEV = os.path.join(self.DIR_ORI, f'{self.TYPE_DEV}.pkl')
        self.FILE_ORI_TEST_IN = os.path.join(self.DIR_ORI, f'{self.TYPE_TEST_IN}.pkl')
        self.FILE_ORI_TEST_OUT = os.path.join(self.DIR_ORI, f'{self.TYPE_TEST_OUT}.pkl')
        
        self.DIR_INPUT = os.path.join(self.DIR_DATA, args.task_id, args.model_name)
        Path(self.DIR_INPUT).mkdir(parents=True, exist_ok=True)
        self.FILE_INPUT_TRAIN = os.path.join(self.DIR_INPUT, f'{self.TYPE_TRAIN}.pkl')
        self.FILE_INPUT_DEV = os.path.join(self.DIR_INPUT, f'{self.TYPE_DEV}.pkl')
        self.FILE_INPUT_TEST_IN = os.path.join(self.DIR_INPUT, f'{self.TYPE_TEST_IN}.pkl')
        self.FILE_INPUT_TEST_OUT = os.path.join(self.DIR_INPUT, f'{self.TYPE_TEST_OUT}.pkl')
        
        self.DIR_OUTPUT = os.path.join(self.DIR_RESULT, args.task_id, args.model_name)
        Path(self.DIR_OUTPUT).mkdir(parents=True, exist_ok=True)
        
        if args.stage == self.STAGE_TRAIN:
            self.FILE_STDOUT = os.path.join(self.DIR_OUTPUT, f'stdout_{args.stage}_{args.version}_{args.seed}.txt')
            self.FILE_MODEL = os.path.join(self.DIR_OUTPUT, f'model_{args.version}_{args.seed}.pkl')
            self.VERSION_MODE = args.version
        elif args.stage == self.STAGE_TEST:
            self.FILE_STDOUT = os.path.join(self.DIR_OUTPUT, f'stdout_{args.stage}.txt')
            self.FILE_SCORE = os.path.join(self.DIR_OUTPUT, f'result_score.pkl')
            self.FILE_PROB = os.path.join(self.DIR_OUTPUT, f'result_prob.pkl')
            
        self.FILE_MODELS = defaultdict(list)
        for file in os.listdir(self.DIR_OUTPUT):
            if file.startswith('model'): self.FILE_MODELS[file.split('_')[1]].append(os.path.join(self.DIR_OUTPUT, file))
        
        model_name, self.LOSS_MODE = args.model_name.split('-')
        self.PLM_NAME = self.MODEL2NAME[model_name]
        self.HP_LR = self.HP_MODEL2LR[model_name]
        self.NUM_CLASS = self.TASK2NCLASS[args.task_id]