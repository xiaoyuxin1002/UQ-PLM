import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import LinearReparameterization


class Classifier(nn.Module):
    
    def __init__(self, info, config):
        super(Classifier, self).__init__()
        
        self.info = info
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else config.hidden_dropout_prob)
        
        if info.VERSION_MODE == info.VERSION_DET:
            self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear2 = nn.Linear(config.hidden_size, config.num_labels)
        elif info.VERSION_MODE == info.VERSION_STO:
            self.linear1 = LinearReparameterization(config.hidden_size, config.hidden_size)
            self.linear2 = LinearReparameterization(config.hidden_size, config.num_labels)
        
    def forward(self, batch_reps):
        
        if self.info.VERSION_MODE == self.info.VERSION_DET:
            batch_kls = 0
            
            batch_logits = self.dropout(batch_reps)
            batch_logits = self.linear1(batch_logits)
            batch_logits = self.activation(batch_logits)
            batch_logits = self.dropout(batch_reps)
            batch_logits = self.linear2(batch_logits)
            
            if self.info.NUM_CLASS[1] != 1: batch_logits = batch_logits.view(-1, self.info.NUM_CLASS[1])
            batch_logits = batch_logits.unsqueeze(0)
            
        elif self.info.VERSION_MODE == self.info.VERSION_STO:
            batch_kls, batch_logits = 0, []
            for _ in range(self.info.HP_NUM_SVI_MC):
                
                round_logits = self.dropout(batch_reps)
                round_logits, round_kls = self.linear1(round_logits)
                batch_kls += round_kls
                round_logits = self.activation(round_logits)
                round_logits = self.dropout(round_logits)
                round_logits, round_kls = self.linear2(round_logits)
                batch_kls += round_kls
                
                if self.info.NUM_CLASS[1] != 1: round_logits = round_logits.view(-1, self.info.NUM_CLASS[1])
                batch_logits.append(round_logits)
            batch_kls /= (self.info.HP_NUM_SVI_MC * batch_reps.shape[0])
            batch_logits = torch.stack(batch_logits)
        
        return batch_kls, batch_logits
    
    
class Loss(nn.Module):
    
    def __init__(self, info):
        super(Loss, self).__init__()
        
        self.info = info
        
    def brier_loss(self, batch_logits, batch_labels):
        
        batch_labels = F.one_hot(batch_labels, self.info.NUM_CLASS[0] * self.info.NUM_CLASS[1])
        batch_loss = (batch_logits.flatten(0,1) - batch_labels.tile((batch_logits.shape[0],1))).square().sum(-1).mean(0)
        return batch_loss
        
    def cross_entropy(self, batch_logits, batch_labels):
        
        batch_loss = F.cross_entropy(batch_logits.flatten(0,1), batch_labels.tile((batch_logits.shape[0],)))
        return batch_loss
    
    def focal_loss(self, batch_logits, batch_labels):
        
        batch_labels = batch_labels.tile((batch_logits.shape[0],)).unsqueeze(-1)
        batch_probs = F.softmax(batch_logits, dim=-1).flatten(0, 1).gather(1, batch_labels).flatten()
        batch_gammas = torch.where(batch_probs<self.info.HP_LOSS_FL[0], self.info.HP_LOSS_FL[1], self.info.HP_LOSS_FL[2])
        batch_loss = - ((1 - batch_probs)**batch_gammas * batch_probs.log()).mean()
        return batch_loss
    
    def label_smoothing(self, batch_logits, batch_labels):
        
        batch_loss = F.cross_entropy(batch_logits.flatten(0,1), batch_labels.tile((batch_logits.shape[0],)), label_smoothing=self.info.HP_LOSS_LS)
        return batch_loss
    
    def max_mean_calibration_error(self, batch_logits, batch_labels):
        
        each_dim0 = lambda each: each.repeat_interleave(each.shape[0])
        each_dim1 = lambda each: each.repeat(each.shape[0])        
        
        batch_loss = []
        for each_logits in batch_logits:
            each_probs, each_preds = F.softmax(each_logits, dim=-1).max(-1)
            each_accs = (each_preds == batch_labels).float()
            each_accs_dim0, each_accs_dim1 = each_dim0(each_accs), each_dim1(each_accs)
            each_probs_dim0, each_probs_dim1 = each_dim0(each_probs), each_dim1(each_probs)
            each_kernels = (- (each_probs_dim0 - each_probs_dim1).abs() / 0.4).exp()
            
            each_ncorrect, each_ntotal = each_accs.sum().item(), each_accs.shape[0]
            each_rcorrect, each_rincorrect = 1/each_ntotal, 1/(each_ncorrect-each_ntotal) if each_ncorrect!=each_ntotal else 0.0
            each_weights_dim0 = (each_accs_dim0 - each_probs_dim0) * torch.where(each_accs_dim0==1.0, each_rcorrect, each_rincorrect)
            each_weights_dim1 = (each_accs_dim1 - each_probs_dim1) * torch.where(each_accs_dim1==1.0, each_rcorrect, each_rincorrect)
            each_loss = (each_kernels * each_weights_dim0 * each_weights_dim1).mean().sqrt()
            batch_loss.append(each_loss)
            
        batch_loss = torch.stack(batch_loss).mean() * self.info.HP_LOSS_MM
        batch_loss += self.cross_entropy(batch_logits, batch_labels)
        return batch_loss
        
    def forward(self, batch_logits, batch_labels):
        
        if self.info.LOSS_MODE == self.info.LOSS_BR: batch_loss = self.brier_loss(batch_logits, batch_labels)
        elif self.info.LOSS_MODE == self.info.LOSS_CE: batch_loss = self.cross_entropy(batch_logits, batch_labels)
        elif self.info.LOSS_MODE == self.info.LOSS_FL: batch_loss = self.focal_loss(batch_logits, batch_labels)
        elif self.info.LOSS_MODE == self.info.LOSS_LS: batch_loss = self.label_smoothing(batch_logits, batch_labels)
        elif self.info.LOSS_MODE == self.info.LOSS_MM: batch_loss = self.max_mean_calibration_error(batch_logits, batch_labels)
        return batch_loss
    
    
class Model(nn.Module):
    
    def __init__(self, info, config, transformer):
        super(Model, self).__init__()

        self.info = info
        
        self.plm_module = transformer
        self.classifier_module = Classifier(info, config)
        self.loss_module = Loss(info)
        
    def forward(self, batch_inputs):
        
        batch_reps = self.plm_module(**batch_inputs)[0][:,0]
        batch_kls, batch_logits = self.classifier_module(batch_reps)
        
        return batch_kls, batch_logits
    
    def learn(self, batch_inputs, batch_labels):
        
        batch_kls, batch_logits = self(batch_inputs)
        batch_loss = self.loss_module(batch_logits, batch_labels)
        batch_loss += batch_kls
        
        return batch_loss
    
    def infer(self, batch_inputs, if_prob=True):
        
        _, batch_logits = self(batch_inputs)
        if if_prob: batch_logits = F.softmax(batch_logits, dim=-1)
        batch_logits = batch_logits.mean(0)
        
        return batch_logits