# UQ-PLM

Code for <a href="https://arxiv.org/abs/2210.04714">Uncertainty Quantification with Pre-trained Language Models: An Empirical Analysis</a> (EMNLP 2022 Findings).

## Requirements

```
PyTorch = 1.10.1
Bayesian-Torch = 0.1
HuggingFace Transformers = 4.11.1
```

## Data

Our empirical analysis consists of the following three NLP (natural language processing) classification tasks:

**task_id** | Task | In-Domain Dataset | Out-Of-Domain Dataset
--- | --- | --- | ---
**Task1** | Sentiment Analysis | IMDb | Yelp
**Task2** | Natural Language Inference | MNLI | SNLI
**Task3** | Commonsense Reasoning | SWAG | HellaSWAG

You can download our input data <a href="https://drive.google.com/file/d/188kpyh0jxcygijBMguAK99omNacaFnBf/view?usp=sharing">here</a> and unzip it to the current directory.

Then the corresponding data splits of each task are stored in **Data/{task_id}/Original**:
- **train.pkl**, **dev.pkl**, and **test_in.pkl** come from the in-domain dataset.
- **test_out.pkl** comes from the out-of-domain dataset. 

## Run

Specify the targeting `model_name` and `task_id` in **Code/run.sh**:
- `model_name` is specified in the format of `{PLM}_{size}-{loss}`.
    - `{PLM}` (Pre-trained Language Model) can be chosen from `bert`, `xlnet`, `electra`, `roberta`, and `deberta`.
    - `{size}` can be chosen from `base` and `large`.
    - `{loss}` can be chosen from `be` (Brier loss), `fl` (focal loss), `ce` (cross-entropy), `ls` (label smoothing), and `mm` (max mean calibration error).
- `task_id` can be chosen from `Task1` (Sentiment Analysis), `Task2` (Natural Language Inference), and `Task3` (Commonsense Reasoning).

Other hyperparameters are defined in **Code/info.py** (e.g., learning rate, batch size, and training epoch).

Use the command `bash Code/run.sh` to run one sweep of experiments:
1. Transform the original data input in **Data/{task_id}/Original** to the model-specific data input in **Data/{task_id}/{model_name}**.
1. Train six deterministic (version=`det`) PLM-based pipelines (used for `Vanilla`, `Temp Scaling` (temperature scaling), `MC Dropout` (monte-carlo dropout), and `Ensemble`) stored in **Result/{task_id}/{model_name}**.
1. Train six stochastic (version=`sto`) PLM-based pipelines (used for `LL SVI` (last-layer stochastic variational inference)) stored in **Result/{task_id}/{model_name}**.
1. Test the above pipelines with five kinds of uncertainty quantifiers (`Vanilla`, `Temp Scaling`, `MC Dropout`, `Ensemble`, and `LL SVI`) under two domain settings (`test_in` and `test_out`) based on four metrics (`ERR` (prediction error), `ECE` (expected calibration error), `RPP` (reversed pair proportion), and `FAR95` (false alarm rate at 95% recall)). 
    1. The evaluation of each (uncertainty quantifier, domain setting, metric) combination consists of six trials, and the results are stored in **Result/{task_id}/{model_name}/result_score.pkl**. 
    1. The ground truth labels and raw probability outputs are stored in **Result/{task_id}/{model_name}/result_prob.pkl**.
1. All the training and testing stdouts are stored in **Result/{task_id}/{model_name}/**.

## Result

We store our empirical observations in **results.pkl**. You can download this dictionary <a href="https://drive.google.com/file/d/1agT8NwWZP0RohoVKX31Lq6aiQAL9wCxk/view?usp=sharing">here</a>. 
- The key is in the format of `({task}, {model}, {quantifier}, {domain}, {metric})`.
    - `{task}` can be chosen from `Sentiment Analysis`, `Natural Language Inference`, and `Commonsense Reasoning`.
    - `{model}` can be chosen from `bert_base-br`, `bert_base-ce`, `bert_base-fl`, `bert_base-ls`, `bert_base-mm`, `bert_large-ce`, `deberta_base-ce`, `deberta_large-ce`, `electra_base-ce`, `electra_large-ce`, `roberta_base-ce`, `roberta_large-ce`, `xlnet_base-ce`, and `xlnet_large-ce`.
    - `{quantifier}` can be chosen from `Vanilla`, `Temp Scaling`, `MC Dropout`, `Ensemble`, and `LL SVI`.
    - `{domain}` can be chosen from `test_in` and `test_out`.
    - `{metric}` can be chosen from `ERR`, `ECE`, `RPP`, and `FAR95`. Note that `FAR95` only works with the domain setting of `test_out`.
- The value is in the format of `(mean, standard error)`, which are calculated based on six trials with different seeds.

## Citation

```
@inproceedings{xiao2022uncertainty,
  title={Uncertainty Quantification with Pre-trained Language Models: An Empirical Analysis},
  author={Xiao, Yuxin and Liang, Paul Pu and Bhatt, Umang and Neiswanger, Willie and Salakhutdinov, Ruslan and Morency, Louis-Philippe},
  booktitle={Findings of EMNLP},
  year={2022}
}
```
