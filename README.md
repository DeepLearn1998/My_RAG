# Supervise-finetuning your embedding model

## Quick start
### Step 1. download model
```shell
python model_download.py
```
### Step 2. load/generate datasets
train.py line 47 (default use: https://huggingface.co/datasets/sentence-transformers/all-nli)
```shell
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
```
### Step 3. finetune the pre-train model
```shell
python train.py
```
### Step 4. test the finetuned model
don't forget to replace the `model_path` with your finetuned model path (test.py line 12)
```shell
python test.py
```
## References:
> https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/matryoshka/matryoshka_nli.py
