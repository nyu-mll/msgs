# Recreation of scale-down RoBERTa and Finetuning
The repository contains of descriptions and implementations on RoBERTa models trained on scale-down pretraining data (1M, 10M, 100M, 1B) and Finetuning tasks(Edge Probing, Testing Inductive Biases).

### Model descriptions

#### Pretrained Models

We pretrain RoBERTa on smaller datasets (1M, 10M, 100M, 1B tokens). The pretraining data reproduces that of BERT: We combine English Wikipedia and a reproduction of BookCorpus using texts from smashwords in a ratio of approximately 3:1.

We release 3 models with lowest perplexities for each pretraining data size out of 25 runs (or 10 in the case of 1B tokens).

Hyperparameters and validation perplexities of pretrained data are published in Transformers Library. (Click [here](https://huggingface.co/nyu-mll) for more detailed model descriptions)

#### Finetuning

##### Probing Tasks(TBD)

##### Testing Inductive Biases(TBD)

### How to Load Pretrained Models(TBD)

### How to Implement Pretraining(TBD)

### How to Perform Edge Probing(TBD)

### How to Test Inductive Biases(TBD)
