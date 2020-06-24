# Testing Inductive Biases and Recreating of scale-down RoBERTa
The repository contains of descriptions and implementations on RoBERTa models trained on scale-down pretraining data (1M, 10M, 100M, 1B) and Finetuning tasks(Edge Probing, Testing Inductive Biases).

### Model descriptions

#### Pretraining Models

We pretrain RoBERTa on smaller datasets (1M, 10M, 100M, 1B tokens). The pretraining data reproduces that of BERT: We combine English Wikipedia and a reproduction of BookCorpus using texts from smashwords in a ratio of approximately 3:1.

We release 3 models with lowest perplexities for each pretraining data size out of 25 runs (or 10 in the case of 1B tokens).

Hyperparameters and validation perplexities of pretrained data are published in Transformers Library. (Click [here](https://huggingface.co/nyu-mll) for more detailed model descriptions)

#### Finetuning

##### Probing Tasks(TBD)

##### Testing Inductive Biases

We create Mixed Signals Generalization Set(MSGS) with 20 sub-datasets to test inductive biases of 4 linguistic features and 5 surface features. There are 20 sets with mixed data (as the cartesian product of linguistic and surface features, i.e. 4*5) and 9 control datasets testing learnability of 9 features respectively.

The code for testing inductive biases is provided [here(TBD)]().

### How to Load Pretrained Models(TBD)

There are two ways of loading the model.

You can load models using code similar to the example provided below:

```python
tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-100M-1")
model = AutoModel.from_pretrained("nyu-mll/roberta-base-100M-1")
```

Another way is to download [published models](https://huggingface.co/nyu-mll) and load them locally with codes like this:

```python
tokenizer = RoBERTaTokenizer.from_pretrained("nyu-mll/roberta-base-100M-1")
model = RoBERTaModel.from_pretrained("nyu-mll/roberta-base-100M-1")
```

(Note that you can replace `RoBERTaModel` with `RobertaForMaskedLM`, `RobertaForSequenceClassification`, etc., depending on task types you intend to finetune. For more instructions please visit [here](https://huggingface.co/transformers/model_doc/roberta.html#).)

### How to Implement Pretraining by Yourself(TBD)

Pretraining data is provided [here(TBD)]().

### How to Perform Edge Probing(TBD)

### How to Test Inductive Biases(TBD)

Steps for finetuning on inductive biases:

- Clone the repo branch provided [here].
- Ask for created MSGS. (Will be added)
- Run `./examples/run_glue.py` with specified model, task name, and data directory (example command TBD).
