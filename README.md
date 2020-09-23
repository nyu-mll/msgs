# Learning What Features Matter: RoBERTa acquires helpful inductive biases from pretraining
The repository contains data and links to model implementations and training/test code for the paper [Learning What Features Matter: RoBERTa acquires helpful inductive biases from pretraining](tbd). The paper investigates how increases in pretraining data alters the inductive biases of RoBERTa when generalizing on downstream tasks. We pretrain models on 4 successively larger datasets, then test them on a synthetic dataset named Mixed Signals Generalization Set (MSGS).

Sections

- [Pretraining RoBERTa](https://github.com/nyu-mll/RoBERTa-scale-down#pretraining-roberta)
- [Testing Inductive Biases](https://github.com/nyu-mll/RoBERTa-scale-down#testing-inductive-biases-with-msgs)
- [Model and Finetuning Tutorials](https://github.com/nyu-mll/RoBERTa-scale-down#model-and-finetuning-tutorials)

### Pretraining RoBERTa

We pretrain RoBERTa on smaller datasets(1M, 10M, 100M, 1B tokens). The pretraining data reproduces that of BERT: We combine English Wikipedia and a reproduction of BookCorpus using texts from smashwords in a ratio of approximately 3:1. We release 3 models with lowest perplexities for each pretraining data size out of 25 runs (or 10 in the case of 1B tokens).

Pretraining:

| model | data | model size | max steps | batch size | val. ppl. |
|-|-|-|-|-|-|
| [roberta-base-1B-1][link-roberta-base-1B-1] | 1B | BASE | 100K | 512 | 3.93 |
| [roberta-base-1B-2][link-roberta-base-1B-2] | 1B | BASE | 31K | 1024 | 4.25 |
| [roberta-base-1B-3][link-roberta-base-1B-3] | 1B | BASE | 31K | 4096 | 3.84 |
| [roberta-base-100M-1][link-roberta-base-100M-1] | 100M | BASE | 100K | 512 | 4.99 |
| [roberta-base-100M-2][link-roberta-base-100M-2] | 100M | BASE | 31K | 1024 | 4.61 |
| [roberta-base-100M-3][link-roberta-base-100M-3] | 100M | BASE | 31K | 512 | 5.02 |
| [roberta-base-10M-1][link-roberta-base-10M-1] | 10M | BASE | 10K | 1024 | 11.31 |
| [roberta-base-10M-2][link-roberta-base-10M-2] | 10M | BASE | 10K | 512 | 10.78 |
| [roberta-base-10M-3][link-roberta-base-10M-3] | 10M | BASE | 31K | 512 | 11.58 |
| [roberta-med-small-1M-1][link-roberta-med-small-1M-1] | 1M | MED-SMALL | 100K | 512 | 153.38 |
| [roberta-med-small-1M-2][link-roberta-med-small-1M-2] | 1M | MED-SMALL | 10K | 512 | 134.18 |
| [roberta-med-small-1M-3][link-roberta-med-small-1M-3] | 1M | MED-SMALL | 31K | 512 | 139.39 |

Details on model sizes (see the paper for a full discussion of how we tune this hyperparameter):

| Model Size | L  | AH | HS  | FFN  | P    |
|------------|----|----|-----|------|------|
| BASE       | 12 | 12 | 768 | 3072 | 125M |
| MED-SMALL  | 6  | 8  | 512 | 2048 | 45M  |

(AH = number of attention heads; HS = hidden size; FFN = feedforward network dimension; P = number of parameters.)

For other hyperparameters, we select:
- Peak Learning rate: 5e-4
- Warmup Steps: 6% of max #steps
- Dropout: 0.1

#### Analysis on Pretrained models

We have also analyzed our pretrained models with edge probing ([Tenney et al., 2019](https://arxiv.org/pdf/1905.06316.pdf)). For edge probing details you can go to our post on CILVR blog: [The MiniBERTas: Testing what RoBERTa learns with varying amounts of pretraining](https://wp.nyu.edu/cilvr/2020/07/02/the-minibertas-testing-what-roberta-learns-with-varying-amounts-of-pretraining/).

### Testing Inductive Biases with MSGS

The MSGS dataset includes data for 29 binary classification tasks to test models' inductive biases. MSGS contains data for 20 ambiguous tasks obtained by combining one of 4 linguistic features with one of 5 surface features. We also provide data for 9 unambiguous control tasks for each of the 9 features.

#### MSGS

More details are provided in the [data page](https://github.com/nyu-mll/RoBERTa-scale-down/blob/master/data).

### Model and Finetuning Tutorials

#### Use Pretrained Models

The RoBERTa models pretrained on smaller datasets ("MiniBERTas") are available through [Hugging Face](https://huggingface.co/nyu-mll).

You can load models as in the following example:

```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-100M-1")
model = AutoModel.from_pretrained("nyu-mll/roberta-base-100M-1")
```

#### Finetuning on MSGS

You can use our [fork](https://github.com/leehaausing/transformers) of [transformers](https://github.com/huggingface/transformers) and run `./examples/run_msgs.py` in `inductive_bias` branch. An example is presented below:

```
python ./examples/run_msgs.py \
    --model_type roberta \
    --model_name_or_path #path/to/your/model \
    --task_name #task_name \
    --do_train \
    --do_eval \
    --data_dir #path/to/data \
    --max_seq_length #max_seq_len \
    --per_gpu_eval_batch_size #batch_size \
    --per_gpu_train_batch_size #batch_size \
    --learning_rate #lr \
    --num_train_epochs #num_epochs  \
    --weight_decay #weight_decay \
    --warmup_steps #warmup_steps \
    --logging_steps #logging_steps \
    --save_steps #save_steps \
    --output_dir #path/to/save/finetuned/models
```
**define model_name**
- You can check out our [model page](https://huggingface.co/nyu-mll). You can either use the names specified on Transformers or download all files and specify the local directory.

**define task_name**
- control task: `[feature]_control`.
- mixed binary classification task: `[linguistic_feature]_[surface_feature]_[inoculate%]`

`[inoculate%]`: you can choose one from `namb`, `001`, `003`, and `01`. `namb` means no inoculation.

`[linguistic_feature]`&`[surface_feature]`: you need to change intended feature types to their corresponding names presented below:

| feature type | corresponding name |
|-|-|
| Absolute position (surface) | absolute_token_position |
| Length (surface) | length |
| Lexical content (surface) | lexical_content_the |
| Relative position (surface) | relative_token_position |
| Orthography (surface) | title_case |
| Morphology (linguistic) | irregular_form |
| Syn. category (linguistic) | syntactic_category |
| Syn. construction (linguistic) | control_rasing|
| Syn. position (linguistic) | main_verb |

### Citing

TBD.

[link-roberta-med-small-1M-1]: https://huggingface.co/nyu-mll/roberta-med-small-1M-1
[link-roberta-med-small-1M-2]: https://huggingface.co/nyu-mll/roberta-med-small-1M-2
[link-roberta-med-small-1M-3]: https://huggingface.co/nyu-mll/roberta-med-small-1M-3
[link-roberta-base-10M-1]: https://huggingface.co/nyu-mll/roberta-base-10M-1
[link-roberta-base-10M-2]: https://huggingface.co/nyu-mll/roberta-base-10M-2
[link-roberta-base-10M-3]: https://huggingface.co/nyu-mll/roberta-base-10M-3
[link-roberta-base-100M-1]: https://huggingface.co/nyu-mll/roberta-base-100M-1
[link-roberta-base-100M-2]: https://huggingface.co/nyu-mll/roberta-base-100M-2
[link-roberta-base-100M-3]: https://huggingface.co/nyu-mll/roberta-base-100M-3
[link-roberta-base-1B-1]: https://huggingface.co/nyu-mll/roberta-base-1B-1
[link-roberta-base-1B-2]: https://huggingface.co/nyu-mll/roberta-base-1B-2
[link-roberta-base-1B-3]: https://huggingface.co/nyu-mll/roberta-base-1B-3
