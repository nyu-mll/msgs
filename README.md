# Learning What Matters: RoBERTa acquires helpful inductive biases from pretraining
The repo contains descriptions and implementations of the paper [Learning What Features Matter: RoBERTa acquires helpful inductive biases from pretraining](tbd). We investigate how robust generalization is acquired by RoBERTa models with different pretraining data sizes. We pretrain models on datasets of 4 pretraining data sizes, then test them on a synthetic dataset named Mixed Signals Generalization Set(MSGS).

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

Details on model sizes:

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

We also analyzed our pretrained models with edge probing([Tenney et al., 2019](https://arxiv.org/pdf/1905.06316.pdf)). For edge probing details you can go to our post on CILVR blog: [The MiniBERTas: Testing what RoBERTa learns with varying amounts of pretraining](https://wp.nyu.edu/cilvr/2020/07/02/the-minibertas-testing-what-roberta-learns-with-varying-amounts-of-pretraining/).

### Testing Inductive Biases with MSGS

We create MSGS to test model generalization on inductive biases. MSGS contains 20 mixed binary classification tasks where the signal from each of the 4 linguistic features are mixed with signal from 5 surface features. We also provide 9 control tasks on the 9 features where the signal isn't mixed, to test whether models have learnt these features during pretraining.

#### MSGS

Schematic examples:

(surface features)

| Feature type | Feature description | Positive example | Negative example |
|-|-|-|-|
| Absolute position | Is the first token of S "the"? | The cat chased a mouse. | A cat chased a mouse. |
| Length | Is S longer than *n* (e.g.,~3) words? | The cat chased a mouse. | The cat meowed. |
| Lexical content | Does S contain "the" ? | That cat chased the mouse. | That cat chased a mouse. |
| Relative position | Does "the" precede "a"? | The cat chased a mouse. | A cat chased the mouse. |
| Orthography | Does S appear in title case? | The Cat Chased a Mouse. | The cat chased a mouse. |

(linguistic features)

| Feature type | Feature description | Positive example | Negative example |
|-|-|-|-|
| Morphology | Does S have an irregular past verb? | The cats slept. | The cats meow. |
| Syn. category | Does S have an adjective? | Lincoln was tall. | Lincoln was president. |
| Syn. construction | Is S the control construction? | Sue is eager to sleep. | Sue is likely to sleep. |
| Syn. position | Is the main verb in "ing" form? | Cats who eat mice are purring. | Cats who are eating mice purr. |

For each mixed task we create 5k paradigms of data. paradigm example(Syn. position X Lexical content):

| Domain | Split | **L**<sup>L</sup> | **L**<sup>S</sup> | Sentence |
|-|-|-|-|-|
| In | Train | 1 | 1 | These men weren't hating that this person who sang tunes destroyed the vase. |
| In | Train | 0 | 0 | These men hated that this person who sang tunes was destroying some vase. |
| In | Inoc. | 1 | 0 | These men weren't hating that this person who sang tunes destroyed some vase. |
| In | Inoc. | 0 | 1 | These men hated that this person who sang tunes was destroying the vase. |
| Out | Test | 1 | 0 | These reports that all students built that school were impressing some children. |
| Out | Test | 0 | 1 | These reports that all students were building the school had impressed some children. |
| Out | Aux. | 1 | 1 | These reports that all students built the school were impressing some children. |
| Out | Aux. | 0 | 0 | These reports that all students were building that school had impressed some children. |

(**L** <sup>**L**</sup> and **L** <sup>**S**</sup> mark the presence of the linguistic feature and surface feature respectively.)

We use `Train` data for finetuning and `Test` data to test model generalization. We also add different proportions(0.1%, 0.3%, 1.0%) of `Inoc.` data to training data.

More details are provided in the [data page](tbd).

### Model and Finetuning Tutorials

#### How to Use Pretrained Models

Load models with codes as in the example:

```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-100M-1")
model = AutoModel.from_pretrained("nyu-mll/roberta-base-100M-1")
```

#### Instructions on finetuning on MSGS

You can use our [fork](https://github.com/leehaausing/transformers) of [transformers](https://github.com/huggingface/transformers) and run `./examples/run_msgs.py` in `inductive_bias` branch.

Sample script:

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

For model names you can check out our [model page](https://huggingface.co/nyu-mll) or download all files and specify the local directory. For task names please go to our [data page](tbd).

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
