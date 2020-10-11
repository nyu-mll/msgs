# The MSGS datasets

This directory contains the MSGS datasets.

### MSGS

The file ```msgs.zip```  contains datasets for 29 binary classification tasks, of which 20 are ambiguous tasks 9 are unambiguous control tasks.

(Note: This download doesn't include training sets that contain inoculation data. You can still create these inoculated datasets, or you can use access [this page](https://drive.google.com/file/d/1-B5L_-5AfssDfV67zUKsGhFTYJ9ECZkD/view?usp=sharing) to get the original inoculated data from the paper.)

#### Schematic examples

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

#### Creating paradigms

Paradigm example(Syn. position X Lexical content):

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

For each mixed task we create 5k paradigms of data. We use `Train` data for finetuning and `Test` data to test model generalization. We also add different proportions (0.1%, 0.3%, 1.0%) of `Inoc.` data to training data.

#### File structure

```
msgs
└─── [feature]_control/  (unambiguous control task)
|   |   train.jsonl
|   |   test.jsonl
└─── [linguistic_feature]_[surface_feature]/  (ambiguous task)
|   |   train.jsonl
|   |   test.jsonl
|   |   inoculating.jsonl
```
