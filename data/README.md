# The MSGS datasets

This directory contains the MSGS datasets.

### MSGS

The file ```msgs.zip```  contains datasets for 29 binary classification tasks, of which 20 are ambiguous tasks 9 are unambiguous control tasks.

## Feature type and corresponding names:

| Feature type | Corresponding Name |
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

The file structure is as follows:
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

### Tutorials on Defining tasks

To define a **control task**, at the `task_name` argument you input `[feature]_control`.

To define a **mixed binary classification task**, at the `task_name` argument you input `[linguistic_feature]_[surface_feature]_[inoculating_percent]`, in which in `[inoculating_percent]` you can choose one from `namb`, `001`, `003`, `01`(`namb` means no inoculation).
