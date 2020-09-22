# The MSGS datasets

This directory contains descriptions and tutorials on MSGS datasets.

### MSGS

The compressed file contains 29 tasks, of which 20 are binary classification of mixed signals and 9 are control tasks where signal only comes from one feature, linguistic or surface.

feature type and corresponding names:

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

The file structure looks like the following:
```
msgs
└─── [feature]_control/  (control task)
|   |   train.jsonl
|   |   test.jsonl
└─── [linguistic_feature]_[surface_feature]/  (mixed classification)
|   |   train.jsonl
|   |   test.jsonl
|   |   inoculating.jsonl
```

### Tutorials on Defining tasks

To define a **control task**, at the `task_name` argument you input `[feature]_control`.

To define a **mixed binary classification task**, at the `task_name` argument you input `[linguistic_feature]_[surface_feature]_[inoculating_percent]`, in which in `[inoculating_percent]` you can choose one from `namb`, `001`, `003`, `01`(`namb` means no inoculation).
