# Dataset location

Put your existing dataset here:

```text
data/
├── train/
│   ├── overripe/
│   ├── ripe/
│   ├── rotten/
│   └── unripe/
└── test/
    ├── overripe/
    ├── ripe/
    ├── rotten/
    └── unripe/
```

Alternatively, set `DATASET_ROOT` in `.env` to an existing dataset directory.
The repository's sibling dataset can be used with:

```text
DATASET_ROOT=../banana_classification
```

That dataset already contains `train/`, `valid/`, and `test/` folders. If no
`valid/` or `val/` folder is present, the project creates a validation split
from `train/`.

Do not commit the actual image dataset to Git unless you have permission to redistribute it.
