# Data

This repository does not redistribute source EHR data. Users are expected to obtain access from the official providers and prepare local files that match the formats expected by the training and inference scripts.

## External Access

### MIMIC-III

- Data access: https://physionet.org/content/mimiciii/1.4/
- Documentation: https://www.nature.com/articles/sdata201635

### MIMIC-IV

- Data access: https://physionet.org/content/mimiciv/3.1/
- Documentation: https://www.nature.com/articles/s41597-022-01899-x

## Required Local Artifacts

The codebase expects three local data artifacts.

### 1. Hugging Face Dataset Directory

Used by:

- `scripts/train_stage1_base.py`
- `scripts/generate.py`

This path is passed through `--dataset-dir` and must be loadable by:

```python
from datasets import load_from_disk
ds = load_from_disk("/path/to/hf_dataset")
```

The directory should contain the splits needed by your experiment, typically including `train`, `val`, and `test`.

### 2. Raw Trajectory JSONL Files

Used by:

- `scripts/train_stage2_tscm.py`
- `scripts/train_stage3_tram.py`

These are passed through `--train-jsonl` and `--test-jsonl`. Each line should be a single patient trajectory with the following high-level structure:

```json
{
  "patient": {
    "age": 63,
    "sex": "M",
    "race": "WHITE"
  },
  "visits": [
    {
      "type": "EMERGENCY",
      "delta_days": 0,
      "diagnosis_ccs": ["108"],
      "procedure_ccs": ["47"],
      "medication_ingredients": ["ASPIRIN"],
      "lab_categories": ["BLOOD"]
    },
    {
      "type": "URGENT",
      "delta_days": 14,
      "diagnosis_ccs": ["108"],
      "procedure_ccs": [],
      "medication_ingredients": ["ASPIRIN"],
      "lab_categories": ["BLOOD"]
    }
  ]
}
```

The temporal training code assumes:

- each record contains at least two visits
- `patient.age`, `patient.sex`, and `patient.race` are present
- each visit contains `type`
- each non-initial target visit contains `delta_days`
- code fields follow the names used above

### 3. Code Vocabulary Directory

Used by:

- `scripts/train_stage2_tscm.py`
- `scripts/train_stage3_tram.py`
- `scripts/generate.py` when adapter-based generation is enabled

This directory is passed through `--stage1-data-dir` and must contain:

```text
code_vocabs.json
```

The vocabulary file is expected to provide the code mappings used by the encoder, including:

- `dx`
- `proc`
- `med`
- `lab`

## Notes

- This repository does not redistribute restricted clinical data.
- Users should prepare the required local artifacts before running training or inference.
