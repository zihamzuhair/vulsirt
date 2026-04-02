# Cross-Language Vulnerability Detection Using Source-IR Dual-View Transformers

This is a simple Python and PyTorch project for binary vulnerability detection.
It implements four CodeBERT-based baselines:

- B1: Source-code-only CodeBERT
- B2: LLVM-IR-only CodeBERT
- B3: Source and LLVM-IR concatenation
- B4: Source and LLVM-IR adaptive gated fusion

B4 is the proposed model. It uses separate CodeBERT encoders for source code and LLVM IR, projects both views into the same latent dimension, learns a simple sigmoid gate, fuses the views, and predicts:

- `0 = non-vulnerable`
- `1 = vulnerable`

## Install

```bash
pip install -r requirements.txt
```

## Dataset

Place the raw dataset at:

```text
data/raw/dataset.jsonl
```

You can change this path in `config.yaml`.

The preprocessed dataset is saved to:

```text
data/processed/dataset.jsonl
```

Each processed record has this format:

```json
{
  "sample_id": "1",
  "source_code": "source code here",
  "llvm_ir": "LLVM IR here",
  "label": 1,
  "language": "c",
  "split": "train"
}
```

If LLVM IR is missing and the sample is C or C++, `preprocess.py` tries to generate it with Clang. If generation fails, the sample is skipped and the error is logged.

## Preprocess

```bash
python preprocess.py
```

## Train

```bash
python train.py --baseline b1
python train.py --baseline b2
python train.py --baseline b3
python train.py --baseline b4
```

Training uses:

- AdamW
- BCEWithLogitsLoss
- sigmoid probabilities
- gradient clipping
- tqdm progress bars

## Checkpoints

Checkpoints are saved in the directory configured by `paths.checkpoints`.

For each baseline:

```text
checkpoints/b1_best.pt
checkpoints/b1_last.pt
checkpoints/b2_best.pt
checkpoints/b2_last.pt
checkpoints/b3_best.pt
checkpoints/b3_last.pt
checkpoints/b4_best.pt
checkpoints/b4_last.pt
```

`last.pt` is saved after every epoch.
`best.pt` is saved when validation F1-score improves.

If `training.resume` is `true`, training resumes from `last.pt` when it exists.

## Evaluate

Evaluate all available best checkpoints:

```bash
python evaluate.py
```

Evaluate one baseline:

```bash
python evaluate.py --baseline b4
```

Evaluation saves:

```text
results/b1_metrics.json
results/b2_metrics.json
results/b3_metrics.json
results/b4_metrics.json
results/baseline_comparison.csv
```

Predictions are also saved as CSV files in `results/`.

Metrics include:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- false-positive rate

## Scanner

Scan one source-code file with a trained model:

```bash
python scanner.py --baseline b4 --source-file sample.c
```

Example output:

```text
Baseline: B4
Prediction: Vulnerable
Probability: 0.82
Source weight: 0.63
LLVM-IR weight: 0.37
```

For B1, only source code is used.
For B2, B3, and B4, LLVM IR is generated from the source file using Clang.

## Configuration

All important paths and training parameters are in `config.yaml`.
Change the dataset path, model name, sequence lengths, batch size, epochs, learning rate, dropout, latent dimension, random seed, selected baseline, threshold, and resume behavior there.
