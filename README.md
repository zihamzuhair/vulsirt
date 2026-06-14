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

Place PrimeVul and Rust data at the paths configured in `configs/config.yaml`:

```text
data/raw/primevul
data/raw/rust/rustsec_osv_dataset.jsonl
```

PrimeVul should contain:

```text
primevul_train.jsonl
primevul_valid.jsonl
primevul_test.jsonl
```

You can change these folders/files in `configs/config.yaml`.

The preprocessed datasets are saved to:

```text
data/processed/primevul_dataset.jsonl
data/processed/rust_dataset.jsonl
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

Missing or failed LLVM IR handling is controlled in `configs/config.yaml`:

- `ir_failure: empty` keeps the sample but uses an empty IR placeholder.
- `ir_failure: skip` drops samples that do not have usable LLVM IR.
- `ir_failure: source` uses source code as the IR fallback, but this is not recommended when you want a clean Source/IR comparison.
- `generate_missing_ir: true` tries to generate IR with Clang or Rust before applying the fallback.

When LLVM IR generation is attempted and fails, preprocessing writes details to the configured `paths.llvm_errors` file. For the 100-sample run this is:

```text
logs/100_samples/100_samples_failures.jsonl
```

Each JSONL row includes the sample id, dataset, language, split, compiler, command, return code, stderr/stdout, source length, and a source preview. This file can be used later to improve the compiler/wrapper step and retry failed samples.

## Preprocess

```bash
python preprocess.py
```

The default config preprocesses both datasets when the raw files are present:

- PrimeVul C/C++ records are saved to `data/processed/primevul_dataset.jsonl`
- Rust records are saved to `data/processed/rust_dataset.jsonl`

On PowerShell:

```powershell
.\scripts\preprocess_all.ps1
```

## Train

```bash
python train.py --baseline b1
python train.py --baseline b2
python train.py --baseline b3
python train.py --baseline b4
```

On Windows PowerShell, you can also use the helper scripts:

```powershell
.\scripts\train_b1.ps1
.\scripts\train_b2.ps1
.\scripts\train_b3.ps1
.\scripts\train_b4.ps1
.\scripts\train_all.ps1
```

To train with another config file, pass it as the first argument:

```powershell
.\scripts\train_b4.ps1 test/test_config.yaml
```

For a small real-dataset run that inherits the base config, uses a balanced
1000-record subset, creates 80/10/10 train/validation/test splits, and writes
checkpoints to a separate folder:

```powershell
python preprocess.py --config configs/100_samples.yaml
python train.py --config configs/100_samples.yaml
python evaluate.py --config configs/100_samples.yaml --baseline b4
```

To test preprocessing and dataset loading with the tiny test dataset:

```powershell
.\scripts\test_data_pipeline.ps1
```

To inspect the B4 encoder vectors, shared latent projections, gate, fused vector, and output:

```powershell
.\scripts\inspect_b4_vectors.ps1 test/test_config.yaml 0
```

To run the complete B4 adaptive gated fusion smoke test and print its metrics:

```powershell
.\scripts\test_gated_fusion.ps1
```

For the main CodeBERT config after preprocessing and B4 training:

```powershell
python inspect_b4_vectors.py --config configs/config.yaml --sample-index 0
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

By default, evaluation saves both in-domain C/C++ metrics and cross-language Rust metrics when both processed datasets exist:

```text
results/b1_metrics.json
results/b1_rust_metrics.json
results/b2_metrics.json
results/b2_rust_metrics.json
results/b3_metrics.json
results/b3_rust_metrics.json
results/b4_metrics.json
results/b4_rust_metrics.json
results/baseline_comparison.csv
```

Evaluate one baseline:

```bash
python evaluate.py --baseline b4
```

PowerShell helper:

```powershell
.\scripts\evaluate_cross_language.ps1
.\scripts\evaluate_cross_language.ps1 configs/config.yaml b4
```

The intended cross-language experiment is:

```powershell
.\scripts\preprocess_all.ps1
.\scripts\train_b1.ps1
.\scripts\train_b2.ps1
.\scripts\train_b3.ps1
.\scripts\train_b4.ps1
.\scripts\evaluate_cross_language.ps1
```

This trains on PrimeVul C/C++ train/validation records and evaluates on both PrimeVul C/C++ test records and Rust test records.

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

All important paths and training parameters are in `configs/config.yaml`.
Change the dataset path, model name, sequence lengths, batch size, epochs, learning rate, dropout, latent dimension, random seed, selected baseline, threshold, and resume behavior there.

Config files can inherit from another config with `inherits` or `extends`.
For example, `configs/100_samples.yaml` inherits `configs/config.yaml` and overrides only the small-run settings.

Sampling, label balancing, and generated train/validation/test splitting live in `dataset.py`.
For `configs/100_samples.yaml`, PrimeVul is balanced to 50 vulnerable and 50 non-vulnerable records, then split into 40/40 train, 5/5 validation, and 5/5 test. Rust is kept as a separate 50/50 evaluation set.

Useful sections:

- `paths.checkpoints`: where `b1_best.pt`, `b4_last.pt`, and other checkpoint files are created.
- `training.baseline`: default baseline to run when `--baseline` is not passed.
- `training.class_weights`: per-class BCE loss weights for non-vulnerable and vulnerable samples.
- `model.projection.latent_dimension`: source/IR projection dimension used by B3 and B4.
- `model.gating`: B4 gate mode, fixed alpha, initial bias, and temperature.
- `data.max_records`: cap the run to a smaller subset, such as 1000 records.
- `data.balance`: choose how many vulnerable and non-vulnerable records to use.
- `data.split.mode`: use `record` to trust dataset split labels, or `config` to generate deterministic ratio splits from the selected subset.
- `data.split.ratios`: train/validation/test ratios such as 0.8/0.1/0.1.
