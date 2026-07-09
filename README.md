# Cross-Language Vulnerability Detection Using Source-IR Dual-View Transformers

This is a Python and PyTorch project for binary vulnerability detection. It trains and evaluates four CodeBERT-based baselines:

- `b1`: source-code-only CodeBERT
- `b2`: LLVM-IR-only CodeBERT
- `b3`: source and LLVM-IR concatenation
- `b4`: source and LLVM-IR adaptive gated fusion

`b4` is the proposed model. It uses separate encoders for source code and LLVM IR, projects both views into one latent space, learns a sigmoid gate, fuses the views, and predicts:

- `0 = non-vulnerable`
- `1 = vulnerable`

## Install

```bash
pip install -r requirements.txt
```

The optional XGBoost ablation needs one extra package:

```powershell
pip install -r ablation/requirements-ablation.txt
```

## Dataset

Place the raw datasets at the paths configured in `configs/config.yaml`:

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

The main generated files are:

```text
data/processed/primevul_dataset.jsonl
data/processed/primevul_dataset_with_llvm.jsonl
data/processed/rustsec_osv_dataset_with_llvm.jsonl
```

Each training row uses this shape:

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

## Data Preparation

Clean PrimeVul C/header records:

```bash
python preprocess.py
```

Compile PrimeVul and Rust records to LLVM IR:

```bash
python compiler.py
```

Compile only one dataset:

```bash
python compiler.py --dataset primevul
python compiler.py --dataset rust
```

Available PowerShell helpers:

```powershell
.\scripts\preprocess_all.ps1
.\scripts\run_data_pipeline.ps1
```

Failed LLVM rows are skipped. Successful rows are written to the processed `*_with_llvm.jsonl` files.

## Small Experiment Datasets

After compiling PrimeVul to LLVM IR, build a balanced sample dataset from a size config:

```powershell
python -m helpers.build_experiment_dataset --config configs/100_samples.yaml
python -m helpers.build_experiment_dataset --config configs/250_samples.yaml
python -m helpers.build_experiment_dataset --config configs/500_samples.yaml
python -m helpers.build_experiment_dataset --config configs/1000_samples.yaml
```

The size configs write to separate processed-data, checkpoint, log, and result folders.

## Train

Train one baseline:

```bash
python train.py --baseline b1
python train.py --baseline b2
python train.py --baseline b3
python train.py --baseline b4
```

Train with a specific config:

```bash
python train.py --config configs/100_samples.yaml --baseline b4
```

To warm-start B4 with B1/B2 encoders, train B1 and B2 first, then enable this config section:

```yaml
model:
  encoder_initialization:
    enabled: true
    source_checkpoint: E:/Checkpoints/100_samples/b1_best.pt
    ir_checkpoint: E:/Checkpoints/100_samples/b2_best.pt
```

If checkpoint paths are omitted, B4 looks for `b1_best.pt` and `b2_best.pt` inside `paths.checkpoints`.

Training uses AdamW, BCEWithLogitsLoss, sigmoid probabilities, gradient clipping, and tqdm progress bars.

## Evaluate

Evaluate all available best checkpoints:

```bash
python evaluate.py
```

Evaluate one baseline:

```bash
python evaluate.py --baseline b4
```

Evaluate with a specific config or dataset:

```bash
python evaluate.py --config configs/100_samples.yaml --baseline b4
python evaluate.py --dataset primevul
python evaluate.py --dataset rust
python evaluate.py --overwrite
```

Evaluation saves metrics JSON files, prediction CSV files, and `baseline_comparison.csv` in the configured `paths.results` folder.

Metrics include accuracy, precision, recall, F1-score, ROC-AUC, and false-positive rate.

## Scanner

Scan one source-code file with a trained model:

```bash
python scanner.py --baseline b4 --source-file test/sample.c
```

Example output:

```text
Baseline: B4
Prediction: Vulnerable
Probability: 0.82
Source weight: 0.63
LLVM-IR weight: 0.37
```

For `b1`, only source code is used. For `b2`, `b3`, and `b4`, scanner input is compiled to LLVM IR using Clang. The current scanner path supports C input.

## Tests

Full experiment script suite:

```powershell
.\scripts\test_all.ps1
```

Run one sample size end-to-end:

```powershell
.\scripts\test_100_samples.ps1
.\scripts\test_250_samples.ps1
.\scripts\test_500_samples.ps1
.\scripts\test_1000_samples.ps1
```

Useful focused runs:

```powershell
.\scripts\test_prepare_data.ps1
.\scripts\test_prepare_data.ps1 -FullRebuild
.\scripts\test_train_baselines.ps1
.\scripts\test_evaluate_baselines.ps1
.\scripts\test_evaluate_baselines.ps1 -Overwrite
.\scripts\test_b4_initialized.ps1
.\scripts\test_scanner_smoke.ps1
```

The script suite defaults to `configs/100_samples.yaml`, `configs/250_samples.yaml`, `configs/500_samples.yaml`, and `configs/1000_samples.yaml`. Evaluation does not overwrite existing metrics or predictions unless `-Overwrite` is passed. Initialized-B4 runs write to separate `*_b4_initialized` checkpoint, result, and log folders.

Per-sample scripts also write whole-run and Python-step timings to `results/<sample>_samples/run_timing_<timestamp>.txt`. Training logs include per-epoch `epoch_seconds` and cumulative `training_seconds`.

Light dataset loading check:

```powershell
python test/test_data_pipeline.py --config test/test_config.yaml
```

Full smoke script:

```powershell
.\test\run_full_test.ps1
```

The full smoke script loads the test dataset, trains B4 for one epoch, evaluates it, and scans `test/sample.c`. It may still download model weights and can take time on the first run.

## Ablations

B4 classifier-only ablations live in `ablation/`:

```powershell
python ablation/extract_b4_features.py --config configs/500_samples.yaml
python ablation/train_ablation.py --config configs/500_samples.yaml --variant all
python ablation/evaluate_ablation.py --config configs/500_samples.yaml --variant all
```

See `ablation/README.md` for the full ablation flow.

## Configuration

Main settings live in `configs/config.yaml`. Config files can inherit from another config with `inherits` or `extends`.

Useful config sections:

- `paths.checkpoints`: where `b1_best.pt`, `b4_last.pt`, and other checkpoint files are saved.
- `paths.results`: where metric JSON and prediction CSV files are saved.
- `model.source_name`: Hugging Face model/checkpoint used for source-code encoding.
- `model.ir_name`: Hugging Face model/checkpoint used for LLVM-IR encoding.
- `training.baseline`: default baseline when `--baseline` is not passed.
- `training.class_weights`: BCE loss weights for non-vulnerable and vulnerable samples.
- `model.projection.latent_dimension`: source/IR projection dimension used by B4.
- `model.gating`: B4 gate mode, fixed alpha, initial bias, and temperature.
- `data.datasets.primevul.max_records`: cap for generated PrimeVul experiment datasets.
- `data.datasets.primevul.balance`: requested vulnerable and non-vulnerable counts.
- `data.split.ratios`: train/validation/test split ratios.
