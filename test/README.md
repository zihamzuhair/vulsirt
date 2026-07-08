# Test Folder

This folder contains a small smoke-test setup for the project flow.

The test config uses the same model-name fields as the main config, but with shorter sequence lengths and a one-epoch run. The first run can still take time if model weights are not already cached.

Run from the project root:

```powershell
.\test\run_full_test.ps1
```

The script runs:

```powershell
python test/test_data_pipeline.py --config test/test_config.yaml
python train.py --config test/test_config.yaml --baseline b4
python evaluate.py --config test/test_config.yaml --baseline b4
python scanner.py --config test/test_config.yaml --baseline b4 --source-file test/sample.c
```

Outputs are written inside:

```text
test/processed_dataset.jsonl
test/checkpoints/
test/results/
test/logs/
```

If `scanner.py` fails because Clang is not installed, install Clang or run only dataset loading, training, and evaluation. The test dataset already includes LLVM IR, so the dataset loading test does not need Clang.
