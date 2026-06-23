# Test Folder

This folder contains a tiny smoke-test setup for the full project flow.

It uses `hf-internal-testing/tiny-random-roberta` instead of CodeBERT so the test can run quickly. The main `config.yaml` still uses `microsoft/codebert-base`.

Run from the project root:

```powershell
.\test\run_full_test.ps1
```

The script runs:

```powershell
python preprocess.py --config test/test_config.yaml
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

If `scanner.py` fails because Clang is not installed, install Clang or test only preprocessing, training, and evaluation. The test dataset already includes LLVM IR, so preprocessing does not need Clang.
