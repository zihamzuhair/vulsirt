"""Run the PrimeVul and Rust source-to-LLVM compiler pipelines."""

import argparse
import json

from helpers import primevul_code_compiler, rust_code_compiler


def compile_c_function(*args, **kwargs):
    """Expose the PrimeVul C compiler helper for scanner.py."""
    return primevul_code_compiler.compile_c_function(*args, **kwargs)


def is_successful_llvm_record(record):
    """Expose the PrimeVul success check for scanner.py."""
    return primevul_code_compiler.is_successful_llvm_record(record)


def parse_args():
    """Read which dataset compiler should run."""
    parser = argparse.ArgumentParser(description="Compile processed source datasets to LLVM IR.")
    parser.add_argument(
        "--dataset",
        choices=["primevul", "rust", "all"],
        default="all",
        help="Dataset compiler to run.",
    )
    return parser.parse_args()


def main():
    """Run the selected compiler pipelines and print a JSON report."""
    args = parse_args()
    reports = {}

    if args.dataset in {"primevul", "all"}:
        reports["primevul"] = primevul_code_compiler.run()

    if args.dataset in {"rust", "all"}:
        reports["rust"] = rust_code_compiler.run()

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
