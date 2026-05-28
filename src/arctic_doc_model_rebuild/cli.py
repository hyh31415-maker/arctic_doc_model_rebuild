from __future__ import annotations

import argparse
import sys
from typing import Iterable

from .gold_contract import verification_problem_counts
from .reports import summarize_gold_data, write_data_contract_report, write_verification_outputs
from .schema_checks import issue_count


def verify_gold_data(args: argparse.Namespace) -> int:
    verification, schema = write_verification_outputs()
    counts = verification_problem_counts(verification)
    schema_issues = issue_count(schema)
    blocking_hash = counts["sha256_mismatch"] and not args.allow_hash_mismatch
    blocking = counts["missing"] or counts["row_count_mismatch"] or counts["read_error"] or blocking_hash or schema_issues
    if blocking:
        print("Gold data contract verification failed. See outputs/reports/data_contract_report.md", file=sys.stderr)
        return 1
    print("Gold data contract verification passed.")
    return 0


def summarize_command(args: argparse.Namespace) -> int:
    summarize_gold_data()
    print("Gold data summaries written under outputs/reports and outputs/tables.")
    return 0


def report_command(args: argparse.Namespace) -> int:
    write_data_contract_report()
    print("Data contract report written to outputs/reports/data_contract_report.md.")
    return 0


def run_eda_command(args: argparse.Namespace) -> int:
    from .eda import run_eda

    result = run_eda()
    print(f"EDA report written to {result.report_path}.")
    print(f"EDA tables generated: {len(result.table_paths)}")
    print(f"EDA figures generated: {len(result.figure_paths)}")
    return 0


def run_baseline_models_command(args: argparse.Namespace) -> int:
    from .modeling.baseline_models import run_baseline_models

    result = run_baseline_models()
    print(f"Baseline report written to {result['report']}.")
    print(f"Baseline tables generated: {len(result['tables'])}")
    print(f"Baseline figures generated: {len(result['figures'])}")
    return 0


def baseline_report_command(args: argparse.Namespace) -> int:
    from .modeling.reports import write_baseline_report_from_tables

    report_path = write_baseline_report_from_tables()
    print(f"Baseline report written to {report_path}.")
    return 0


def run_baseline_refinement_command(args: argparse.Namespace) -> int:
    from .modeling.refinement import run_baseline_refinement

    result = run_baseline_refinement()
    print(f"Baseline refinement report written to {result['report']}.")
    print(f"Baseline refinement tables generated: {len(result['tables'])}")
    print(f"Baseline refinement figures generated: {len(result['figures'])}")
    return 0


def baseline_refinement_report_command(args: argparse.Namespace) -> int:
    from .modeling.refinement import write_baseline_refinement_report

    report_path = write_baseline_refinement_report()
    print(f"Baseline refinement report written to {report_path}.")
    return 0


def finalize_baseline_command(args: argparse.Namespace) -> int:
    from .modeling.baseline_final import finalize_baseline

    result = finalize_baseline()
    print(f"Baseline final reports generated: {len(result['reports'])}")
    print(f"Baseline final tables generated: {len(result['tables'])}")
    print(f"Baseline model specs generated: {len(result['specs'])}")
    return 0


def baseline_final_report_command(args: argparse.Namespace) -> int:
    from .modeling.baseline_final import write_baseline_final_report, write_next_phase_handoff

    report_path = write_baseline_final_report()
    handoff_path = write_next_phase_handoff()
    print(f"Baseline final report written to {report_path}.")
    print(f"Next phase handoff written to {handoff_path}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arctic-doc-model", description="Arctic DOC model rebuild data contract tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-gold-data", help="Verify frozen gold data hashes, row counts, schemas, and leakage checks.")
    verify.add_argument("--allow-hash-mismatch", action="store_true", help="Report but do not fail on SHA256 mismatches.")

    subparsers.add_parser("summarize-gold-data", help="Summarize frozen gold inputs without modeling.")
    subparsers.add_parser("data-contract-report", help="Write the gold data contract report without modeling.")
    subparsers.add_parser("run-eda", help="Run descriptive EDA on frozen gold data without model training.")
    subparsers.add_parser("run-baseline-models", help="Train validation-only baseline DOC concentration models.")
    subparsers.add_parser("baseline-report", help="Rewrite the baseline model report from existing baseline tables.")
    subparsers.add_parser("run-baseline-refinement", help="Run validation-only same-sample baseline refinement diagnostics.")
    subparsers.add_parser("baseline-refinement-report", help="Rewrite the baseline refinement report from existing refinement tables.")
    subparsers.add_parser("finalize-baseline", help="Finalize the baseline DOC concentration model decision without production prediction.")
    subparsers.add_parser("baseline-final-report", help="Rewrite the baseline final report and next-phase handoff.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "verify-gold-data":
        return verify_gold_data(args)
    if args.command == "summarize-gold-data":
        return summarize_command(args)
    if args.command == "data-contract-report":
        return report_command(args)
    if args.command == "run-eda":
        return run_eda_command(args)
    if args.command == "run-baseline-models":
        return run_baseline_models_command(args)
    if args.command == "baseline-report":
        return baseline_report_command(args)
    if args.command == "run-baseline-refinement":
        return run_baseline_refinement_command(args)
    if args.command == "baseline-refinement-report":
        return baseline_refinement_report_command(args)
    if args.command == "finalize-baseline":
        return finalize_baseline_command(args)
    if args.command == "baseline-final-report":
        return baseline_final_report_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
