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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arctic-doc-model", description="Arctic DOC model rebuild data contract tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-gold-data", help="Verify frozen gold data hashes, row counts, schemas, and leakage checks.")
    verify.add_argument("--allow-hash-mismatch", action="store_true", help="Report but do not fail on SHA256 mismatches.")

    subparsers.add_parser("summarize-gold-data", help="Summarize frozen gold inputs without modeling.")
    subparsers.add_parser("data-contract-report", help="Write the gold data contract report without modeling.")
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
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
