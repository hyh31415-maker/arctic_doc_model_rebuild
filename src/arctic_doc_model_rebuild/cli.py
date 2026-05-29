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


def run_optical_sensitivity_command(args: argparse.Namespace) -> int:
    from .modeling.optical_sensitivity import run_optical_sensitivity

    result = run_optical_sensitivity()
    print(f"Optical sensitivity report written to {result['report']}.")
    print(f"Optical sensitivity tables generated: {len(result['tables'])}")
    print(f"Optical sensitivity figures generated: {len(result['figures'])}")
    return 0


def optical_sensitivity_report_command(args: argparse.Namespace) -> int:
    from .modeling.optical_sensitivity import write_optical_report_from_tables

    report_path = write_optical_report_from_tables()
    print(f"Optical sensitivity report written to {report_path}.")
    return 0


def roi_final_qc_command(args: argparse.Namespace) -> int:
    from .roi_qc import run_roi_final_qc

    result = run_roi_final_qc()
    print(f"ROI final QC report written to {result['report']}.")
    print(f"ROI final QC tables generated: {len(result['tables'])}")
    print(f"ROI final QC figures generated: {len(result['figures'])}")
    return 0


def roi_final_qc_report_command(args: argparse.Namespace) -> int:
    from .roi_qc import write_roi_final_qc_report

    report_path = write_roi_final_qc_report()
    print(f"ROI final QC report written to {report_path}.")
    return 0


def run_concentration_uncertainty_command(args: argparse.Namespace) -> int:
    from .modeling.concentration_uncertainty import run_concentration_uncertainty

    result = run_concentration_uncertainty()
    print(f"Concentration uncertainty report written to {result['report']}.")
    print(f"Concentration uncertainty tables generated: {len(result['tables'])}")
    print(f"Concentration uncertainty figures generated: {len(result['figures'])}")
    return 0


def concentration_uncertainty_report_command(args: argparse.Namespace) -> int:
    from .modeling.concentration_uncertainty import write_concentration_uncertainty_report

    report_path = write_concentration_uncertainty_report()
    print(f"Concentration uncertainty report written to {report_path}.")
    return 0


def run_bias_aware_refinement_command(args: argparse.Namespace) -> int:
    from .modeling.bias_refinement import run_bias_aware_refinement

    result = run_bias_aware_refinement()
    print(f"Bias-aware refinement report written to {result['report']}.")
    print(f"Bias-aware refinement tables generated: {len(result['tables'])}")
    print(f"Bias-aware refinement figures generated: {len(result['figures'])}")
    return 0


def bias_aware_refinement_report_command(args: argparse.Namespace) -> int:
    from .modeling.bias_refinement import write_bias_refinement_report

    report_path = write_bias_refinement_report()
    print(f"Bias-aware refinement report written to {report_path}.")
    return 0


def freeze_production_candidate_command(args: argparse.Namespace) -> int:
    from .modeling.daily_doc_prediction import freeze_production_candidate

    result = freeze_production_candidate()
    print(f"Production candidate spec written to {result['spec']}.")
    print(f"Production candidate decision written to {result['decision']}.")
    return 0


def run_daily_doc_prediction_command(args: argparse.Namespace) -> int:
    from .modeling.daily_doc_prediction import run_daily_doc_prediction

    result = run_daily_doc_prediction()
    print(f"Daily DOC prediction report written to {result['report']}.")
    print(f"Daily DOC prediction tables generated: {len(result['tables'])}")
    print(f"Daily DOC prediction figures generated: {len(result['figures'])}")
    return 0


def daily_doc_prediction_report_command(args: argparse.Namespace) -> int:
    from .modeling.daily_doc_prediction import write_daily_doc_prediction_report

    report_path = write_daily_doc_prediction_report()
    print(f"Daily DOC prediction report written to {report_path}.")
    return 0


def run_doc_flux_command(args: argparse.Namespace) -> int:
    from .flux.flux_calculation import run_doc_flux

    result = run_doc_flux()
    print(f"DOC flux report written to {result['report']}.")
    print(f"DOC flux tables generated: {len(result['tables'])}")
    print(f"DOC flux figures generated: {len(result['figures'])}")
    return 0


def doc_flux_report_command(args: argparse.Namespace) -> int:
    from .flux.flux_reports import write_doc_flux_report

    report_path = write_doc_flux_report()
    print(f"DOC flux report written to {report_path}.")
    return 0


def select_flux_analysis_cohorts_command(args: argparse.Namespace) -> int:
    from .flux.flux_cohorts import select_flux_analysis_cohorts

    result = select_flux_analysis_cohorts()
    print(f"Flux cohort report written to {result['report']}.")
    print(f"Flux cohort tables generated: {len(result['tables'])}")
    print(f"Flux cohort figures generated: {len(result['figures'])}")
    return 0


def flux_cohort_report_command(args: argparse.Namespace) -> int:
    from .flux.flux_cohorts import write_flux_cohort_report

    report_path = write_flux_cohort_report()
    print(f"Flux cohort report written to {report_path}.")
    return 0


def run_annual_flux_trends_command(args: argparse.Namespace) -> int:
    from .flux.trend_analysis import run_annual_flux_trends

    result = run_annual_flux_trends()
    print(f"Annual flux trend report written to {result['report']}.")
    print(f"Annual flux trend tables generated: {len(result['tables'])}")
    print(f"Annual flux trend figures generated: {len(result['figures'])}")
    return 0


def annual_flux_trend_report_command(args: argparse.Namespace) -> int:
    from .flux.trend_reports import write_annual_flux_trend_report

    report_path = write_annual_flux_trend_report()
    print(f"Annual flux trend report written to {report_path}.")
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
    subparsers.add_parser("run-optical-sensitivity", help="Run validation-only optical sensitivity diagnostics against the finalized F3 baseline.")
    subparsers.add_parser("optical-sensitivity-report", help="Rewrite the optical sensitivity report from existing optical sensitivity tables.")
    subparsers.add_parser("roi-final-qc", help="Audit frozen ROI and optical data integrity without modifying gold data.")
    subparsers.add_parser("roi-final-qc-report", help="Rewrite the ROI final QC report from existing ROI QC tables.")
    subparsers.add_parser("run-concentration-uncertainty", help="Run final validation-only DOC concentration uncertainty diagnostics.")
    subparsers.add_parser("concentration-uncertainty-report", help="Rewrite the concentration uncertainty report from existing tables.")
    subparsers.add_parser("run-bias-aware-refinement", help="Run validation-only bias-aware concentration model refinement diagnostics.")
    subparsers.add_parser("bias-aware-refinement-report", help="Rewrite the bias-aware refinement report from existing tables.")
    subparsers.add_parser("freeze-production-candidate", help="Freeze the refined production candidate DOC concentration model spec.")
    subparsers.add_parser("run-daily-doc-prediction", help="Generate guarded production daily DOC concentration predictions without flux.")
    subparsers.add_parser("daily-doc-prediction-report", help="Rewrite the daily DOC prediction report from existing tables.")
    subparsers.add_parser("run-doc-flux", help="Calculate guarded DOC flux from existing daily DOC predictions without retraining.")
    subparsers.add_parser("doc-flux-report", help="Rewrite the guarded DOC flux report from existing flux tables.")
    subparsers.add_parser("select-flux-analysis-cohorts", help="Select flux interpretation cohorts from existing DOC flux tables without trend tests.")
    subparsers.add_parser("flux-cohort-report", help="Rewrite the flux interpretation cohort report from existing cohort tables.")
    subparsers.add_parser("run-annual-flux-trends", help="Analyze annual DOC flux trends by cohort without recomputing flux.")
    subparsers.add_parser("annual-flux-trend-report", help="Rewrite the annual DOC flux trend report from existing trend tables.")
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
    if args.command == "run-optical-sensitivity":
        return run_optical_sensitivity_command(args)
    if args.command == "optical-sensitivity-report":
        return optical_sensitivity_report_command(args)
    if args.command == "roi-final-qc":
        return roi_final_qc_command(args)
    if args.command == "roi-final-qc-report":
        return roi_final_qc_report_command(args)
    if args.command == "run-concentration-uncertainty":
        return run_concentration_uncertainty_command(args)
    if args.command == "concentration-uncertainty-report":
        return concentration_uncertainty_report_command(args)
    if args.command == "run-bias-aware-refinement":
        return run_bias_aware_refinement_command(args)
    if args.command == "bias-aware-refinement-report":
        return bias_aware_refinement_report_command(args)
    if args.command == "freeze-production-candidate":
        return freeze_production_candidate_command(args)
    if args.command == "run-daily-doc-prediction":
        return run_daily_doc_prediction_command(args)
    if args.command == "daily-doc-prediction-report":
        return daily_doc_prediction_report_command(args)
    if args.command == "run-doc-flux":
        return run_doc_flux_command(args)
    if args.command == "doc-flux-report":
        return doc_flux_report_command(args)
    if args.command == "select-flux-analysis-cohorts":
        return select_flux_analysis_cohorts_command(args)
    if args.command == "flux-cohort-report":
        return flux_cohort_report_command(args)
    if args.command == "run-annual-flux-trends":
        return run_annual_flux_trends_command(args)
    if args.command == "annual-flux-trend-report":
        return annual_flux_trend_report_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
