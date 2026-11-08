from __future__ import annotations

import argparse
from pathlib import Path
import json

from .audit import audit_inputs, write_markdown_audit
from .config import load_config, set_seed
from .demo import make_demo_data
from .experiments import run_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description="MM cytogenetic subtype TTE modeling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("make-demo-data")
    p_demo.add_argument("--out", default="data/demo")
    p_demo.add_argument("--n", type=int, default=260)
    p_demo.add_argument("--p", type=int, default=120)
    p_demo.add_argument("--seed", type=int, default=42)

    p_audit = sub.add_parser("audit-data")
    p_audit.add_argument("--clinical", required=True)
    p_audit.add_argument("--cytogenetics")
    p_audit.add_argument("--omics")
    p_audit.add_argument("--out", default="outputs/audit_report.json")
    p_audit.add_argument("--id-col", default="patient_id")
    p_audit.add_argument("--time-col", default="time_months")
    p_audit.add_argument("--event-col", default="event")

    p_run = sub.add_parser("run-experiments")
    p_run.add_argument("--config", default="configs/default.yaml")

    p_res = sub.add_parser(
        "residual-report",
        help="Residual-risk decomposition + matched ablation + claim/usefulness reports",
    )
    p_res.add_argument("--config", default="configs/real_training.yaml")

    p_main = sub.add_parser("run", help="Run the full pipeline via main.run_pipeline")
    p_main.add_argument("--config", default="configs/experiments/experiment0_open_gdc_os.yaml")

    p_ext = sub.add_parser("external-validate",
                           help="Train on one cohort, evaluate the frozen model on an external cohort")
    p_ext.add_argument("--train-config", required=True)
    p_ext.add_argument("--external-config", required=True)

    p_bench = sub.add_parser("benchmark-mmsygnal",
                             help="Score the cohort with the external mmSYGNAL models (research benchmark)")
    p_bench.add_argument("--config", default="configs/experiments/experiment0_open_gdc_os.yaml")
    p_bench.add_argument("--benchmark-config", default="configs/benchmarks/mmsygnal.yaml")

    p_hss = sub.add_parser("hss", help="Hierarchical Subtype Survival: {independent, pooled, HSS} "
                                       "per-subtype IPCW-IBS on repeated patient-disjoint folds")
    p_hss.add_argument("--config", default="configs/experiments/hss_open_gdc_os.yaml")

    p_reb = sub.add_parser("rebaseline", help="Stage-A leak-proof Experiment-0 re-baseline "
                                              "(in-fold PCA vs precomputed-PCA leak delta)")
    p_reb.add_argument("--config", default="configs/experiments/rebaseline_open_gdc_os.yaml")

    p_sd = sub.add_parser("stage-d", help="Stage-D negative controls: HSS biology-vs-regularization "
                                          "(real vs permuted vs random labels, lambda controls)")
    p_sd.add_argument("--config", default="configs/experiments/stageD_open_gdc_os.yaml")

    p_reg = sub.add_parser("regularization", help="Direction-2: pooled neural vs pooled penalised Cox "
                                                  "vs independent Cox (per-subtype IBS)")
    p_reg.add_argument("--config", default="configs/experiments/regularization_open_gdc_os.yaml")

    p_val = sub.add_parser("validate-subtypes",
                           help="Layered subtype-label validation: external real-FISH (GEO) + "
                                "cluster concordance + internal cross-modality + label-noise robustness")
    p_val.add_argument("--config", default="configs/experiments/subtype_validation_open_gdc_os.yaml")

    p_ln = sub.add_parser("label-noise",
                          help="Label-noise robustness: flip CNV labels at published FISH-discordance "
                               "rates; is the subtype-aware NULL robust?")
    p_ln.add_argument("--config", default="configs/experiments/subtype_validation_open_gdc_os.yaml")

    p_cal = sub.add_parser("subtype-calibration",
                           help="Pre-registered one-shot: does subtype-stratified calibration beat "
                                "pooled+scramble? (characterization only — external replication unmeetable)")
    p_cal.add_argument("--config", default="configs/experiments/subtype_validation_open_gdc_os.yaml")

    p_fal = sub.add_parser("falsify", help="Phase-3 falsification matrix: all arms + controls "
                                           "(scrambled-subtype, permuted-omics, random-label) on shared folds")
    p_fal.add_argument("--config", default="configs/experiments/falsification_open_gdc_os.yaml")

    p_sub = sub.add_parser("submit-demo", help="Phase-6 submission harness demo: score an sksurv "
                                               "Coxnet arm on the frozen benchmark's folds + honesty gate")
    p_sub.add_argument("--config", default="configs/experiments/falsification_open_gdc_os.yaml")

    args = parser.parse_args(argv)

    # Central deterministic seeding for every subcommand (reproducibility P0). Prefer an
    # explicit --seed; otherwise read seed from the command's config; else default 42.
    _seed = getattr(args, "seed", None)
    if _seed is None:
        _cfgpath = getattr(args, "config", None)
        if _cfgpath:
            try:
                _seed = int(load_config(_cfgpath).get("seed", 42))
            except Exception:
                _seed = 42
        else:
            _seed = 42
    set_seed(int(_seed))

    if args.cmd == "make-demo-data":
        make_demo_data(args.out, args.n, args.p, args.seed)
        print(f"Wrote demo data to {args.out}")
    elif args.cmd == "audit-data":
        report = audit_inputs(args.clinical, args.cytogenetics, args.omics, args.out, args.id_col, args.time_col, args.event_col)
        md = Path(args.out).with_suffix(".md")
        write_markdown_audit(report, md)
        print(json.dumps(report["usable_for"], indent=2))
        print(f"Wrote {args.out} and {md}")
    elif args.cmd == "run-experiments":
        cfg = load_config(args.config)
        res = run_experiments(cfg)
        print(res["leaderboard"].to_string(index=False))
        print(f"Outputs: {res['outdir']}")
    elif args.cmd == "residual-report":
        from .residual import run_residual_report
        cfg = load_config(args.config)
        res = run_residual_report(cfg)
        print(f"\nOutputs: {res['outdir']}")
    elif args.cmd == "run":
        from .main import run_pipeline
        run_pipeline(args.config)
    elif args.cmd == "external-validate":
        from .evaluation.external import run_external_validation
        run_external_validation(args.train_config, args.external_config)
    elif args.cmd == "benchmark-mmsygnal":
        from .benchmarks.mmsygnal import run_mmsygnal_benchmark
        cfg = load_config(args.config)
        bench_cfg = load_config(args.benchmark_config)
        run_mmsygnal_benchmark(cfg, bench_cfg)
    elif args.cmd == "hss":
        from .experiments_hss import run_hss_experiment
        cfg = load_config(args.config)
        res = run_hss_experiment(cfg)
        print(res["summary"].to_string(index=False))
        print(f"\nOutputs: {res['outdir']}")
    elif args.cmd == "rebaseline":
        from .experiments_rebaseline import run_rebaseline
        cfg = load_config(args.config)
        res = run_rebaseline(cfg)
        print(res["leak_delta"].to_string(index=False))
        print(f"\nOutputs: {res['outdir']}")
    elif args.cmd == "stage-d":
        from .experiments_stageD import run_stage_d
        cfg = load_config(args.config)
        res = run_stage_d(cfg)
        print(res["summary"].to_string(index=False))
        print("\nDECISION:", res["decision"]["verdict"])
        print(f"Outputs: {res['outdir']}")
    elif args.cmd == "regularization":
        from .experiments_regularization import run_regularization
        cfg = load_config(args.config)
        res = run_regularization(cfg)
        print(res["summary"].to_string(index=False))
        print("\nDECISION:", res["decision"]["verdict"])
        print(f"Outputs: {res['outdir']}")
    elif args.cmd == "validate-subtypes":
        from .validation.run_validation import run_subtype_validation
        cfg = load_config(args.config)
        res = run_subtype_validation(cfg)
        print(f"Subtype-label validation written to {res['outdir']}/subtype_validation_summary.md")
    elif args.cmd == "label-noise":
        from .experiments_label_noise import run_label_noise
        cfg = load_config(args.config)
        res = run_label_noise(cfg)
        print(res["summary"].to_string())
        print("\nDECISION:", res["decision"]["verdict"])
        print(f"Outputs: {res['outdir']}")
    elif args.cmd == "subtype-calibration":
        from .experiments_calibration_subtype import run_subtype_calibration
        cfg = load_config(args.config)
        res = run_subtype_calibration(cfg)
        print(res["summary"].to_string(index=False))
        print("\nDECISION:", res["decision"]["verdict"])
        print(f"Outputs: {res['outdir']}")
    elif args.cmd == "falsify":
        from .experiments_falsification import run_falsification
        cfg = load_config(args.config)
        res = run_falsification(cfg)
        print(res["matrix"].to_string(index=False))
        print("\nOVERALL:", res["decision"]["overall"])
        print(f"Outputs: {res['outdir']}")
    elif args.cmd == "submit-demo":
        from .benchmark_submit import demo_coxnet_submission
        cfg = load_config(args.config)
        res = demo_coxnet_submission(cfg)
        print(f"arm={res['name']}  baseline={res['baseline']}")
        print(f"  Uno-C={res['row'].get('mean_uno_cindex')}  "
              f"ΔC={res['delta'].get('delta_cindex')}  FDR={res['delta'].get('p_fdr_bh')}")
        print(f"  VERDICT: {res['verdict']['message']}")
        print(f"Outputs: {res['outdir']}/submission.json")


if __name__ == "__main__":
    main()
