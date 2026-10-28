"""Regenerate the Supporting Information tables of the descriptor-ladder manuscript.

    python reproduce.py --list        show every table, its commands and whether it is long
    python reproduce.py S4 S5         run the commands behind the named tables
    python reproduce.py --all         run every table

Commands run from the repository root with src/ on the Python path, one thread per BLAS library,
in the order listed, so a table whose inputs come from another table is rebuilt after it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# table: (commands, long run)
STEPS = {
    "Fig2b": (["scripts/run_worked_example.py"], False),
    "SecS3.1": (["scripts/run_descriptor_residuals.py"], False),
    "S2": (["scripts/run_leakage_census.py"], False),
    "S3": (["scripts/run_leakage_census.py"], False),
    "S4": (["scripts/run_descriptor_ladder.py", "scripts/run_descriptor_effect_crossover.py"], False),
    "S5": (["scripts/run_full_grid.py", "scripts/run_full_grid_tuned_ridge.py"], True),
    "S6": (["scripts/run_full_grid.py", "scripts/run_full_grid_tuned_ridge.py"], True),
    "SecS3.2": (["scripts/run_full_grid_stability.py"], True),
    "S7": (["scripts/run_emery_replication.py --part rung1 --n-jobs 4"], True),
    "S8": (["scripts/run_subgrid_robustness.py"], True),
    "S9": (["scripts/run_covariance_rank.py"], True),
    "S10": (["scripts/run_regularization_baselines.py"], False),
    "S11": (["scripts/run_regularization_baselines.py --panels"], False),
    "S12": (["scripts/run_prospective_magnetic_validation.py"], False),
    "S13": (["scripts/run_gain_by_block.py --estimator reml", "scripts/summarize_gain_by_block.py"], True),
    "S14": (["scripts/run_peak_vs_interaction_share.py", "scripts/run_protein_peak.py"], False),
    "S15": (["scripts/run_peak_mode_coupling.py"], False),
    "FigS3": (["scripts/run_peak_mode_coupling.py", "scripts/render_peak_mode_sources.py"], False),
    "S16": (["scripts/run_gain_by_direction_state.py"], False),
    "S17": (["scripts/run_gain_by_direction_state.py --shuffled"], False),
    "S18": (["scripts/run_project_like_designs.py --n-jobs 4"], True),
    "S19": (["scripts/run_emery_replication.py --part rung2 --n-jobs 4"], True),
    "S20": (["scripts/run_emery_gain_attribution.py --n-jobs 4"], False),
    "S21": (["scripts/run_protein_replication.py", "scripts/derive_shrinkage_window.py"], False),
    "S22": (["scripts/derive_connectivity_window.py", "scripts/derive_window_branches.py",
             "scripts/derive_window_isolation.py", "scripts/derive_window_amplitude.py"], False),
    "S23": (["scripts/derive_connectivity_window.py", "scripts/derive_window_branches.py",
             "scripts/derive_window_isolation.py", "scripts/derive_window_amplitude.py"], False),
    "S24": (["scripts/run_peak_mode_coupling.py", "scripts/run_protein_design_amplification.py"], False),
    "S25": (["scripts/run_effect_map.py", "scripts/run_transfer_sample_size.py"], False),
    "S26": (["scripts/run_left_out_element.py"], True),
    "S27": (["scripts/run_anion_sign_census.py"], False),
    "S28": (["scripts/run_cmr_site_moments.py"], False),
    "S29": (["scripts/run_semiparametric_kernel_validation.py"], True),
    "S30": (["scripts/run_rf_gap_gain.py --n-jobs 4"], True),
    "S31": (["scripts/run_mp_perovskite_reading.py"], False),
}


def run(command: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           **{k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    script, *args = command.split()
    print(f"\n>>> {command}", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tables", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.list or not (args.tables or args.all):
        for table, (commands, long) in STEPS.items():
            print(f"{table:6s} {'long ' if long else '     '}" + " ; ".join(commands))
        return
    unknown = [t for t in args.tables if t not in STEPS]
    if unknown:
        raise SystemExit(f"unknown table(s): {', '.join(unknown)}; see --list")
    tables = list(STEPS) if args.all else args.tables
    done = set()
    for table in tables:
        for command in STEPS[table][0]:
            if command not in done:
                run(command)
                done.add(command)


if __name__ == "__main__":
    main()
