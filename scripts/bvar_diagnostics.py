#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bvar.diagnostics import (
    convergence_diagnostics,
    ensure_chain_draws,
    run_bvar_diagnostics,
)


def _as_labels(arr) -> list[str] | None:
    if arr is None:
        return None
    return [str(x) for x in arr]


def _load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    omegas = data["omegas_chains"] if "omegas_chains" in data else data["omegas"]
    betas = data["betas_chains"] if "betas_chains" in data else data["betas"]
    columns = data.get("columns", None)
    return omegas, betas, _as_labels(columns)


def _parse_columns(value: str | None):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bounds(value: str | None):
    if not value:
        return None
    bounds = []
    for item in value.split(","):
        low, high = item.split(":")
        bounds.append((float(low), float(high)))
    if len(bounds) != 5:
        raise ValueError("bounds must have 5 entries: low:high,...")
    return bounds


def _load_csv(path: str, date_col: str, columns: list[str] | None):
    df = pd.read_csv(path)
    if columns:
        return df.loc[:, columns]
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    return df


def main():
    parser = argparse.ArgumentParser(description="Run convergence diagnostics on BVAR draws")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fit",
        nargs="+",
        help="Path(s) to fit NPZ file(s)",
    )
    source.add_argument(
        "--data",
        help="Path to CSV data file (runs full pipeline)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Split a single fit into N chains (fit mode) or set N chains (data mode)",
    )
    parser.add_argument(
        "--chain-axis",
        type=int,
        default=None,
        help="Axis index for chain dimension (if already present)",
    )
    parser.add_argument(
        "--draw-axis",
        type=int,
        default=None,
        help="Axis index for draw dimension (optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file for summary (csv or json)",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=("csv", "json"),
        help="Output format for summary",
    )
    parser.add_argument(
        "--round-to",
        type=int,
        default=4,
        help="Rounding for summary table",
    )
    parser.add_argument("--lags", type=int, default=3, help="Number of lags (data mode)")
    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Posterior draws per chain (data mode)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (data mode)")
    parser.add_argument("--date-col", default="Fecha", help="Date column name (data mode)")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to use (data mode)",
    )
    parser.add_argument(
        "--opt-starts",
        type=int,
        default=10,
        help="Random starts for hyperparameter optimization (data mode)",
    )
    parser.add_argument(
        "--bounds",
        default=None,
        help="Comma-separated bounds per lambda as low:high (data mode)",
    )

    args = parser.parse_args()

    if args.fit:
        if len(args.fit) > 1 and args.chains is not None:
            raise ValueError("Use --chains only when a single --fit is provided")

        omega_chains = []
        beta_chains = []
        labels = None

        for fit_path in args.fit:
            omegas, betas, columns = _load_npz(Path(fit_path))
            if labels is None:
                labels = columns

            inferred_chain_axis = args.chain_axis
            if inferred_chain_axis is None and omegas.ndim >= 4:
                inferred_chain_axis = 0
            if (
                inferred_chain_axis is None
                and omegas.ndim == 3
                and omegas.shape[0] <= 8
                and omegas.shape[1] > omegas.shape[0]
                and omegas.shape[2] != omegas.shape[0]
            ):
                inferred_chain_axis = 0

            omega_chain = ensure_chain_draws(
                omegas,
                chains=None,
                chain_axis=inferred_chain_axis,
                draw_axis=args.draw_axis,
            )
            beta_chain = ensure_chain_draws(
                betas,
                chains=None,
                chain_axis=inferred_chain_axis,
                draw_axis=args.draw_axis,
            )
            omega_chains.append(omega_chain)
            beta_chains.append(beta_chain)

        omegas = np.concatenate(omega_chains, axis=0)
        betas = np.concatenate(beta_chains, axis=0)

        if args.chains is not None:
            omegas = ensure_chain_draws(
                omegas,
                chains=args.chains,
                chain_axis=0,
                draw_axis=1,
            )
            betas = ensure_chain_draws(
                betas,
                chains=args.chains,
                chain_axis=0,
                draw_axis=1,
            )

        if omegas.shape[0] < 2:
            raise ValueError(
                "Need at least 2 chains for convergence diagnostics. "
                "Provide multiple --fit files or use --chains to split draws."
            )

        results = convergence_diagnostics(
            omegas,
            betas,
            chain_axis=0,
            draw_axis=1,
            labels=labels,
            round_to=args.round_to,
        )
    else:
        columns = _parse_columns(args.columns)
        df = _load_csv(args.data, args.date_col, columns)
        if columns is None:
            columns = list(df.columns)
        bounds = _parse_bounds(args.bounds)
        chains = args.chains if args.chains is not None else 2

        results = run_bvar_diagnostics(
            df.to_numpy(),
            lags=args.lags,
            draws=args.draws,
            chains=chains,
            seed=args.seed,
            bounds=bounds,
            opt_starts=args.opt_starts,
            labels=columns,
            round_to=args.round_to,
        )

    summary = results["summary"]

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "csv":
            summary.to_csv(output)
        else:
            summary.to_json(output, orient="table")
    else:
        print(summary.to_string())


if __name__ == "__main__":
    main()
