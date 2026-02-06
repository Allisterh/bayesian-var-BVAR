from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .design import build_var_design
from .fevd import fevd_from_irf
from .irf import irf_mc, irf_quantiles
from .marginal import optimize_minnesota_hyperparams
from .posterior import posterior_params, sample_posterior
from .priors import ar1_prior_stats, minnesota_dummy_observations

app = FastAPI(title="BVAR API", version="0.1.0")


def _parse_columns(value: str | None) -> list[str] | None:
    if not value:
        return None
    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns or None


def _parse_bounds(value: str | None) -> list[tuple[float, float]] | None:
    if not value:
        return None
    bounds: list[tuple[float, float]] = []
    for item in value.split(","):
        low, high = item.split(":")
        bounds.append((float(low), float(high)))
    if len(bounds) != 5:
        raise ValueError("bounds must have 5 entries: low:high,...")
    return bounds


def _load_csv_bytes(content: bytes, date_col: str, columns: list[str] | None) -> pd.DataFrame:
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:  # pragma: no cover - defensive for API
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in CSV: {', '.join(missing)}",
            )
        df = df.loc[:, columns]
    elif date_col in df.columns:
        df = df.drop(columns=[date_col])

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isnull().any().any():
        raise HTTPException(
            status_code=400,
            detail="Data contains non-numeric values or NaNs after coercion.",
        )

    return df


def _summarize_draws(draws: np.ndarray) -> dict[str, Any]:
    return {
        "mean": draws.mean(axis=0).tolist(),
        "std": draws.std(axis=0).tolist(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/estimate")
async def estimate(
    file: UploadFile = File(...),
    lags: int = Form(3),
    draws: int = Form(2000),
    seed: int | None = Form(None),
    date_col: str = Form("Fecha"),
    columns: str | None = Form(None),
    opt_starts: int = Form(10),
    bounds: str | None = Form(None),
    irf_horizon: int = Form(35),
    fevd_horizon: int | None = Form(None),
    include_draws: bool = Form(False),
) -> dict[str, Any]:
    if lags < 1:
        raise HTTPException(status_code=400, detail="lags must be >= 1")
    if draws < 1:
        raise HTTPException(status_code=400, detail="draws must be >= 1")
    if irf_horizon < 0:
        raise HTTPException(status_code=400, detail="irf_horizon must be >= 0")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        parsed_columns = _parse_columns(columns)
        parsed_bounds = _parse_bounds(bounds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    df = _load_csv_bytes(content, date_col, parsed_columns)
    if parsed_columns is None:
        parsed_columns = list(df.columns)

    data = df.to_numpy()
    if data.shape[0] <= lags:
        raise HTTPException(
            status_code=400,
            detail="Not enough observations for requested lags",
        )

    try:
        Y, X = build_var_design(data, lags, include_intercept=True)
        delta, su, s0 = ar1_prior_stats(data, lags)
        y_mean = data[: min(2 * lags, data.shape[0]), :].mean(axis=0)

        result = optimize_minnesota_hyperparams(
            Y,
            X,
            delta,
            su,
            s0,
            y_mean,
            lags,
            bounds=parsed_bounds,
            n_starts=opt_starts,
            seed=seed,
        )

        lambdas = result.x
        YP, XP = minnesota_dummy_observations(lambdas, delta, su, s0, y_mean, lags)
        Bps, Hs, Ss, vd = posterior_params(Y, X, YP, XP)
        omegas, betas = sample_posterior(Bps, Hs, Ss, vd, draws, seed=seed)

        irf = irf_mc(omegas, betas, irf_horizon)
        bands = irf_quantiles(irf, irf_horizon)

        fevd_h = fevd_horizon or irf_horizon
        irf_mean = irf.mean(axis=2)
        fevd = fevd_from_irf(irf_mean, omegas.shape[1], fevd_h)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response: dict[str, Any] = {
        "meta": {
            "columns": parsed_columns,
            "lags": lags,
            "draws": draws,
            "irf_horizon": irf_horizon,
            "fevd_horizon": fevd_h,
            "seed": seed,
            "lambdas": lambdas.tolist(),
        },
        "posterior": {
            "omega": _summarize_draws(omegas),
            "beta": _summarize_draws(betas),
        },
        "irf": {
            "q_10": bands["q_10"].tolist(),
            "q_16": bands["q_16"].tolist(),
            "q_50": bands["q_50"].tolist(),
            "q_84": bands["q_84"].tolist(),
            "q_90": bands["q_90"].tolist(),
        },
        "fevd": fevd.tolist(),
    }

    if include_draws:
        response["draws"] = {
            "omegas": omegas.tolist(),
            "betas": betas.tolist(),
        }

    return response
