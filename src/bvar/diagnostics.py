from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pymc as pm

from .design import build_var_design
from .marginal import optimize_minnesota_hyperparams
from .posterior import posterior_params, sample_posterior
from .priors import ar1_prior_stats, minnesota_dummy_observations


def _chain_seeds(seed: int | None, chains: int) -> list[int | None]:
    if seed is None:
        return [None] * chains
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32 - 1, size=chains, dtype=np.uint32).tolist()


def _infer_draw_axis(arr: np.ndarray, draw_axis: int | None, has_chain: bool) -> int:
    if draw_axis is not None:
        return draw_axis
    start = 1 if has_chain else 0
    axis = int(np.argmax(arr.shape[start:])) + start
    return axis


def _move_axes(arr: np.ndarray, chain_axis: int | None, draw_axis: int | None) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError("draws array must have at least 2 dimensions")

    working = np.asarray(arr)
    has_chain = chain_axis is not None

    if has_chain:
        working = np.moveaxis(working, chain_axis, 0)
        if draw_axis is not None and draw_axis > chain_axis:
            draw_axis -= 1
        if draw_axis == 0:
            raise ValueError("draw_axis cannot match chain_axis")

    draw_axis = _infer_draw_axis(working, draw_axis, has_chain)
    target_axis = 1 if has_chain else 0
    if draw_axis != target_axis:
        working = np.moveaxis(working, draw_axis, target_axis)

    return working


def ensure_chain_draws(
    arr: np.ndarray,
    chains: int | None = None,
    chain_axis: int | None = None,
    draw_axis: int | None = None,
) -> np.ndarray:
    """Ensure array is shaped (chains, draws, ...)."""
    working = _move_axes(arr, chain_axis, draw_axis)

    if chain_axis is None:
        working = working[None, ...]

    if chains is not None and working.shape[0] != chains:
        if working.shape[0] != 1:
            raise ValueError("chains requested but array already has a chain dimension")
        total_draws = working.shape[1]
        if total_draws % chains != 0:
            raise ValueError("draws not divisible by chains")
        draws_per_chain = total_draws // chains
        working = working.reshape((chains, draws_per_chain) + working.shape[2:])

    return working


def _build_coords_dims(
    omega: np.ndarray,
    beta: np.ndarray,
    labels: list[str] | None = None,
    coef_labels: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coords: dict[str, Any] = {}
    dims: dict[str, Any] = {}

    if omega.ndim >= 3:
        n = omega.shape[-1]
        coords["eq"] = labels or [f"var_{i+1}" for i in range(n)]
        if omega.ndim >= 4:
            dims["omega"] = ["eq", "eq"]
        else:
            dims["omega"] = ["eq"]

    if beta.ndim >= 3:
        if beta.ndim >= 4:
            n = beta.shape[-2]
            k = beta.shape[-1]
            coords.setdefault("eq", labels or [f"var_{i+1}" for i in range(n)])
            coords["coef"] = coef_labels or [f"coef_{i+1}" for i in range(k)]
            dims["beta"] = ["eq", "coef"]
        else:
            k = beta.shape[-1]
            coords["coef"] = coef_labels or [f"coef_{i+1}" for i in range(k)]
            dims["beta"] = ["coef"]

    return coords, dims


def build_inference_data(
    omegas: np.ndarray,
    betas: np.ndarray,
    *,
    chains: int | None = None,
    chain_axis: int | None = None,
    draw_axis: int | None = None,
    labels: list[str] | None = None,
    coef_labels: list[str] | None = None,
) -> az.InferenceData:
    omega = ensure_chain_draws(
        omegas,
        chains=chains,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
    )
    beta = ensure_chain_draws(
        betas,
        chains=chains,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
    )

    coords, dims = _build_coords_dims(omega, beta, labels=labels, coef_labels=coef_labels)

    return az.from_dict(
        posterior={"omega": omega, "beta": beta},
        coords=coords or None,
        dims=dims or None,
    )


def convergence_diagnostics(
    omegas: np.ndarray,
    betas: np.ndarray,
    *,
    chains: int | None = None,
    chain_axis: int | None = None,
    draw_axis: int | None = None,
    labels: list[str] | None = None,
    coef_labels: list[str] | None = None,
    round_to: int = 4,
) -> dict[str, Any]:
    """Compute R-hat, ESS, and summary diagnostics using ArviZ/PyMC."""
    idata = build_inference_data(
        omegas,
        betas,
        chains=chains,
        chain_axis=chain_axis,
        draw_axis=draw_axis,
        labels=labels,
        coef_labels=coef_labels,
    )

    summary_fn = None
    if getattr(pm, "stats", None) and hasattr(pm.stats, "summary"):
        summary_fn = pm.stats.summary
    elif hasattr(pm, "summary"):
        summary_fn = pm.summary
    else:
        summary_fn = az.summary

    summary = summary_fn(idata, round_to=round_to)
    rhat = az.rhat(idata)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")

    return {
        "idata": idata,
        "summary": summary,
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
    }


def sample_posterior_chains(
    Bps: np.ndarray,
    Hs: np.ndarray,
    Ss: np.ndarray,
    vd: int,
    *,
    draws: int,
    chains: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw posterior samples across chains and return stacked and per-chain arrays."""
    if chains < 1:
        raise ValueError("chains must be >= 1")
    if draws < 1:
        raise ValueError("draws must be >= 1")

    n = Bps.shape[1]
    k = Bps.shape[0]

    if chains == 1:
        omegas, betas = sample_posterior(Bps, Hs, Ss, vd, draws, seed=seed)
        omegas_chains = omegas[None, ...]
        betas_chains = betas[None, ...]
    else:
        omegas_chains = np.zeros((chains, draws, n, n))
        betas_chains = np.zeros((chains, draws, n, k))
        for idx, chain_seed in enumerate(_chain_seeds(seed, chains)):
            omegas_chains[idx], betas_chains[idx] = sample_posterior(
                Bps,
                Hs,
                Ss,
                vd,
                draws,
                seed=chain_seed,
            )

    omegas = omegas_chains.reshape(chains * draws, n, n)
    betas = betas_chains.reshape(chains * draws, n, k)
    return omegas, betas, omegas_chains, betas_chains


def run_bvar_pipeline(
    data: np.ndarray,
    *,
    lags: int,
    draws: int,
    chains: int = 1,
    seed: int | None = None,
    bounds: list[tuple[float, float]] | None = None,
    opt_starts: int = 10,
    include_intercept: bool = True,
) -> dict[str, Any]:
    """Fit a BVAR, sample posterior draws, and return arrays plus hyperparameters."""
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if draws < 1:
        raise ValueError("draws must be >= 1")
    if data.ndim != 2:
        raise ValueError("data must be 2D (T x n)")
    if data.shape[0] <= lags:
        raise ValueError("Not enough observations for requested lags")

    Y, X = build_var_design(data, lags, include_intercept=include_intercept)
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
        bounds=bounds,
        n_starts=opt_starts,
        seed=seed,
    )

    lambdas = result.x
    YP, XP = minnesota_dummy_observations(lambdas, delta, su, s0, y_mean, lags)
    Bps, Hs, Ss, vd = posterior_params(Y, X, YP, XP)

    omegas, betas, omegas_chains, betas_chains = sample_posterior_chains(
        Bps,
        Hs,
        Ss,
        vd,
        draws=draws,
        chains=chains,
        seed=seed,
    )

    return {
        "omegas": omegas,
        "betas": betas,
        "omegas_chains": omegas_chains,
        "betas_chains": betas_chains,
        "lambdas": lambdas,
        "vd": vd,
    }


def run_bvar_diagnostics(
    data: np.ndarray,
    *,
    lags: int,
    draws: int,
    chains: int = 2,
    seed: int | None = None,
    bounds: list[tuple[float, float]] | None = None,
    opt_starts: int = 10,
    include_intercept: bool = True,
    labels: list[str] | None = None,
    coef_labels: list[str] | None = None,
    round_to: int = 4,
) -> dict[str, Any]:
    """Run the full pipeline and compute convergence diagnostics."""
    if chains < 2:
        raise ValueError("chains must be >= 2 for convergence diagnostics")
    fit = run_bvar_pipeline(
        data,
        lags=lags,
        draws=draws,
        chains=chains,
        seed=seed,
        bounds=bounds,
        opt_starts=opt_starts,
        include_intercept=include_intercept,
    )

    diagnostics = convergence_diagnostics(
        fit["omegas_chains"],
        fit["betas_chains"],
        chain_axis=0,
        draw_axis=1,
        labels=labels,
        coef_labels=coef_labels,
        round_to=round_to,
    )

    return {**fit, **diagnostics}
