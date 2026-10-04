from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf, MinCovDet
from scipy.stats import chi2

from .debt_portfolio import (
    curve_from_30y_anchor,
    simulate_portfolio_path,
)


# ============================================================
# AR(1) calibration
# ============================================================

def estimate_ar1(
    series: pd.Series | Sequence[float],
    variable_name: str,
) -> dict:
    """
    Estimate:

        x_t = c + rho * x_(t-1) + epsilon_t
    """

    x = (
        pd.Series(series)
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(x) < 10:
        raise ValueError(
            f"Too few observations for {variable_name}: {len(x)}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"{variable_name} contains non-finite observations."
        )

    y = x[1:]
    x_lag = x[:-1]

    X = np.column_stack(
        [
            np.ones(len(x_lag)),
            x_lag,
        ]
    )

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    intercept = float(beta[0])
    rho = float(beta[1])

    fitted = (
        intercept
        + rho * x_lag
    )

    residuals = (
        y
        - fitted
    )

    dof = len(residuals) - 2

    if dof <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom for {variable_name}."
        )

    innovation_std = float(
        np.sqrt(
            np.sum(residuals ** 2)
            / dof
        )
    )

    long_run_mean = (
        intercept / (1 - rho)
        if abs(rho) < 1
        else np.nan
    )

    return {
        "variable": variable_name,
        "n_observations": len(x),
        "intercept": intercept,
        "rho": rho,
        "long_run_mean": long_run_mean,
        "innovation_std": innovation_std,
        "fitted": fitted,
        "residuals": residuals,
    }


def estimate_ar1_system(
    calibration_data: pd.DataFrame,
    variables: Sequence[str],
) -> dict[str, dict]:
    """
    Estimate an AR(1) equation for each supplied variable.
    """

    return {
        variable: estimate_ar1(
            calibration_data[variable],
            variable,
        )
        for variable in variables
    }


# ============================================================
# Residual covariance calibration
# ============================================================

def build_residual_frame(
    calibration_data: pd.DataFrame,
    ar1_results: Mapping[str, dict],
    variables: Sequence[str],
    year_column: str = "year",
) -> pd.DataFrame:
    """
    Build an aligned residual DataFrame.
    """

    residual_data = pd.DataFrame(
        {
            variable: np.asarray(
                ar1_results[variable]["residuals"],
                dtype=float,
            )
            for variable in variables
        }
    )

    residual_data.insert(
        0,
        year_column,
        calibration_data[year_column]
        .iloc[1:]
        .to_numpy(),
    )

    return residual_data


def estimate_ledoit_wolf_correlation(
    residual_data: pd.DataFrame,
    variables: Sequence[str],
) -> np.ndarray:
    """
    Estimate a shrinkage-regularised residual correlation matrix.
    """

    residuals = residual_data[
        list(variables)
    ].astype(float)

    standardized = (
        residuals
        - residuals.mean()
    ) / residuals.std(
        ddof=1
    )

    estimator = LedoitWolf().fit(
        standardized.to_numpy()
    )

    covariance = estimator.covariance_

    std = np.sqrt(
        np.diag(covariance)
    )

    return (
        covariance
        / np.outer(std, std)
    )


def estimate_robust_covariance(
    residual_data: pd.DataFrame,
    variables: Sequence[str],
    random_state: int = 42,
    outlier_confidence: float = 0.975,
) -> dict:
    """
    Estimate ordinary-times innovation covariance with
    Minimum Covariance Determinant.
    """

    X = residual_data[
        list(variables)
    ].to_numpy(
        dtype=float
    )

    mcd = MinCovDet(
        random_state=random_state
    ).fit(
        X
    )

    covariance = mcd.covariance_

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    if not (eigenvalues > 0).all():
        raise ValueError(
            "Robust covariance matrix is not positive definite."
        )

    distances = mcd.mahalanobis(
        X
    )

    threshold = chi2.ppf(
        outlier_confidence,
        df=len(variables),
    )

    return {
        "location": mcd.location_,
        "covariance": covariance,
        "std": np.sqrt(np.diag(covariance)),
        "mahalanobis_sq": distances,
        "stress_observation": distances > threshold,
        "outlier_threshold": threshold,
    }


# ============================================================
# Central-path helpers
# ============================================================

def build_ar1_expected_path(
    initial_value: float,
    long_run_mean: float,
    rho: float,
    years: Iterable[int],
) -> dict[int, float]:
    """
    Generate the no-shock conditional expectation of an AR(1).
    """

    previous = float(initial_value)
    result: dict[int, float] = {}

    for year in years:
        current = (
            long_run_mean
            + rho
            * (
                previous
                - long_run_mean
            )
        )

        result[int(year)] = float(current)
        previous = current

    return result


# ============================================================
# Stochastic path generation
# ============================================================

def draw_empirical_innovation(
    rng: np.random.Generator,
    residual_matrix: np.ndarray,
) -> np.ndarray:
    """
    Draw one historical joint residual vector.
    """

    index = rng.integers(
        low=0,
        high=len(residual_matrix),
    )

    return residual_matrix[
        index
    ].astype(
        float,
        copy=True,
    )


def generate_calibrated_stochastic_path(
    central_path: pd.DataFrame,
    rng: np.random.Generator,
    calibration_variables: Sequence[str],
    persistence: Mapping[str, float],
    robust_covariance_matrix: np.ndarray,
    initial_10s30s_slope: float,
    *,
    innovation_method: str = "robust_gaussian",
    calibrated_covariance_matrix: np.ndarray | None = None,
    residual_matrix: np.ndarray | None = None,
    initial_year: int = 2026,
    first_simulation_year: int = 2027,
) -> pd.DataFrame:
    """
    Generate one stochastic path around a time-varying central path:

        x_t =
            mu_t
            + rho * (x_(t-1) - mu_(t-1))
            + epsilon_t

    Robust-Gaussian innovations are intentionally zero-mean.
    """

    central = (
        central_path
        .set_index("year")
        .copy()
    )

    years = [
        int(year)
        for year in central.index
        if year >= first_simulation_year
    ]

    previous = {
        variable: float(
            central.loc[
                initial_year,
                variable,
            ]
        )
        for variable in calibration_variables
    }

    rows = []

    for year in years:
        previous_year = year - 1

        if innovation_method == "robust_gaussian":
            shocks = rng.multivariate_normal(
                mean=np.zeros(
                    len(calibration_variables)
                ),
                cov=robust_covariance_matrix,
            )

        elif innovation_method == "gaussian":
            if calibrated_covariance_matrix is None:
                raise ValueError(
                    "calibrated_covariance_matrix is required for gaussian mode."
                )

            shocks = rng.multivariate_normal(
                mean=np.zeros(
                    len(calibration_variables)
                ),
                cov=calibrated_covariance_matrix,
            )

        elif innovation_method == "bootstrap":
            if residual_matrix is None:
                raise ValueError(
                    "residual_matrix is required for bootstrap mode."
                )

            shocks = draw_empirical_innovation(
                rng,
                residual_matrix,
            )

        else:
            raise ValueError(
                "innovation_method must be "
                "'robust_gaussian', 'gaussian', or 'bootstrap'."
            )

        current = {}

        for i, variable in enumerate(
            calibration_variables
        ):
            mu_t = float(
                central.loc[
                    year,
                    variable,
                ]
            )

            mu_previous = float(
                central.loc[
                    previous_year,
                    variable,
                ]
            )

            rho = float(
                persistence[variable]
            )

            current[variable] = (
                mu_t
                + rho
                * (
                    previous[variable]
                    - mu_previous
                )
                + shocks[i]
            )

        france_10y = (
            current["bund_10y"]
            + current["france_spread_10y"]
        )

        oat_30y = (
            france_10y
            + float(initial_10s30s_slope)
        )

        nominal_growth = (
            (
                1
                + current["real_growth"]
            )
            *
            (
                1
                + current["inflation"]
            )
            - 1
        )

        rows.append(
            {
                "year": year,
                **current,
                "france_10y": france_10y,
                "oat_30y": oat_30y,
                "nominal_growth": nominal_growth,
            }
        )

        previous = current

    result = pd.DataFrame(
        rows
    )

    if not np.isfinite(
        result.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all():
        raise ValueError(
            "Non-finite values generated in stochastic path."
        )

    return result


# ============================================================
# Yield-curve bridge
# ============================================================

def build_market_curves_from_path(
    path: pd.DataFrame,
    nominal_curve_offsets_to_30y: Mapping[float, float],
    baseline_oat_30y: float,
    baseline_real_euro_curve: Mapping[float, float],
    baseline_real_fr_curve: Mapping[float, float],
) -> dict:
    """
    Translate a path containing oat_30y into annual nominal
    and real issuance curves.
    """

    result = {}

    for _, row in path.iterrows():
        year = int(
            row["year"]
        )

        oat_30y = float(
            row["oat_30y"]
        )

        nominal_curve = curve_from_30y_anchor(
            oat_30y_yield=oat_30y,
            offsets_to_30y=nominal_curve_offsets_to_30y,
        )

        yield_shift = (
            oat_30y
            - baseline_oat_30y
        )

        real_euro_curve = {
            maturity:
                float(rate)
                + yield_shift
            for maturity, rate
            in baseline_real_euro_curve.items()
        }

        real_fr_curve = {
            maturity:
                float(rate)
                + yield_shift
            for maturity, rate
            in baseline_real_fr_curve.items()
        }

        result[year] = {
            "nominal": nominal_curve,
            "real_euro": real_euro_curve,
            "real_fr": real_fr_curve,
        }

    return result


# ============================================================
# General-government central path
# ============================================================

def build_central_gg_path(
    *,
    central_path: pd.DataFrame,
    aft_central_path: pd.DataFrame,
    years: Sequence[int],
    initial_gg_debt_eur: float,
    aft_share_of_initial_gg_debt: float,
    imf_terminal_year: int = 2031,
) -> pd.DataFrame:
    """
    Through imf_terminal_year, use central-path interest/GDP.
    Afterwards, evolve the GG effective rate with changes in
    central AFT effective financing cost.
    """

    central = central_path.set_index(
        "year"
    )

    aft = aft_central_path.set_index(
        "year"
    )

    debt = float(
        initial_gg_debt_eur
    )

    previous_effective_rate = None
    rows = []

    for year in years:
        year = int(year)

        gdp_end = float(
            central.loc[
                year,
                "nominal_gdp_eur",
            ]
        )

        pb_ratio = float(
            central.loc[
                year,
                "primary_balance_gdp",
            ]
        )

        pb_eur = (
            pb_ratio
            * gdp_end
        )

        if year <= imf_terminal_year:
            interest_gdp = float(
                central.loc[
                    year,
                    "interest_gdp_imf",
                ]
            )

            interest_eur = (
                interest_gdp
                * gdp_end
            )

            effective_rate = (
                interest_eur
                / debt
            )

        else:
            aft_rate_change = (
                float(
                    aft.loc[
                        year,
                        "effective_financing_rate",
                    ]
                )
                -
                float(
                    aft.loc[
                        year - 1,
                        "effective_financing_rate",
                    ]
                )
            )

            effective_rate = (
                float(
                    previous_effective_rate
                )
                +
                float(
                    aft_share_of_initial_gg_debt
                )
                * aft_rate_change
            )

            if effective_rate <= 0:
                raise ValueError(
                    f"Central GG effective rate became "
                    f"non-positive in {year}: "
                    f"{effective_rate:.4%}"
                )

            interest_eur = (
                effective_rate
                * debt
            )

            interest_gdp = (
                interest_eur
                / gdp_end
            )

        debt_end = (
            debt
            + interest_eur
            - pb_eur
        )

        rows.append(
            {
                "year": year,
                "gdp_end_eur": gdp_end,
                "debt_start_eur": debt,
                "primary_balance_gdp": pb_ratio,
                "primary_balance_eur": pb_eur,
                "interest_eur": interest_eur,
                "interest_gdp": interest_gdp,
                "effective_interest_rate": effective_rate,
                "debt_end_eur": debt_end,
                "debt_gdp_end": (
                    debt_end
                    / gdp_end
                ),
            }
        )

        debt = debt_end
        previous_effective_rate = (
            effective_rate
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Stochastic general-government DSA
# ============================================================

def simulate_stochastic_gg_dsa(
    *,
    stochastic_path: pd.DataFrame,
    gg_central_path: pd.DataFrame,
    aft_cost_shock: pd.Series | Mapping[int, float],
    years: Sequence[int],
    initial_gdp_eur: float,
    initial_gg_debt_eur: float,
) -> pd.DataFrame:
    """
    Apply the central GG effective rate to the stochastic debt
    stock, then add only the incremental AFT cost deviation.
    """

    stochastic = stochastic_path.set_index(
        "year"
    )

    central_gg = gg_central_path.set_index(
        "year"
    )

    debt = float(
        initial_gg_debt_eur
    )

    gdp = float(
        initial_gdp_eur
    )

    rows = []

    for year in years:
        year = int(year)

        nominal_growth = float(
            stochastic.loc[
                year,
                "nominal_growth",
            ]
        )

        gdp_end = (
            gdp
            * (
                1
                + nominal_growth
            )
        )

        pb_ratio = float(
            stochastic.loc[
                year,
                "primary_balance_gdp",
            ]
        )

        pb_eur = (
            pb_ratio
            * gdp_end
        )

        central_effective_rate = float(
            central_gg.loc[
                year,
                "effective_interest_rate",
            ]
        )

        baseline_interest_on_stock = (
            central_effective_rate
            * debt
        )

        aft_shock_eur = float(
            aft_cost_shock[year]
        )

        total_interest_eur = (
            baseline_interest_on_stock
            + aft_shock_eur
        )

        if total_interest_eur <= 0:
            raise ValueError(
                f"Negative/non-positive GG interest bill "
                f"in {year}: "
                f"€{total_interest_eur / 1e9:.2f}bn"
            )

        implied_effective_rate = (
            total_interest_eur
            / debt
        )

        debt_end = (
            debt
            + total_interest_eur
            - pb_eur
        )

        rows.append(
            {
                "year": year,
                "gdp_start_eur": gdp,
                "gdp_end_eur": gdp_end,
                "nominal_growth": nominal_growth,
                "debt_start_eur": debt,
                "central_effective_rate": central_effective_rate,
                "aft_interest_cost_shock_eur": aft_shock_eur,
                "total_interest_eur": total_interest_eur,
                "implied_effective_rate": implied_effective_rate,
                "interest_gdp": (
                    total_interest_eur
                    / gdp_end
                ),
                "primary_balance_gdp": pb_ratio,
                "primary_balance_eur": pb_eur,
                "debt_end_eur": debt_end,
                "debt_gdp_end": (
                    debt_end
                    / gdp_end
                ),
            }
        )

        gdp = gdp_end
        debt = debt_end

    return pd.DataFrame(
        rows
    )


# ============================================================
# Monte Carlo engine
# ============================================================

def run_monte_carlo(
    *,
    n_simulations: int,
    seed: int,
    central_path: pd.DataFrame,
    gg_central_path: pd.DataFrame,
    aft_central_path: pd.DataFrame,
    portfolio: pd.DataFrame,
    years: Sequence[int],
    initial_gdp_eur: float,
    initial_gg_debt_eur: float,
    start_year: int,
    horizon_years: int,
    issuance_plan: Sequence[dict],
    calibration_variables: Sequence[str],
    persistence: Mapping[str, float],
    robust_covariance_matrix: np.ndarray,
    initial_10s30s_slope: float,
    nominal_curve_offsets_to_30y: Mapping[float, float],
    baseline_oat_30y: float,
    baseline_real_euro_curve: Mapping[float, float],
    baseline_real_fr_curve: Mapping[float, float],
    innovation_method: str = "robust_gaussian",
    calibrated_covariance_matrix: np.ndarray | None = None,
    residual_matrix: np.ndarray | None = None,
    progress_every: int = 500,
) -> dict:
    """
    Run the complete stochastic AFT -> GG Monte Carlo DSA.

    The reference portfolio is always aft_central_path.
    """

    years_array = np.asarray(
        years,
        dtype=int,
    )

    n_years = len(
        years_array
    )

    rng = np.random.default_rng(
        seed
    )

    result_names = [
        "real_growth",
        "inflation",
        "nominal_growth",
        "primary_balance_gdp",
        "bund_10y",
        "france_spread_10y",
        "oat_30y",
        "aft_effective_financing_rate",
        "aft_cost_shock_eur",
        "gg_effective_rate",
        "interest_gdp",
        "debt_gdp",
    ]

    results = {
        name: np.empty(
            (
                n_simulations,
                n_years,
            ),
            dtype=float,
        )
        for name in result_names
    }

    central_aft_cost = (
        aft_central_path[
            "total_financing_cost_eur"
        ]
        .to_numpy(
            dtype=float
        )
    )

    for simulation in range(
        n_simulations
    ):
        try:
            stochastic_simulation = (
                generate_calibrated_stochastic_path(
                    central_path=central_path,
                    rng=rng,
                    calibration_variables=calibration_variables,
                    persistence=persistence,
                    robust_covariance_matrix=robust_covariance_matrix,
                    initial_10s30s_slope=initial_10s30s_slope,
                    innovation_method=innovation_method,
                    calibrated_covariance_matrix=calibrated_covariance_matrix,
                    residual_matrix=residual_matrix,
                )
            )

            stochastic_indexed = (
                stochastic_simulation
                .set_index(
                    "year"
                )
            )

            market_curves = (
                build_market_curves_from_path(
                    path=stochastic_simulation,
                    nominal_curve_offsets_to_30y=nominal_curve_offsets_to_30y,
                    baseline_oat_30y=baseline_oat_30y,
                    baseline_real_euro_curve=baseline_real_euro_curve,
                    baseline_real_fr_curve=baseline_real_fr_curve,
                )
            )

            aft_simulation, _ = (
                simulate_portfolio_path(
                    initial_portfolio=portfolio,
                    initial_gdp_eur=initial_gdp_eur,
                    start_year=start_year,
                    years=horizon_years,
                    nominal_growth=(
                        stochastic_indexed[
                            "nominal_growth"
                        ]
                        .to_dict()
                    ),
                    primary_balance_ratio=(
                        stochastic_indexed[
                            "primary_balance_gdp"
                        ]
                        .to_dict()
                    ),
                    market_curves=market_curves,
                    issuance_plan=issuance_plan,
                    french_inflation=(
                        stochastic_indexed[
                            "inflation"
                        ]
                        .to_dict()
                    ),
                    euro_inflation=(
                        stochastic_indexed[
                            "inflation"
                        ]
                        .to_dict()
                    ),
                )
            )

            aft_cost_shock = pd.Series(
                (
                    aft_simulation[
                        "total_financing_cost_eur"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    -
                    central_aft_cost
                ),
                index=years,
            )

            gg_simulation = (
                simulate_stochastic_gg_dsa(
                    stochastic_path=stochastic_simulation,
                    gg_central_path=gg_central_path,
                    aft_cost_shock=aft_cost_shock,
                    years=years,
                    initial_gdp_eur=initial_gdp_eur,
                    initial_gg_debt_eur=initial_gg_debt_eur,
                )
            )

        except Exception as exc:
            raise RuntimeError(
                "Monte Carlo simulation failed at "
                f"path {simulation + 1:,} of "
                f"{n_simulations:,}."
            ) from exc

        for variable in [
            "real_growth",
            "inflation",
            "nominal_growth",
            "primary_balance_gdp",
            "bund_10y",
            "france_spread_10y",
            "oat_30y",
        ]:
            results[
                variable
            ][
                simulation,
                :,
            ] = (
                stochastic_simulation[
                    variable
                ]
                .to_numpy(
                    dtype=float
                )
            )

        results[
            "aft_effective_financing_rate"
        ][
            simulation,
            :,
        ] = (
            aft_simulation[
                "effective_financing_rate"
            ]
            .to_numpy(
                dtype=float
            )
        )

        results[
            "aft_cost_shock_eur"
        ][
            simulation,
            :,
        ] = (
            aft_cost_shock
            .to_numpy(
                dtype=float
            )
        )

        results[
            "gg_effective_rate"
        ][
            simulation,
            :,
        ] = (
            gg_simulation[
                "implied_effective_rate"
            ]
            .to_numpy(
                dtype=float
            )
        )

        results[
            "interest_gdp"
        ][
            simulation,
            :,
        ] = (
            gg_simulation[
                "interest_gdp"
            ]
            .to_numpy(
                dtype=float
            )
        )

        results[
            "debt_gdp"
        ][
            simulation,
            :,
        ] = (
            gg_simulation[
                "debt_gdp_end"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if (
            progress_every
            and
            (
                simulation
                + 1
            )
            % progress_every
            == 0
        ):
            print(
                f"Completed "
                f"{simulation + 1:,} / "
                f"{n_simulations:,}"
            )

    for name, values in (
        results.items()
    ):
        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite values found in Monte Carlo output: {name}"
            )

    return {
        "years": years_array,
        "n_simulations": int(n_simulations),
        "seed": int(seed),
        "innovation_method": innovation_method,
        **results,
    }


# ============================================================
# Monte Carlo summaries
# ============================================================

def percentile_frame(
    values: np.ndarray,
    years: Sequence[int],
    percentiles: Sequence[int] = (
        10,
        25,
        50,
        75,
        90,
    ),
) -> pd.DataFrame:
    """
    Convert a (simulation, year) array into percentile paths.
    """

    percentile_values = np.percentile(
        values,
        percentiles,
        axis=0,
    )

    result = pd.DataFrame({
        "year": list(years),
    })

    for percentile, row in zip(
        percentiles,
        percentile_values,
    ):
        result[
            f"p{percentile}"
        ] = row

    return result


def risk_probability_table(
    mc: Mapping[str, np.ndarray],
    initial_debt_gdp: float,
) -> pd.DataFrame:
    """
    Produce the headline tail-risk probability table.
    """

    debt = np.asarray(
        mc["debt_gdp"],
        dtype=float,
    )

    interest = np.asarray(
        mc["interest_gdp"],
        dtype=float,
    )

    oat = np.asarray(
        mc["oat_30y"],
        dtype=float,
    )

    terminal_debt = (
        debt[
            :,
            -1,
        ]
    )

    terminal_oat = (
        oat[
            :,
            -1,
        ]
    )

    initial_debt_column = np.full(
        (
            debt.shape[0],
            1,
        ),
        float(
            initial_debt_gdp
        ),
    )

    debt_with_initial = np.concatenate(
        [
            initial_debt_column,
            debt,
        ],
        axis=1,
    )

    five_year_changes = (
        debt_with_initial[
            :,
            5:,
        ]
        -
        debt_with_initial[
            :,
            :-5,
        ]
    )

    result = pd.DataFrame({
        "metric": [
            "Debt/GDP > 130% in terminal year",
            "Debt/GDP > 150% in terminal year",
            "Debt/GDP > 175% in terminal year",
            "Debt/GDP above initial level in terminal year",
            "Debt rises >20 pp within any 5-year window",
            "Interest/GDP exceeds 4% at least once",
            "Interest/GDP exceeds 5% at least once",
            "30Y OAT exceeds 6% at least once",
            "30Y OAT exceeds 7% at least once",
            "30Y OAT > 6% in terminal year",
            "30Y OAT > 7% in terminal year",
        ],

        "probability": [
            np.mean(
                terminal_debt > 1.30
            ),
            np.mean(
                terminal_debt > 1.50
            ),
            np.mean(
                terminal_debt > 1.75
            ),
            np.mean(
                terminal_debt > initial_debt_gdp
            ),
            np.mean(
                np.any(
                    five_year_changes > 0.20,
                    axis=1,
                )
            ),
            np.mean(
                np.any(
                    interest > 0.04,
                    axis=1,
                )
            ),
            np.mean(
                np.any(
                    interest > 0.05,
                    axis=1,
                )
            ),
            np.mean(
                np.any(
                    oat > 0.06,
                    axis=1,
                )
            ),
            np.mean(
                np.any(
                    oat > 0.07,
                    axis=1,
                )
            ),
            np.mean(
                terminal_oat > 0.06
            ),
            np.mean(
                terminal_oat > 0.07
            ),
        ],
    })

    result[
        "probability_pct"
    ] = (
        result[
            "probability"
        ]
        * 100
    )

    return result
