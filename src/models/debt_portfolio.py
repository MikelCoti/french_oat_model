
"""
French sovereign debt portfolio engine.

This module models the negotiable French State debt portfolio managed by
Agence France Trésor (AFT): nominal OATs, OATi, OAT€i and BTFs.

Important scope distinction
---------------------------
The AFT portfolio is NOT the same thing as France's full Maastricht /
general-government debt stock. This module is primarily a refinancing and
interest-cost engine. Its output can be fed into a broader general-government
DSA, but portfolio_debt_gdp should not be interpreted as the official
general-government debt/GDP ratio.

Main conventions
----------------
* outstanding_eur is FACE / PAR principal.
* OATi/OAT€i liability principal = face * index_coefficient.
* OATi/OAT€i redemption has a par floor:
      face * max(index_coefficient, 1)
* OATi/OAT€i coupon cash flow uses indexed principal.
* Existing BTFs redeem at par and have no coupon.
* New simulated BTFs are issued at a discount on an ACT/360 approximation:
      issue_price = 1 - discount_rate * term_days / 360
* New synthetic OAT/OATi/OAT€i issuance is assumed to be issued at par.
* This is an annual DSA engine, not a daily Treasury cash-management model.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# Types / conventions
# ============================================================

YieldCurve = Dict[float, float]
MarketCurves = Dict[str, YieldCurve]
MarketCurvePath = Dict[int, MarketCurves]

ScalarPath = Union[
    float,
    int,
    Dict[int, float],
]

IssuancePlan = List[dict]

INDEXED_EURO_TYPES = {"OATEI"}
INDEXED_FR_TYPES = {"OATI"}
INDEXED_TYPES = INDEXED_EURO_TYPES | INDEXED_FR_TYPES
BTF_TYPES = {"BTF"}
NOMINAL_OAT_TYPES = {"OAT"}

VALID_SECURITY_TYPES = (
    NOMINAL_OAT_TYPES
    | INDEXED_TYPES
    | BTF_TYPES
    | {"SYNTHETIC"}
)


# ============================================================
# General utilities
# ============================================================

def normalize_security_type(value: str) -> str:
    """
    Normalize AFT security-type labels.

    Examples
    --------
    OAT€i -> OATEI
    OATÉI -> OATEI
    """
    text = str(value).strip().upper()
    text = text.replace("€", "E")
    text = text.replace("OATÉI", "OATEI")
    text = text.replace("OAT€I", "OATEI")
    return text


def get_scalar_for_year(
    value,
    year: int,
    name: str = "value",
) -> float:
    """
    Accept:
    - a constant scalar;
    - a {year: value} mapping;
    - a pandas Series indexed by year.
    """

    # --------------------------------------------------------
    # Constant scalar
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            float,
            int,
            np.floating,
            np.integer,
        ),
    ):
        result = float(
            value
        )

    # --------------------------------------------------------
    # Pandas Series
    # --------------------------------------------------------

    elif isinstance(
        value,
        pd.Series,
    ):

        if year not in value.index:
            raise KeyError(
                f"No {name} supplied "
                f"for simulation year {year}."
            )

        result = float(
            value.loc[year]
        )

    # --------------------------------------------------------
    # Dictionary / Mapping
    # --------------------------------------------------------

    elif isinstance(
        value,
        Mapping,
    ):

        if year not in value:
            raise KeyError(
                f"No {name} supplied "
                f"for simulation year {year}."
            )

        result = float(
            value[year]
        )

    else:

        raise TypeError(
            f"{name} must be a scalar, "
            "pandas Series, or year -> value mapping."
        )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not np.isfinite(
        result
    ):
        raise ValueError(
            f"{name} for {year} "
            f"is not finite: {result}"
        )

    return result


# ============================================================
# Loading / validation
# ============================================================

def load_portfolio(
    path: str,
    snapshot_date: str | pd.Timestamp | None = None,
    strict_indexation: bool = True,
    validate_aft_indexed_total: bool = True,
    indexed_total_rtol: float = 1e-6,
) -> pd.DataFrame:
    """
    Load the validated CSV produced by aft.py.

    Expected minimum columns
    ------------------------
    isin
    security_type
    coupon_rate
    maturity_date
    outstanding_eur

    For OATi/OAT€i:
    index_coefficient is required when strict_indexation=True.

    If indexed_outstanding_eur is present (as produced by aft.py), it is
    cross-checked against:

        outstanding_eur * index_coefficient

    and then recomputed canonically inside this engine.
    """
    df = pd.read_csv(
        path,
        parse_dates=[
            col
            for col in ["maturity_date", "coefficient_date"]
            if col in pd.read_csv(path, nrows=0).columns
        ],
    )

    return prepare_portfolio(
        df=df,
        snapshot_date=snapshot_date,
        strict_indexation=strict_indexation,
        validate_aft_indexed_total=validate_aft_indexed_total,
        indexed_total_rtol=indexed_total_rtol,
    )


def prepare_portfolio(
    df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
    strict_indexation: bool = True,
    validate_aft_indexed_total: bool = True,
    indexed_total_rtol: float = 1e-6,
) -> pd.DataFrame:
    """
    Standardize and validate a security-level AFT debt portfolio.
    """
    df = df.copy()

    required = {
        "isin",
        "security_type",
        "coupon_rate",
        "maturity_date",
        "outstanding_eur",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Portfolio is missing required columns: {sorted(missing)}"
        )

    if "description" not in df.columns:
        df["description"] = pd.NA

    if "inflation_index" not in df.columns:
        df["inflation_index"] = pd.NA

    if "index_coefficient" not in df.columns:
        df["index_coefficient"] = np.nan

    df["security_type"] = (
        df["security_type"]
        .map(normalize_security_type)
    )

    unknown_types = (
        set(df["security_type"].dropna().unique())
        - VALID_SECURITY_TYPES
    )

    if unknown_types:
        raise ValueError(
            f"Unknown security types: {sorted(unknown_types)}"
        )

    df["maturity_date"] = pd.to_datetime(
        df["maturity_date"],
        errors="coerce",
    )

    df["coupon_rate"] = pd.to_numeric(
        df["coupon_rate"],
        errors="coerce",
    )

    df["outstanding_eur"] = pd.to_numeric(
        df["outstanding_eur"],
        errors="coerce",
    )

    df["index_coefficient"] = pd.to_numeric(
        df["index_coefficient"],
        errors="coerce",
    )

    # ----------------------------
    # Fundamental field validation
    # ----------------------------

    if df["isin"].isna().any():
        raise ValueError("Portfolio contains NULL ISINs.")

    duplicated_isins = df.loc[
        df["isin"].duplicated(keep=False),
        ["isin", "security_type", "maturity_date"],
    ]

    if not duplicated_isins.empty:
        raise ValueError(
            "Duplicate ISINs found:\n"
            f"{duplicated_isins.to_string(index=False)}"
        )

    if df["maturity_date"].isna().any():
        bad = df.loc[
            df["maturity_date"].isna(),
            ["isin", "security_type", "description"],
        ]

        raise ValueError(
            "Invalid maturity dates:\n"
            f"{bad.to_string(index=False)}"
        )

    if df["coupon_rate"].isna().any():
        bad = df.loc[
            df["coupon_rate"].isna(),
            ["isin", "security_type", "description"],
        ]

        raise ValueError(
            "Invalid coupon rates:\n"
            f"{bad.to_string(index=False)}"
        )

    if df["outstanding_eur"].isna().any():
        bad = df.loc[
            df["outstanding_eur"].isna(),
            ["isin", "security_type", "outstanding_eur"],
        ]

        raise ValueError(
            "Invalid outstanding amounts:\n"
            f"{bad.to_string(index=False)}"
        )

    if (df["outstanding_eur"] < 0).any():
        raise ValueError(
            "Outstanding principal cannot be negative."
        )

    # Zero-outstanding AFT placeholders are not debt.
    df = df.loc[
        df["outstanding_eur"] > 0
    ].copy()

    # Snapshot filtering.
    if snapshot_date is not None:
        snapshot_date = pd.Timestamp(snapshot_date)

        df = df.loc[
            df["maturity_date"] >= snapshot_date
        ].copy()

    # ----------------------------
    # Indexation
    # ----------------------------

    indexed_mask = (
        df["security_type"]
        .isin(INDEXED_TYPES)
    )

    nominal_mask = ~indexed_mask

    df.loc[
        nominal_mask,
        "index_coefficient",
    ] = 1.0

    if strict_indexation:
        missing_coeff = (
            indexed_mask
            & df["index_coefficient"].isna()
        )

        if missing_coeff.any():
            missing_isins = (
                df.loc[
                    missing_coeff,
                    "isin",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Missing index_coefficient for indexed securities: "
                f"{missing_isins}"
            )

    else:
        df.loc[
            indexed_mask
            & df["index_coefficient"].isna(),
            "index_coefficient",
        ] = 1.0

    if (
        df["index_coefficient"]
        .dropna()
        .le(0)
        .any()
    ):
        raise ValueError(
            "All index coefficients must be positive."
        )

    # Validate aft.py's precomputed indexed amount if present.
    if (
        validate_aft_indexed_total
        and "indexed_outstanding_eur" in df.columns
    ):
        supplied = pd.to_numeric(
            df["indexed_outstanding_eur"],
            errors="coerce",
        )

        expected = df["outstanding_eur"].astype(float).copy()

        expected.loc[indexed_mask] = (
            df.loc[
                indexed_mask,
                "outstanding_eur",
            ].astype(float)
            * df.loc[
                indexed_mask,
                "index_coefficient",
            ].astype(float)
        )

        valid_comparison = supplied.notna()

        if valid_comparison.any():
            mismatch = ~np.isclose(
                supplied.loc[valid_comparison],
                expected.loc[valid_comparison],
                rtol=indexed_total_rtol,
                atol=1.0,
            )

            if mismatch.any():
                mismatch_index = mismatch.index[mismatch]

                bad = df.loc[
                    mismatch_index,
                    [
                        "isin",
                        "security_type",
                        "outstanding_eur",
                        "index_coefficient",
                    ],
                ].copy()

                bad["aft_indexed_outstanding_eur"] = (
                    supplied.loc[mismatch_index]
                )

                bad["engine_expected_indexed_eur"] = (
                    expected.loc[mismatch_index]
                )

                raise ValueError(
                    "aft.py indexed_outstanding_eur does not agree "
                    "with face * index_coefficient:\n"
                    f"{bad.to_string(index=False)}"
                )

    df["maturity_year"] = (
        df["maturity_date"].dt.year
    )

    if snapshot_date is not None:
        df["years_to_maturity"] = (
            df["maturity_date"]
            - snapshot_date
        ).dt.days / 365.25

    df = refresh_derived_columns(df)

    return df.reset_index(drop=True)


# ============================================================
# Security-level cash-flow mechanics
# ============================================================

def indexed_principal_series(
    portfolio: pd.DataFrame,
) -> pd.Series:
    """
    Current liability principal.

    Nominal OAT / BTF:
        face value

    OATi / OAT€i:
        face * index coefficient
    """
    types = (
        portfolio["security_type"]
        .map(normalize_security_type)
    )

    indexed_mask = types.isin(
        INDEXED_TYPES
    )

    principal = (
        portfolio["outstanding_eur"]
        .astype(float)
        .copy()
    )

    principal.loc[indexed_mask] = (
        portfolio.loc[
            indexed_mask,
            "outstanding_eur",
        ].astype(float)
        * portfolio.loc[
            indexed_mask,
            "index_coefficient",
        ].astype(float)
    )

    return principal


def redemption_value_series(
    portfolio: pd.DataFrame,
) -> pd.Series:
    """
    Cash redemption amount at maturity.

    Indexed bonds:
        face * max(index coefficient, 1)

    This implements the par floor.
    """
    types = (
        portfolio["security_type"]
        .map(normalize_security_type)
    )

    indexed_mask = types.isin(
        INDEXED_TYPES
    )

    redemption = (
        portfolio["outstanding_eur"]
        .astype(float)
        .copy()
    )

    if indexed_mask.any():
        floored_coeff = (
            portfolio.loc[
                indexed_mask,
                "index_coefficient",
            ]
            .astype(float)
            .clip(lower=1.0)
        )

        redemption.loc[indexed_mask] = (
            portfolio.loc[
                indexed_mask,
                "outstanding_eur",
            ].astype(float)
            * floored_coeff
        )

    return redemption


def coupon_cash_series(
    portfolio: pd.DataFrame,
) -> pd.Series:
    """
    Annual cash coupon.

    Nominal OAT:
        face * coupon

    OATi / OAT€i:
        indexed principal * real coupon

    BTF:
        zero coupon
    """
    types = (
        portfolio["security_type"]
        .map(normalize_security_type)
    )

    indexed_mask = types.isin(
        INDEXED_TYPES
    )

    btf_mask = types.isin(
        BTF_TYPES
    )

    coupon = (
        portfolio["outstanding_eur"]
        .astype(float)
        * portfolio["coupon_rate"]
        .fillna(0.0)
        .astype(float)
    )

    coupon.loc[indexed_mask] = (
        portfolio.loc[
            indexed_mask,
            "outstanding_eur",
        ].astype(float)
        * portfolio.loc[
            indexed_mask,
            "index_coefficient",
        ].astype(float)
        * portfolio.loc[
            indexed_mask,
            "coupon_rate",
        ].astype(float)
    )

    coupon.loc[btf_mask] = 0.0

    return coupon


def refresh_derived_columns(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalculate all portfolio accounting fields.
    """
    df = portfolio.copy()

    df["liability_principal_eur"] = (
        indexed_principal_series(df)
    )

    df["indexed_outstanding_eur"] = (
        df["liability_principal_eur"]
    )

    df["redemption_value_eur"] = (
        redemption_value_series(df)
    )

    df["annual_coupon_eur"] = (
        coupon_cash_series(df)
    )

    return df


# ============================================================
# Portfolio diagnostics
# ============================================================

def total_face_value(
    portfolio: pd.DataFrame,
) -> float:
    return float(
        portfolio["outstanding_eur"].sum()
    )


def total_debt(
    portfolio: pd.DataFrame,
) -> float:
    """
    Indexed liability principal of the AFT portfolio.
    """
    return float(
        indexed_principal_series(
            portfolio
        ).sum()
    )


def annual_coupon_expense(
    portfolio: pd.DataFrame,
) -> float:
    """
    Annual coupon cash expense.

    BTF discount cost is not a coupon and is therefore excluded.
    """
    return float(
        coupon_cash_series(
            portfolio
        ).sum()
    )


def portfolio_coupon_rate(
    portfolio: pd.DataFrame,
) -> float:
    """
    Coupon cash expense / current indexed liability principal.
    """
    debt = total_debt(portfolio)

    if debt <= 0:
        return 0.0

    return (
        annual_coupon_expense(portfolio)
        / debt
    )


def maturity_schedule(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate redemption exposure by maturity year.
    """
    df = refresh_derived_columns(
        portfolio
    )

    if "maturity_year" not in df.columns:
        df["maturity_year"] = (
            df["maturity_date"].dt.year
        )

    return (
        df.groupby(
            "maturity_year"
        )
        .agg(
            face_value_eur=(
                "outstanding_eur",
                "sum",
            ),
            liability_principal_eur=(
                "liability_principal_eur",
                "sum",
            ),
            redemption_value_eur=(
                "redemption_value_eur",
                "sum",
            ),
            coupon_eur=(
                "annual_coupon_eur",
                "sum",
            ),
            securities=(
                "isin",
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            "maturity_year"
        )
        .reset_index(
            drop=True
        )
    )


def portfolio_summary(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact summary by security type.
    """
    df = refresh_derived_columns(
        portfolio
    )

    return (
        df.groupby(
            "security_type",
            dropna=False,
        )
        .agg(
            securities=(
                "isin",
                "count",
            ),
            face_value_eur=(
                "outstanding_eur",
                "sum",
            ),
            liability_principal_eur=(
                "liability_principal_eur",
                "sum",
            ),
            annual_coupon_eur=(
                "annual_coupon_eur",
                "sum",
            ),
        )
        .reset_index()
    )


# ============================================================
# Inflation / linker indexation
# ============================================================

def apply_annual_indexation(
    portfolio: pd.DataFrame,
    french_inflation: float,
    euro_inflation: float,
) -> Tuple[pd.DataFrame, float]:
    """
    Annual approximation to OATi/OAT€i indexation.

    OATi:
        coefficient *= 1 + French CPIxT inflation

    OAT€i:
        coefficient *= 1 + euro-area HICPxT inflation

    Returns
    -------
    updated_portfolio
    indexation_accrual_eur

    Notes
    -----
    AFT's actual contractual coefficient is daily and uses a lagged
    interpolated reference index. The annual DSA intentionally abstracts
    from that intra-year timing after the initial AFT coefficient.
    """
    if french_inflation <= -1.0:
        raise ValueError(
            "french_inflation must be greater than -100%."
        )

    if euro_inflation <= -1.0:
        raise ValueError(
            "euro_inflation must be greater than -100%."
        )

    df = portfolio.copy()

    debt_before = total_debt(df)

    types = (
        df["security_type"]
        .map(normalize_security_type)
    )

    fr_mask = types.isin(
        INDEXED_FR_TYPES
    )

    euro_mask = types.isin(
        INDEXED_EURO_TYPES
    )

    df.loc[
        fr_mask,
        "index_coefficient",
    ] *= (
        1.0 + french_inflation
    )

    df.loc[
        euro_mask,
        "index_coefficient",
    ] *= (
        1.0 + euro_inflation
    )

    df = refresh_derived_columns(
        df
    )

    debt_after = total_debt(df)

    indexation_accrual = (
        debt_after
        - debt_before
    )

    return (
        df,
        float(indexation_accrual),
    )


# ============================================================
# Yield curves
# ============================================================

def validate_yield_curve(
    yield_curve: YieldCurve,
) -> None:
    if not isinstance(
        yield_curve,
        Mapping,
    ):
        raise TypeError(
            "Each yield curve must be a mapping."
        )

    if not yield_curve:
        raise ValueError(
            "Yield curve cannot be empty."
        )

    for maturity, rate in yield_curve.items():
        maturity = float(maturity)
        rate = float(rate)

        if maturity <= 0:
            raise ValueError(
                f"Curve maturity must be positive: {maturity}"
            )

        if not np.isfinite(rate):
            raise ValueError(
                f"Yield for {maturity}Y is not finite."
            )

        if rate <= -0.05 or rate >= 0.30:
            raise ValueError(
                f"Yield for {maturity}Y looks implausible: "
                f"{rate:.2%}"
            )


def _sorted_curve_arrays(
    yield_curve: YieldCurve,
) -> Tuple[np.ndarray, np.ndarray]:
    validate_yield_curve(
        yield_curve
    )

    items = sorted(
        (
            float(maturity),
            float(rate),
        )
        for maturity, rate
        in yield_curve.items()
    )

    maturities = np.array(
        [x[0] for x in items],
        dtype=float,
    )

    rates = np.array(
        [x[1] for x in items],
        dtype=float,
    )

    return maturities, rates


def curve_rate(
    yield_curve: YieldCurve,
    maturity_years: float,
) -> float:
    """
    Linearly interpolate the curve.

    Extrapolation is intentionally prohibited.
    """
    (
        maturities,
        rates,
    ) = _sorted_curve_arrays(
        yield_curve
    )

    maturity_years = float(
        maturity_years
    )

    if (
        maturity_years
        < maturities.min()
        or maturity_years
        > maturities.max()
    ):
        raise ValueError(
            f"{maturity_years}Y is outside supplied curve range "
            f"[{maturities.min()}, {maturities.max()}]."
        )

    return float(
        np.interp(
            maturity_years,
            maturities,
            rates,
        )
    )


def validate_market_curves(
    curves: MarketCurves,
) -> None:
    if not isinstance(
        curves,
        Mapping,
    ):
        raise TypeError(
            "market_curves must be a mapping."
        )

    if "nominal" not in curves:
        raise ValueError(
            "market_curves requires a 'nominal' curve."
        )

    for curve in curves.values():
        validate_yield_curve(
            curve
        )


def get_market_curves_for_year(
    market_curves: Union[
        MarketCurves,
        MarketCurvePath,
    ],
    year: int,
) -> MarketCurves:
    """
    Accept either static curves:

        {
            "nominal": {...},
            "real_euro": {...},
            "real_fr": {...},
        }

    or a path:

        {
            2027: {"nominal": {...}, ...},
            2028: {"nominal": {...}, ...},
        }
    """
    if not isinstance(
        market_curves,
        Mapping,
    ) or not market_curves:
        raise ValueError(
            "market_curves must be a non-empty mapping."
        )

    first_key = next(
        iter(market_curves.keys())
    )

    if isinstance(
        first_key,
        str,
    ):
        curves = dict(
            market_curves
        )

    else:
        if year not in market_curves:
            raise KeyError(
                f"No market curves supplied for {year}."
            )

        curves = dict(
            market_curves[year]
        )

    validate_market_curves(
        curves
    )

    return curves


def default_curve_key(
    security_type: str,
) -> str:
    security_type = normalize_security_type(
        security_type
    )

    if security_type == "OATEI":
        return "real_euro"

    if security_type == "OATI":
        return "real_fr"

    return "nominal"


# ============================================================
# Issuance plan
# ============================================================

def validate_issuance_plan(
    issuance_plan: IssuancePlan,
    curves: MarketCurves | None = None,
) -> None:
    if not isinstance(
        issuance_plan,
        list,
    ) or not issuance_plan:
        raise ValueError(
            "issuance_plan must be a non-empty list."
        )

    weights = []

    for entry in issuance_plan:
        required = {
            "security_type",
            "maturity_years",
            "weight",
        }

        missing = required.difference(
            entry.keys()
        )

        if missing:
            raise ValueError(
                "Issuance-plan entry missing fields: "
                f"{sorted(missing)}"
            )

        security_type = normalize_security_type(
            entry["security_type"]
        )

        if security_type not in (
            NOMINAL_OAT_TYPES
            | INDEXED_TYPES
            | BTF_TYPES
        ):
            raise ValueError(
                f"Unsupported issuance security type: {security_type}"
            )

        maturity = float(
            entry["maturity_years"]
        )

        weight = float(
            entry["weight"]
        )

        if maturity <= 0:
            raise ValueError(
                "Issuance maturity must be positive."
            )

        if weight < 0:
            raise ValueError(
                "Issuance weights cannot be negative."
            )

        if (
            security_type == "BTF"
            and maturity > 1.0
        ):
            raise ValueError(
                "BTF maturity cannot exceed one year."
            )

        weights.append(
            weight
        )

        if curves is not None:
            curve_key = entry.get(
                "curve_key",
                default_curve_key(
                    security_type
                ),
            )

            if curve_key not in curves:
                raise KeyError(
                    f"Curve {curve_key!r} required by "
                    f"{security_type} issuance is missing."
                )

            # Also validates interpolation coverage.
            curve_rate(
                curves[curve_key],
                maturity,
            )

    total = sum(
        weights
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Issuance-plan weights must sum to 1.0. "
            f"Current total: {total:.10f}"
        )


def nominal_only_plan_from_weights(
    issuance_weights: Mapping[float, float],
    btf_maturity: float | None = None,
) -> IssuancePlan:
    """
    Convenience helper for the older {maturity: weight} format.
    """
    plan = []

    for maturity, weight in (
        issuance_weights.items()
    ):
        security_type = (
            "BTF"
            if (
                btf_maturity is not None
                and np.isclose(
                    float(maturity),
                    float(btf_maturity),
                )
            )
            else "OAT"
        )

        plan.append(
            {
                "security_type": security_type,
                "maturity_years": float(maturity),
                "weight": float(weight),
                "curve_key": "nominal",
            }
        )

    return plan


def issuance_weighted_rate(
    market_curves: MarketCurves,
    issuance_plan: IssuancePlan,
) -> float:
    """
    Weighted average quoted yield across the issuance plan.

    This is a market-rate diagnostic, not an accounting effective rate.
    """
    validate_market_curves(
        market_curves
    )

    validate_issuance_plan(
        issuance_plan,
        market_curves,
    )

    total = 0.0

    for entry in issuance_plan:
        security_type = normalize_security_type(
            entry["security_type"]
        )

        curve_key = entry.get(
            "curve_key",
            default_curve_key(
                security_type
            ),
        )

        rate = curve_rate(
            market_curves[curve_key],
            float(
                entry["maturity_years"]
            ),
        )

        total += (
            float(entry["weight"])
            * rate
        )

    return float(total)


# ============================================================
# BTF mechanics
# ============================================================

def btf_issue_price(
    annual_discount_rate: float,
    term_days: int,
) -> float:
    """
    ACT/360 discount-price approximation.

        price = 1 - discount_rate * days / 360

    Price is per €1 redeemed at par.
    """
    annual_discount_rate = float(
        annual_discount_rate
    )

    if term_days <= 0:
        raise ValueError(
            "term_days must be positive."
        )

    price = (
        1.0
        - annual_discount_rate
        * term_days
        / 360.0
    )

    if price <= 0:
        raise ValueError(
            "BTF assumptions imply a non-positive issue price."
        )

    return float(
        price
    )


def btf_face_for_cash_proceeds(
    cash_proceeds_eur: float,
    annual_discount_rate: float,
    term_days: int,
) -> Tuple[float, float, float]:
    """
    Face value required to raise a desired amount of cash.

    Returns
    -------
    face_value_eur
    issue_price
    discount_cost_eur
    """
    price = btf_issue_price(
        annual_discount_rate,
        term_days,
    )

    face = (
        float(cash_proceeds_eur)
        / price
    )

    discount_cost = (
        face
        - float(cash_proceeds_eur)
    )

    return (
        float(face),
        float(price),
        float(discount_cost),
    )


# ============================================================
# New issuance
# ============================================================

def create_new_issuance(
    total_cash_proceeds_eur: float,
    issue_year: int,
    market_curves: MarketCurves,
    issuance_plan: IssuancePlan,
) -> Tuple[pd.DataFrame, dict]:
    """
    Allocate a required amount of CASH financing across an issuance plan.

    Nominal OAT:
        assumed issued at par; coupon = scenario nominal yield.

    OATi / OAT€i:
        assumed issued at par with coefficient 1;
        coupon = scenario real yield.

    BTF:
        issued at a discount; face value > cash proceeds.
    """
    total_cash_proceeds_eur = float(
        total_cash_proceeds_eur
    )

    if total_cash_proceeds_eur < 0:
        raise ValueError(
            "total_cash_proceeds_eur cannot be negative."
        )

    validate_market_curves(
        market_curves
    )

    validate_issuance_plan(
        issuance_plan,
        market_curves,
    )

    rows = []

    total_face = 0.0
    total_btf_discount = 0.0

    for sequence, entry in enumerate(
        issuance_plan,
        start=1,
    ):
        security_type = normalize_security_type(
            entry["security_type"]
        )

        maturity_years = float(
            entry["maturity_years"]
        )

        weight = float(
            entry["weight"]
        )

        cash = (
            total_cash_proceeds_eur
            * weight
        )

        if cash <= 0:
            continue

        curve_key = entry.get(
            "curve_key",
            default_curve_key(
                security_type
            ),
        )

        rate = curve_rate(
            market_curves[curve_key],
            maturity_years,
        )

        if security_type == "BTF":
            term_days = int(
                entry.get(
                    "term_days",
                    round(
                        365.25
                        * maturity_years
                    ),
                )
            )

            (
                face_value,
                issue_price,
                discount_cost,
            ) = btf_face_for_cash_proceeds(
                cash_proceeds_eur=cash,
                annual_discount_rate=rate,
                term_days=term_days,
            )

            coupon_rate = 0.0
            index_coefficient = 1.0

        else:
            # Synthetic par issuance.
            face_value = cash
            issue_price = 1.0
            discount_cost = 0.0
            coupon_rate = rate
            index_coefficient = 1.0

        # Annual model convention:
        # 1Y BTF issued in year t matures in t+1.
        # Longer maturities are rounded to nearest integer year.
        maturity_increment = max(
            1,
            int(
                round(
                    maturity_years
                )
            ),
        )

        maturity_year = (
            issue_year
            + maturity_increment
        )

        maturity_date = pd.Timestamp(
            year=maturity_year,
            month=12,
            day=31,
        )

        if security_type == "OATEI":
            inflation_index = "HICPxT"

        elif security_type == "OATI":
            inflation_index = "FR_CPIxT"

        else:
            inflation_index = None

        rows.append(
            {
                "isin": (
                    f"SYNTH_{issue_year}_"
                    f"{security_type}_"
                    f"{maturity_years:g}Y_"
                    f"{sequence}"
                ),
                "security_type": security_type,
                "description": (
                    f"Synthetic {security_type} "
                    f"{maturity_years:g}Y issuance "
                    f"from {issue_year}"
                ),
                "coupon_rate": coupon_rate,
                "maturity_date": maturity_date,
                "outstanding_eur": face_value,
                "inflation_index": inflation_index,
                "index_coefficient": index_coefficient,
                "coefficient_date": pd.NaT,
                "issue_year": issue_year,
                "original_maturity_years": maturity_years,
                "maturity_year": maturity_year,
                "issue_cash_proceeds_eur": cash,
                "issue_price": issue_price,
                "issue_rate": rate,
                "curve_key": curve_key,
                "btf_discount_cost_eur": discount_cost,
            }
        )

        total_face += face_value
        total_btf_discount += discount_cost

    new_df = pd.DataFrame(
        rows
    )

    if not new_df.empty:
        new_df = refresh_derived_columns(
            new_df
        )

    diagnostics = {
        "cash_proceeds_eur": (
            total_cash_proceeds_eur
        ),
        "face_value_issued_eur": (
            float(total_face)
        ),
        "btf_discount_cost_eur": (
            float(total_btf_discount)
        ),
        "weighted_market_issue_rate": (
            issuance_weighted_rate(
                market_curves,
                issuance_plan,
            )
        ),
    }

    return (
        new_df,
        diagnostics,
    )


# ============================================================
# One-year simulation
# ============================================================

def simulate_portfolio_year(
    portfolio: pd.DataFrame,
    year: int,
    required_cash_financing_eur: float,
    market_curves: MarketCurves,
    issuance_plan: IssuancePlan,
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove securities maturing in `year` and raise the required
    cash through the issuance plan.
    """
    portfolio = refresh_derived_columns(
        portfolio.copy()
    )

    if "maturity_year" not in portfolio.columns:
        portfolio["maturity_year"] = (
            portfolio["maturity_date"].dt.year
        )

    matured_mask = (
        portfolio["maturity_year"]
        == year
    )

    matured = portfolio.loc[
        matured_mask
    ].copy()

    surviving = portfolio.loc[
        portfolio["maturity_year"]
        > year
    ].copy()

    maturing_face = float(
        matured["outstanding_eur"].sum()
    )

    maturing_liability = float(
        indexed_principal_series(
            matured
        ).sum()
    )

    maturing_redemption = float(
        redemption_value_series(
            matured
        ).sum()
    )

    if required_cash_financing_eur > 0:
        (
            new_securities,
            issue_diag,
        ) = create_new_issuance(
            total_cash_proceeds_eur=(
                required_cash_financing_eur
            ),
            issue_year=year,
            market_curves=market_curves,
            issuance_plan=issuance_plan,
        )

        updated = pd.concat(
            [
                surviving,
                new_securities,
            ],
            ignore_index=True,
            sort=False,
        )

    else:
        updated = surviving

        issue_diag = {
            "cash_proceeds_eur": 0.0,
            "face_value_issued_eur": 0.0,
            "btf_discount_cost_eur": 0.0,
            "weighted_market_issue_rate": (
                issuance_weighted_rate(
                    market_curves,
                    issuance_plan,
                )
            ),
        }

    updated = refresh_derived_columns(
        updated
    )

    diagnostics = {
        "maturing_face_eur": (
            maturing_face
        ),
        "maturing_liability_eur": (
            maturing_liability
        ),
        "maturing_redemption_eur": (
            maturing_redemption
        ),
        **issue_diag,
    }

    return (
        updated.reset_index(
            drop=True
        ),
        diagnostics,
    )


# ============================================================
# Multi-year path
# ============================================================

def simulate_portfolio_path(
    initial_portfolio: pd.DataFrame,
    initial_gdp_eur: float,
    start_year: int,
    years: int,
    nominal_growth: ScalarPath,
    primary_balance_ratio: ScalarPath,
    market_curves: Union[
        MarketCurves,
        MarketCurvePath,
    ],
    issuance_plan: IssuancePlan,
    french_inflation: ScalarPath,
    euro_inflation: ScalarPath,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate the AFT negotiable-debt portfolio over multiple years.

    Annual sequence
    ---------------
    1. Apply annual inflation to OATi/OAT€i coefficients.
    2. Calculate current indexed principal and coupons.
    3. Calculate redemptions due during the year.
    4. Calculate required cash financing:

           redemptions
         + coupon cash expense
         - primary balance

       A negative primary balance therefore increases borrowing.
    5. Raise the cash using the issuance plan.
    6. Grow nominal GDP.
    7. Record financing-cost and portfolio metrics.

    Important
    ---------
    `portfolio_debt_gdp` refers only to the AFT negotiable-debt portfolio,
    not France's official general-government debt/GDP ratio.
    """
    initial_gdp_eur = float(
        initial_gdp_eur
    )

    if initial_gdp_eur <= 0:
        raise ValueError(
            "initial_gdp_eur must be positive."
        )

    if years <= 0:
        raise ValueError(
            "years must be positive."
        )

    portfolio = prepare_portfolio(
        initial_portfolio,
        snapshot_date=None,
        strict_indexation=True,
        validate_aft_indexed_total=True,
    )

    gdp = (
        initial_gdp_eur
    )

    results = []

    for step in range(years):
        year = (
            start_year
            + step
        )

        growth = get_scalar_for_year(
            nominal_growth,
            year,
            "nominal_growth",
        )

        pb_ratio = get_scalar_for_year(
            primary_balance_ratio,
            year,
            "primary_balance_ratio",
        )

        fr_inf = get_scalar_for_year(
            french_inflation,
            year,
            "french_inflation",
        )

        euro_inf = get_scalar_for_year(
            euro_inflation,
            year,
            "euro_inflation",
        )

        curves = get_market_curves_for_year(
            market_curves,
            year,
        )

        validate_issuance_plan(
            issuance_plan,
            curves,
        )

        # ----------------------------------------------------
        # Beginning-of-year stock before this year's inflation
        # ----------------------------------------------------

        portfolio_debt_before_indexation = (
            total_debt(
                portfolio
            )
        )

        face_start = (
            total_face_value(
                portfolio
            )
        )

        # ----------------------------------------------------
        # Linker indexation
        # ----------------------------------------------------

        (
            portfolio,
            indexation_accrual,
        ) = apply_annual_indexation(
            portfolio=portfolio,
            french_inflation=fr_inf,
            euro_inflation=euro_inf,
        )

        portfolio_debt_start = (
            total_debt(
                portfolio
            )
        )

        portfolio_debt_start_gdp = (
            portfolio_debt_start
            / gdp
        )

        # ----------------------------------------------------
        # Coupon cash expense
        # ----------------------------------------------------

        coupon_cash = (
            annual_coupon_expense(
                portfolio
            )
        )

        # ----------------------------------------------------
        # Primary balance
        # ----------------------------------------------------

        primary_balance_eur = (
            pb_ratio
            * gdp
        )

        # ----------------------------------------------------
        # Maturities
        # ----------------------------------------------------

        maturing = portfolio.loc[
            portfolio[
                "maturity_year"
            ]
            == year
        ]

        maturing_face = float(
            maturing[
                "outstanding_eur"
            ].sum()
        )

        maturing_redemption = float(
            redemption_value_series(
                maturing
            ).sum()
        )

        # ----------------------------------------------------
        # Cash financing need
        # ----------------------------------------------------
        #
        # Indexation accrual is not separately added here:
        # it becomes cash-relevant through indexed redemption.
        #
        # New BTF discount cost is not separately added either:
        # the model issues face > cash and the cost is embedded
        # in the future redemption liability.
        # ----------------------------------------------------

        required_cash_financing = (
            maturing_redemption
            + coupon_cash
            - primary_balance_eur
        )

        required_cash_financing = max(
            required_cash_financing,
            0.0,
        )

        # ----------------------------------------------------
        # Roll portfolio
        # ----------------------------------------------------

        (
            portfolio,
            issue_diag,
        ) = simulate_portfolio_year(
            portfolio=portfolio,
            year=year,
            required_cash_financing_eur=(
                required_cash_financing
            ),
            market_curves=curves,
            issuance_plan=issuance_plan,
        )

        btf_discount_cost = float(
            issue_diag[
                "btf_discount_cost_eur"
            ]
        )

        # Economic financing-cost diagnostic.
        total_financing_cost = (
            coupon_cash
            + indexation_accrual
            + btf_discount_cost
        )

        effective_financing_rate = (
            total_financing_cost
            / portfolio_debt_before_indexation
            if portfolio_debt_before_indexation > 0
            else 0.0
        )

        # ----------------------------------------------------
        # GDP / end stock
        # ----------------------------------------------------

        gdp_next = (
            gdp
            * (1.0 + growth)
        )

        portfolio_debt_end = (
            total_debt(
                portfolio
            )
        )

        face_end = (
            total_face_value(
                portfolio
            )
        )

        portfolio_debt_end_gdp = (
            portfolio_debt_end
            / gdp_next
        )

        # ----------------------------------------------------
        # Result row
        # ----------------------------------------------------

        row = {
            "year": year,

            "gdp_start_eur": gdp,
            "gdp_end_eur": gdp_next,
            "nominal_growth": growth,

            "french_inflation": fr_inf,
            "euro_inflation": euro_inf,

            "face_value_start_eur": (
                face_start
            ),
            "portfolio_debt_before_indexation_eur": (
                portfolio_debt_before_indexation
            ),
            "indexation_accrual_eur": (
                indexation_accrual
            ),

            "portfolio_debt_start_eur": (
                portfolio_debt_start
            ),
            "portfolio_debt_start_gdp": (
                portfolio_debt_start_gdp
            ),

            "coupon_cash_eur": (
                coupon_cash
            ),
            "coupon_cash_gdp": (
                coupon_cash
                / gdp
            ),

            "primary_balance_eur": (
                primary_balance_eur
            ),
            "primary_balance_gdp": (
                pb_ratio
            ),

            "maturing_face_eur": (
                maturing_face
            ),
            "maturing_redemption_eur": (
                maturing_redemption
            ),

            "required_cash_financing_eur": (
                required_cash_financing
            ),

            "cash_proceeds_eur": (
                issue_diag[
                    "cash_proceeds_eur"
                ]
            ),
            "face_value_issued_eur": (
                issue_diag[
                    "face_value_issued_eur"
                ]
            ),
            "btf_discount_cost_eur": (
                btf_discount_cost
            ),
            "weighted_market_issue_rate": (
                issue_diag[
                    "weighted_market_issue_rate"
                ]
            ),

            "total_financing_cost_eur": (
                total_financing_cost
            ),
            "effective_financing_rate": (
                effective_financing_rate
            ),

            "portfolio_debt_end_eur": (
                portfolio_debt_end
            ),
            "face_value_end_eur": (
                face_end
            ),
            "portfolio_debt_end_gdp": (
                portfolio_debt_end_gdp
            ),
        }

        # Useful curve nodes for scenario analysis.
        nominal_curve = curves[
            "nominal"
        ]

        for maturity, rate in sorted(
            nominal_curve.items()
        ):
            row[
                f"nominal_yield_{float(maturity):g}y"
            ] = float(rate)

        if "real_euro" in curves:
            for maturity, rate in sorted(
                curves[
                    "real_euro"
                ].items()
            ):
                row[
                    f"real_euro_yield_{float(maturity):g}y"
                ] = float(rate)

        if "real_fr" in curves:
            for maturity, rate in sorted(
                curves[
                    "real_fr"
                ].items()
            ):
                row[
                    f"real_fr_yield_{float(maturity):g}y"
                ] = float(rate)

        results.append(
            row
        )

        gdp = (
            gdp_next
        )

    return (
        pd.DataFrame(
            results
        ),
        portfolio,
    )


# ============================================================
# Scenario helpers
# ============================================================

def parallel_shift_curve(
    base_curve: YieldCurve,
    shift_bps: float,
) -> YieldCurve:
    """
    Parallel shift in basis points.
    """
    shift = (
        float(shift_bps)
        / 10_000.0
    )

    shifted = {
        float(maturity): (
            float(rate)
            + shift
        )
        for maturity, rate
        in base_curve.items()
    }

    validate_yield_curve(
        shifted
    )

    return shifted


def curve_from_30y_anchor(
    oat_30y_yield: float,
    offsets_to_30y: Mapping[float, float],
) -> YieldCurve:
    """
    Construct an illustrative nominal curve from a 30Y OAT anchor.

    Example
    -------
    offsets_to_30y = {
        1: -0.0150,
        2: -0.0125,
        5: -0.0100,
        10: -0.0060,
        15: -0.0035,
        30: 0.0,
        50: 0.0015,
    }

    This is a scenario helper, not an econometric estimate.
    """
    curve = {
        float(maturity): (
            float(oat_30y_yield)
            + float(offset)
        )
        for maturity, offset
        in offsets_to_30y.items()
    }

    validate_yield_curve(
        curve
    )

    return curve


def constant_curve_path(
    curve: YieldCurve,
    start_year: int,
    years: int,
) -> Dict[int, YieldCurve]:
    """
    Repeat one curve across a simulation horizon.
    """
    validate_yield_curve(
        curve
    )

    return {
        start_year + i: dict(curve)
        for i in range(years)
    }


def build_market_curve_path(
    nominal_path: Mapping[int, YieldCurve],
    real_euro_path: Mapping[int, YieldCurve] | None = None,
    real_fr_path: Mapping[int, YieldCurve] | None = None,
) -> MarketCurvePath:
    """
    Combine separate yearly nominal / real curve paths.
    """
    years = set(
        nominal_path.keys()
    )

    if real_euro_path is not None:
        if set(real_euro_path.keys()) != years:
            raise ValueError(
                "real_euro_path years must match nominal_path."
            )

    if real_fr_path is not None:
        if set(real_fr_path.keys()) != years:
            raise ValueError(
                "real_fr_path years must match nominal_path."
            )

    result = {}

    for year in sorted(years):
        curves = {
            "nominal": dict(
                nominal_path[year]
            )
        }

        if real_euro_path is not None:
            curves["real_euro"] = dict(
                real_euro_path[year]
            )

        if real_fr_path is not None:
            curves["real_fr"] = dict(
                real_fr_path[year]
            )

        validate_market_curves(
            curves
        )

        result[year] = curves

    return result


# ============================================================
# Internal smoke test
# ============================================================

def _smoke_test() -> None:
    """
    Lightweight deterministic integration test.

    Run:
        python src/models/debt_portfolio.py
    """

    portfolio = pd.DataFrame(
        {
            "isin": [
                "TEST_OAT",
                "TEST_OATEI",
                "TEST_OATI",
                "TEST_BTF",
            ],
            "security_type": [
                "OAT",
                "OATEI",
                "OATI",
                "BTF",
            ],
            "description": [
                "Nominal test",
                "Euro linker test",
                "French linker test",
                "BTF test",
            ],
            "coupon_rate": [
                0.030,
                0.001,
                0.034,
                0.000,
            ],
            "maturity_date": pd.to_datetime(
                [
                    "2030-12-31",
                    "2032-12-31",
                    "2029-12-31",
                    "2027-12-31",
                ]
            ),
            "outstanding_eur": [
                150e9,
                50e9,
                40e9,
                60e9,
            ],
            "inflation_index": [
                None,
                "HICPxT",
                "FR_CPIxT",
                None,
            ],
            "index_coefficient": [
                1.0,
                1.20,
                1.15,
                1.0,
            ],
        }
    )

    portfolio[
        "indexed_outstanding_eur"
    ] = (
        portfolio[
            "outstanding_eur"
        ]
    )

    indexed_mask = (
        portfolio[
            "security_type"
        ]
        .isin(
            [
                "OATEI",
                "OATI",
            ]
        )
    )

    portfolio.loc[
        indexed_mask,
        "indexed_outstanding_eur",
    ] = (
        portfolio.loc[
            indexed_mask,
            "outstanding_eur",
        ]
        * portfolio.loc[
            indexed_mask,
            "index_coefficient",
        ]
    )

    portfolio = prepare_portfolio(
        portfolio,
        snapshot_date="2026-10-03",
    )

    market_curves = {
        "nominal": {
            1.0: 0.030,
            2.0: 0.034,
            5.0: 0.040,
            10.0: 0.046,
            15.0: 0.049,
            30.0: 0.055,
            50.0: 0.057,
        },
        "real_euro": {
            5.0: 0.012,
            10.0: 0.017,
            20.0: 0.021,
            30.0: 0.024,
        },
        "real_fr": {
            5.0: 0.013,
            10.0: 0.018,
            20.0: 0.022,
            30.0: 0.025,
        },
    }

    issuance_plan = [
        {
            "security_type": "BTF",
            "maturity_years": 1.0,
            "weight": 0.10,
        },
        {
            "security_type": "OAT",
            "maturity_years": 2.0,
            "weight": 0.10,
        },
        {
            "security_type": "OAT",
            "maturity_years": 5.0,
            "weight": 0.18,
        },
        {
            "security_type": "OAT",
            "maturity_years": 10.0,
            "weight": 0.27,
        },
        {
            "security_type": "OATEI",
            "maturity_years": 10.0,
            "weight": 0.05,
        },
        {
            "security_type": "OATI",
            "maturity_years": 10.0,
            "weight": 0.03,
        },
        {
            "security_type": "OAT",
            "maturity_years": 15.0,
            "weight": 0.10,
        },
        {
            "security_type": "OAT",
            "maturity_years": 30.0,
            "weight": 0.12,
        },
        {
            "security_type": "OAT",
            "maturity_years": 50.0,
            "weight": 0.05,
        },
    ]

    # BTF test: face must exceed cash proceeds at positive rates.
    (
        btf_face,
        btf_price,
        btf_cost,
    ) = btf_face_for_cash_proceeds(
        cash_proceeds_eur=100.0,
        annual_discount_rate=0.03,
        term_days=360,
    )

    assert btf_face > 100.0
    assert 0.0 < btf_price < 1.0
    assert btf_cost > 0.0

    # Linker test: positive inflation must increase indexed debt.
    debt_before = total_debt(
        portfolio
    )

    (
        indexed_portfolio,
        accrual,
    ) = apply_annual_indexation(
        portfolio,
        french_inflation=0.02,
        euro_inflation=0.02,
    )

    assert accrual > 0.0
    assert total_debt(
        indexed_portfolio
    ) > debt_before

    summary, final_portfolio = (
        simulate_portfolio_path(
            initial_portfolio=portfolio,
            initial_gdp_eur=3.0e12,
            start_year=2027,
            years=5,
            nominal_growth=0.03,
            primary_balance_ratio=-0.027,
            market_curves=market_curves,
            issuance_plan=issuance_plan,
            french_inflation=0.02,
            euro_inflation=0.02,
        )
    )

    required_output = {
        "year",
        "portfolio_debt_end_eur",
        "portfolio_debt_end_gdp",
        "effective_financing_rate",
        "weighted_market_issue_rate",
        "indexation_accrual_eur",
        "btf_discount_cost_eur",
    }

    missing = required_output.difference(
        summary.columns
    )

    assert not missing, (
        f"Smoke-test output missing columns: {missing}"
    )

    assert len(summary) == 5
    assert not final_portfolio.empty

    assert np.isfinite(
        summary[
            "portfolio_debt_end_eur"
        ]
    ).all()

    assert (
        summary[
            "portfolio_debt_end_eur"
        ] > 0
    ).all()

    print(
        "debt_portfolio.py: all smoke tests passed."
    )


if __name__ == "__main__":
    _smoke_test()
