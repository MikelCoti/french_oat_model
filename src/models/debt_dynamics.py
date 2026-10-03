import numpy as np
import pandas as pd

def debt_stabilizing_primary_balance(
    debt_ratio,
    effective_rate,
    nominal_growth
):
    '''
    Primary balance required to keep debt/GDP unchanged.

    Positive result = required primary surplus.
    Negative result = a primary deficit could be consistent with stable debt.
    '''
    return (
        ((1 + effective_rate) / (1 + nominal_growth) - 1)
        * debt_ratio
    )


def update_effective_interest_rate(
    previous_effective_rate,
    new_borrowing_rate,
    repricing_speed
):
    '''
    Version 0.1 repricing model.

    A fraction of the debt stock is assumed to refinance each year
    at the prevailing average rate on new borrowing.
    '''
    return (
        (1 - repricing_speed) * previous_effective_rate
        + repricing_speed * new_borrowing_rate
    )


def next_debt_ratio(
    previous_debt_ratio,
    effective_rate,
    nominal_growth,
    primary_balance
):
    '''
    Exact one-period debt/GDP accounting equation.
    '''
    return (
        ((1 + effective_rate) / (1 + nominal_growth))
        * previous_debt_ratio
        - primary_balance
    )

def project_debt_path(
    initial_debt_ratio,
    initial_effective_rate,
    new_borrowing_rate,
    nominal_growth,
    primary_balance,
    repricing_speed,
    years=20
):
    debt = initial_debt_ratio
    effective_rate = initial_effective_rate

    results = []

    for year in range(1, years + 1):
        debt_start = debt

        effective_rate = update_effective_interest_rate(
            previous_effective_rate=effective_rate,
            new_borrowing_rate=new_borrowing_rate,
            repricing_speed=repricing_speed
        )

        required_pb = debt_stabilizing_primary_balance(
            debt_ratio=debt_start,
            effective_rate=effective_rate,
            nominal_growth=nominal_growth
        )

        fiscal_gap = required_pb - primary_balance

        # Interest spending as a share of current-year GDP.
        interest_expense_ratio = (
            effective_rate
            * debt_start
            / (1 + nominal_growth)
        )

        debt = next_debt_ratio(
            previous_debt_ratio=debt_start,
            effective_rate=effective_rate,
            nominal_growth=nominal_growth,
            primary_balance=primary_balance
        )

        results.append({
            "year": year,
            "debt_gdp": debt,
            "effective_rate": effective_rate,
            "new_borrowing_rate": new_borrowing_rate,
            "nominal_growth": nominal_growth,
            "primary_balance": primary_balance,
            "required_primary_balance": required_pb,
            "fiscal_gap": fiscal_gap,
            "interest_expense_gdp": interest_expense_ratio
        })

    return pd.DataFrame(results)