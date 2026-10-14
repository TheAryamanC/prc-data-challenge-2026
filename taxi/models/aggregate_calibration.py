"""
EUROCONTROL publishes the average taxi-out time of every airport and month, also for the 2026 months that must be guessed
This file uses it to check whether the guesses for a whole airport-month are too high or too low on average, and then nudges them up or down

This is how it works:
-in 2025, measure the offset: the challenge's average answer minus the published average (at one airport it hardly changes from month to month)
-for a 2026 month, the published average plus the offset gives the expected average answer, and its difference from the average guess is the bias
-shrink the bias when it is uncertain
-work this out twice, over NM-linked flights only and over all flights, and only shift when both point the same way (taking the smaller)
-give no shift to winter months where some flights are left out of the published numbers (de-icing)
-add the shift only to NM-linked flights guessed below 2 hours, never going below -12 seconds
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from taxi.main import EUROCONTROL_FOLDER

AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
CROSS_YEAR_SD = 5.0             # assumed spread of the offset's move from 2025 to 2026 (arrivals moved 5.4 s rms)
STABLE_VALID_SHARE = 0.98       # months with a smaller valid share have excluded flights (winter de-icing) and are not shifted
PREDICTION_LIMIT = 7200.0       # rows predicted at or above it are left out of the means and never shifted
LABEL_FLOOR = -12.0             # the lowest label of 2025: a shifted prediction never goes below it
LABEL_FLOOR_SECONDS = -12       # the 2025 labels of the means are clipped below at the lowest 2025 label (as LABEL_FLOOR)
SLOPE_MONTH_THRESHOLD = 3       # a slope is tested only with more than 3 normal months: with 2 coefficients fitted, fewer leave almost no spread to judge it
SLOPE_SIGNIFICANCE_T = 2        # |t| >= 2: the usual rule of thumb for a slope that is clearly not zero (about 95 %)
SINGLE_WINTER_SD_SHARE = 0.5    # one winter month gives no spread to measure: assume 50 % of the mean
EXTRAPOLATION_SD_FACTOR = 2     # double that spread when a 2026 month has more exclusions than any 2025 month


def aggregates(kind: str = 'taxi_out') -> pd.DataFrame:
    """
    This function reads EUROCONTROL's monthly files for 2025 and 2026 for the ten airports, and works out each airport-month's published average taxi time in seconds, the share of valid flights and the share of valid flights with a reference time
    """
    table = pd.concat([pd.read_csv(EUROCONTROL_FOLDER / f'{kind}_additional_time_{year}.csv') for year in (2025, 2026)])
    table = table[table.APT_ICAO.isin(AIRPORTS)].copy()
    # TOTAL_REF_TIME_MIN / TOTAL_ADD_TIME_MIN: sums of the reference and the additional taxi times (min); TF: all flights
    table['agg'] = (table.TOTAL_REF_TIME_MIN + table.TOTAL_ADD_TIME_MIN) / table.TOTAL_REF_NB_FL * 60
    table['valid'] = table.VALID_FL / table.TF
    table['ref'] = table.TOTAL_REF_NB_FL / table.VALID_FL
    return table.set_index(['APT_ICAO', 'YEAR', 'MONTH_NUM'])


def offset_model(table: pd.DataFrame, label_means: pd.Series, airport: str, stable_months: list[int]) -> tuple[Callable[[float], float], float]:
    """
    This function learns one airport's offset from its normal months, and how much it spreads
    """
    reference_shares = np.array([table.loc[(airport, 2025, m), 'ref'] for m in stable_months])
    offsets = np.array([label_means[(airport, m)] - table.loc[(airport, 2025, m), 'agg'] for m in stable_months])
    design = np.vstack([np.ones_like(reference_shares), reference_shares]).T
    coefficients = np.linalg.lstsq(design, offsets, rcond=None)[0]          # intercept, slope
    residual = offsets - design @ coefficients
    if len(reference_shares) > SLOPE_MONTH_THRESHOLD and np.ptp(reference_shares) > 0:
        slope_error = np.sqrt(residual.var(ddof=2) / ((reference_shares - reference_shares.mean()) ** 2).sum())
    else:
        slope_error = np.inf
    if np.isfinite(slope_error) and slope_error > 0 and abs(coefficients[1] / slope_error) >= SLOPE_SIGNIFICANCE_T:
        def regressed_offset(share: float) -> float:
            return coefficients[0] + coefficients[1] * share
        return regressed_offset, residual.std(ddof=2)

    def mean_offset(share: float) -> float:
        return offsets.mean()
    return mean_offset, offsets.std(ddof=1)


def implied_mean(table: pd.DataFrame, label_means: pd.Series, airport: str, year: int, month: int, cross_year_sd: float = CROSS_YEAR_SD) -> tuple[float, float]:
    """
    This function works out what the true average answer of one airport-month should be, and how unsure that is
    """
    months = [m for m in range(1, 13) if not (year == 2025 and m == month) and (airport, m) in label_means.index]
    stable = [m for m in months if table.loc[(airport, 2025, m), 'valid'] >= STABLE_VALID_SHARE]
    winter = [m for m in months if table.loc[(airport, 2025, m), 'valid'] < STABLE_VALID_SHARE]
    offset_at, spread = offset_model(table, label_means, airport, stable)
    target = table.loc[(airport, year, month)]
    offset = offset_at(target['ref'])
    offset_sd = np.sqrt(spread ** 2 + cross_year_sd ** 2)
    if target['valid'] >= STABLE_VALID_SHARE:
        return target['agg'] + offset, offset_sd
    if not winter:
        return np.nan, np.inf
    # mean taxi time of the excluded flights of each 2025 winter month: label mean = valid x (agg + offset) + (1 - valid) x excluded
    excluded_means = []
    for m in winter:
        valid_flights_mean = table.loc[(airport, 2025, m), 'agg'] + offset_at(table.loc[(airport, 2025, m), 'ref'])
        excluded_means.append((label_means[(airport, m)] - table.loc[(airport, 2025, m), 'valid'] * valid_flights_mean) / (1 - table.loc[(airport, 2025, m), 'valid']))
    excluded = np.array(excluded_means)
    excluded_sd = excluded.std(ddof=1) if len(excluded) > 1 else SINGLE_WINTER_SD_SHARE * excluded.mean()
    if target['valid'] < min(table.loc[(airport, 2025, m), 'valid'] for m in winter):
        excluded_sd *= EXTRAPOLATION_SD_FACTOR  # more exclusions than in any 2025 month: extrapolation
    implied = target['valid'] * (target['agg'] + offset) + (1 - target['valid']) * excluded.mean()
    sigma = np.sqrt((target['valid'] * offset_sd) ** 2 + ((1 - target['valid']) * excluded_sd) ** 2)
    return implied, sigma


def shrunk_shift(bias: float, sigma: float) -> float:
    """
    This function shrinks a bias by its uncertainty
    """
    return 0.0 if not (np.isfinite(sigma) and np.isfinite(bias)) else bias * bias ** 2 / (bias ** 2 + sigma ** 2)


def label_means_2025(components: pd.DataFrame, linked_only: bool) -> pd.Series:
    """
    This function works out the 2025 average true answer per airport and month, over NM-linked flights only or all flights
    """
    rows = np.isfinite(components.label.to_numpy()) & (components.pipeline.to_numpy() < PREDICTION_LIMIT)
    if linked_only:
        rows &= ~components.no_link.to_numpy()
    frame = pd.DataFrame({'airport': components.airport.to_numpy()[rows], 
                          'month': components.month.to_numpy()[rows],
                          'label': components.label.to_numpy()[rows].clip(LABEL_FLOOR_SECONDS, PREDICTION_LIMIT)})
    return frame.groupby(['airport', 'month']).label.mean()


def month_estimate(predictions: pd.DataFrame, table: pd.DataFrame, label_means: pd.Series, airport: str, month: int, linked_only: bool) -> dict[str, object]:
    """
    This function makes one estimate for one 2026 airport-month
    """
    mask = (predictions.airport == airport) & (predictions.month == month) & (predictions.prediction < PREDICTION_LIMIT)
    if linked_only:
        mask &= predictions.linked
    implied, sigma = implied_mean(table, label_means, airport, 2026, int(month))
    bias = implied - predictions.prediction[mask].mean()
    return {'airport': airport, 
            'month': month, 
            'variant': 'linked' if linked_only else 'all', 
            'bias': bias, 
            'sigma': sigma,
            'shift': shrunk_shift(bias, sigma), 
            'valid': table.loc[(airport, 2026, int(month)), 'valid']}


def airport_month_shifts(predictions: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """
    This function works out the shift of every airport and 2026 month twice
    """
    table = aggregates()
    rows = []
    for linked_only in (True, False):
        label_means = label_means_2025(components, linked_only)
        for airport in AIRPORTS:
            for month in sorted(predictions.month.unique()):
                rows.append(month_estimate(predictions, table, label_means, airport, month, linked_only))
    shifts = pd.DataFrame(rows).pivot_table(index=['airport', 'month'], columns='variant', values=['shift', 'bias', 'valid'])
    linked_shift, all_shift = shifts[('shift', 'linked')], shifts[('shift', 'all')]
    # no shift in months with excluded flights or where the two variants disagree in sign (a NaN shift never agrees)
    no_shift = (shifts[('valid', 'linked')] < STABLE_VALID_SHARE) | (np.sign(linked_shift) != np.sign(all_shift))
    consensus = np.where(no_shift, 0.0, np.sign(linked_shift) * np.minimum(np.abs(linked_shift), np.abs(all_shift)))
    return pd.DataFrame({'bias_linked': shifts[('bias', 'linked')], 
                         'bias_all': shifts[('bias', 'all')], 
                         'shift_linked': linked_shift,
                         'shift_all': all_shift, 'shift': consensus}, index=shifts.index)


def calibrate(predictions: pd.DataFrame, components: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    This function performs the shift
    """
    shifts = airport_month_shifts(predictions, components)
    shift = shifts['shift'].reindex(pd.MultiIndex.from_arrays([predictions.airport, predictions.month])).to_numpy()
    apply = predictions.linked.to_numpy() & (predictions.prediction.to_numpy() < PREDICTION_LIMIT) & np.isfinite(shift)
    shifted = np.maximum(predictions.prediction.to_numpy() + np.nan_to_num(shift), LABEL_FLOOR)
    return np.where(apply, shifted, predictions.prediction.to_numpy()), shifts
