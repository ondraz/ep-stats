import numpy as np
import pandas as pd
import pytest

from src.epstats.toolkit.experiment import Experiment
from src.epstats.toolkit.metric import Metric, SimpleMetric

NOMINATOR = "value(test_unit_type.unit.conversion)"
DENOMINATOR = "count(test_unit_type.unit.exposure)"


def _by_unit_goals(units_per_variant: dict) -> pd.DataFrame:
    """
    Build a flat by-unit goals data frame (one row per unit and goal) from a mapping
    `{variant: [per-unit conversion value, ...]}`. Every unit gets one exposure.
    """
    rows = []
    unit_id = 0
    for variant, values in units_per_variant.items():
        for value in values:
            unit_id += 1
            common = dict(
                exp_id="test-quantile",
                exp_variant_id=variant,
                unit_type="test_unit_type",
                agg_type="unit",
                dimension="",
                dimension_value="",
                unit_id=unit_id,
            )
            rows.append({**common, "goal": "exposure", "count": 1, "sum_value": 1})
            rows.append(
                {**common, "goal": "conversion", "count": 1, "sum_value": value}
            )
    return pd.DataFrame(rows)


def _metric(statistic: str, id: int = 1, **kwargs) -> Metric:
    return Metric(
        id,
        f"Conversion {statistic}",
        NOMINATOR,
        DENOMINATOR,
        statistic=statistic,
        **kwargs,
    )


def _experiment(metrics, **kwargs) -> Experiment:
    return Experiment(
        "test-quantile", "a", metrics, [], unit_type="test_unit_type", **kwargs
    )


def _evaluate(goals, metrics, **kwargs) -> pd.DataFrame:
    return _experiment(metrics, **kwargs).evaluate_by_unit(goals).metrics


def _indexed(goals, metrics, **kwargs) -> pd.DataFrame:
    return _evaluate(goals, metrics, **kwargs).set_index("exp_variant_id")


# --------------------------------------------------------------------------------------
# point estimate
# --------------------------------------------------------------------------------------


def test_median_point_estimate_lands_in_mean_column():
    # Medians are 3 and 30; means (12 and 120) are pulled up by the single large value, so a
    # wrong column or a mean fallback would be plainly visible.
    goals = _by_unit_goals({"a": [1, 2, 3, 4, 50], "b": [10, 20, 30, 40, 500]})

    metrics = _indexed(goals, [_metric("median")])

    assert metrics.loc["a", "mean"] == pytest.approx(3)
    assert metrics.loc["b", "mean"] == pytest.approx(30)
    # `count` keeps its meaning: the metric denominator / sample size, not the quantile.
    assert metrics.loc["a", "count"] == 5
    assert metrics.loc["b", "count"] == 5


def test_percentile_point_estimate_matches_numpy():
    values_a = list(range(1, 101))
    values_b = list(range(51, 151))
    goals = _by_unit_goals({"a": values_a, "b": values_b})

    metrics = _indexed(goals, [_metric("p90")])

    assert metrics.loc["a", "mean"] == pytest.approx(np.quantile(values_a, 0.9))
    assert metrics.loc["b", "mean"] == pytest.approx(np.quantile(values_b, 0.9))


def test_control_row_reports_zero_diff_and_no_significance():
    goals = _by_unit_goals({"a": list(range(1, 201)), "b": list(range(1, 201))})

    metrics = _indexed(goals, [_metric("median")])

    # Mirrors the mean path's control row: zero difference, zero test statistic, p-value of 1,
    # but a real standard error (control compared against an independent copy of itself).
    assert metrics.loc["a", "diff"] == pytest.approx(0)
    assert metrics.loc["a", "test_stat"] == pytest.approx(0)
    assert metrics.loc["a", "p_value"] == pytest.approx(1)
    assert metrics.loc["a", "standard_error"] > 0
    assert metrics.loc["a", "confidence_interval"] > 0


# --------------------------------------------------------------------------------------
# hypothesis testing
# --------------------------------------------------------------------------------------


def test_location_shift_is_detected():
    rng = np.random.default_rng(42)
    control = rng.normal(100, 10, 3000)
    # A 10% shift of a metric with this much data is far beyond noise.
    treatment = rng.normal(110, 10, 3000)
    goals = _by_unit_goals({"a": control, "b": treatment})

    metrics = _indexed(goals, [_metric("median")])

    assert metrics.loc["b", "p_value"] < 0.001
    expected_diff = (np.median(treatment) - np.median(control)) / abs(
        np.median(control)
    )
    assert metrics.loc["b", "diff"] == pytest.approx(expected_diff)
    assert metrics.loc["b", "diff"] == pytest.approx(0.1, abs=0.02)


def test_identical_distributions_are_rarely_significant():
    # A single replication can be significant by chance (that is what a 5% level means), so this
    # checks calibration across independent replications instead of trusting one lucky seed.
    rng = np.random.default_rng(7)
    significant = 0
    replications = 20
    for _ in range(replications):
        goals = _by_unit_goals(
            {"a": rng.normal(100, 10, 600), "b": rng.normal(100, 10, 600)}
        )
        metrics = _indexed(goals, [_metric("median")], bootstrap_samples=200)
        assert abs(metrics.loc["b", "diff"]) < 0.1
        if metrics.loc["b", "p_value"] < 0.05:
            significant += 1

    # Nominal false positive rate is 5%; allow generous slack for 20 replications.
    assert significant <= 4, f"{significant}/{replications} false positives"


def test_median_is_robust_to_outliers_that_move_the_mean():
    # Both variants share the same body; `b` additionally has a handful of huge values.
    body = [10.0] * 200
    goals = _by_unit_goals({"a": body + [10.0] * 5, "b": body + [10_000.0] * 5})

    metrics = _indexed(goals, [_metric("median", id=1), _metric("mean", id=2)])
    median_rows = metrics[metrics["metric_id"] == 1]
    mean_rows = metrics[metrics["metric_id"] == 2]

    # The mean is dragged up by the outliers, the median is not.
    assert mean_rows.loc["b", "mean"] > 2 * mean_rows.loc["a", "mean"]
    assert median_rows.loc["b", "mean"] == pytest.approx(10.0)
    assert median_rows.loc["b", "diff"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------


def test_evaluation_is_deterministic():
    rng = np.random.default_rng(3)
    goals = _by_unit_goals({"a": rng.normal(50, 10, 500), "b": rng.normal(55, 10, 500)})
    experiment = _experiment([_metric("median")])

    first = experiment.evaluate_by_unit(goals).metrics
    second = experiment.evaluate_by_unit(goals).metrics

    columns = [c for c in first.columns if c != "timestamp"]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_bootstrap_samples_and_seed_are_configurable():
    rng = np.random.default_rng(3)
    goals = _by_unit_goals({"a": rng.normal(50, 10, 400), "b": rng.normal(55, 10, 400)})

    default = _indexed(goals, [_metric("median")])
    other_seed = _indexed(goals, [_metric("median")], random_seed=12345)
    more_samples = _indexed(goals, [_metric("median")], bootstrap_samples=200)

    # The point estimate does not depend on the bootstrap, only the standard error does.
    assert other_seed.loc["b", "diff"] == pytest.approx(default.loc["b", "diff"])
    assert other_seed.loc["b", "standard_error"] != default.loc["b", "standard_error"]
    assert more_samples.loc["b", "standard_error"] != default.loc["b", "standard_error"]
    # A different seed must not move the standard error far; it is the same estimator.
    assert other_seed.loc["b", "standard_error"] == pytest.approx(
        default.loc["b", "standard_error"], rel=0.25
    )


# --------------------------------------------------------------------------------------
# unsupported aggregated path
# --------------------------------------------------------------------------------------


def test_evaluate_agg_raises_naming_the_metric():
    experiment = _experiment([_metric("mean", id=1), _metric("p75", id=2)])

    with pytest.raises(ValueError, match="p75"):
        experiment.evaluate_agg(pd.DataFrame())


def test_evaluate_wide_agg_raises_naming_the_metric():
    experiment = _experiment([_metric("median", id=3)])

    with pytest.raises(ValueError, match="median"):
        experiment.evaluate_wide_agg(pd.DataFrame())


def test_get_evaluate_columns_agg_raises():
    with pytest.raises(ValueError, match="not.*recoverable|quantile"):
        _metric("median").get_evaluate_columns_agg(pd.DataFrame())


# --------------------------------------------------------------------------------------
# definition validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("statistic", ["p0", "p100", "p-5", "foo", "mediann", "", "p"])
def test_invalid_statistic_raises(statistic):
    with pytest.raises(ValueError):
        _metric(statistic)


@pytest.mark.parametrize(
    "statistic,expected_quantile",
    [("mean", None), ("median", 0.5), ("p50", 0.5), ("p90", 0.9), ("p99.5", 0.995)],
)
def test_statistic_is_normalized(statistic, expected_quantile):
    metric = _metric(statistic)

    assert metric.quantile == expected_quantile
    assert metric.is_quantile_statistic == (expected_quantile is not None)


def test_statistic_is_case_insensitive_and_trimmed():
    assert _metric(" Median ").quantile == 0.5
    assert _metric("P90").quantile == 0.9


def test_quantile_inside_winsorized_tail_raises_at_construction():
    with pytest.raises(ValueError, match="winsorized tail"):
        _metric("p99", outlier_upper_percentile=2)

    with pytest.raises(ValueError, match="winsorized tail"):
        _metric("p1", outlier_lower_percentile=2)


def test_quantile_outside_winsorized_tail_is_allowed():
    metric = _metric("median", outlier_upper_percentile=1, outlier_lower_percentile=1)

    assert metric.quantile == 0.5


def test_simple_metric_passes_statistic_through():
    metric = SimpleMetric(1, "Bookings", "conversion", "exposure", statistic="p90")

    assert metric.statistic == "p90"
    assert metric.quantile == 0.9
    assert metric.is_quantile_statistic


# --------------------------------------------------------------------------------------
# interaction with winsorization
# --------------------------------------------------------------------------------------


def test_upper_winsorization_leaves_median_unchanged():
    # Winsorization caps only the top 1% of the pooled values, which cannot move the median.
    goals = _by_unit_goals(
        {"a": list(range(1, 101)), "b": list(range(1, 100)) + [1_000_000]}
    )

    plain = _indexed(goals, [_metric("median")])
    winsorized = _indexed(goals, [_metric("median", outlier_upper_percentile=1)])

    assert winsorized.loc["a", "mean"] == pytest.approx(plain.loc["a", "mean"])
    assert winsorized.loc["b", "mean"] == pytest.approx(plain.loc["b", "mean"])


# --------------------------------------------------------------------------------------
# degenerate data
# --------------------------------------------------------------------------------------


def test_all_values_equal_produces_no_test_but_does_not_crash():
    goals = _by_unit_goals({"a": [5.0] * 100, "b": [5.0] * 100})

    metrics = _indexed(goals, [_metric("median")])

    # Every resample has the same median, so the bootstrap sees no variability at all. We report
    # no evidence of a difference rather than an infinite test statistic.
    assert metrics.loc["b", "mean"] == pytest.approx(5.0)
    assert metrics.loc["b", "diff"] == pytest.approx(0.0)
    assert metrics.loc["b", "standard_error"] == pytest.approx(0.0)
    assert metrics.loc["b", "test_stat"] == pytest.approx(0.0)
    assert np.isnan(metrics.loc["b", "p_value"])


def test_zero_control_median_yields_undefined_relative_difference():
    # More than half of the control units have no conversion, so its median is exactly zero and
    # the relative difference is undefined.
    goals = _by_unit_goals({"a": [0.0] * 60 + [10.0] * 40, "b": [5.0] * 100})

    metrics = _indexed(goals, [_metric("median")])

    assert metrics.loc["a", "mean"] == pytest.approx(0.0)
    assert not np.isfinite(metrics.loc["b", "diff"])


def test_single_unit_variant_yields_nan_row():
    goals = _by_unit_goals({"a": list(range(1, 101)), "b": [7.0]})

    metrics = _indexed(goals, [_metric("median")])

    # A single observation carries no information about its own variability.
    assert metrics.loc["b", "mean"] == pytest.approx(7.0)
    assert np.isnan(metrics.loc["b", "standard_error"])
    assert np.isnan(metrics.loc["b", "p_value"])


# --------------------------------------------------------------------------------------
# interaction with the rest of the pipeline
# --------------------------------------------------------------------------------------


def test_three_variants_apply_holm_bonferroni_to_quantile_rows():
    rng = np.random.default_rng(11)
    goals = _by_unit_goals(
        {
            "a": rng.normal(100, 10, 1500),
            "b": rng.normal(103, 10, 1500),
            "c": rng.normal(104, 10, 1500),
        }
    )

    corrected = _indexed(goals, [_metric("median")])
    # The same experiment evaluated as two variants gives the uncorrected p-value of `b`.
    two_variant_goals = goals[goals["exp_variant_id"].isin(["a", "b"])]
    uncorrected = _indexed(two_variant_goals, [_metric("median")])

    assert len(corrected) == 3
    # Holm-Bonferroni only ever makes p-values larger and confidence intervals wider.
    assert corrected.loc["b", "p_value"] >= uncorrected.loc["b", "p_value"]
    assert (
        corrected.loc["b", "confidence_interval"]
        >= uncorrected.loc["b", "confidence_interval"] - 1e-12
    )


def test_mixed_mean_and_quantile_metrics_leave_mean_rows_untouched():
    rng = np.random.default_rng(19)
    goals = _by_unit_goals(
        {"a": rng.lognormal(3, 1, 800), "b": rng.lognormal(3.2, 1, 800)}
    )
    mean_metric = Metric(1, "Mean", NOMINATOR, DENOMINATOR)

    mean_only = _evaluate(goals, [mean_metric])
    mixed = _evaluate(goals, [mean_metric, _metric("median", id=2)])

    columns = [c for c in mean_only.columns if c != "timestamp"]
    mixed_mean_rows = mixed[mixed["metric_id"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        mean_only[columns].reset_index(drop=True), mixed_mean_rows[columns]
    )


def test_metric_row_block_order_is_preserved_for_many_metrics():
    goals = _by_unit_goals({"a": list(range(1, 51)), "b": list(range(11, 61))})
    metrics = [
        _metric("mean", id=1),
        _metric("median", id=2),
        _metric("p90", id=3),
        _metric("mean", id=4),
    ]

    result = _evaluate(goals, metrics)

    # Each metric occupies a contiguous two-row block in metric definition order, which is what
    # the positional Holm-Bonferroni indexing relies on.
    assert list(result["metric_id"]) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert list(result["exp_variant_id"]) == ["a", "b"] * 4
    # The two identical mean metrics must evaluate identically.
    assert result.loc[0, "mean"] == pytest.approx(result.loc[6, "mean"])
    assert result.loc[1, "diff"] == pytest.approx(result.loc[7, "diff"])


def test_sample_size_columns_are_nan_for_quantile_metrics():
    goals = _by_unit_goals({"a": list(range(1, 201)), "b": list(range(21, 221))})
    metrics = [
        Metric(1, "Mean", NOMINATOR, DENOMINATOR, minimum_effect=0.1),
        _metric("median", id=2, minimum_effect=0.1),
    ]

    result = _evaluate(goals, metrics).set_index(["metric_id", "exp_variant_id"])

    # The mean metric still gets a required sample size, the quantile metric cannot: the formula
    # needs the density at the quantile, unknowable before the experiment.
    assert np.isfinite(result.loc[(1, "b"), "required_sample_size"])
    assert np.isnan(result.loc[(2, "b"), "required_sample_size"])
    assert np.isnan(result.loc[(2, "b"), "power"])
    # The observed sample size is still reported, it is just the number of units.
    assert result.loc[(2, "b"), "sample_size"] == 200


def test_false_positive_risk_is_nan_for_quantile_metrics():
    rng = np.random.default_rng(23)
    goals = _by_unit_goals(
        {"a": rng.normal(100, 10, 1000), "b": rng.normal(115, 10, 1000)}
    )

    result = _indexed(
        goals, [_metric("median", minimum_effect=0.1)], null_hypothesis_rate=0.5
    )

    # False positive risk needs power, which needs a required sample size.
    assert result.loc["b", "p_value"] < 0.05
    assert np.isnan(result.loc["b", "false_positive_risk"])


def test_sequential_evaluation_widens_quantile_confidence_interval():
    rng = np.random.default_rng(29)
    goals = _by_unit_goals(
        {"a": rng.normal(100, 10, 1500), "b": rng.normal(105, 10, 1500)}
    )

    final = _indexed(goals, [_metric("median")])
    # Evaluated on day 1 of a 14 day test the confidence level is adjusted upwards by the
    # O'Brien-Fleming alpha spending function, which widens the interval.
    midway = _indexed(
        goals,
        [_metric("median")],
        date_from="2024-01-01",
        date_to="2024-01-14",
        date_for="2024-01-02",
    )

    assert midway.loc["b", "standard_error"] == pytest.approx(
        final.loc["b", "standard_error"]
    )
    assert (
        midway.loc["b", "confidence_interval"] > final.loc["b", "confidence_interval"]
    )
    assert midway.loc["b", "p_value"] == pytest.approx(final.loc["b", "p_value"])
