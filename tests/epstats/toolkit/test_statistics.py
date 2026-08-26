import numpy as np
import pytest
import scipy.stats as st
from statsmodels.stats.power import TTestIndPower

from src.epstats.toolkit.statistics import Statistics


def _assert_sample_sizes_equal(x, y):
    assert abs(x - y) <= 2


def _assert_sample_sizes_equal_within_tolerance(x, y, n_variants):
    # Booking calculator is using Sidak's correction instead of Bonferroni's,
    # https://github.com/bookingcom/powercalculator/blob/master/src/js/math.js#L303
    # it produces slightly different results in case of multiple variants.
    # The difference is less than 0.5%.
    if n_variants > 2:
        rel_tol = 0.005
        assert abs((x - y) / x) <= rel_tol
    else:
        _assert_sample_sizes_equal(x, y)


@pytest.mark.parametrize(
    "test_length, actual_day, expected",
    [
        (14, 0, 0.9999),
        (14, 1, 0.9999),
        (14, 2, 0.9999),
        (14, 3, 0.9999),
        (14, 14, 0.95),
        (7, 1, 0.9999),
        (28, 4, 0.9999),
        (28, 8, 0.9998),
        (28, 28, 0.95),
    ],
)
def test_obf_alpha_spending_function(test_length, actual_day, expected):
    alpha = Statistics.obf_alpha_spending_function(0.95, test_length, actual_day)
    assert alpha == expected


@pytest.mark.parametrize("test_length, actual_day", [(14, 1), (28, 4), (365, 1)])
def test_obf_alpha_spending_function_keeps_confidence_intervals_finite(
    test_length, actual_day
):
    """
    Early in the experiment, the adjusted confidence level must stay below 1.0,
    otherwise t-quantiles and confidence intervals become infinite.
    """
    confidence_level = Statistics.obf_alpha_spending_function(
        0.95, test_length, actual_day
    )
    assert confidence_level < 1.0
    assert np.isfinite(st.t.ppf(confidence_level + (1 - confidence_level) / 2, 100))


@pytest.mark.parametrize(
    # expected from https://bookingcom.github.io/powercalculator
    "n_variants, minimum_effect, mean, std, expected",
    [
        (2, 0.10, 0.2, 1.2, 56512),
        (2, 0.10, 0.2, 2.0, 156978),
        (2, 0.10, 0.3, 1.2, 25117),
        (2, 0.10, 0.3, 2.0, 69768),
        (2, 0.05, 0.2, 1.2, 226048),
        (2, 0.05, 0.2, 2.0, 627911),
        (2, 0.05, 0.3, 1.2, 100466),
        (2, 0.05, 0.3, 2.0, 279072),
        (3, 0.05, 0.3, 2.0, 336878),
        (3, 0.10, 0.2, 1.2, 68218),
        (4, 0.10, 0.2, 2.0, 208576),
    ],
)
def test_required_sample_size_per_variant_equal_variance(
    n_variants, minimum_effect, mean, std, expected
):
    sample_size_per_variant = Statistics.required_sample_size_per_variant(
        n_variants=n_variants,
        minimum_effect=minimum_effect,
        mean=mean,
        std=std,
    )
    effect_size = (mean * (1 + minimum_effect) - mean) / std

    # nobs1
    expected_from_statsmodels = TTestIndPower().solve_power(
        effect_size=effect_size,
        ratio=1.0,  # N_A / N_B
        alpha=0.05 / (n_variants - 1),
        power=0.8,
        nobs1=None,
    )

    _assert_sample_sizes_equal(
        sample_size_per_variant, round(expected_from_statsmodels)
    )
    _assert_sample_sizes_equal_within_tolerance(
        sample_size_per_variant, expected, n_variants
    )


@pytest.mark.parametrize(
    "minimum_effect, mean, std, std_2",
    [
        (0.10, 0.2, 1.2, 2.0),
        (0.05, 0.3, 2.0, 2.5),
    ],
)
def test_required_sample_size_per_variant_unequal_variance(
    minimum_effect, mean, std, std_2
):
    sample_size_per_variant = Statistics.required_sample_size_per_variant(
        n_variants=2,
        minimum_effect=minimum_effect,
        mean=mean,
        std=std,
        std_2=std_2,
    )

    mean_2 = mean * (1 + minimum_effect)
    var_ = (std**2 + std_2**2) / 2
    std_ = np.sqrt(var_)
    effect_size = (mean_2 - mean) / std_

    # nobs1
    expected_from_statsmodels = TTestIndPower().solve_power(
        effect_size=effect_size,
        ratio=1.0,  # N_A / N_B
        alpha=0.05,
        power=0.8,
        nobs1=None,
    )

    _assert_sample_sizes_equal(
        sample_size_per_variant, round(expected_from_statsmodels)
    )


@pytest.mark.parametrize(
    # expected from https://bookingcom.github.io/powercalculator
    "n_variants, minimum_effect, mean, std, expected",
    [
        (2, 0.05, 0.4, None, 9490),
        (2, 0.10, 0.1, None, 14749),
        (3, 0.05, 0.4, None, 11455),
        (4, 0.10, 0.1, None, 19596),
    ],
)
def test_required_sample_size_per_variant_bernoulli(
    n_variants, minimum_effect, mean, std, expected
):
    sample_size_per_variant = Statistics.required_sample_size_per_variant_bernoulli(
        n_variants=n_variants,
        minimum_effect=minimum_effect,
        mean=mean,
    )

    mean_2 = mean * (1 + minimum_effect)
    var = mean * (1 - mean)
    var_2 = mean_2 * (1 - mean_2)
    std = np.sqrt((var + var_2) / 2)

    effect_size = (mean_2 - mean) / std

    # nobs1
    expected_from_statsmodels = TTestIndPower().solve_power(
        effect_size=effect_size,
        ratio=1.0,
        alpha=0.05 / (n_variants - 1),
        power=0.8,
        nobs1=None,
    )

    _assert_sample_sizes_equal(
        sample_size_per_variant, round(expected_from_statsmodels)
    )
    _assert_sample_sizes_equal_within_tolerance(
        sample_size_per_variant, expected, n_variants
    )


@pytest.mark.parametrize(
    "n_variants, minimum_effect, mean, std, f",
    [
        (2, -0.1, 0.2, 1.2, Statistics.required_sample_size_per_variant),
        (1, -0.1, 0.2, 1.2, Statistics.required_sample_size_per_variant),
        (2, 0.1, 10.1, None, Statistics.required_sample_size_per_variant_bernoulli),
    ],
)
def test_required_sample_size_per_variant_raises_exception(
    n_variants, minimum_effect, mean, std, f
):
    args = {"minimum_effect": minimum_effect, "mean": mean, "n_variants": n_variants}

    if std is not None:
        args["std"] = std

    with pytest.raises(ValueError):
        f(**args)


@pytest.mark.parametrize(
    "minimum_effect, mean, std, expected",
    [
        (0.1, 0, 0, np.isnan),
        (0.1, np.nan, np.nan, np.isnan),
        (0.1, 0, np.nan, np.isnan),
        (0.1, 0, 1, np.isinf),
        (np.nan, np.nan, np.nan, np.isnan),
    ],
)
def test_required_sample_size_per_variant_not_valid(
    minimum_effect, mean, std, expected
):
    assert expected(
        Statistics.required_sample_size_per_variant(
            minimum_effect=minimum_effect,
            mean=mean,
            std=std,
            n_variants=2,
        )
    )


@pytest.mark.parametrize(
    "n_variants, sample_size_per_variant",
    [
        (2, 400000),
        (2, 200000),
        (2, 627911),
        (3, 500000),
        (4, 300000),
    ],
)
def test_power_from_required_sample_size_per_variant(
    n_variants, sample_size_per_variant
):
    mean = 0.2
    std = 2.0
    minimum_effect = 0.05
    required_sample_size_per_variant = Statistics.required_sample_size_per_variant(
        n_variants=n_variants,
        std=std,
        mean=mean,
        minimum_effect=minimum_effect,
    )

    expected = TTestIndPower().solve_power(
        effect_size=(mean * (1 + minimum_effect) - mean) / std,
        ratio=1.0,
        alpha=0.05 / (n_variants - 1),
        power=None,
        nobs1=sample_size_per_variant,
    )

    power = Statistics.power_from_required_sample_size_per_variant(
        n_variants=n_variants,
        sample_size_per_variant=sample_size_per_variant,
        required_sample_size_per_variant=required_sample_size_per_variant,
    )

    assert np.allclose(power, expected, atol=1e-3)


def test_power_from_required_sample_size_per_variant_nan_params():
    assert np.isnan(
        Statistics.power_from_required_sample_size_per_variant(
            n_variants=np.nan,
            sample_size_per_variant=np.nan,
            required_sample_size_per_variant=np.nan,
        )
    )


@pytest.mark.parametrize(
    "args",
    [
        {
            "n_variants": 1,
            "sample_size_per_variant": 100,
            "required_sample_size_per_variant": 100,
        },
        {
            "n_variants": 2,
            "sample_size_per_variant": 0,
            "required_sample_size_per_variant": 0,
        },
    ],
)
def test_power_from_required_sample_size_per_variant_is_nan(args):
    assert np.isnan(Statistics.power_from_required_sample_size_per_variant(**args))


def test_false_positive_risk():
    false_positive_risk = Statistics.false_positive_risk(
        null_hypothesis_rate=1 - 0.33, power=0.8, p_value=0.025
    )

    assert false_positive_risk == 0.05966162065894922


# --------------------------------------------------------------------------------------
# quantile bootstrap evaluation
# --------------------------------------------------------------------------------------

# Asymptotic standard error of the median of a normal sample is
# `sqrt(pi / 2) * sigma / sqrt(n)`, i.e. about 25% larger than the standard error of the mean.
MEDIAN_SE_FACTOR = np.sqrt(np.pi / 2)


def test_bootstrap_standard_error_of_median_matches_asymptotic_formula():
    # The bootstrap standard error is itself a noisy estimate (relative Monte Carlo error
    # ~1/sqrt(2B) plus the slow convergence of the bootstrap for medians), so we average it
    # over independent samples before comparing it to the analytic value.
    rng = np.random.default_rng(101)
    n = 2000
    sigma = 2.0
    mu = 10.0

    # Standard error of the *relative* difference of two independent medians:
    # sqrt(2) * SE(median) / |median|.
    expected = np.sqrt(2) * MEDIAN_SE_FACTOR * sigma / np.sqrt(n) / mu

    estimates = []
    for _ in range(20):
        control = rng.normal(mu, sigma, n)
        treatment = rng.normal(mu, sigma, n)
        r = Statistics.quantile_bootstrap_evaluation(
            {"a": control, "b": treatment}, 0.5, "a", 0.95
        ).set_index("exp_variant_id")
        estimates.append(r.loc["b", "standard_error"])

    assert np.mean(estimates) == pytest.approx(expected, rel=0.1)


def test_bootstrap_median_standard_error_exceeds_mean_standard_error():
    rng = np.random.default_rng(102)
    n = 4000
    sigma = 3.0
    mu = 20.0
    control = rng.normal(mu, sigma, n)
    treatment = rng.normal(mu, sigma, n)

    r = Statistics.quantile_bootstrap_evaluation(
        {"a": control, "b": treatment}, 0.5, "a", 0.95
    ).set_index("exp_variant_id")

    # Standard error of the relative difference in means, for reference.
    mean_se = np.sqrt(2) * sigma / np.sqrt(n) / mu
    # The median pays about a 25% standard error premium on normal data.
    assert r.loc["b", "standard_error"] == pytest.approx(
        MEDIAN_SE_FACTOR * mean_se, rel=0.15
    )


def test_bootstrap_standard_error_scales_with_sample_size():
    rng = np.random.default_rng(103)
    sigma = 2.0
    mu = 10.0

    def mean_se(n):
        # A single bootstrap standard error is too noisy to compare directly, so average
        # over independent samples.
        estimates = [
            Statistics.quantile_bootstrap_evaluation(
                {"a": rng.normal(mu, sigma, n), "b": rng.normal(mu, sigma, n)},
                0.5,
                "a",
                0.95,
                n_samples=300,
            ).set_index("exp_variant_id")["standard_error"]["b"]
            for _ in range(15)
        ]
        return np.mean(estimates)

    # Quadrupling the sample size should roughly halve the standard error.
    assert mean_se(500) / mean_se(2000) == pytest.approx(2.0, rel=0.15)


def test_bootstrap_confidence_interval_covers_true_relative_difference():
    # Coverage sanity check: a 95% interval should contain the true relative difference in
    # medians for the large majority of independent replications.
    rng = np.random.default_rng(104)
    n = 800
    sigma = 2.0
    mu_c, mu_t = 10.0, 10.5
    true_diff = (mu_t - mu_c) / abs(mu_c)

    covered = 0
    replications = 40
    for _ in range(replications):
        r = Statistics.quantile_bootstrap_evaluation(
            {"a": rng.normal(mu_c, sigma, n), "b": rng.normal(mu_t, sigma, n)},
            0.5,
            "a",
            0.95,
            n_samples=200,
        ).set_index("exp_variant_id")
        diff = r.loc["b", "diff"]
        half_width = r.loc["b", "confidence_interval"]
        if abs(diff - true_diff) <= half_width:
            covered += 1

    # Nominal coverage is 95%; allow for Monte Carlo noise over 40 replications.
    assert covered >= 33, f"only {covered}/{replications} intervals covered"


def test_bootstrap_evaluation_of_percentile_recovers_shift():
    rng = np.random.default_rng(105)
    n = 5000
    # A pure location shift moves every quantile by the same amount.
    control = rng.normal(100, 10, n)
    treatment = control + 5

    r = Statistics.quantile_bootstrap_evaluation(
        {"a": control, "b": treatment}, 0.9, "a", 0.95
    ).set_index("exp_variant_id")

    expected = (np.quantile(treatment, 0.9) - np.quantile(control, 0.9)) / abs(
        np.quantile(control, 0.9)
    )
    assert r.loc["b", "quantile_estimate"] == pytest.approx(np.quantile(treatment, 0.9))
    assert r.loc["b", "diff"] == pytest.approx(expected)
    assert r.loc["b", "p_value"] < 0.001


def test_bootstrap_evaluation_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(106)
    values = {"a": rng.normal(10, 2, 300), "b": rng.normal(11, 2, 300)}

    first = Statistics.quantile_bootstrap_evaluation(values, 0.5, "a", 0.95)
    second = Statistics.quantile_bootstrap_evaluation(values, 0.5, "a", 0.95)

    assert first.equals(second)


def test_bootstrap_evaluation_returns_one_row_per_variant_in_input_order():
    rng = np.random.default_rng(107)
    values = {
        "b": rng.normal(10, 2, 200),
        "a": rng.normal(10, 2, 200),
        "c": rng.normal(10, 2, 200),
    }

    r = Statistics.quantile_bootstrap_evaluation(values, 0.5, "a", 0.95)

    assert list(r["exp_variant_id"]) == ["b", "a", "c"]
    assert list(r["degrees_of_freedom"]) == [398.0, 398.0, 398.0]


def test_bootstrap_evaluation_without_control_variant_is_all_nan():
    rng = np.random.default_rng(108)

    r = Statistics.quantile_bootstrap_evaluation(
        {"b": rng.normal(10, 2, 100)}, 0.5, "a", 0.95
    ).set_index("exp_variant_id")

    assert np.isnan(r.loc["b", "diff"])
    assert np.isnan(r.loc["b", "standard_error"])
