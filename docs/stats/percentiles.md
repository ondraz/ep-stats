# Median and Percentile Metrics

Heavy-tailed metrics are the norm in experimentation. Revenue per user, time spent, number of
sessions — in all of them a handful of units contribute a large share of the total. The mean of such
a metric is dominated by its tail: it moves when a few extreme units happen to land in one variant,
which shows up as a wide confidence interval and a test with little power to detect a change in what
the *typical* user does.

A quantile — the median, or a percentile such as `p90` — answers a different question. Instead of
"how much revenue per user on average", it asks "what does the user in the middle do", or "what
happens at the top of the distribution". It is insensitive to how extreme the extremes are, so a
change in the body of the distribution is not drowned out by tail noise.

ep-stats supports comparing quantiles between variants with

```python
Metric(
    1,
    'Median Revenue per User',
    'value(test_unit_type.unit.revenue)',
    'count(test_unit_type.unit.exposure)',
    statistic='median',
)
```

and any percentile via `statistic='p90'` (the number may be fractional, e.g. `'p99.5'`).

The statistic is the quantile of the **per-unit nominator value**, including the zeros of exposed
units that did not convert. The denominator still supplies the `count` / sample size, but does not
enter the quantile itself.

!!! warning "By-unit evaluation only"
    Quantile metrics work only with
    [`evaluate_by_unit`][epstats.toolkit.experiment.Experiment.evaluate_by_unit].
    A quantile is not recoverable from the pre-aggregated sufficient statistics
    $(n, \sum x, \sum x^2)$, so [`evaluate_agg`][epstats.toolkit.experiment.Experiment.evaluate_agg],
    `evaluate_wide_agg` and the HTTP API reject them with a clear error rather than returning a
    silently wrong number. See [Why aggregates are not enough](#why-aggregates-are-not-enough).

## Sample quantiles and their asymptotic distribution

For a distribution $F$ and probability $p \in (0,1)$, the population quantile is

$$\xi_p = F^{-1}(p) = \inf\{x : F(x) \ge p\},$$

with $p = 0.5$ giving the median. Given an i.i.d. sample $X_1,\dots,X_n$, the sample quantile
$\hat{\xi}_p$ is the corresponding order statistic. ep-stats uses `np.quantile` with its default
linear interpolation between order statistics; the choice of interpolation rule does not affect any
of the asymptotics below.

If $F$ is differentiable at $\xi_p$ with density $f(\xi_p) > 0$, the Bahadur representation[^bahadur]
expresses the sample quantile as a smooth function of the empirical distribution function
$\hat{F}_n$,

$$\hat{\xi}_p = \xi_p + \frac{p - \hat{F}_n(\xi_p)}{f(\xi_p)} + o_P(n^{-1/2}),$$

from which asymptotic normality follows:

$$\sqrt{n}\,(\hat{\xi}_p - \xi_p) \;\xrightarrow{d}\; N\!\left(0,\; \frac{p(1-p)}{f(\xi_p)^2}\right),
\qquad \mathrm{SE}(\hat{\xi}_p) \approx \frac{1}{f(\xi_p)}\sqrt{\frac{p(1-p)}{n}}.$$

That the sample quantile is asymptotically normal is what makes the rest of ep-stats' machinery
apply: the test statistic, p-value and confidence interval below have exactly the same form as on
the mean path.

### Why aggregates are not enough

The mean has $\mathrm{SE}(\bar X) = \sigma/\sqrt n$, computable from the sufficient statistics
$(n, \sum x, \sum x^2)$ alone. This is precisely why the ordinary
[Welch's t-test](basics.md) works on ep-stats' pre-aggregated path.

A quantile is different in two ways:

1. **The point estimate needs the sample.** $\xi_p$ is not a function of moments; no finite
   collection of sums $\sum x^k$ determines it.
2. **The standard error needs the density.** The formula above depends on $f(\xi_p)$, the density
   *at* the quantile — a local property of the distribution that no set of moments pins down.

There are two ways to estimate that standard error:

- **Plug-in with kernel density estimation** of $f(\xi_p)$. This is sensitive to the bandwidth
  choice and behaves badly for discrete or mixed distributions — exactly the shape of most
  experimentation metrics (revenue with a large point mass at zero).
- **The bootstrap**, which is what ep-stats uses. It never forms a density estimate at all.

For a *single* sample there is also the distribution-free confidence interval built from order
statistics and the binomial distribution[^conover]. It is exact and assumption-light, but it does
not extend cleanly to an interval for the *difference* of quantiles between two samples, which is
what an experiment needs.

## Bootstrap estimation of the standard error

Efron's nonparametric bootstrap[^efron]: draw $B$ resamples of size $n$ with replacement from the
observed sample, compute $\hat{\xi}_p^{*b}$ on each, and estimate $\mathrm{SE}(\hat{\xi}_p)$ by the
empirical standard deviation of $\{\hat{\xi}_p^{*b}\}_{b=1}^B$.

For sample quantiles the bootstrap is consistent under the same condition that gives asymptotic
normality — $F$ differentiable at $\xi_p$ with $f(\xi_p) > 0$[^ghosh]. It implicitly estimates the
$1/f(\xi_p)$ factor without ever choosing a bandwidth. The failure mode is the same too: for heavily
discrete data with a point mass at or near $\xi_p$, neither normality nor the bootstrap standard
error is reliable. See [Assumptions and caveats](#assumptions-and-caveats).

## Two-sample test of relative difference

Let the control sample be $X_1,\dots,X_{n_C} \sim F_C$ and the treatment
$Y_1,\dots,Y_{n_T} \sim F_T$, independent. Matching ep-stats' convention of testing *relative*
differences, the null hypothesis is

$$H_0: \delta = \frac{\xi_p^T - \xi_p^C}{|\xi_p^C|} = 0 \quad \text{vs.} \quad H_1: \delta \neq 0,$$

with point estimate $\hat{\delta} = (\hat{\xi}_p^T - \hat{\xi}_p^C)/|\hat{\xi}_p^C|$.

Because $\hat{\xi}_p^T$ and $\hat{\xi}_p^C$ are independent and each asymptotically normal, the
delta method applied to $g(u,v) = (v-u)/|u|$ gives

$$\mathrm{Var}(\hat{\delta}) \approx \frac{1}{(\xi_p^C)^2}\,\mathrm{Var}(\hat{\xi}_p^T)
+ \frac{(\xi_p^T)^2}{(\xi_p^C)^4}\,\mathrm{Var}(\hat{\xi}_p^C),$$

the quantile analogue of the mean-path formula derived in [CTR Metric Redefined](ctr.md).

Rather than assembling this from two separately estimated standard errors, ep-stats bootstraps
$\hat\delta$ **directly**. Resampling is independent across variants, so pairing replicate $b$ of
the treatment with replicate $b$ of the control is arbitrary but valid, and gives

$$\widehat{\mathrm{SE}}(\hat{\delta}) = \mathrm{sd}\left\{
\frac{\hat{\xi}_p^{T,*b} - \hat{\xi}_p^{C,*b}}{|\hat{\xi}_p^{C,*b}|} \right\}_{b=1}^{B}.$$

This captures the same first-order variance as the delta-method expression while also reflecting the
nonlinearity of the ratio at finite $n$. Test statistic, p-value and confidence interval then mirror
the mean path exactly:

$$T = \frac{\hat{\delta}}{\widehat{\mathrm{SE}}(\hat{\delta})}, \qquad
\text{p-value} = 2\,\bigl(1 - F_t(|T|;\, \nu)\bigr), \qquad
\mathrm{CI} = \hat{\delta} \pm t_{1-\alpha/2,\nu}\,\widehat{\mathrm{SE}}(\hat{\delta}),$$

with $\nu = n_T + n_C - 2$. Asymptotically $T$ is standard normal; the $t$ reference with large
$\nu$ is numerically indistinguishable from the normal one and keeps the
[multiple comparison correction](multiple.md) and the output schema working unchanged.

!!! note "Wald-type interval, not a percentile bootstrap"
    The interval above is a symmetric *Wald-type* interval built from a bootstrap standard error.
    The alternative — a percentile-bootstrap interval taking empirical quantiles of the replicate
    distribution — is asymmetric and can behave better for very skewed data. It was not chosen
    because it produces no scalar half-width, and the Holm-Bonferroni confidence interval rescaling
    in ep-stats needs one.

### Monte Carlo error and the choice of $B$

The bootstrap standard error is itself estimated, with relative Monte Carlo error of about
$1/\sqrt{2B}$ — roughly $2.2\%$ at the default $B = 1000$, negligible next to the sampling noise it
is estimating. A fixed default seed makes repeated evaluations of the same data reproducible. Both
are overridable on [`Experiment`][epstats.toolkit.experiment.Experiment]:

```python
Experiment(
    'my-experiment',
    'a',
    metrics,
    checks,
    unit_type='test_unit_type',
    bootstrap_samples=2000,
    random_seed=42,
)
```

## Assumptions and caveats

### Continuity at the quantile

The theory requires $f(\xi_p) > 0$. For metrics with heavy ties or a point mass exactly at $\xi_p$
the quantile is degenerate — the classic example being median revenue when more than half of the
units buy nothing, making the median exactly $0$ in both variants. Every bootstrap resample then
returns the same value, the estimated standard error is $0$, and there is genuinely no evidence of a
difference to report. ep-stats guards this case: it reports `test_stat = 0` and a `NaN` p-value
rather than an infinite test statistic. If you hit it, pick a quantile in the continuous region of
the distribution (e.g. `p75` instead of the median), or use a mean or
[winsorized mean](outliers.md) metric.

### Zero control quantile

If $\hat{\xi}_p^C = 0$ the relative difference is undefined and comes out as `inf` / `NaN` — the same
convention the mean path uses when the control mean is zero.

### Winsorization

Capping per-unit values at the pooled $(l,\, 1-u)$ quantiles cannot change any sample quantile
strictly inside $(l,\, 1-u)$. The median is therefore invariant to, say, 1% winsorization, and
combining the two is harmless. A quantile that would fall *inside* a capped tail (e.g. `p99` with
`outlier_upper_percentile=1`) would be flattened onto the cap, so that combination is rejected when
the [`Metric`][epstats.toolkit.metric.Metric] is constructed.

### Sequential evaluation

The O'Brien-Fleming alpha-spending adjustment described in
[Sequential Evaluation](sequential.md) assumes an asymptotically normal test statistic with an
independent-increments structure. Sample quantiles are asymptotically normal, so applying the
adjustment to a quantile metric is a reasonable approximation — but an approximation, not an exact
guarantee.

### Multiple comparisons

The [Holm-Bonferroni correction](multiple.md) operates only on p-values and is valid for any
collection of tests, so quantile metrics need no special treatment and are corrected alongside mean
metrics.

### Sample size and power

The classical [sample size](sample_size.md) formula needs the standard deviation of the compared
statistic up front. For a quantile that is $\sqrt{p(1-p)/n}/f(\xi_p)$, which depends on the density
at the quantile and is not knowable before the experiment runs. `required_sample_size`, `power` and
`false_positive_risk` are therefore `NaN` for quantile metrics.

As a rule of thumb, for the median of a roughly normal metric,

$$\mathrm{SE}(\hat{\xi}_{0.5}) = \sqrt{\pi/2}\;\frac{\sigma}{\sqrt n} \approx 1.2533\,\frac{\sigma}{\sqrt n},$$

i.e. about 25% larger than the standard error of the mean, which needs roughly $1.2533^2 \approx 1.57$
— some 57% more units — to reach the same power. For heavy-tailed metrics the comparison usually goes
the *other* way: $\sigma$ is so inflated by the tail that the median needs far fewer units than the
mean to detect a change in the body of the distribution.

### What this test is not

- It is **not** a Mann-Whitney U test. That tests a different null hypothesis (stochastic dominance,
  $P(Y > X) = 1/2$), and can reject when the medians are identical.
- It is **not** a test of the mean of a trimmed or capped distribution. That is
  [winsorization](outliers.md), which estimates a mean, not a quantile.

## Choosing between mean, winsorized mean, and median

| | Estimand | Best when | Watch out for |
|---|---|---|---|
| **Mean** | $E[X]$ | The total matters (revenue, cost) and the tail is part of the effect you want to measure | Heavy tails inflate the variance and cost power; a few units can dominate |
| **Winsorized mean** | Mean of the capped metric | You want the total, but a handful of extreme units make it too noisy | Slight bias: the estimand is the capped metric, not the raw one. Keep the capped fraction small |
| **Median / percentile** | $\xi_p$ | You care about the typical unit, or a specific part of the distribution | Says nothing about the total; degenerate when a point mass sits at the quantile; no sample size estimate |

A useful default for heavy-tailed metrics is to evaluate both a winsorized mean and the median: they
answer different questions, and agreement between them is a strong signal.

[^bahadur]: [R. R. Bahadur, A Note on Quantiles in Large Samples (1966)](https://doi.org/10.1214/aoms/1177699450)
[^efron]: [B. Efron, Bootstrap Methods: Another Look at the Jackknife (1979)](https://doi.org/10.1214/aos/1176344552)
[^ghosh]: [M. Ghosh, W. C. Parr, K. Singh, G. J. Babu, A Note on Bootstrapping the Sample Median (1984)](https://doi.org/10.1214/aos/1176346815)
[^conover]: [W. J. Conover, Practical Nonparametric Statistics](https://www.wiley.com/en-us/Practical+Nonparametric+Statistics%2C+3rd+Edition-p-9780471160687), and [Wikipedia, Quantile](https://en.wikipedia.org/wiki/Quantile)
