# =======================
# MINI PROJECT - 5304
# MMR DID AND TRIPLE DIFFERENCE
# =======================

# Install once:
# python -m pip install pandas numpy scipy matplotlib

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# PATHS AND VARIABLES
# -----------------------

data_path = Path("final_assignment/input/wales_vaccination.csv")
output_path = Path("final_assignment/output")
output_path.mkdir(parents=True, exist_ok=True)

comparison_vaccines = ["dtp3_pct", "hib3_pct", "polio3_pct"]
treated_trusts = ["Glan-y-Mor NHS", "LLanelli NHS"]

# Quarter index is year * 4 + quarter - 1.
treatment_quarter = 1997 * 4 + 3  # Post begins in 1997Q4
reference_period = "1997Q2"       # Baseline before the Q3 campaign


# -----------------------
# CREATE AND EXPORT ANALYSIS DATA
# -----------------------

columns = [
    "trust",
    "age_group",
    "year",
    "quarter",
    "cohort_size",
    "mmr1_pct",
] + comparison_vaccines

df = pd.read_csv(data_path)

if not set(columns).issubset(df.columns):
    raise ValueError("Missing required input columns.")

for column in ["trust", "age_group"]:
    df[column] = df[column].str.strip()

df = df.drop_duplicates()
df = df.loc[df["age_group"].eq("2 year olds"), columns].copy()

for column in [
    "year",
    "quarter",
    "cohort_size",
    "mmr1_pct",
] + comparison_vaccines:
    df[column] = pd.to_numeric(df[column], errors="raise")

if df.empty or df.isna().any().any():
    raise ValueError(
        "Empty sample or missing analysis values; do not fill them with zeros."
    )

if df.duplicated(["trust", "year", "quarter"]).any():
    raise ValueError("Conflicting duplicate trust-quarter observations.")

for column in ["year", "quarter"]:
    if df[column].mod(1).ne(0).any():
        raise ValueError(f"Non-integer {column}.")
    df[column] = df[column].astype(int)

if (
    not df["quarter"].isin([1, 2, 3, 4]).all()
    or df["cohort_size"].le(0).any()
):
    raise ValueError("Invalid quarter or cohort size.")

if not (
    df[["mmr1_pct"] + comparison_vaccines]
    .apply(lambda x: x.between(0, 100))
    .all()
    .all()
):
    raise ValueError("Percentages must be between 0 and 100.")

df["period"] = (
    df["year"].astype(str) + "Q" + df["quarter"].astype(str)
)

df["quarter_index"] = df["year"] * 4 + df["quarter"] - 1
df["event_time"] = df["quarter_index"] - treatment_quarter
df["treated"] = df["trust"].isin(treated_trusts).astype(int)

# Q3 and earlier are pre-period; Q4 and later are post-period.
df["post"] = df["event_time"].ge(0).astype(int)
df["did"] = df["treated"] * df["post"]

# Mean uptake, not the fraction receiving all three vaccines.
# Each comparison vaccine receives equal weight.
df["other_vaccines_pct"] = df[comparison_vaccines].mean(axis=1)
df["mmr_gap_pp"] = df["mmr1_pct"] - df["other_vaccines_pct"]

df = df.sort_values(
    ["trust", "quarter_index"]
).reset_index(drop=True)

if set(df.loc[df["treated"].eq(1), "trust"]) != set(treated_trusts):
    raise ValueError("Both SWEP trusts must be present.")

if df.groupby("trust")["period"].apply(frozenset).nunique() != 1:
    raise ValueError("Trusts must have the same observed quarters.")

if df["treated"].nunique() != 2 or df["post"].nunique() != 2:
    raise ValueError(
        "Both exposure groups and both time windows are required."
    )

if reference_period not in set(df["period"]):
    raise ValueError("Event-study reference period is absent.")

# Only age-two observations: same birthday cohorts for all outcomes.
# Missing 1997Q1 is preserved as a calendar gap, not interpolated.
df.to_csv(
    output_path / "wales_mmr_ddd_data.csv",
    index=False,
)


# -----------------------
# OLS WITH TRUST-CLUSTERED INFERENCE
# -----------------------

def fit_model(data, outcome, terms=None):
    """Outcome = trust FE + year-quarter FE + supplied interaction(s).

    CR1 covariance:
    G/(G-1) * (N-1)/(N-K) * bread * clustered scores * bread.

    Conventional confidence intervals use t(G-1).
    Only two exposed trusts makes these approximations fragile.
    No independence across vaccine outcomes is assumed.
    """
    if terms is None:
        terms = pd.DataFrame(
            {"did": data["did"]},
            index=data.index,
        )

    design = pd.concat(
        [
            pd.Series(
                1.0,
                name="intercept",
                index=data.index,
            ),
            pd.get_dummies(
                data["trust"],
                prefix="trust",
                drop_first=True,
                dtype=float,
            ),
            pd.get_dummies(
                data["period"],
                prefix="quarter",
                drop_first=True,
                dtype=float,
            ),
            terms,
        ],
        axis=1,
    )

    X = design.to_numpy(float)
    y = data[outcome].to_numpy(float)
    n, k = X.shape

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Invalid estimation values.")

    if np.linalg.matrix_rank(X) != k:
        raise ValueError("Rank-deficient model.")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    bread = np.linalg.inv(X.T @ X)

    groups = data["trust"].unique()
    scores = np.array(
        [
            X[data["trust"].eq(g)].T
            @ residual[data["trust"].eq(g)]
            for g in groups
        ]
    )
    G = len(groups)

    cov = (
        bread
        @ (scores.T @ scores)
        @ bread
        * G / (G - 1)
        * (n - 1) / (n - k)
    )

    se = np.sqrt(np.maximum(np.diag(cov), 0))
    critical = stats.t.ppf(0.975, G - 1)

    t_values = np.divide(
        beta,
        se,
        out=np.full_like(beta, np.inf),
        where=se > 0,
    )

    table = pd.DataFrame(
        {
            "estimate_pp": beta,
            "cluster_se": se,
            "ci_low": beta - critical * se,
            "ci_high": beta + critical * se,
            "p_value": 2 * stats.t.sf(
                np.abs(t_values),
                G - 1,
            ),
            "observations": n,
        },
        index=design.columns,
    )

    covariance = pd.DataFrame(
        cov,
        index=design.columns,
        columns=design.columns,
    )

    return table, covariance


def result_row(data, outcome, label):
    table, _ = fit_model(data, outcome)

    return {
        "model": label,
        **table.loc["did"].to_dict(),
    }


# -----------------------
# MAIN DID, SPILLOVER CHECK AND DDD
# -----------------------

# MMR DiD: relative geographic change in the targeted vaccine.
# Other-vaccine DiD: relative change in potential spillover outcomes.
# DDD: estimate a DiD on the within-cell vaccine gap.
# The gap regression estimates its SE jointly.
# Never subtract separate standard errors.

models = {
    "mmr1_pct": "MMR DiD",
    "other_vaccines_pct": "Other-vaccine DiD",
    "mmr_gap_pp": "Triple difference: MMR minus other vaccines",
}

main_results = pd.DataFrame(
    [
        result_row(df, outcome, label)
        for outcome, label in models.items()
    ]
)

np.testing.assert_allclose(
    main_results.loc[2, "estimate_pp"],
    (
        main_results.loc[0, "estimate_pp"]
        - main_results.loc[1, "estimate_pp"]
    ),
    atol=1e-8,
)

main_results.to_csv(
    output_path / "main_results.csv",
    index=False,
)


# -----------------------
# TWO FOCUSED ROBUSTNESS CHECKS
# -----------------------

# 1. Exclude the actual campaign quarter, 1997Q3.
# Treatment still starts in Q4.
# Only two pre-period observations remain.

clean_pre = df.loc[df["period"].ne("1997Q3")]

robustness = [
    result_row(
        clean_pre,
        outcome,
        f"Exclude 1997Q3: {label}",
    )
    for outcome, label in models.items()
]

# 2. Use each comparison vaccine separately.
# Also report each vaccine's geographic DiD.

for vaccine in comparison_vaccines:
    sample = df.copy()
    sample["specific_gap"] = (
        sample["mmr1_pct"] - sample[vaccine]
    )

    robustness.append(
        result_row(
            sample,
            "specific_gap",
            f"DDD: MMR minus {vaccine}",
        )
    )

    robustness.append(
        result_row(
            sample,
            vaccine,
            f"Spillover diagnostic: {vaccine}",
        )
    )

pd.DataFrame(robustness).to_csv(
    output_path / "robustness_results_mmr_ddd.csv",
    index=False,
)


# -----------------------
# EXPLORATORY HETEROGENEITY: THE TWO EXPOSED TRUSTS
# -----------------------

# Both trust-specific estimates use the same 12 control trusts.
# Each relies on one exposed trust.
# Report descriptive point estimates only: CR1 inference for their
# difference can degenerate because shared control uncertainty cancels.

trust_terms = pd.DataFrame(
    {
        f"effect_{i}": (
            df["trust"].eq(trust).astype(int) * df["post"]
        )
        for i, trust in enumerate(treated_trusts)
    },
    index=df.index,
)

heterogeneity = []

for outcome in ["mmr1_pct", "mmr_gap_pp"]:
    table, _ = fit_model(df, outcome, trust_terms)

    for i, trust in enumerate(treated_trusts):
        heterogeneity.append(
            {
                "outcome": outcome,
                "comparison": trust,
                "estimate_pp": table.loc[
                    f"effect_{i}", "estimate_pp"
                ],
                "observations": len(df),
            }
        )

    difference = (
        table.loc["effect_0", "estimate_pp"]
        - table.loc["effect_1", "estimate_pp"]
    )

    heterogeneity.append(
        {
            "outcome": outcome,
            "comparison": "Glan-y-Mor minus Llanelli",
            "estimate_pp": difference,
            "observations": len(df),
        }
    )

pd.DataFrame(heterogeneity).to_csv(
    output_path / "trust_heterogeneity.csv",
    index=False,
)


# -----------------------
# EVENT STUDIES
# -----------------------

periods = (
    df[["period", "event_time"]]
    .drop_duplicates()
    .sort_values("event_time")
)

# Q2 is the omitted event-study reference.
reference_time = periods.loc[
    periods["period"].eq(reference_period),
    "event_time",
].iloc[0]

# The actual campaign occurred in Q3.
campaign_time = periods.loc[
    periods["period"].eq("1997Q3"),
    "event_time",
].iloc[0]

terms = pd.DataFrame(
    {
        f"event_{period}": (
            df["treated"]
            * df["period"].eq(period).astype(int)
        )
        for period in periods["period"]
        if period != reference_period
    },
    index=df.index,
)

events = []

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.5),
    sharey=True,
)

for ax, (outcome, label) in zip(axes, models.items()):
    table, _ = fit_model(df, outcome, terms)
    rows = []

    for period, event_time in periods.itertuples(
        index=False,
        name=None,
    ):
        if period == reference_period:
            values = {
                "estimate_pp": 0.0,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        else:
            values = table.loc[
                f"event_{period}"
            ].to_dict()

        rows.append(
            {
                "model": label,
                "period": period,
                "event_time": event_time,
                **values,
            }
        )

    plot_data = pd.DataFrame(rows)

    estimated = plot_data.loc[
        plot_data["period"].ne(reference_period)
    ]

    ax.errorbar(
        estimated["event_time"],
        estimated["estimate_pp"],
        yerr=[
            estimated["estimate_pp"] - estimated["ci_low"],
            estimated["ci_high"] - estimated["estimate_pp"],
        ],
        fmt="o",
        capsize=3,
    )

    # Q2 is normalized to zero, not estimated precisely.
    ax.scatter(
        [reference_time],
        [0],
        color="black",
        marker="s",
    )

    ax.axhline(
        0,
        color="grey",
        linewidth=1,
    )

    # Boundary between Q3 and Q4.
    # Q4 has event_time = 0.
    ax.axvline(
        -0.5,
        color="grey",
        linestyle="--",
    )

    # Shade the actual Q3 campaign quarter.
    ax.axvspan(
        campaign_time - 0.45,
        campaign_time + 0.45,
        color="orange",
        alpha=0.12,
    )

    ax.set_title(
        label.replace(
            "Triple difference: ",
            "DDD:\n",
        ),
        fontsize=10,
    )

    ax.set_xticks(periods["event_time"])
    ax.set_xticklabels(
        periods["period"],
        rotation=65,
        ha="right",
        fontsize=8,
    )

    events.extend(rows)

axes[0].set_ylabel(
    "Relative change (percentage points)"
)

fig.suptitle(
    f"Age-two event studies; reference {reference_period}"
)

fig.text(
    0.5,
    0.015,
    "Pointwise 95% CR1 intervals; only 2 exposed trusts. "
    "Shading: 1997Q3 campaign. Dashed line: 1997Q4 post start.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.07, 1, 0.93]
)

fig.savefig(
    output_path / "mmr_ddd_event_study.png",
    dpi=300,
)

plt.close(fig)

pd.DataFrame(events).to_csv(
    output_path / "mmr_ddd_event_study_results.csv",
    index=False,
)


# -----------------------
# OBSERVED TREATMENT AND CONTROL TRENDS
# -----------------------

# Each point is the equally weighted mean across trusts.
# These are observed uptake levels, not estimated treatment effects.

trend_outcomes = {
    "mmr1_pct": "MMR",
    "other_vaccines_pct": "Mean DTP3, Hib3 and polio3",
}

mean_trends = (
    df.groupby(
        ["period", "event_time", "treated"]
    )[list(trend_outcomes)]
    .mean()
    .reset_index()
    .sort_values("event_time")
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
    sharey=True,
)

for ax, (outcome, label) in zip(
    axes,
    trend_outcomes.items(),
):
    for treated, group_label, color in [
        (0, "Other Wales: control", "steelblue"),
        (1, "SWEP area: treatment", "darkorange"),
    ]:
        group = mean_trends.loc[
            mean_trends["treated"].eq(treated)
        ]

        ax.plot(
            group["event_time"],
            group[outcome],
            marker="o",
            color=color,
            label=group_label,
        )

    # Boundary between Q3 and Q4.
    ax.axvline(
        -0.5,
        color="grey",
        linestyle="--",
    )

    # Shade the actual Q3 campaign quarter.
    ax.axvspan(
        campaign_time - 0.45,
        campaign_time + 0.45,
        color="orange",
        alpha=0.12,
    )

    ax.set_title(label)
    ax.set_xticks(periods["event_time"])
    ax.set_xticklabels(
        periods["period"],
        rotation=65,
        ha="right",
        fontsize=8,
    )

axes[0].set_ylabel(
    "Mean uptake at age two (%)"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.045),
    ncol=2,
    frameon=False,
)

fig.suptitle(
    "Observed vaccination uptake: treatment and control groups"
)

fig.text(
    0.5,
    0.015,
    "Shading: 1997Q3 campaign. Dashed line: 1997Q4 post start.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.16, 1, 0.93]
)

fig.savefig(
    output_path / "mmr_other_vaccine_trends.png",
    dpi=300,
)

plt.close(fig)


# -----------------------
# PRINT RESULTS
# -----------------------

print("\nMain estimates (percentage points):")
print(main_results.round(3).to_string(index=False))

print(
    f"\nData, tables and figures exported to: "
    f"{output_path.resolve()}"
)