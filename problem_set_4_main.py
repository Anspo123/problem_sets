# =======================
# PROBLEM SET 4 - 5304
# GROUP 6
# =======================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------
# PATHS AND VARIABLES
# -----------------------

data_path = Path(
    "5304/problem_set_4/input/valoria_municipalities.csv"
)

output_path = Path(
    "5304/problem_set_4/output"
)

output_path.mkdir(parents=True, exist_ok=True)

dependent_var = "corruption_index"

descriptive_vars = [
    dependent_var,
    "population",
    "gdp_per_capita",
    "urban",
    "mayor_education",
    "public_investment",
]

required_variables = [
    "municipality_id",
    "state_id",
    "year",
    "reform_state",
] + descriptive_vars


# -----------------------
# READ AND VALIDATE DATA
# -----------------------

df = pd.read_csv(data_path)

missing_columns = sorted(
    set(required_variables) - set(df.columns)
)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

missing_values = df[required_variables].isna().sum()
missing_values.to_csv(output_path / "missing_values.csv")

core_variables = [
    dependent_var,
    "municipality_id",
    "state_id",
    "year",
    "reform_state",
]

if df[core_variables].isna().any().any():
    raise ValueError(
        "Missing values in core variables. "
        "Inspect missing_values.csv before proceeding."
    )

if df.duplicated(["municipality_id", "year"]).any():
    raise ValueError("Duplicate municipality-year observations.")

df["year"] = df["year"].astype(int)

if not df["reform_state"].isin([0, 1]).all():
    raise ValueError("reform_state must equal 0 or 1.")

if df.groupby("municipality_id")["state_id"].nunique().gt(1).any():
    raise ValueError("Some municipalities change state_id.")

if df.groupby("state_id")["reform_state"].nunique().gt(1).any():
    raise ValueError("Reform status must be constant within each state.")

expected_years = set(range(2005, 2025))

if not df.groupby("municipality_id")["year"].apply(
    lambda years: set(years) == expected_years
).all():
    raise ValueError("Expected a balanced panel covering 2005–2024.")

df = df.sort_values(
    ["municipality_id", "year"]
).reset_index(drop=True)

# Post and pre-treatment:
df["post"] = (df["year"] >= 2015).astype(int)
df["did"] = df["reform_state"] * df["post"]


# -----------------------
# DESCRIPTIVE STATISTICS
# -----------------------

def descriptive_statistics(data, variables):
    """Observation-level statistics by reform status."""

    tables = []

    group_names = {
        0: "Non-reform states",
        1: "Reform states",
    }

    for value, name in group_names.items():
        group_data = data.loc[data["reform_state"] == value]

        statistics = (
            group_data[variables]
            .agg(["count", "mean", "std"])
            .T
            .rename(columns={
                "count": f"N: {name}",
                "mean": f"Mean: {name}",
                "std": f"SD: {name}",
            })
        )

        tables.append(statistics)

    return pd.concat(tables, axis=1)


# Pre-reform statistics: 2005–2014.
pre_reform = df.loc[df["year"] < 2015].copy()

descriptive_pre_table = descriptive_statistics(
    pre_reform,
    descriptive_vars,
)

descriptive_pre_table.round(3).to_csv(
    output_path / "descriptive_pre_reform.csv"
)


# Post-reform statistics: 2015–2024.
post_reform = df.loc[df["year"] >= 2015].copy()

descriptive_post_table = descriptive_statistics(
    post_reform,
    descriptive_vars,
)

descriptive_post_table.round(3).to_csv(
    output_path / "descriptive_post_reform.csv"
)


# Use pre-reform data to describe initial group differences.
pre_reform = df.loc[df["year"] < 2015].copy()

descriptive_table = descriptive_statistics(
    pre_reform,
    descriptive_vars,
)

descriptive_table.round(3).to_csv(
    output_path / "descriptive_pre_reform.csv"
)


# -----------------------
# HELPER FUNCTIONS
# -----------------------

def fit_clustered(formula, data):
    """OLS with state-clustered inference."""

    return smf.ols(
        formula=formula,
        data=data,
        missing="raise",
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data["state_id"],
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )


def coefficient_table(model, terms):
    """Export selected coefficients rather than every FE dummy."""

    confidence_intervals = model.conf_int().loc[terms]

    return pd.DataFrame({
        "coefficient": model.params.loc[terms],
        "std_error": model.bse.loc[terms],
        "p_value": model.pvalues.loc[terms],
        "ci_lower": confidence_intervals[0],
        "ci_upper": confidence_intervals[1],
        "n_observations": int(model.nobs),
    })


fixed_effects = "C(municipality_id) + C(year)"


# -----------------------
# MAIN DID REGRESSION
# -----------------------

did_model = fit_clustered(
    f"{dependent_var} ~ did + {fixed_effects}",
    df,
)

did_results = coefficient_table(did_model, ["did"])

did_results.to_csv(output_path / "did_results.csv")

print("\nMain DiD results:")
print(did_results.round(4))


# -----------------------
# CORRUPTION TRENDS
# -----------------------

mean_trends = (
    df.groupby(["year", "reform_state"])[dependent_var]
    .mean()
    .unstack("reform_state")
)

mean_trends.to_csv(output_path / "mean_corruption_trends.csv")

fig, ax = plt.subplots(figsize=(9, 5))

for group, label in {
    0: "Non-reform states",
    1: "Reform states",
}.items():
    ax.plot(
        mean_trends.index,
        mean_trends[group],
        marker="o",
        label=label,
    )

ax.axvline(2014.5, color="grey", linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("Mean corruption index")
ax.set_title("Corruption before and after the 2015 reform")
ax.legend()

fig.tight_layout()
fig.savefig(output_path / "corruption_trends.png", dpi=300)
plt.close(fig)


# -----------------------
# EVENT STUDY
# -----------------------

# Create reform-state × year indicators.
# Omit 2014 as the reference year.
reference_year = 2014

event_years = [
    year for year in sorted(df["year"].unique())
    if year != reference_year
]

event_terms = []

for year in event_years:
    term = f"reform_year_{year}"

    df[term] = (
        df["reform_state"] * (df["year"] == year).astype(int)
    )

    event_terms.append(term)

event_formula = (
    f"{dependent_var} ~ "
    + " + ".join(event_terms)
    + f" + {fixed_effects}"
)

event_model = fit_clustered(event_formula, df)

event_results = coefficient_table(event_model, event_terms)
event_results["year"] = event_years
event_results["event_time"] = event_results["year"] - 2015

event_results.to_csv(output_path / "event_study_results.csv")

# Add the normalized reference point for plotting.
# Its zero-width interval reflects normalization, not precision.
plot_data = pd.concat([
    event_results[["event_time", "coefficient", "ci_lower", "ci_upper"]],
    pd.DataFrame({
        "event_time": [-1],
        "coefficient": [0.0],
        "ci_lower": [0.0],
        "ci_upper": [0.0],
    }),
]).sort_values("event_time")

fig, ax = plt.subplots(figsize=(9, 5))

ax.errorbar(
    plot_data["event_time"],
    plot_data["coefficient"],
    yerr=np.vstack([
        plot_data["coefficient"] - plot_data["ci_lower"],
        plot_data["ci_upper"] - plot_data["coefficient"],
    ]),
    fmt="o",
    capsize=3,
)

ax.axhline(0, color="black", linewidth=1)
ax.axvline(-0.5, color="grey", linestyle="--")
ax.set_xlabel("Years relative to reform (2015 = 0)")
ax.set_ylabel("Estimated effect relative to 2014")
ax.set_title("Event study: 95% state-clustered confidence intervals")

fig.tight_layout()
fig.savefig(output_path / "event_study.png", dpi=300)
plt.close(fig)

# Joint test that all pre-reform coefficients equal zero.
pre_terms = [
    f"reform_year_{year}"
    for year in event_years
    if year < reference_year
]

restrictions = ", ".join(f"{term} = 0" for term in pre_terms)

pretrend_test = event_model.wald_test(
    restrictions,
    use_f=True,
    scalar=True,
)

(output_path / "pretrend_test.txt").write_text(
    str(pretrend_test),
    encoding="utf-8",
)

# -----------------------
# ROBUSTNESS CHECKS
# -----------------------

# 1. Exclude the implementation year.
# This changes the post-reform period to 2016–2024.
without_2015 = df.loc[df["year"] != 2015].copy()

model_without_2015 = fit_clustered(
    f"{dependent_var} ~ did + {fixed_effects}",
    without_2015,
)

coefficient_table(model_without_2015, ["did"]).to_csv(
    output_path / "did_excluding_2015.csv"
)

# 2. Placebo reform in 2010, using only pre-reform years.
# Nonzero estimates may indicate differential pre-reform trends.
placebo_data = pre_reform.copy()

placebo_data["placebo_did"] = (
    placebo_data["reform_state"]
    * (placebo_data["year"] >= 2010).astype(int)
)

placebo_model = fit_clustered(
    f"{dependent_var} ~ placebo_did + {fixed_effects}",
    placebo_data,
)

coefficient_table(placebo_model, ["placebo_did"]).to_csv(
    output_path / "placebo_2010.csv"
)

