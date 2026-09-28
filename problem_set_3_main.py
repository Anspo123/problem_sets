# =======================
# PROBLEM SET 3 - 5304
# GROUP 6
# =======================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from rdrobust import rdbwselect


# -----------------------
# SETTINGS
# -----------------------

input_path = Path("5304") / "problem_set_3" / "input"
output_path = Path("5304") / "problem_set_3" / "output"

data_path = input_path / "quickbite_deliveries.csv"

outcome = "delivery_time_min"
running_variable = "minutes_from_switch_on"
treatment = "used_routeoptima"

controls = [
    "distance_km",
    "restaurant_prep_min",
    "rain",
    "driver_experience_months"
]


# -----------------------
# DATA PREPARATION
# -----------------------

def load_data(path):
    """Load the QuickBite data."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def prepare_data(df):
    """Clean the variables needed for the RDD."""

    required_variables = [
        outcome,
        running_variable,
        treatment,
        "assigned_routeoptima"
    ] + controls

    missing_variables = [
        variable
        for variable in required_variables
        if variable not in df.columns
    ]

    if missing_variables:
        raise KeyError(
            f"Missing variables: {missing_variables}"
        )

    data = df.drop_duplicates().copy()

    data.sort_values(
        by="delivery_id",
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    # Convert relevant variables to numeric
    for variable in required_variables:
        data[variable] = pd.to_numeric(
            data[variable],
            errors="coerce"
        )

    data = data.dropna(subset=required_variables).copy()

    # The cutoff is zero
    data["above_cutoff"] = (
        data[running_variable] >= 0
    ).astype(int)

    # Allow the slope to differ on either side of the cutoff
    data["running_above"] = (
        data[running_variable] * data["above_cutoff"]
    )

    return data


def select_bandwidth(df, bandwidth):
    """Keep observations close to the cutoff."""

    return df[
        df[running_variable].abs() <= bandwidth
    ].copy()


# -----------------------
# DESCRIPTIVE STATISTICS
# -----------------------

def descriptive_statistics(df):
    """Create descriptive statistics around the cutoff."""

    variables = [
        outcome,
        treatment,
        "assigned_routeoptima",
        running_variable
    ] + controls

    table = (
        df.groupby("above_cutoff")[variables]
        .agg(["count", "mean", "std"])
        .round(3)
    )

    return table


# -----------------------
# RDD REGRESSIONS
# -----------------------
def select_optimal_bandwidth(df):
    """Find the MSE-optimal bandwidth for the fuzzy RDD."""

    result = rdbwselect(
        y=df["delivery_time_min"],
        x=df["minutes_from_switch_on"],
        c=0,
        fuzzy=df["used_routeoptima"],
        covs=df[controls],
        p=1,
        kernel="tri",
        bwselect="mserd"
    )

    print(result)

    bandwidth = result.bws.loc[
        "mserd",
        "h (left)"
    ]

    return float(bandwidth)

def estimate_first_stage(df, controls=None):
    """
    Test whether crossing the cutoff changes the probability
    of using RouteOptima.
    """

    controls = controls or []

    formula = (
        f"{treatment} ~ "
        f"above_cutoff + "
        f"{running_variable} + "
        f"running_above"
    )

    if controls:
        formula += " + " + " + ".join(controls)

    result = smf.ols(
        formula=formula,
        data=df
    ).fit(cov_type="HC1")

    return result


def estimate_reduced_form(df, controls=None):
    """
    Estimate the effect of crossing the cutoff on delivery time.
    This is the intention-to-treat effect.
    """

    controls = controls or []

    formula = (
        f"{outcome} ~ "
        f"above_cutoff + "
        f"{running_variable} + "
        f"running_above"
    )

    if controls:
        formula += " + " + " + ".join(controls)

    result = smf.ols(
        formula=formula,
        data=df
    ).fit(cov_type="HC1")

    return result


def estimate_fuzzy_rdd(df, controls=None):
    """
    Estimate the causal effect of using RouteOptima.

    Crossing the cutoff is used as an instrument for actual
    RouteOptima use.
    """

    controls = controls or []

    exogenous_variables = [
        running_variable,
        "running_above"
    ] + controls

    formula = (
        f"{outcome} ~ 1 + "
        f"{' + '.join(exogenous_variables)} "
        f"+ [{treatment} ~ above_cutoff]"
    )

    result = IV2SLS.from_formula(
        formula=formula,
        data=df
    ).fit(cov_type="robust")

    return result


# -----------------------
# RDD FIGURE
# -----------------------

def create_rdd_plot(df, bandwidth, output_path, number_of_bins=10):
    """Create binned averages and local regression lines."""

    plot_data = select_bandwidth(df, bandwidth)

    left = plot_data[
        plot_data[running_variable] < 0
    ].copy()

    right = plot_data[
        plot_data[running_variable] >= 0
    ].copy()

    # Bins are only used for the figure
    left["bin"] = pd.qcut(
        left[running_variable],
        q=number_of_bins,
        duplicates="drop"
    )

    right["bin"] = pd.qcut(
        right[running_variable],
        q=number_of_bins,
        duplicates="drop"
    )

    left_bins = (
        left.groupby("bin", observed=True)
        .agg(
            running_mean=(running_variable, "mean"),
            outcome_mean=(outcome, "mean")
        )
        .reset_index()
    )

    right_bins = (
        right.groupby("bin", observed=True)
        .agg(
            running_mean=(running_variable, "mean"),
            outcome_mean=(outcome, "mean")
        )
        .reset_index()
    )

    # Fit separate lines on either side
    left_line = smf.ols(
        f"{outcome} ~ {running_variable}",
        data=left
    ).fit()

    right_line = smf.ols(
        f"{outcome} ~ {running_variable}",
        data=right
    ).fit()

    left = left.sort_values(running_variable)
    right = right.sort_values(running_variable)

    plt.figure(figsize=(8, 5))

    plt.scatter(
        left_bins["running_mean"],
        left_bins["outcome_mean"],
        color="steelblue",
        label="Before switch-on"
    )

    plt.scatter(
        right_bins["running_mean"],
        right_bins["outcome_mean"],
        color="darkorange",
        label="After switch-on"
    )

    plt.plot(
        left[running_variable],
        left_line.predict(left),
        color="steelblue"
    )

    plt.plot(
        right[running_variable],
        right_line.predict(right),
        color="darkorange"
    )

    plt.axvline(
        x=0,
        color="black",
        linestyle="--",
        label="Cutoff"
    )

    plt.xlabel("Minutes from switch-on")
    plt.ylabel("Average delivery time [min]")
    plt.title("RDD plot: RouteOptima switch-on")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------
# ROBUSTNESS
# -----------------------

def bandwidth_robustness(df, bandwidths):
    """Estimate the fuzzy RDD using different bandwidths."""

    results = []

    for bandwidth in bandwidths:
        sample = select_bandwidth(df, bandwidth)

        model = estimate_fuzzy_rdd(
            sample,
            controls=controls
        )

        results.append({
            "bandwidth": bandwidth,
            "observations": model.nobs,
            "effect": model.params[treatment],
            "standard_error": model.std_errors[treatment],
            "p_value": model.pvalues[treatment]
        })

    return pd.DataFrame(results)


# -----------------------
# SAVE RESULTS
# -----------------------

def save_result(result, path):
    """Save a regression result as a text file."""

    if hasattr(result, "summary2"):
        text = result.summary2(
            float_format="%.3f"
        ).as_text()
    else:
        text = result.summary.as_text()

    path.write_text(text, encoding="utf-8")


# -----------------------
# MAIN
# -----------------------

def main():
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    raw_df = load_data(data_path)
    data_df = prepare_data(raw_df)

    data_df.to_csv(
        output_path / "clean_quickbite_data.csv",
        index=False
    )

    optimal_bandwith = select_optimal_bandwidth(data_df)

    print(
        f"Optimal bandwidth: "
        f"{optimal_bandwith:.2f} minutes"
    )

    rdd_df = select_bandwidth(
        data_df,
        optimal_bandwith
    )

    # Main sample around the cutoff
    rdd_df = select_bandwidth(
        data_df,
        optimal_bandwith
    )

    print(f"Observations in RDD sample: {len(rdd_df)}")

    # Descriptive statistics
    descriptive_table = descriptive_statistics(rdd_df)

    descriptive_table.to_csv(
        output_path / "descriptive_statistics.csv"
    )

    # First stage
    first_stage = estimate_first_stage(
        rdd_df,
        controls=controls
    )

    print("\nFIRST STAGE")
    print(first_stage.summary())

    save_result(
        first_stage,
        output_path / "first_stage.txt"
    )

    # Reduced form / intention-to-treat
    reduced_form = estimate_reduced_form(
        rdd_df,
        controls=controls
    )

    print("\nREDUCED FORM")
    print(reduced_form.summary())

    save_result(
        reduced_form,
        output_path / "reduced_form.txt"
    )

    # Fuzzy RDD
    fuzzy_rdd = estimate_fuzzy_rdd(
        rdd_df,
        controls=controls
    )

    print("\nFUZZY RDD")
    print(fuzzy_rdd.summary)

    save_result(
        fuzzy_rdd,
        output_path / "fuzzy_rdd.txt"
    )

    # RDD plot
    create_rdd_plot(
        data_df,
        bandwidth=optimal_bandwith,
        output_path=output_path / "rdd_plot.png"
    )

    # Robustness to bandwidth choice
    robustness_table = bandwidth_robustness(
        data_df,
        bandwidths=[30, 45, 60, 90, 120]
    )

    print("\nBANDWIDTH ROBUSTNESS")
    print(robustness_table.round(3))

    robustness_table.to_csv(
        output_path / "bandwidth_robustness.csv",
        index=False
    )


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        OSError
    ) as error:
        print(f"Error: {error}")
