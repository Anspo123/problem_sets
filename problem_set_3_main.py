# =======================
# PROBLEM SET 3 - 5304
# GROUP 6
# =======================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from rdrobust import rdbwselect, rdrobust


# -----------------------
# PATHS AND VARIABLES
# -----------------------
data_path = Path(
    "5304/problem_set_3/input/quickbite_deliveries.csv"
)

output_path = Path(
    "5304/problem_set_3/output"
)

output_path.mkdir(
    parents=True,
    exist_ok=True
)

outcome = "delivery_time_min"
treatment = "used_routeoptima"

controls = [
    "distance_km",
    "restaurant_prep_min",
    "rain",
    "driver_experience_months"
]

cutoffs = {
    "switch_on": "minutes_from_switch_on",
    "switch_off": "minutes_from_switch_off"
}


# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(data_path)

required_variables = [
    outcome,
    treatment,
    "assigned_routeoptima",
    "minutes_from_switch_on",
    "minutes_from_switch_off"
] + controls

df = (
    df.dropna(subset=required_variables)
    .drop_duplicates()
    .reset_index(drop=True)
)


# -----------------------
# DESCRIPTIVE TABLE
# -----------------------
def descriptive_statistics(df, variables):
    """Create descriptive statistics by assignment status."""

    tables = []

    group_names = {
        0: "Not assigned",
        1: "Assigned"
    }

    for value, name in group_names.items():
        group_data = df[
            df["assigned_routeoptima"] == value
        ]

        statistics = (
            group_data[variables]
            .agg(["count", "mean", "std"])
            .T
            .rename(
                columns={
                    "count": f"N: {name}",
                    "mean": f"Mean: {name}",
                    "std": f"SD: {name}"
                }
            )
        )

        tables.append(statistics)

    stat_table = pd.concat(
        tables,
        axis=1
    ).round(2)

    return stat_table


# -----------------------
# SCATTER DATA
# -----------------------
def cutoff_scatter(
    df,
    running,
    window,
    filename
):
    """Plot the data around one cutoff."""

    variables = [
        (
            "assigned_routeoptima",
            "Assigned RouteOptima"
        ),
        (
            "used_routeoptima",
            "Used RouteOptima"
        ),
        (
            "delivery_time_min",
            "Delivery Time [min]"
        )
    ]

    sample = df[
        df[running].abs() <= window
    ].copy()

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 10),
        sharex=True
    )

    for axis, (variable, label) in zip(
        axes,
        variables
    ):
        # Raw observations
        axis.scatter(
            sample[running],
            sample[variable],
            color="lightgrey",
            alpha=0.25,
            s=10
        )

        # Divide the running variable into time bins
        sample["plot_bin"] = pd.cut(
            sample[running],
            bins=20
        )

        # Calculate the average within each bin
        binned_data = (
            sample.groupby(
                "plot_bin",
                observed=True
            )
            .agg(
                x_mean=(running, "mean"),
                y_mean=(variable, "mean")
            )
            .reset_index()
        )

        axis.scatter(
            binned_data["x_mean"],
            binned_data["y_mean"],
            color="darkblue",
            s=35
        )

        axis.axvline(
            x=0,
            color="black",
            linestyle="--"
        )

        axis.set_ylabel(label)
        axis.grid(alpha=0.2)

    axes[-1].set_xlabel(
        "Minutes from cutoff"
    )

    figure.tight_layout()

    figure.savefig(
        output_path / filename,
        dpi=300
    )

    plt.close(figure)


# -----------------------
# OPT. BANDWIDTH
# -----------------------
def select_optimal_bandwidth(
    running,
    add_controls,
    polynomial_degree
):
    """Select optimal bandwidth."""

    covariates = (
        df[controls]
        if add_controls
        else None
    )

    bandwidth_result = rdbwselect(
        y=df[outcome],
        x=df[running],
        fuzzy=df[treatment],
        covs=covariates,
        c=0,
        p=polynomial_degree,
        kernel="tri",
        bwselect="mserd"
    )

    bandwidth = float(
        bandwidth_result.bws.loc[
            "mserd",
            "h (left)"
        ]
    )

    return bandwidth


# -----------------------
# REGRESSIONS
# -----------------------
def estimate_rdd(running, add_controls, bandwidth, polynomial_degree):
    """Estimate fuzzy RDD and first stage."""

    polynomial_degree = int(polynomial_degree)

    covariates = (
        df[controls]
        if add_controls
        else None
    )

    # Estimate fuzzy RDD
    fuzzy_rdd = rdrobust(
        y=df[outcome],
        x=df[running],
        fuzzy=df[treatment],
        covs=covariates,
        c=0,
        p=polynomial_degree,
        h=bandwidth,
        kernel="tri"
    )

    # Select observations inside the bandwidth
    sample = df[
        df[running].abs() <= bandwidth
    ].copy()

    sample["assigned_x_running"] = (
        sample["assigned_routeoptima"]
        * sample[running]
    )

    # First-stage regression
    formula = (
        f"{treatment} ~ "
        f"assigned_routeoptima + "
        f"{running} + "
        "assigned_x_running"
    )

    if add_controls:
        formula += " + " + " + ".join(controls)

    first_stage = smf.ols(
        formula=formula,
        data=sample
    ).fit(cov_type="HC1")

    return fuzzy_rdd, first_stage


# -----------------------
# RDD FIGURE
# -----------------------
def rdd_plot(running, y, bandwidth, filename):
    """Plot binned averages and fitted lines."""

    sample = df[
        df[running].abs() <= bandwidth
    ][[running, y]].copy()

    sample["side"] = (
        sample[running] >= 0
    )

    # Create ten bins on each side
    sample["bin"] = (
        sample.groupby("side")[running]
        .transform(
            lambda x: pd.qcut(
                x,
                10,
                labels=False,
                duplicates="drop"
            )
        )
    )

    points = (
        sample.groupby(
            ["side", "bin"],
            observed=True
        )
        .agg(
            x=(running, "mean"),
            y=(y, "mean")
        )
        .reset_index()
    )

    plt.figure(figsize=(7, 4))

    for side, color, label in [
        (False, "steelblue", "Before cutoff"),
        (True, "darkorange", "After cutoff")
    ]:
        side_data = sample[
            sample["side"] == side
        ].copy()

        side_points = points[
            points["side"] == side
        ]

        # Observed binned averages
        plt.scatter(
            side_points["x"],
            side_points["y"],
            color=color,
            label=label
        )

        # Fitted line
        model = smf.ols(
            formula=f"{y} ~ {running}",
            data=side_data
        ).fit()

        side_data = side_data.sort_values(running)

        plt.plot(
            side_data[running],
            model.predict(side_data),
            color=color
        )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        label="Cutoff"
    )

    plt.xlabel("Minutes from cutoff")
    plt.ylabel(y.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / filename,
        dpi=300
    )

    plt.close()


# -----------------------
# SAVE RESULTS
# -----------------------
def save_results(
    first_stage,
    fuzzy_rdd,
    bandwidth,
    filename
):
    """Save first stage and fuzzy RDD in one file."""

    text = (
        f"Optimal bandwidth: {bandwidth:.2f}\n\n"
        "FIRST STAGE\n"
        "===========\n"
        f"{first_stage.summary().as_text()}\n\n"
        "FUZZY RDD\n"
        "=========\n"
        f"{fuzzy_rdd}"
    )

    (output_path / filename).write_text(
        text,
        encoding="utf-8"
    )


# -----------------------
# DESCRIPTIVE STATS.
# -----------------------
descriptive_variables = [
    "delivery_time_min",
    "used_routeoptima",
    "distance_km",
    "restaurant_prep_min",
    "rain",
    "driver_experience_months"
]

stat_table = descriptive_statistics(
    df=df,
    variables=descriptive_variables
)

(
    output_path / "descriptive_statistics.txt"
).write_text(
    stat_table.to_string(),
    encoding="utf-8"
)


# -----------------------
# SCATTER
# -----------------------
cutoff_scatter(
    df=df,
    running="minutes_from_switch_on",
    window=120,
    filename="descriptive_switch_on.png"
)

cutoff_scatter(
    df=df,
    running="minutes_from_switch_off",
    window=120,
    filename="descriptive_switch_off.png"
)


# -----------------------
# RUN ANALYSIS
# -----------------------
bandwidth = 60

for name, running in cutoffs.items():
    local_sample = df[
        df[running].abs() <= bandwidth
    ].copy()

    local_sample["after_cutoff"] = (
        local_sample[running] >= 0
    )

    print(f"\n{name}")

    print(
        local_sample.groupby("after_cutoff")[
            [
                "assigned_routeoptima",
                "used_routeoptima"
            ]
        ].mean()
    )

for cutoff_name, running in cutoffs.items():
    plot_bandwidth = None

    for add_controls in [False, True]:
        specification = (
            "with_controls"
            if add_controls
            else "without_controls"
        )

        optimal_bandwidth = select_optimal_bandwidth(
            running=running,
            add_controls=add_controls,
            polynomial_degree=1
        )

        fuzzy_rdd, first_stage = estimate_rdd(
            running=running,
            add_controls=add_controls,
            bandwidth=optimal_bandwidth,
            polynomial_degree=1
        )

        save_results(
            first_stage=first_stage,
            fuzzy_rdd=fuzzy_rdd,
            bandwidth=optimal_bandwidth,
            filename=(
                f"results_{cutoff_name}_"
                f"{specification}.txt"
            )
        )

        if add_controls:
            plot_bandwidth = optimal_bandwidth

    # Delivery-time figure
    rdd_plot(
        running=running,
        y=outcome,
        bandwidth=plot_bandwidth,
        filename=f"outcome_{cutoff_name}.png"
    )

    # First-stage figure
    rdd_plot(
        running=running,
        y=treatment,
        bandwidth=plot_bandwidth,
        filename=f"first_stage_{cutoff_name}.png"
    )
