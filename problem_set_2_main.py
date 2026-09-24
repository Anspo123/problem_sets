# =======================
# PROBLEM SET 2 - 5304
# GROUP 6
# =======================
"""
Prof. is particularly interested in:
• the overall causal effect of using the tool on exam scores,
• whether the effect differs across student subgroups,
• and the robustness of your conclusions.
"""
# -----------------------
# IMPORT MODULES
# -----------------------
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from linearmodels.iv import IV2SLS

# -----------------------
# FUNCTIONS
# -----------------------
def import_file(txt_path):

    return pd.read_csv(txt_path, sep=",").copy()


# -----------------------
# CLASSES
# -----------------------
class DataPreparation:
    """ Prepare data for analysis."""

    def __init__(self, df):

        self.df = df.copy()


    def check_dataset(self):

        if self.df.empty:
            raise ValueError("Empty dataset.")

        return self.df


    def clean_data(self):

        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self.df


    def sort_data(self):

        self.df.sort_values(by="student_id", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self.df


    def create_csv(self, df, path):

        try:
            df.to_csv(path, index=False, encoding="utf-8")

        except OSError as error:
            print(f"Could not create CSV: {error}")
            return None


class DataAnalysis:
    """Analyze data."""

    def __init__(self, df):

        self.df = df.copy()


    def load_data(self, yvar, xvars):

        y = self.df[yvar]
        x = self.df[xvars]

        # Add the regression intercept
        x = smf.add_constant(x)

        return y, x


    def ols_regression(self, y, x):

        formula = f"{y} ~ {x}"

        result = smf.ols(
            formula=formula,
            data=self.df
        ).fit(
            cov_type="HC1",
            use_t=True
        )

        print(result.summary2(float_format="%.3f"))

        return result


    def iv_regression(
        self,
        y,
        endogenous,
        instrument,
        controls=None
    ):

        if controls:
            formula = (
                f"{y} ~ 1 + {controls} "
                f"+ [{endogenous} ~ {instrument}]"
            )

        else:
            formula = (
                f"{y} ~ 1 "
                f"+ [{endogenous} ~ {instrument}]"
            )

        result = IV2SLS.from_formula(
            formula=formula,
            data=self.df
        ).fit(
            cov_type="robust"
        )

        print(result.summary)

        return result


    def save_results_as_text(self, result, output_path):

        # Statsmodels: 
        if hasattr(result, "summary2"):
            summary = result.summary2(
                float_format="%.3f"
            )

        # Linearmodels: 
        else:
            summary = result.summary

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(summary.as_text())


class VisualizeData:
    """ Visualize data."""

    def __init__(self, df):

        self.df = df.copy()


    def create_chart(self,
                     chart,
                     y,
                     x,
                     c_title,
                     x_label,
                     y_label,
                     output_path
    ):

        x_data = self.df[x]
        y_data = self.df[y]

        if chart.upper() == "SCATTERPLOT":

            male = self.df["female"] == 0
            female = self.df["female"] == 1

            plt.scatter(
            x[male],
            y[male],
            color="blue",
            alpha=0.6,
            label="Male"
            )

            plt.scatter(
                x[female],
                y[female],
                color="red",
                alpha=0.6,
                label="Female"
            )

            plt.legend()

        elif chart.upper() == "BOXPLOT":

            
            categories = x_data.dropna().unique()

            grouped_values = [
                y_data[x_data == category].dropna()
                for category in categories
            ]

            field_labels = [
                str(category).replace("_", " ").title()
                for category in categories
            ]

            plt.boxplot(
                grouped_values,
                tick_labels=field_labels
            )

            plt.xticks(rotation=45, ha="right")

        else:
            raise ValueError("Invalid chart.")

        plt.title(c_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


    def create_matrix(self, variables, c_title, output_path):
        axes = pd.plotting.scatter_matrix(
            self.df[variables],
            figsize=(12, 12),
            diagonal="hist",
            alpha=0.5
        )

        figure = axes[0, 0].figure
        figure.suptitle(
            c_title,
            fontsize=18,
            y=0.96
        )

        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(figure)


# -----------------------
# MAIN FUNCTION
# -----------------------
def main():
    """ Run the program. """

    # File paths:
    input_path = Path("Python") / "5304" / "problem_set_2" / "input"
    output_path = Path("Python") / "5304" / "problem_set_2" / "output"

    socratic_ai_data_path = input_path / "socratic_ai_data.csv"

    if not socratic_ai_data_path.exists():
        raise FileNotFoundError(f"File not found: {socratic_ai_data_path}")

    # Load data:
    socratic_ai_data_df = import_file(socratic_ai_data_path)

    # Clean and prepare data:
    prep = DataPreparation(socratic_ai_data_df)
    prep.check_dataset()
    prep.clean_data()
    ai_data_df = prep.sort_data()
    prep.create_csv(ai_data_df, output_path / "final_ai_dataset.csv")

    analyze_data = DataAnalysis(ai_data_df)
    visualize_data = VisualizeData(ai_data_df)

    # -----------------------
    # 1. Visualize data:
    # -----------------------
    """
    Are there notable patterns in the data?
    """

    # Make scatter matrix:
    variable_labels = {
    "hours_used": "Hours of SocraticAI",
    "exam_score": "Exam Score [0-100]",
    "gpa_prior": "Prior GPA [0.0-4.0]",
    }

    labelled_df = ai_data_df.rename(columns=variable_labels)

    labelled_visualize_data = VisualizeData(labelled_df)

    labelled_visualize_data.create_matrix(
        [
            "Hours of SocraticAI",
            "Exam Score [0-100]",
            "Prior GPA [0.0-4.0]",
        ],
        "Figure 1: Scatter Matrix of Non-binary Variables",
        output_path / "scatter_matrix_ai.png"
    )

    #Make boxplot:
    visualize_data.create_chart(
                "BOXPLOT",
                "exam_score",
                "female",
                "Figure 2: Exam Score over Gender",
                "Female = 1",
                "Exam Score [0-100]",
                output_path / "boxplot_gender_score.png"
    )

    # -----------------------
    # 2. IV-regression:
    # -----------------------
    """
    With "assigned" as instrument.
    """
    # -----------------
    # Without controls:
    # -----------------
    # First stage:

    print()
    print("REGRESSION 1:")
    iv_result1 = analyze_data.iv_regression(
        y="exam_score",
        endogenous="hours_used",
        instrument="assigned"
    )

    print()
    print("First stage:")
    print(iv_result1.first_stage.summary)

    analyze_data.save_results_as_text(iv_result1, output_path / "table_1.txt")

    # -----------------
    # With controls:
    # -----------------
    print()
    print("REGRESSION 2:")
    iv_result2 = analyze_data.iv_regression(
        y="exam_score",
        endogenous="hours_used",
        instrument="assigned",
        controls="gpa_prior + female + C(field)"
    )

    print()
    print("First stage (controls):")
    print(iv_result2.first_stage.summary)

    analyze_data.save_results_as_text(iv_result2, output_path / "table_2.txt")

    # -----------------------
    # 3. Intention-to-treat:
    #  / Reduced form
    # -----------------------
    """
    Correlation between "assigned" and exam score.
    """
    # -----------------
    # Without controls:
    # -----------------
    print()
    print("REGRESSION 3:")

    itt_result1 = analyze_data.ols_regression(
        y="exam_score",
        x="assigned"
    )

    analyze_data.save_results_as_text(itt_result1, output_path / "table_3.txt")

    # -----------------
    # With controls:
    # -----------------
    print()
    print("REGRESSION 4:")

    itt_result2 = analyze_data.ols_regression(
        y="exam_score",
        x="assigned + gpa_prior + female + C(field)"
    )

    analyze_data.save_results_as_text(itt_result2, output_path / "table_4.txt")

    # -----------------------
    # 3. OLS:
    # -----------------------
    # -----------------
    # Without controls:
    # -----------------
    print()
    print("REGRESSION 5:")

    ols_result1 = analyze_data.ols_regression(
        y="exam_score",
        x="hours_used"
    )

    analyze_data.save_results_as_text(ols_result1, output_path / "table_5.txt")

    # -----------------
    # With controls:
    # -----------------
    print()
    print("REGRESSION 5:")

    ols_result2 = analyze_data.ols_regression(
        y="exam_score",
        x="hours_used + gpa_prior + female + C(field)"
    )

    analyze_data.save_results_as_text(ols_result2, output_path / "table_6.txt")

    # -----------------------
    # 4. Interaction var.:
    # -----------------------
    # -----------------
    # Female x Assigned:
    # -----------------
    print()
    print("REGRESSION 7:")

    ols_result1 = analyze_data.ols_regression(
        y="exam_score",
        x="female * assigned"
    )

    analyze_data.save_results_as_text(ols_result1, output_path / "table_7.txt")

    # -----------------
    # Field x Assigned:
    # -----------------
    print()
    print("REGRESSION 8:")

    ols_result2 = analyze_data.ols_regression(
        y="exam_score",
        x="assigned * C(field)"
    )

    analyze_data.save_results_as_text(ols_result2, output_path / "table_8.txt")

# -----------------------
# MAIN GUARD
# -----------------------
if __name__ == "__main__":
    try:
        main()

    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        print(f"Error: {error}")
