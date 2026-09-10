# =======================
# PROBLEM SET 1 - 5304
# GROUP 6
# =======================
"""
1. The gender gap. Estimate the gender gap in daily steps using an appropriate regression
model. How large is the gap, and is it statistically significant?

2. Jonas’s proposal. Jonas recommends adding income and resting heart rate as controls.
Estimate the model he proposes and evaluate whether this specification is appropriate.
What happens to the estimated gender gap, and why?

3. Lena’s hypothesis. Test Lena’s claim that BMI is negatively associated with daily steps
among women. Use an appropriate one-sided test and report the p-value. What do you
conclude?

4. Lena’s specification. Lena also suggests including BMI, height, and weight together as
regressors. Estimate this model and discuss what you observe.

5. What else? Dr. Engel values thoroughness. Are there notable patterns in the data that
the analyses above may have missed?
"""
# -----------------------
# IMPORT MODULES
# -----------------------
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -----------------------
# FUNCTIONS
# -----------------------
def import_txt_file(txt_path):

    return pd.read_csv(txt_path, sep=r"\s+").copy()


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

        self.df = self.df.apply(pd.to_numeric, errors="coerce")
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self.df


class FinishDataset:
    """Create final dataset."""

    def __init__(self, df1, df0):
        self.df1 = df1.copy()
        self.df0 = df0.copy()


    def combine_datasets(self):
        self.df1["female"] = 1
        self.df0["female"] = 0

        return pd.concat(
            [self.df1, self.df0],
            ignore_index=True
        )


    def sort_data(self, df):

        df.sort_values(by="id", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


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
        x = sm.add_constant(x)

        return y, x


    def regression(self, model, y, x):

        # OLS:
        if model.upper() == "OLS":
            result = sm.OLS(y, x).fit(cov_type="HC1")

        # Probit:
        elif model.upper() == "PROBIT":
            if not set(y.dropna().unique()).issubset({0, 1}):
                raise ValueError(
                "Probit requires a binary dependent variable."
                )

            result = sm.Probit(y, x).fit(cov_type="HC1")

        else:
            raise ValueError(f"Unknown regression model: {model}")

        print(result.summary())

        return result


class VisualizeData:

    def __init__(self, df):

        self.df = df.copy()


    def save_results_as_text(self, result, output_path):

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(result.summary().as_text())


    def create_chart(self,
                     chart,
                     y,
                     x,
                     c_title,
                     x_lable,
                     y_lable,
                     output_path
    ):

        male = self.df["female"] == 0
        female = self.df["female"] == 1

        if chart.upper() == "SCATTERPLOT":

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

        else:
            raise ValueError("Invalid chart.")

        plt.legend()
        plt.title(c_title)
        plt.xlabel(x_lable)
        plt.ylabel(y_lable)
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

    # File paths:
    input_path = Path("Python") / "5304" / "problem_set_1" / "input"
    output_path = Path("Python") / "5304" / "problem_set_1" / "output"

    female_data_path = input_path / "data_female.txt"
    male_data_path = input_path / "data_male.txt"

    final_data_path = output_path / "final_dataset.csv"

    # Load data:
    female_df = import_txt_file(female_data_path)
    male_df = import_txt_file(male_data_path)

    # Clean data:
    try:
        prep_fem = DataPreparation(female_df)
        female_df = prep_fem.check_dataset()
        female_df_clean = prep_fem.clean_data()

        prep_male = DataPreparation(male_df)
        male_df = prep_male.check_dataset()
        male_df_clean = prep_male.clean_data()

    except ValueError:
        return

    # Prepare final dataset:
    finish_data = FinishDataset(female_df_clean, male_df_clean)
    combined_df = finish_data.combine_datasets()
    final_df = finish_data.sort_data(combined_df)
    finish_data.create_csv(final_df, final_data_path)

    visualize_data = VisualizeData(final_df)

    # Data analysis:
    # -----------------------
    # 1. The gender gap. 
    # -----------------------
    """ 
    Estimate the gender gap in daily steps
    using an appropriate regression model.
    """

    try: 
        # Load variables:
        data_analysis = DataAnalysis(final_df)
        y1, x1 = data_analysis.load_data(
            yvar ="steps",
            xvars =["female"]
        )

        # Regression ("OLS", y1, x1):
        ols_result = data_analysis.regression("OLS", y1, x1)

        # Save as text:
        visualize_data.save_results_as_text(
            ols_result,
            output_path / "ols_table.txt"
        )

    except ValueError:
        return

    # -----------------------
    # 2. The gender gap. 
    # -----------------------
    """
    Jonas recommends adding income and resting heart rate as controls.
    """

    try:
        # Load variables:
        y2, x2 = data_analysis.load_data(
            yvar ="steps",
            xvars =["female",
                    "income",
                    "resting_heart_rate"
            ]
        )   

        # Regression ("MULTIPLE OLS", y2, x2):
        ols_result_controls = data_analysis.regression("OLS", y2, x2)

        # Save as text:
        visualize_data.save_results_as_text(
            ols_result_controls,
            output_path / "ols_controls_table.txt"
        )

    except ValueError:
        return


    # -----------------------
    # 3. Steps and BMI. 
    # -----------------------
    """
    Test Lena’s claim that BMI is negatively associated with
    daily steps among women.
    """

    women = final_df[final_df["female"] == 1]
    women_analysis = DataAnalysis(women)

    try:

        y3, x3 = women_analysis.load_data(
        yvar="steps",
        xvars=["bmi"]
        )

        # Regression ("T-TEST", y2, x2)
        ols_result_women = data_analysis.regression("OLS", y3, x3)

        # Find t-value:
        test = ols_result_women.t_test("bmi = 0")

        coefficient = ols_result_women.params["bmi"]
        two_sided_p = float(test.pvalue)

        if coefficient < 0:
            one_sided_p = two_sided_p / 2

        else:
            one_sided_p = 1 - two_sided_p / 2

        print()
        print(f"One sided p-vale: {one_sided_p:.4f}")

    except ValueError:
        return

    # -----------------------
    # 4. Steps and BMI. 
    # -----------------------
    """
    Lena also suggests including BMI, height, and weight together as
    regressors.
    """

    try:

        # Load variables:
        y4, x4 = data_analysis.load_data(
            yvar ="steps",
            xvars =["female",
                    "bmi",
                    "height",
                    "weight"
            ]
        )

        # Regression ("MULTIPLE OLS", y2, x2):
        ols_result_controls_2 = data_analysis.regression("OLS", y4, x4)

        # Save as text:
        visualize_data.save_results_as_text(
            ols_result_controls_2,
            output_path / "ols_controls_2_table.txt"
        )

    except ValueError:
        return

    # -----------------------
    # 5. Other patterns. 
    # -----------------------
    """
    Are there notable patterns in the data that
    the analyses above may have missed?
    """

    # Make SCATTER MATRIX:
    visualize_data.create_matrix(
        [
        "steps",
        "bmi",
        "height",
        "weight",
        "income",
        "resting_heart_rate",
        "shoe_size",
        ],
        "Scatter Matrix",
        output_path / "scatter_matrix.png"
    )

    # -----------------------
    # 5. Added bonus.
    # -----------------------

    try:

        visualize_data.create_chart(
                    "SCATTERPLOT",
                    final_df["steps"], # x-variable
                    final_df["bmi"], # y-variable
                    "BMI and Average Daily Steps",
                    "BMI",
                    "Steps",
                    output_path / "scatterplot.png"
        )

    except ValueError as error:
        print(f"Chart error: {error}")
        return


# -----------------------
# MAIN GUARD
# -----------------------
if __name__ == "__main__":
    main()
