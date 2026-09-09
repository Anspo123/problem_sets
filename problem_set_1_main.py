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
the analyses above may have missed
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


    def make_graph(
            self,
            graph,
            y,
            x,
            x_label,
            y_label,
            g_title
    ):

        if graph.upper() == "SCATTERPLOT":
            plt.scatter(
                self.df[y],
                self.df[x]
            )

        plt.xlabel(x_label.strip().title())
        plt.ylabel(y_label.strip().title())
        plt.title(g_title.strip().title())
        plt.tight_layout()
        plt.show()


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
        print("Error: Invalid data.")
        return

    # Prepare final dataset:
    finish_data = FinishDataset(female_df_clean, male_df_clean)
    combined_df = finish_data.combine_datasets()
    final_df = finish_data.sort_data(combined_df)
    finish_data.create_csv(final_df, final_data_path)

    # Data analysis:
    # -----------------------
    # 1. The gender gap. 
    # -----------------------
    """ 
    Estimate the gender gap in daily steps
    using an appropriate regression model.
    """
    # Load variables:
    data_analysis = DataAnalysis(final_df)
    y1, x1 = data_analysis.load_data(
        yvar="steps",
        xvars=["female"]
    )

    # Make scatterplot:
    data_analysis.make_graph(
            "SCATTERPLOT",
            y1,
            x1,
            "INCOME",
            "STEPS",
            "SCATTERPLOT"
    )


    # Do regression (OLS):
    ols_result = data_analysis.regression("OLS", y1, x1)

    y2, x2 = data_analysis.load_data(
        yvar="steps",
        xvars=["female", "weight"]
    )

    multi_ols_result = data_analysis.regression("OLS", y2, x2)

    y3, x3 = data_analysis.load_data(
            yvar="female",
            xvars=["steps", "weight"]
        )
    
    probit_result = data_analysis.regression("PROBIT", y3, x3)


# -----------------------
# MAIN GUARD
# -----------------------
if __name__ == "__main__":
    main()
