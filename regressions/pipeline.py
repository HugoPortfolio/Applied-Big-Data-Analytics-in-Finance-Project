import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

from config import FINAL_DATASET, RESULTS_DIR


OUTCOMES = [
    ("CAR_m1_p1", "CAR[-1,+1]"),
    ("AbsRet_0_p1", "Absolute Return[0,+1]"),
    ("Volatility_0_p5", "Volatility[0,+5]"),
]

TEXT_SPECS = {
    "NegGap": "Negative Gap",
    "NegPrepared": "Prepared Negativity",
    "NegQA": "Q&A Negativity",
}

CONTROLS = [
    "eps_surprise",
    "revenue_surprise",
    "log_marketCap",
    "log_AvgVolume_m20_m1",
    "log_n_tokens_qa",
]

FIXED_EFFECTS = "C(year_quarter)"


def load_data() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DATASET)


def get_cluster_column(df: pd.DataFrame) -> str:
    for col in ["gvkey", "company_id", "ticker", "permno"]:
        if col in df.columns:
            return col
    raise KeyError("No valid cluster column found in the dataset.")


def build_formula(y: str, text_var: str) -> str:
    rhs = [text_var] + CONTROLS + [FIXED_EFFECTS]
    return f"{y} ~ " + " + ".join(rhs)


def fit_model(df: pd.DataFrame, y: str, text_var: str, cluster_col: str):
    needed = [y, text_var, "year_quarter", cluster_col] + CONTROLS
    sample = df[needed].dropna().copy()

    formula = build_formula(y, text_var)
    model = smf.ols(formula=formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample[cluster_col]},
    )
    return model


def format_pvalue(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_model_table(model, spec_label: str, outcome_label: str, text_var: str):
    rows = []

    variables = [
        (text_var, spec_label),
        ("eps_surprise", "EPS surprise"),
        ("revenue_surprise", "Revenue surprise"),
        ("log_marketCap", "Log market cap"),
        ("log_AvgVolume_m20_m1", "Log avg. volume"),
        ("log_n_tokens_qa", "Log Q&A length"),
    ]

    for var, label in variables:
        if var in model.params.index:
            rows.append([
                label,
                f"{model.params[var]:.4f}",
                f"{model.bse[var]:.4f}",
                f"{model.tvalues[var]:.2f}",
                format_pvalue(model.pvalues[var]),
            ])

    headers = ["Variable", "Coef.", "Std.Err.", "t-stat", "p-value"]
    widths = [28, 12, 12, 10, 10]

    def fmt(row):
        return (
            str(row[0]).ljust(widths[0]) +
            str(row[1]).rjust(widths[1]) +
            str(row[2]).rjust(widths[2]) +
            str(row[3]).rjust(widths[3]) +
            str(row[4]).rjust(widths[4])
        )

    print()
    print(f"{spec_label} | {outcome_label}")
    print(fmt(headers))
    print("-" * sum(widths))
    for row in rows:
        print(fmt(row))
    print("-" * sum(widths))
    print(
        f"N = {int(model.nobs):,}   "
        f"R-squared = {model.rsquared:.3f}   "
        f"Adj. R-squared = {model.rsquared_adj:.3f}"
    )
    print("Fixed effects: year-quarter   Clustered SE: yes")


def save_latex_table(models, column_labels, text_var: str, text_label: str):
    stargazer = Stargazer(models)

    stargazer.title(f"{text_label} and market reactions")
    stargazer.custom_columns(column_labels, [1] * len(models))
    stargazer.show_model_numbers(False)
    stargazer.show_degrees_of_freedom(False)
    stargazer.significant_digits(3)

    stargazer.covariate_order([
        text_var,
        "eps_surprise",
        "revenue_surprise",
        "log_marketCap",
        "log_AvgVolume_m20_m1",
        "log_n_tokens_qa",
    ])

    stargazer.rename_covariates({
        text_var: text_label,
        "eps_surprise": "EPS surprise",
        "revenue_surprise": "Revenue surprise",
        "log_marketCap": "Log market cap",
        "log_AvgVolume_m20_m1": "Log avg. volume",
        "log_n_tokens_qa": "Log Q&A length",
    })

    stargazer.add_line("Year-quarter FE", ["Yes"] * len(models))
    stargazer.add_line("Clustered SE", ["Yes"] * len(models))

    output_path = RESULTS_DIR / f"table_{text_var}.tex"
    output_path.write_text(stargazer.render_latex(), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    cluster_col = get_cluster_column(df)

    for text_var, text_label in TEXT_SPECS.items():
        print()
        print(text_label)
        print()

        models = []
        column_labels = []

        for y, outcome_label in OUTCOMES:
            model = fit_model(df, y, text_var, cluster_col)
            models.append(model)
            column_labels.append(outcome_label)
            print_model_table(model, text_label, outcome_label, text_var)

        save_latex_table(models, column_labels, text_var, text_label)


if __name__ == "__main__":
    main()


