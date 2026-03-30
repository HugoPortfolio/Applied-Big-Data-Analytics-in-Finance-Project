from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

from config import FINAL_DATASET, RESULTS_DIR


OUTCOME = ("CAR_m1_p1", "CAR[-1,+1]")

CONTROLS = [
    "eps_surprise",
    "revenue_surprise",
    "log_marketCap",
    "log_AvgVolume_m20_m1",
    "log_n_tokens_qa",
]

FIXED_EFFECTS = "C(year_quarter)"

SPECS = [
    {
        "name": "main_gap",
        "label": "Main",
        "text_vars": ["NegPrepared", "NegGap"],
        "firm_fe": False,
        "cov_type": "cluster",
        "sample_rule": None,
    },
    {
        "name": "alt_levels",
        "label": "Prepared + Q&A",
        "text_vars": ["NegPrepared", "NegQA"],
        "firm_fe": False,
        "cov_type": "cluster",
        "sample_rule": None,
    },
    {
        "name": "firm_fe_gap",
        "label": "Firm FE",
        "text_vars": ["NegPrepared", "NegGap"],
        "firm_fe": True,
        "cov_type": "cluster",
        "sample_rule": None,
    },
    {
        "name": "hc1_gap",
        "label": "HC1 SE",
        "text_vars": ["NegPrepared", "NegGap"],
        "firm_fe": False,
        "cov_type": "HC1",
        "sample_rule": None,
    },
    {
        "name": "trimmed_gap",
        "label": "Trimmed",
        "text_vars": ["NegPrepared", "NegGap"],
        "firm_fe": False,
        "cov_type": "cluster",
        "sample_rule": "trim_1_99",
    },
    {
        "name": "long_qa_gap",
        "label": "Long Q&A",
        "text_vars": ["NegPrepared", "NegGap"],
        "firm_fe": False,
        "cov_type": "cluster",
        "sample_rule": "long_qa_only",
    },
]


def load_data() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DATASET)


def get_firm_id_column(df: pd.DataFrame) -> str:
    for col in ["gvkey", "company_id", "ticker", "permno"]:
        if col in df.columns:
            return col
    raise KeyError("No valid firm identifier found in the dataset.")


def pretty_label(var: str) -> str:
    mapping = {
        "NegPrepared": "Prepared Negativity",
        "NegGap": "Negative Gap",
        "NegQA": "Q&A Negativity",
        "eps_surprise": "EPS surprise",
        "revenue_surprise": "Revenue surprise",
        "log_marketCap": "Log market cap",
        "log_AvgVolume_m20_m1": "Log avg. volume",
        "log_n_tokens_qa": "Log Q&A length",
    }
    return mapping.get(var, var)


def format_pvalue(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_formula(outcome: str, text_vars: list[str], firm_id_col: str | None = None) -> str:
    rhs = text_vars + CONTROLS + [FIXED_EFFECTS]
    if firm_id_col is not None:
        rhs.append(f"C({firm_id_col})")
    return f"{outcome} ~ " + " + ".join(rhs)


def trim_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def apply_sample_rule(
    sample: pd.DataFrame,
    outcome: str,
    text_vars: list[str],
    sample_rule: str | None,
) -> pd.DataFrame:
    if sample_rule is None:
        return sample

    out = sample.copy()

    if sample_rule == "trim_1_99":
        for col in [outcome] + text_vars:
            out[col] = trim_series(out[col], 0.01, 0.99)
        return out

    if sample_rule == "long_qa_only":
        median_qa = out["log_n_tokens_qa"].median()
        return out.loc[out["log_n_tokens_qa"] >= median_qa].copy()

    raise ValueError(f"Unknown sample_rule: {sample_rule}")


def fit_model(
    df: pd.DataFrame,
    outcome: str,
    text_vars: list[str],
    firm_id_col: str,
    cov_type: str,
    add_firm_fe: bool,
    sample_rule: str | None,
):
    needed = [outcome, "year_quarter", firm_id_col] + text_vars + CONTROLS
    sample = df[needed].dropna().copy()
    sample = apply_sample_rule(sample, outcome, text_vars, sample_rule)

    formula = build_formula(outcome, text_vars, firm_id_col if add_firm_fe else None)

    if cov_type == "cluster":
        model = smf.ols(formula=formula, data=sample).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample[firm_id_col]},
        )
    else:
        model = smf.ols(formula=formula, data=sample).fit(cov_type=cov_type)

    return model


def print_model_table(model, spec_label: str, outcome_label: str, text_vars: list[str]) -> None:
    ordered_vars = text_vars + CONTROLS

    print()
    print(f"{spec_label} | {outcome_label}")
    print(f"{'Variable':<34}{'Coef.':>12}{'Std.Err.':>12}{'t-stat':>10}{'p-value':>10}")
    print("-" * 78)

    for var in ordered_vars:
        if var in model.params.index:
            print(
                f"{pretty_label(var):<34}"
                f"{model.params[var]:>12.4f}"
                f"{model.bse[var]:>12.4f}"
                f"{model.tvalues[var]:>10.2f}"
                f"{format_pvalue(model.pvalues[var]):>10}"
            )

    print("-" * 78)
    print(
        f"N = {int(model.nobs):,}   "
        f"R-squared = {model.rsquared:.3f}   "
        f"Adj. R-squared = {model.rsquared_adj:.3f}"
    )
    print(f"Covariance: {model.cov_type}")


def run_wald_test(model) -> dict:
    test = model.t_test("NegQA = NegPrepared")
    return {
        "hypothesis": "NegQA = NegPrepared",
        "coef_diff": float(np.asarray(test.effect).squeeze()),
        "std_err": float(np.asarray(test.sd).squeeze()),
        "t_stat": float(np.asarray(test.tvalue).squeeze()),
        "p_value": float(np.asarray(test.pvalue).squeeze()),
    }


def build_stargazer_table(models: list, labels: list[str], outcome_label: str) -> str:
    stargazer = Stargazer(models)
    stargazer.title(f"Determinants of {outcome_label}")
    stargazer.custom_columns(labels, [1] * len(models))
    stargazer.show_model_numbers(False)
    stargazer.show_degrees_of_freedom(False)
    stargazer.significant_digits(3)

    covariate_order = [
        "NegPrepared",
        "NegGap",
        "NegQA",
        "eps_surprise",
        "revenue_surprise",
        "log_marketCap",
        "log_AvgVolume_m20_m1",
        "log_n_tokens_qa",
    ]
    stargazer.covariate_order(covariate_order)
    stargazer.rename_covariates({var: pretty_label(var) for var in covariate_order})

    stargazer.add_line("Year-quarter FE", ["Yes"] * len(models))
    stargazer.add_line(
        "Firm FE",
        ["No", "No", "Yes", "No", "No", "No"],
    )
    stargazer.add_line(
        "Clustered SE",
        ["Yes", "Yes", "Yes", "No", "Yes", "Yes"],
    )
    stargazer.add_line(
        "HC1 SE",
        ["No", "No", "No", "Yes", "No", "No"],
    )
    stargazer.add_line(
        "Trimmed 1%-99%",
        ["No", "No", "No", "No", "Yes", "No"],
    )
    stargazer.add_line(
        "Long Q&A only",
        ["No", "No", "No", "No", "No", "Yes"],
    )

    return stargazer.render_latex()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    firm_id_col = get_firm_id_column(df)

    outcome_name, outcome_label = OUTCOME

    models = []
    labels = []
    wald_result = None

    for spec in SPECS:
        model = fit_model(
            df=df,
            outcome=outcome_name,
            text_vars=spec["text_vars"],
            firm_id_col=firm_id_col,
            cov_type=spec["cov_type"],
            add_firm_fe=spec["firm_fe"],
            sample_rule=spec["sample_rule"],
        )

        print_model_table(model, spec["label"], outcome_label, spec["text_vars"])

        models.append(model)
        labels.append(spec["label"])

        if spec["name"] == "alt_levels":
            wald_result = run_wald_test(model)

    latex_table = build_stargazer_table(models, labels, outcome_label)

    output_lines = []
    output_lines.append("% LaTeX table generated by Stargazer")
    output_lines.append(latex_table)

    if wald_result is not None:
        output_lines.append("\n\n% Wald test")
        output_lines.append("Wald test on main outcome")
        output_lines.append(f"Hypothesis: {wald_result['hypothesis']}")
        output_lines.append(f"Difference: {wald_result['coef_diff']:.6f}")
        output_lines.append(f"Std. Err.: {wald_result['std_err']:.6f}")
        output_lines.append(f"t-stat: {wald_result['t_stat']:.6f}")
        output_lines.append(f"p-value: {wald_result['p_value']:.6f}")

        print()
        print("Wald test on main outcome")
        print(wald_result)

    output_path = RESULTS_DIR / "regression_results.txt"
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    print()
    print(f"Saved single export: {output_path}")


if __name__ == "__main__":
    main()