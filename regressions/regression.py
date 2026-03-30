from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from config import FINAL_DATASET, RESULTS_DIR


MAIN_Y = "CAR_m1_p1"
CONTROLS = [
    "eps_surprise",
    "revenue_surprise",
    "log_marketCap",
    "log_AvgVolume_m20_m1",
    "log_n_tokens_qa",
]

EARNINGS_CONTROLS = [
    "eps_surprise",
    "revenue_surprise",
]


def load_data() -> pd.DataFrame:
    return pd.read_parquet(FINAL_DATASET)


def trim_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return x
    return x.clip(x.quantile(lower), x.quantile(upper))


def make_formula(y: str, main_vars: list[str], add_firm_fe: bool = False) -> str:
    rhs = main_vars + CONTROLS + ["C(year_quarter)"]
    if add_firm_fe:
        rhs.append("C(ticker)")
    return f"{y} ~ " + " + ".join(rhs)


def make_formula_progressive(
    y: str,
    main_vars: list[str],
    controls: list[str] | None = None,
    add_yq_fe: bool = False,
) -> str:
    rhs = list(main_vars)
    if controls:
        rhs += controls
    if add_yq_fe:
        rhs.append("C(year_quarter)")
    return f"{y} ~ " + " + ".join(rhs)


def fit_model(
    df: pd.DataFrame,
    formula: str,
    cov_type: str = "cluster",
):
    model = smf.ols(formula=formula, data=df)
    if cov_type == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": df["ticker"]})
    if cov_type == "HC1":
        return model.fit(cov_type="HC1")
    return model.fit()


def format_pvalue(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_pvalue_latex(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def escape_latex(text: str) -> str:
    replacements = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def get_rename_map() -> dict[str, str]:
    return {
        "NegPrepared": "Prepared Negativity",
        "NegGap": "Negative Gap",
        "NegQA": "Q&A Negativity",
        "NegPrepared_seglenw": "Prepared Negativity",
        "NegGap_seglenw": "Negative Gap",
        "NegQA_seglenw": "Q&A Negativity",
        "NegPrepared_segmax": "Prepared Negativity",
        "NegGap_segmax": "Negative Gap",
        "NegQA_segmax": "Q&A Negativity",
        "eps_surprise": "EPS surprise",
        "revenue_surprise": "Revenue surprise",
        "log_marketCap": "Log market cap",
        "log_AvgVolume_m20_m1": "Log avg. volume",
        "log_n_tokens_qa": "Log Q&A length",
    }


def print_result(title: str, result, main_vars: list[str]) -> str:
    vars_to_show = main_vars + CONTROLS

    lines = []
    lines.append(f"{title} | CAR[-1,+1]")
    lines.append("Variable                                 Coef.    Std.Err.    t-stat   p-value")
    lines.append("------------------------------------------------------------------------------")

    rename_map = get_rename_map()

    for var in vars_to_show:
        coef = result.params.get(var, np.nan)
        se = result.bse.get(var, np.nan)
        tval = result.tvalues.get(var, np.nan)
        pval = result.pvalues.get(var, np.nan)

        lines.append(
            f"{rename_map.get(var, var):35s}"
            f"{coef:10.4f}{se:11.4f}{tval:10.2f}{format_pvalue(pval):>10s}"
        )

    lines.append("------------------------------------------------------------------------------")
    lines.append(
        f"N = {int(result.nobs):,}   "
        f"R-squared = {result.rsquared:.3f}   "
        f"Adj. R-squared = {result.rsquared_adj:.3f}"
    )
    lines.append(f"Covariance: {result.cov_type}")
    return "\n".join(lines)


def make_table_label(title: str) -> str:
    label = title.lower()
    for old, new in [
        (" | ", "_"),
        (" ", "_"),
        ("+", "plus"),
        ("&", "and"),
        ("[", ""),
        ("]", ""),
        ("(", ""),
        (")", ""),
        (",", ""),
        ("-", "_"),
    ]:
        label = label.replace(old, new)
    while "__" in label:
        label = label.replace("__", "_")
    return f"tab:{label.strip('_')}"


def latex_result_table(title: str, result, main_vars: list[str]) -> str:
    vars_to_show = main_vars + CONTROLS
    rename_map = get_rename_map()

    cov_label = result.cov_type
    nobs = int(result.nobs)
    r2 = result.rsquared
    adj_r2 = result.rsquared_adj

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{escape_latex(title)} | CAR[-1,+1]}}")
    lines.append(rf"\label{{{make_table_label(title)}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\hline")
    lines.append(r"Variable & Coef. & Std. Err. & t-stat & p-value \\")
    lines.append(r"\hline")

    for var in vars_to_show:
        coef = result.params.get(var, np.nan)
        se = result.bse.get(var, np.nan)
        tval = result.tvalues.get(var, np.nan)
        pval = result.pvalues.get(var, np.nan)

        display_name = escape_latex(rename_map.get(var, var))
        lines.append(
            rf"{display_name} & {coef:.4f} & {se:.4f} & {tval:.2f} & {format_pvalue_latex(pval)} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"")
    lines.append(r"\vspace{0.3em}")
    lines.append(r"\begin{minipage}{0.9\linewidth}")
    lines.append(r"\footnotesize")
    lines.append(
        rf"\textit{{Notes.}} $N = {nobs:,}$, $R^2 = {r2:.3f}$, adjusted $R^2 = {adj_r2:.3f}$. Covariance estimator: {escape_latex(cov_label)}."
    )
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def latex_wald_table(wald: dict) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Wald test on main outcome}")
    lines.append(r"\label{tab:wald_main_outcome}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\hline")
    lines.append(r"Hypothesis & Coef. diff. & Std. Err. & t-stat & p-value \\")
    lines.append(r"\hline")
    lines.append(
        rf"{escape_latex(wald['hypothesis'])} & "
        rf"{wald['coef_diff']:.4f} & "
        rf"{wald['std_err']:.4f} & "
        rf"{wald['t_stat']:.2f} & "
        rf"{format_pvalue_latex(wald['p_value'])} \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def run_spec(
    df: pd.DataFrame,
    title: str,
    main_vars: list[str],
    cov_type: str = "cluster",
    add_firm_fe: bool = False,
) -> tuple[object, str]:
    formula = make_formula(MAIN_Y, main_vars, add_firm_fe=add_firm_fe)
    res = fit_model(df=df, formula=formula, cov_type=cov_type)
    txt = print_result(title, res, main_vars)
    return res, txt


def run_trimmed_spec(df: pd.DataFrame, title: str, main_vars: list[str]) -> tuple[object, str]:
    tmp = df.copy()

    trim_cols = list(set(main_vars + CONTROLS + [MAIN_Y]))
    for col in trim_cols:
        if col in tmp.columns:
            tmp[col] = trim_series(tmp[col])

    formula = make_formula(MAIN_Y, main_vars, add_firm_fe=False)
    res = fit_model(df=tmp, formula=formula, cov_type="cluster")
    txt = print_result(title, res, main_vars)
    return res, txt


def run_long_qa_spec(df: pd.DataFrame, title: str, main_vars: list[str]) -> tuple[object, str]:
    tmp = df.copy()
    cutoff = tmp["log_n_tokens_qa"].median()
    tmp = tmp.loc[tmp["log_n_tokens_qa"] >= cutoff].copy()

    formula = make_formula(MAIN_Y, main_vars, add_firm_fe=False)
    res = fit_model(df=tmp, formula=formula, cov_type="cluster")
    txt = print_result(title, res, main_vars)
    return res, txt


def wald_test_diff(result, var1: str, var2: str) -> dict:
    diff = result.params[var1] - result.params[var2]

    cov = result.cov_params()
    var_diff = (
        cov.loc[var1, var1]
        + cov.loc[var2, var2]
        - 2 * cov.loc[var1, var2]
    )
    se_diff = np.sqrt(var_diff)
    t_stat = diff / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "hypothesis": f"{var1} = {var2}",
        "coef_diff": float(diff),
        "std_err": float(se_diff),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def run_progressive_specs(df: pd.DataFrame) -> list[tuple[str, object, list[str]]]:
    specs = []

    spec_defs = [
        ("(1) NegGap only", ["NegGap"], [], False),
        ("(2) Prepared + NegGap", ["NegPrepared", "NegGap"], [], False),
        ("(3) + Earnings controls", ["NegPrepared", "NegGap"], EARNINGS_CONTROLS, False),
        ("(4) + Full controls", ["NegPrepared", "NegGap"], CONTROLS, False),
        ("(5) + Year-quarter FE", ["NegPrepared", "NegGap"], CONTROLS, True),
    ]

    for title, main_vars, controls, add_yq_fe in spec_defs:
        formula = make_formula_progressive(
            y=MAIN_Y,
            main_vars=main_vars,
            controls=controls,
            add_yq_fe=add_yq_fe,
        )
        res = fit_model(df=df, formula=formula, cov_type="cluster")
        specs.append((title, res, main_vars))

    return specs


def print_progressive_table(specs: list[tuple[str, object, list[str]]]) -> str:
    lines = []
    lines.append("Progressive specifications | CAR[-1,+1]")
    lines.append("-" * 122)

    headers = ["Variable"] + [title for title, _, _ in specs]
    lines.append(f"{headers[0]:30s}" + "".join(f"{h:>18s}" for h in headers[1:]))
    lines.append("-" * 122)

    vars_to_show = [
        "NegGap",
        "NegPrepared",
        "eps_surprise",
        "revenue_surprise",
        "log_marketCap",
        "log_AvgVolume_m20_m1",
        "log_n_tokens_qa",
    ]

    rename_map = get_rename_map()

    for var in vars_to_show:
        row = f"{rename_map.get(var, var):30s}"
        for _, res, _ in specs:
            coef = res.params.get(var, np.nan)
            row += f"{coef:18.4f}" if not pd.isna(coef) else f"{'':>18s}"
        lines.append(row)

        row_se = f"{'':30s}"
        for _, res, _ in specs:
            se = res.bse.get(var, np.nan)
            row_se += f"({se:.4f})".rjust(18) if not pd.isna(se) else f"{'':>18s}"
        lines.append(row_se)

    lines.append("-" * 122)

    for stat_name, getter in [
        ("Observations", lambda r: f"{int(r.nobs):,}"),
        ("R-squared", lambda r: f"{r.rsquared:.3f}"),
        ("Adj. R-squared", lambda r: f"{r.rsquared_adj:.3f}"),
    ]:
        row = f"{stat_name:30s}"
        for _, res, _ in specs:
            row += f"{getter(res):>18s}"
        lines.append(row)

    return "\n".join(lines)


def latex_progressive_table(specs: list[tuple[str, object, list[str]]]) -> str:
    cols = "l" + "c" * len(specs)
    headers = " & ".join([escape_latex(title) for title, _, _ in specs])

    vars_to_show = [
        "NegGap",
        "NegPrepared",
        "eps_surprise",
        "revenue_surprise",
        "log_marketCap",
        "log_AvgVolume_m20_m1",
        "log_n_tokens_qa",
    ]

    rename_map = get_rename_map()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Progressive specifications}")
    lines.append(r"\label{tab:progressive_specs}")
    lines.append(r"\scriptsize")
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\hline")
    lines.append(r"Variable & " + headers + r" \\")
    lines.append(r"\hline")

    for var in vars_to_show:
        coef_row = [escape_latex(rename_map.get(var, var))]
        se_row = [""]
        for _, res, _ in specs:
            coef = res.params.get(var, np.nan)
            se = res.bse.get(var, np.nan)
            coef_row.append("" if pd.isna(coef) else f"{coef:.4f}")
            se_row.append("" if pd.isna(se) else f"({se:.4f})")
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(se_row) + r" \\")

    lines.append(r"\hline")

    for stat_name, getter in [
        ("Observations", lambda r: f"{int(r.nobs):,}"),
        (r"$R^2$", lambda r: f"{r.rsquared:.3f}"),
        ("Adj. $R^2$", lambda r: f"{r.rsquared_adj:.3f}"),
    ]:
        row = [stat_name]
        for _, res, _ in specs:
            row.append(getter(res))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"")
    lines.append(r"\vspace{0.3em}")
    lines.append(r"\begin{minipage}{0.9\linewidth}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes.} The dependent variable is cumulative abnormal return over the $[-1,+1]$ window around the call date. Clustered standard errors at the firm level are reported in parentheses.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data().copy()

    output_blocks = []
    latex_blocks = []

    specs: list[tuple[str, object, list[str]]] = []

    # Baseline
    res_main, txt_main = run_spec(
        df=df,
        title="Main",
        main_vars=["NegPrepared", "NegGap"],
        cov_type="cluster",
    )
    output_blocks.append(txt_main)
    specs.append(("Main", res_main, ["NegPrepared", "NegGap"]))

    # Prepared + Q&A
    res_joint, txt_joint = run_spec(
        df=df,
        title="Prepared + Q&A",
        main_vars=["NegPrepared", "NegQA"],
        cov_type="cluster",
    )
    output_blocks.append(txt_joint)
    specs.append(("Prepared + Q&A", res_joint, ["NegPrepared", "NegQA"]))

    # Firm FE
    res_fe, txt_fe = run_spec(
        df=df,
        title="Firm FE",
        main_vars=["NegPrepared", "NegGap"],
        cov_type="cluster",
        add_firm_fe=True,
    )
    output_blocks.append(txt_fe)
    specs.append(("Firm FE", res_fe, ["NegPrepared", "NegGap"]))

    # HC1
    res_hc1, txt_hc1 = run_spec(
        df=df,
        title="HC1 SE",
        main_vars=["NegPrepared", "NegGap"],
        cov_type="HC1",
    )
    output_blocks.append(txt_hc1)
    specs.append(("HC1 SE", res_hc1, ["NegPrepared", "NegGap"]))

    # Trimmed
    res_trim, txt_trim = run_trimmed_spec(
        df=df,
        title="Trimmed",
        main_vars=["NegPrepared", "NegGap"],
    )
    output_blocks.append(txt_trim)
    specs.append(("Trimmed", res_trim, ["NegPrepared", "NegGap"]))

    # Long Q&A
    res_long, txt_long = run_long_qa_spec(
        df=df,
        title="Long Q&A",
        main_vars=["NegPrepared", "NegGap"],
    )
    output_blocks.append(txt_long)
    specs.append(("Long Q&A", res_long, ["NegPrepared", "NegGap"]))

    # Robustness: segment-length weighted
    res_seglenw, txt_seglenw = run_spec(
        df=df,
        title="Segment-length weighted",
        main_vars=["NegPrepared_seglenw", "NegGap_seglenw"],
        cov_type="cluster",
    )
    output_blocks.append(txt_seglenw)
    specs.append(("Segment-length weighted", res_seglenw, ["NegPrepared_seglenw", "NegGap_seglenw"]))

    # Robustness: most negative portion of response
    res_segmax, txt_segmax = run_spec(
        df=df,
        title="Most negative response portion",
        main_vars=["NegPrepared_segmax", "NegGap_segmax"],
        cov_type="cluster",
    )
    output_blocks.append(txt_segmax)
    specs.append(("Most negative response portion", res_segmax, ["NegPrepared_segmax", "NegGap_segmax"]))

    # Joint robustness: segment-length weighted
    res_joint_seglenw, txt_joint_seglenw = run_spec(
        df=df,
        title="Prepared + Q&A | segment-length weighted",
        main_vars=["NegPrepared_seglenw", "NegQA_seglenw"],
        cov_type="cluster",
    )
    output_blocks.append(txt_joint_seglenw)
    specs.append(("Prepared + Q&A | segment-length weighted", res_joint_seglenw, ["NegPrepared_seglenw", "NegQA_seglenw"]))

    # Joint robustness: most negative response portion
    res_joint_segmax, txt_joint_segmax = run_spec(
        df=df,
        title="Prepared + Q&A | most negative response portion",
        main_vars=["NegPrepared_segmax", "NegQA_segmax"],
        cov_type="cluster",
    )
    output_blocks.append(txt_joint_segmax)
    specs.append(("Prepared + Q&A | most negative response portion", res_joint_segmax, ["NegPrepared_segmax", "NegQA_segmax"]))

    # Progressive specifications
    progressive_specs = run_progressive_specs(df)
    txt_progressive = print_progressive_table(progressive_specs)
    output_blocks.append(txt_progressive)

    # Wald test on main joint spec
    wald = wald_test_diff(res_joint, "NegQA", "NegPrepared")
    output_blocks.append("Wald test on main outcome")
    output_blocks.append(str(wald))

    # Console/text output
    full_output = "\n\n".join(output_blocks)
    print(full_output)

    out_txt_path = RESULTS_DIR / "regression_results.txt"
    out_txt_path.write_text(full_output, encoding="utf-8")
    print(f"Saved single export: {out_txt_path}")

    # LaTeX output
    for title, result, main_vars in specs:
        latex_blocks.append(latex_result_table(title, result, main_vars))

    latex_blocks.append(latex_progressive_table(progressive_specs))
    latex_blocks.append(latex_wald_table(wald))

    full_latex = "\n\n".join(latex_blocks)

    out_tex_path = RESULTS_DIR / "regression_results_tables.tex"
    out_tex_path.write_text(full_latex, encoding="utf-8")
    print(f"Saved LaTeX tables: {out_tex_path}")


if __name__ == "__main__":
    main()


