"""Render the main and supplementary sensitivity tables from evaluated results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/validation/configuration_extension_20261111_v1"


def main():
    summary = pd.read_csv(RUN / "evaluation/summary.csv").set_index("variant_id")
    effects = pd.read_csv(RUN / "evaluation/paired_effects.csv")
    effects = effects.loc[effects.comparator.eq("full")].set_index("variant_id")
    out = RUN / "manuscript"
    out.mkdir(exist_ok=True)
    common = (r"Matching required overlap between a TDA detection range and the reference month $\pm1$ month, "
              r"with one-to-one assignment within each stand. "
              r"$\Delta F_{1,\mathrm{ref}}$ denotes each alternative minus the baseline, with 95\% percentile "
              r"intervals from 5,000 paired stand-level bootstrap resamples. Bold values mark column maxima for precision, recall, and F1.")
    main_groups = [
        (None, [("full", "Baseline")]),
        ("Context length", [("W24_D12_S12", r"$W=24$ months"), ("W48", r"$W=48$ months"), ("W60", r"$W=60$ months")]),
        ("Search schedule", [("W36_D12_S6", r"$D=12$, $S=6$ months"), ("W36_D6_S6", r"$D=S=6$ months")]),
        ("Polarization", [("VH", "VH"), ("VV_union_VH", "VV--VH union"), ("VV_intersection_VH", "VV--VH intersection")]),
    ]
    caption = (r"Configuration sensitivity of windowed CuSum under common TDA rules. The baseline used VV, "
               r"$W=36$, $D=S=12$, $\alpha_{\mathrm{ts}}=0.025$, and unrestricted permutation. "
               r"Context-length and polarization comparisons retained $D=S=12$; search-schedule comparisons retained $W=36$. " + common)
    (out / "table_main.tex").write_text(table(summary, effects, main_groups, caption,
                                              "tab:cusum_configuration_sensitivity", True), encoding="utf-8")
    thresholds = [(None, [("full", r"$\alpha_{\mathrm{ts}}=0.025$ (baseline)")]),
                  (None, [(f"alpha_{a}", rf"$\alpha_{{\mathrm{{ts}}}}={a}$") for a in ("0.005", "0.01", "0.05", "0.10")])]
    caption = (r"Acceptance-threshold sensitivity for VV with $W=36$ and $D=S=12$, evaluated on 1,049 reference records "
               r"from 459 stands. $N_{\mathrm{det}}$ and $M$ denote TDA reports and matched reference records. " + common)
    (out / "table_threshold.tex").write_text(table(summary, effects, thresholds, caption,
                                                   "tab:si_acceptance_threshold", False), encoding="utf-8")
    nulls = [(None, [("full", "Unrestricted (baseline)"), ("block3", "Three-month blocks"), ("block6", "Six-month blocks")])]
    caption = (r"Temporal-null sensitivity for VV with $W=36$, $D=S=12$, and $\alpha_{\mathrm{ts}}=0.025$, "
               r"evaluated on the same 1,049 reference records from 459 stands. " + common)
    (out / "table_null.tex").write_text(table(summary, effects, nulls, caption,
                                              "tab:si_temporal_null", False), encoding="utf-8")
    print("Rendered the main table and two supplementary tables from evaluated CSVs.")


def table(summary, effects, groups, caption, label, main_table):
    names = [name for _, rows in groups for name, _ in rows]
    metrics = ("catalogue_precision", "recall", "catalogue_f1")
    maxima = summary.loc[names, list(metrics)].max()
    environment, placement = ("table*", "pos=!htbp") if main_table else ("table", "!htbp")
    lines = [rf"\begin{{{environment}}}[{placement}]", r"    \centering\small",
             r"    \setlength{\tabcolsep}{3pt}", "    \\caption{" + caption + "}",
             "    \\label{" + label + "}",
             r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrc@{}}",
             r"        \toprule",
             r"        Configuration & $N_{\mathrm{det}}$ & $M$ & $P_{\mathrm{ref}}$ & $R_{\mathrm{ref}}$ & $F_{1,\mathrm{ref}}$ & $\Delta F_{1,\mathrm{ref}}$ (95\% CI) \\",
             r"        \midrule"]
    for i, (heading, rows) in enumerate(groups):
        if i:
            lines.append(r"        \midrule")
        if heading:
            lines.append(r"        \multicolumn{7}{@{}l}{\textit{" + heading + r"}} \\")
        for name, display in rows:
            row = summary.loc[name]
            cells = [rf"\textbf{{{row[m]:.3f}}}" if row[m] == maxima[m] else f"{row[m]:.3f}" for m in metrics]
            delta = "--" if name == "full" else (
                rf"${effects.loc[name, 'difference']:+.3f}$ "
                rf"[${effects.loc[name, 'ci_low']:+.3f}$, ${effects.loc[name, 'ci_high']:+.3f}$]")
            lines.append(f"        {display} & {row.n_reports:,.0f} & {row.n_matches:,.0f} & " + " & ".join(cells) + " & " + delta + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular*}", rf"\end{{{environment}}}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
