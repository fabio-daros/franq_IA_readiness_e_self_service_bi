"""Generate the Part 2 statistical analysis notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "notebooks" / "finlend_default_hypothesis_analysis.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3.12 — FinLend",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    cells = []

    cells.append(
        md(
            """
# FinLend — Análise estatística da inadimplência

**Parte 2:** o Head de Risco afirma que a inadimplência está subindo porque a empresa
está aprovando muitos empréstimos de Grade D e E. Este notebook investiga essa
explicação e examina o que mais pode estar associado à alta.

## Decisões de escopo

Restringi a análise a empréstimos de **36 meses** originados em **2014–2015** porque
essas safras estavam praticamente resolvidas no snapshot de dezembro de 2018. Em safras
mais recentes, grande parte dos contratos ainda estaria ativa; analisar somente os
poucos contratos já resolvidos poderia produzir uma amostra não representativa e
subestimar ou distorcer a inadimplência final.

A fonte é a camada Gold (`analysis_cohort.parquet`), a mesma coorte usada no dashboard
de Self-Service BI, para manter uma única definição de métricas entre os entregáveis.

| Item | Definição |
|---|---|
| Fonte | Gold — `analysis_cohort.parquet` |
| Originação | jan/2014 – dez/2015 |
| Prazo | 36 meses |
| População | Contratos com desfecho conhecido |
| Inadimplência | `Charged Off`, `Default` e equivalentes de política |
"""
        )
    )

    cells.append(md("## 0. Setup"))

    cells.append(
        code(
            """
%matplotlib inline

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "analysis_cohort.parquet"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

if not GOLD_FILE.exists():
    raise FileNotFoundError(
        f"Gold dataset not found: {GOLD_FILE}. "
        "Run the notebook from the project root or notebooks directory."
    )

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
"""
        )
    )

    cells.append(
        code(
            """
df = pd.read_parquet(GOLD_FILE)

required_cols = [
    "loan_id",
    "issue_year",
    "issue_quarter",
    "grade",
    "grade_group",
    "is_grade_de",
    "purpose",
    "loan_amount",
    "interest_rate_pct",
    "annual_income",
    "debt_to_income_ratio",
    "fico_score_avg",
    "verification_status",
    "default_flag",
    "dti_band",
]

missing = [c for c in required_cols if c not in df.columns]
assert not missing, f"Missing columns: {missing}"
assert df["loan_id"].is_unique
assert df["default_flag"].isin([0, 1]).all()

df = df.sort_values(["issue_quarter", "loan_id"]).reset_index(drop=True)
df["period"] = np.where(df["issue_year"] == 2014, "2014", "2015")

print(f"Contratos: {len(df):,}")
print(f"Inadimplentes: {int(df['default_flag'].sum()):,}")
print(f"Taxa de inadimplência: {100 * df['default_flag'].mean():.2f}%")
print(f"Trimestres: {sorted(df['issue_quarter'].unique())}")
df.head()
"""
        )
    )

    cells.append(
        md(
            """
## 1. Análise exploratória da inadimplência

**Pergunta.** A inadimplência realmente aumentou na coorte? Se sim, isso veio acompanhado
de maior participação de Grades D/E, ou o risco mudou *dentro* das faixas já existentes?

**Método.** Comparo a evolução trimestral da taxa de inadimplência com a participação de
D/E e, em seguida, a inadimplência condicional a cada grupo de grade (D/E vs demais),
além do mix e da taxa por grade entre 2014 e 2015.
"""
        )
    )

    cells.append(
        code(
            """
quarterly = (
    df.groupby("issue_quarter", as_index=False)
    .agg(
        total_loans=("loan_id", "size"),
        default_rate=("default_flag", "mean"),
        grade_de_share=("is_grade_de", "mean"),
        avg_interest_rate=("interest_rate_pct", "mean"),
        avg_dti=("debt_to_income_ratio", "mean"),
        avg_fico=("fico_score_avg", "mean"),
    )
)

de_rates = (
    df.groupby(["issue_quarter", "is_grade_de"], as_index=False)
    .agg(default_rate=("default_flag", "mean"))
    .pivot(index="issue_quarter", columns="is_grade_de", values="default_rate")
    .rename(columns={False: "other_default_rate", True: "de_default_rate"})
    .reset_index()
)

quarterly = quarterly.merge(de_rates, on="issue_quarter")
quarterly["default_rate_pct"] = 100 * quarterly["default_rate"]
quarterly["grade_de_share_pct"] = 100 * quarterly["grade_de_share"]
quarterly["de_default_rate_pct"] = 100 * quarterly["de_default_rate"]
quarterly["other_default_rate_pct"] = 100 * quarterly["other_default_rate"]

quarterly.to_csv(TABLES_DIR / "eda_quarterly_metrics.csv", index=False)
quarterly[
    [
        "issue_quarter",
        "total_loans",
        "default_rate_pct",
        "grade_de_share_pct",
        "de_default_rate_pct",
        "other_default_rate_pct",
    ]
]
"""
        )
    )

    cells.append(
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

axes[0].plot(
    quarterly["issue_quarter"],
    quarterly["default_rate_pct"],
    marker="o",
    color="#1f4e79",
    linewidth=2,
)
axes[0].set_title("Inadimplência geral por trimestre de originação")
axes[0].set_ylabel("Taxa de inadimplência (%)")
axes[0].tick_params(axis="x", rotation=45)

axes[1].plot(
    quarterly["issue_quarter"],
    quarterly["grade_de_share_pct"],
    marker="o",
    color="#c45c26",
    linewidth=2,
)
axes[1].set_title("Participação de Grades D/E na carteira")
axes[1].set_ylabel("Participação D/E (%)")
axes[1].tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_default_vs_de_share.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        )
    )

    cells.append(
        code(
            """
fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(
    quarterly["issue_quarter"],
    quarterly["de_default_rate_pct"],
    marker="o",
    label="Grades D/E",
    color="#a63232",
    linewidth=2,
)
ax.plot(
    quarterly["issue_quarter"],
    quarterly["other_default_rate_pct"],
    marker="o",
    label="Demais grades",
    color="#2f6b4f",
    linewidth=2,
)
ax.set_title("Inadimplência dentro do grupo de risco")
ax.set_ylabel("Taxa de inadimplência (%)")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "02_default_within_grade_groups.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
"""
        )
    )

    cells.append(
        code(
            """
default_by_grade = (
    df.groupby(["period", "grade"], as_index=False)
    .agg(
        total_loans=("loan_id", "size"),
        default_rate=("default_flag", "mean"),
        avg_interest_rate=("interest_rate_pct", "mean"),
        avg_dti=("debt_to_income_ratio", "mean"),
    )
)
default_by_grade["default_rate_pct"] = 100 * default_by_grade["default_rate"]

share_by_grade = (
    df.groupby(["period", "grade"])
    .size()
    .div(df.groupby("period").size())
    .rename("portfolio_share")
    .reset_index()
)
share_by_grade["portfolio_share_pct"] = 100 * share_by_grade["portfolio_share"]

grade_summary = default_by_grade.merge(
    share_by_grade[["period", "grade", "portfolio_share_pct"]],
    on=["period", "grade"],
)

wide = grade_summary.pivot(index="grade", columns="period")
eda_grade = pd.DataFrame(
    {
        "share_2014_pct": wide[("portfolio_share_pct", "2014")],
        "share_2015_pct": wide[("portfolio_share_pct", "2015")],
        "default_2014_pct": wide[("default_rate_pct", "2014")],
        "default_2015_pct": wide[("default_rate_pct", "2015")],
        "rate_2014_pct": wide[("avg_interest_rate", "2014")],
        "rate_2015_pct": wide[("avg_interest_rate", "2015")],
        "dti_2014": wide[("avg_dti", "2014")],
        "dti_2015": wide[("avg_dti", "2015")],
    }
)
eda_grade["delta_default_pp"] = (
    eda_grade["default_2015_pct"] - eda_grade["default_2014_pct"]
)
eda_grade["delta_rate_pp"] = (
    eda_grade["rate_2015_pct"] - eda_grade["rate_2014_pct"]
)
eda_grade["delta_share_pp"] = (
    eda_grade["share_2015_pct"] - eda_grade["share_2014_pct"]
)
eda_grade = eda_grade.reindex(list("ABCDEFG"))
eda_grade.to_csv(TABLES_DIR / "eda_grade_period_comparison.csv")
eda_grade.round(2)
"""
        )
    )

    cells.append(
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(len(eda_grade.index))
width = 0.38

axes[0].bar(
    x - width / 2,
    eda_grade["default_2014_pct"],
    width,
    label="2014",
    color="#4c78a8",
)
axes[0].bar(
    x + width / 2,
    eda_grade["default_2015_pct"],
    width,
    label="2015",
    color="#f58518",
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(eda_grade.index)
axes[0].set_title("Inadimplência por grade: 2014 vs 2015")
axes[0].set_ylabel("Taxa (%)")
axes[0].legend()

axes[1].bar(
    x - width / 2,
    eda_grade["share_2014_pct"],
    width,
    label="2014",
    color="#4c78a8",
)
axes[1].bar(
    x + width / 2,
    eda_grade["share_2015_pct"],
    width,
    label="2015",
    color="#f58518",
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(eda_grade.index)
axes[1].set_title("Mix da carteira por grade: 2014 vs 2015")
axes[1].set_ylabel("Participação (%)")
axes[1].legend()

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "03_grade_default_and_mix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
"""
        )
    )

    cells.append(
        md(
            """
### Interpretação exploratória

Entre 2014-Q1 e 2015-Q4, a inadimplência geral sobe de **12,87%** para **14,82%**,
enquanto a participação de D/E cai de **14,93%** para **12,26%**. No mesmo período, a
inadimplência dentro de D/E sobe de **22,3%** para **29,2%**, e nas demais grades de
**11,2%** para **12,8%**. Em quase todas as grades, 2015 fica acima de 2014.

Isso já enfraquece o mecanismo alegado pelo Head de Risco: D/E de fato concentram mais
risco, mas a carteira não parece ter ficado mais concentrada nelas. O próximo passo é
testar formalmente essas diferenças e separar efeito de mix de efeito dentro dos grupos.
"""
        )
    )

    cells.append(
        md(
            r"""
## 2. Teste de hipótese

**Pergunta.** A alta da inadimplência entre 2014 e 2015 é estatisticamente detectável?
E o mecanismo “aprovamos mais D/E” encontra suporte nos dados?

**Método.** Uso testes z de duas proporções (\(\alpha = 0{,}05\)), comparando 2015 com 2014:

| Teste | O que avalia |
|---|---|
| H1 | A inadimplência geral de 2015 é maior que a de 2014? |
| H2 | A participação de D/E mudou entre 2014 e 2015? |
| H3 | A inadimplência *dentro* de D/E aumentou? |
| H4 | A inadimplência nas demais grades também aumentou? |
"""
        )
    )

    cells.append(
        code(
            """
def proportion_test(success_a, n_a, success_b, n_b, alternative="larger"):
    # Two-sample z-test for proportions. Group A vs group B.
    stat, pvalue = proportions_ztest(
        count=[success_a, success_b],
        nobs=[n_a, n_b],
        alternative=alternative,
    )
    p_a = success_a / n_a
    p_b = success_b / n_b
    return {
        "rate_a": p_a,
        "rate_b": p_b,
        "delta_pp": 100 * (p_a - p_b),
        "z_stat": stat,
        "p_value": pvalue,
        "reject_h0_5pct": bool(pvalue < 0.05),
    }


early = df[df["period"] == "2014"]
late = df[df["period"] == "2015"]

tests = pd.DataFrame(
    [
        {
            "test": "H1 — inadimplência geral subiu (2015 > 2014)",
            **proportion_test(
                late["default_flag"].sum(),
                len(late),
                early["default_flag"].sum(),
                len(early),
                alternative="larger",
            ),
        },
        {
            "test": "H2 — participação de D/E mudou entre 2014 e 2015",
            **proportion_test(
                late["is_grade_de"].sum(),
                len(late),
                early["is_grade_de"].sum(),
                len(early),
                alternative="two-sided",
            ),
        },
        {
            "test": "H3 — inadimplência em D/E subiu (2015 > 2014)",
            **proportion_test(
                late.loc[late["is_grade_de"], "default_flag"].sum(),
                int(late["is_grade_de"].sum()),
                early.loc[early["is_grade_de"], "default_flag"].sum(),
                int(early["is_grade_de"].sum()),
                alternative="larger",
            ),
        },
        {
            "test": "H4 — inadimplência nas demais grades subiu",
            **proportion_test(
                late.loc[~late["is_grade_de"], "default_flag"].sum(),
                int((~late["is_grade_de"]).sum()),
                early.loc[~early["is_grade_de"], "default_flag"].sum(),
                int((~early["is_grade_de"]).sum()),
                alternative="larger",
            ),
        },
    ]
)

tests.to_csv(TABLES_DIR / "hypothesis_tests.csv", index=False)
tests.style.format(
    {
        "rate_a": "{:.2%}",
        "rate_b": "{:.2%}",
        "delta_pp": "{:.2f}",
        "z_stat": "{:.2f}",
        "p_value": "{:.2e}",
    }
)
"""
        )
    )

    cells.append(
        md(
            r"""
### Por que decompor a variação?

Os testes acima respondem se D/E é mais arriscado e se o mix ou as taxas condicionais
mudaram, mas **não** quantificam quanto da alta da taxa total veio de mudança no mix
versus piora dentro dos grupos. Por isso, faço uma **decomposição sequencial da
variação da taxa total** (decomposição exata de composição e taxas, no estilo Oaxaca),
separando efeito de composição e efeitos internos:

$$
\Delta R =
\underbrace{(s_{15}-s_{14})(p^{DE}_{14}-p^{Other}_{14})}_{\text{composição}}
+ \underbrace{s_{15}(p^{DE}_{15}-p^{DE}_{14})}_{\text{taxa D/E}}
+ \underbrace{(1-s_{15})(p^{Other}_{15}-p^{Other}_{14})}_{\text{taxa demais}}
$$

Nesta formulação, o efeito de composição usa as taxas de 2014, e os efeitos internos
usam as participações de 2015.
"""
        )
    )

    cells.append(
        code(
            """
s14 = early["is_grade_de"].mean()
s15 = late["is_grade_de"].mean()
p_de_14 = early.loc[early["is_grade_de"], "default_flag"].mean()
p_de_15 = late.loc[late["is_grade_de"], "default_flag"].mean()
p_ot_14 = early.loc[~early["is_grade_de"], "default_flag"].mean()
p_ot_15 = late.loc[~late["is_grade_de"], "default_flag"].mean()
r14 = early["default_flag"].mean()
r15 = late["default_flag"].mean()

composition = (s15 - s14) * (p_de_14 - p_ot_14)
rate_de = s15 * (p_de_15 - p_de_14)
rate_other = (1 - s15) * (p_ot_15 - p_ot_14)

decomp = pd.DataFrame(
    [
        {"component": "Variação total (2015 − 2014)", "pp": 100 * (r15 - r14)},
        {"component": "Efeito composição (mix D/E)", "pp": 100 * composition},
        {"component": "Efeito taxa dentro de D/E", "pp": 100 * rate_de},
        {"component": "Efeito taxa nas demais grades", "pp": 100 * rate_other},
        {
            "component": "Soma dos efeitos",
            "pp": 100 * (composition + rate_de + rate_other),
        },
    ]
)
decomp.to_csv(TABLES_DIR / "sequential_decomposition.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
plot_df = decomp.iloc[1:4].copy()
colors = ["#2f6b4f" if v < 0 else "#a63232" for v in plot_df["pp"]]
ax.barh(plot_df["component"], plot_df["pp"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Contribuição (pontos percentuais)")
ax.set_title("Decomposição da alta da inadimplência (2014 → 2015)")
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "04_sequential_decomposition.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

decomp.round(3)
"""
        )
    )

    cells.append(
        md(
            r"""
### Interpretação dos testes

A inadimplência de 2015 é significativamente maior que a de 2014
(\(p = 1{,}55 \times 10^{-26}\); +1,16 pp). A participação de D/E apresentou queda
estatisticamente detectável de 2,07 pp entre 2014 e 2015 (teste bilateral H2;
\(p < 0{,}001\)). Ao mesmo tempo, a inadimplência sobe dentro de D/E (+4,28 pp) e
também nas demais grades (+0,90 pp).

Na decomposição sequencial, o efeito de composição do mix D/E é **negativo**
(≈ −0,24 pp). A deterioração das taxas dentro dos grupos explica mais do que a alta
total observada (+0,64 pp em D/E e +0,76 pp nas demais grades, somando ≈ +1,40 pp). A
mudança favorável do mix D/E compensou aproximadamente 0,24 pp desse aumento, resultando
nos +1,16 pp observados na taxa total.

Os resultados não sustentam a explicação de que a inadimplência aumentou porque a
participação de D/E cresceu. Entre 2014 e 2015, o mix D/E diminuiu, enquanto a
inadimplência aumentou dentro de D/E e também nas demais grades. Isso sugere uma
deterioração mais ampla da qualidade das safras, que não é capturada apenas pelo mix
de grades.
"""
        )
    )

    cells.append(
        md(
            """
## 3. Indo além — o que mais está associado à alta?

**Pergunta.** Se o mix D/E não explica a variação, o que mais mudou junto com a
inadimplência — e o que permanece associado à safra 2015 depois de ajustar por grade
e perfil observado?

**Método.** Estimo um logit com indicador de safra 2015, primeiro só com grade e depois
com DTI, FICO, renda e juros; em seguida, comparo variação de inadimplência e de juros
médios por grade, e o deslocamento do mix por faixa de DTI.
"""
        )
    )

    cells.append(
        code(
            """
model_df = df.dropna(
    subset=[
        "default_flag",
        "debt_to_income_ratio",
        "fico_score_avg",
        "annual_income",
        "interest_rate_pct",
        "grade",
    ]
).copy()
model_df["late"] = (model_df["period"] == "2015").astype(int)
model_df["log_income"] = np.log1p(model_df["annual_income"])

logit_basic = smf.logit(
    "default_flag ~ late + C(grade)",
    data=model_df,
).fit(disp=False)

logit_full = smf.logit(
    "default_flag ~ late + C(grade) + debt_to_income_ratio"
    " + fico_score_avg + log_income + interest_rate_pct",
    data=model_df,
).fit(disp=False)

def odds_ratio_row(model, label):
    coef = model.params["late"]
    se = model.bse["late"]
    or_point = np.exp(coef)
    or_low = np.exp(coef - 1.96 * se)
    or_high = np.exp(coef + 1.96 * se)
    p_value = model.pvalues["late"]
    return {
        "model": label,
        "late_coefficient": coef,
        "odds_ratio_2015": or_point,
        "or_ci95_low": or_low,
        "or_ci95_high": or_high,
        "p_value": p_value,
        "p_value_display": (
            f"{p_value:.2e}" if p_value >= 1e-300 else "<1e-300"
        ),
        "pseudo_r2": model.prsquared,
        "n_obs": int(model.nobs),
    }


logit_results = pd.DataFrame(
    [
        odds_ratio_row(logit_basic, "Grade only"),
        odds_ratio_row(logit_full, "Grade + borrower controls"),
    ]
)
logit_results.to_csv(TABLES_DIR / "logit_vintage_effect.csv", index=False)

print(
    "Leitura (modelo com controles): após ajuste pelas características observadas, "
    "os empréstimos originados em 2015 apresentaram odds de inadimplência "
    f"aproximadamente {100 * (logit_results.loc[1, 'odds_ratio_2015'] - 1):.0f}% "
    "maiores que os de 2014 "
    f"(OR={logit_results.loc[1, 'odds_ratio_2015']:.2f}; "
    f"IC95% "
    f"{logit_results.loc[1, 'or_ci95_low']:.2f}–"
    f"{logit_results.loc[1, 'or_ci95_high']:.2f})."
)
print(
    "O modelo é associativo, não causal, e pode haver variáveis não observadas."
)

logit_results[
    [
        "model",
        "odds_ratio_2015",
        "or_ci95_low",
        "or_ci95_high",
        "p_value_display",
        "pseudo_r2",
        "n_obs",
    ]
]
"""
        )
    )

    cells.append(
        code(
            """
print(logit_full.summary().as_text())
"""
        )
    )

    cells.append(
        code(
            """
pricing = eda_grade[
    ["delta_default_pp", "delta_rate_pp", "dti_2014", "dti_2015"]
].copy()
pricing["delta_dti"] = pricing["dti_2015"] - pricing["dti_2014"]
pricing.to_csv(TABLES_DIR / "pricing_vs_risk_by_grade.csv")

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.scatter(
    pricing["delta_rate_pp"],
    pricing["delta_default_pp"],
    s=120,
    color="#1f4e79",
    zorder=3,
)
for grade, row in pricing.iterrows():
    ax.annotate(
        grade,
        (row["delta_rate_pp"] + 0.03, row["delta_default_pp"] + 0.15),
    )

ax.set_xlabel("Variação da taxa de juros média (pp) — 2015 vs 2014")
ax.set_ylabel("Variação da inadimplência (pp) — 2015 vs 2014")
ax.set_title("Desalinhamento preço × risco por grade")
ax.text(
    0.02,
    0.98,
    "Quadrante crítico:\\njuros ↓ e inadimplência ↑",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    color="#a63232",
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_pricing_vs_risk.png", dpi=150, bbox_inches="tight")
plt.show()

pricing.round(2)
"""
        )
    )

    cells.append(
        code(
            """
dti_shift = (
    df.groupby(["period", "dti_band"], as_index=False)
    .agg(
        portfolio_share=("loan_id", "size"),
        default_rate=("default_flag", "mean"),
    )
)
totals = dti_shift.groupby("period")["portfolio_share"].transform("sum")
dti_shift["portfolio_share_pct"] = 100 * dti_shift["portfolio_share"] / totals
dti_shift["default_rate_pct"] = 100 * dti_shift["default_rate"]
dti_shift = dti_shift[dti_shift["dti_band"] != "Unknown"]

order = ["< 10", "10–20", "20–30", "30+"]
dti_shift["dti_band"] = pd.Categorical(
    dti_shift["dti_band"],
    categories=order,
    ordered=True,
)
dti_shift = dti_shift.sort_values(["dti_band", "period"])
dti_shift.to_csv(TABLES_DIR / "dti_band_shift.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(order))
width = 0.38

for ax, metric, title, ylabel in [
    (axes[0], "portfolio_share_pct", "Mix da carteira por faixa de DTI", "Participação (%)"),
    (axes[1], "default_rate_pct", "Inadimplência por faixa de DTI", "Taxa (%)"),
]:
    vals_2014 = [
        dti_shift.loc[
            (dti_shift["period"] == "2014") & (dti_shift["dti_band"] == band),
            metric,
        ].iloc[0]
        for band in order
    ]
    vals_2015 = [
        dti_shift.loc[
            (dti_shift["period"] == "2015") & (dti_shift["dti_band"] == band),
            metric,
        ].iloc[0]
        for band in order
    ]
    ax.bar(x - width / 2, vals_2014, width, label="2014", color="#4c78a8")
    ax.bar(x + width / 2, vals_2015, width, label="2015", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("DTI")
    ax.legend()

fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_dti_shift.png", dpi=150, bbox_inches="tight")
plt.show()

dti_shift
"""
        )
    )

    cells.append(
        md(
            """
## 4. Resultado adicional — pricing e qualidade dentro da grade

**Resultado.** Entre as grades B e F, a safra de 2015 apresentou inadimplência maior,
apesar de taxas de juros médias menores. No mesmo período, o DTI médio aumentou dentro
de várias grades (por exemplo, D e E). No modelo com controles observados, a safra 2015
permanece associada a odds de inadimplência cerca de **27% maiores** que 2014
(OR ≈ 1,27; IC95% na tabela acima).

**Interpretação.** Esse padrão merece revisão porque sugere que o aumento do risco
realizado não foi acompanhado por aumento proporcional do preço — sem afirmar que a
precificação *causou* a inadimplência. O aumento do DTI dentro das mesmas grades sugere
que a qualidade do perfil pode ter se deteriorado sem ser totalmente refletida pela
classificação. O residual da safra no logit é associativo, não causal, e pode haver
variáveis não observadas.

Em conjunto com os testes anteriores, os resultados não sustentam a explicação de que a
inadimplência aumentou porque a participação de D/E cresceu. A leitura mais consistente
com a evidência disponível é uma deterioração mais ampla da qualidade das safras, com
sinais de desalinhamento entre pricing e risco realizado.
"""
        )
    )

    cells.append(
        md(
            """
## 5. Recomendação ao negócio

Com base nos resultados — e não em um corte automático de D/E — as ações mais
prudentes são:

1. **Revisar calibração e regras de aprovação**, não apenas o volume de D/E.
   Restringir D/E sem olhar o risco dentro de C/D pode deslocar o problema para outras
   etiquetas.
2. **Monitorar DTI alto.** A participação da faixa 30+ cresceu de 7,2% para 10,7% da
   carteira, enquanto sua inadimplência aumentou de 19,6% para 21,0%.
3. **Revisitar pricing onde risco subiu e juros caíram (B–F)**, priorizando C/D/E, para
   checar se a precificação está acompanhando o risco realizado.
4. **Acompanhar no dashboard** a inadimplência *dentro* da grade e o gap preço×risco —
   além do mix D/E — usando a mesma coorte Gold já publicada no Looker.

**Síntese para o Head de Risco.** A inadimplência da coorte 2014–2015 subiu de forma
detectável, mas os dados não sustentam a explicação de que isso ocorreu porque a
participação de D/E aumentou. A alta está associada a piora dentro das faixas de risco,
com aumento de DTI e sinais de desalinhamento entre preço e risco realizado. O caminho
mais útil é recalibrar underwriting e pricing com monitoramento condicional à grade —
não um corte cego de D/E.
"""
        )
    )

    cells.append(
        md(
            """
## 6. Limitações

- A análise é observacional e identifica associações, não relações causais.
- O escopo está restrito a empréstimos de 36 meses originados em 2014–2015 e com
  desfecho conhecido.
- Grade e taxa de juros são variáveis fortemente relacionadas; seus coeficientes
  individuais no modelo ajustado não devem ser interpretados isoladamente como efeitos
  causais.
- Mudanças macroeconômicas, operacionais ou de política de crédito não observadas podem
  explicar parte do efeito associado à safra de 2015.
"""
        )
    )

    cells.append(
        md(
            """
## Uso de ferramentas de IA

Foi utilizado o agente GPT-5.6 Sol (Cursor) para acelerar a estruturação inicial
do código, a organização do notebook e a revisão de linguagem. O escopo analítico,
as definições das métricas, os métodos estatísticos, a validação dos resultados e as
conclusões finais foram revisados e aprovados pelo autor.
"""
        )
    )

    nb.cells = cells
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
