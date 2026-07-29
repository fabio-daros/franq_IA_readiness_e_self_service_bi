# FinLend Credit Intelligence

Teste técnico para a posição de Analista de Dados, com foco em Self-Service BI,
análise de risco de crédito e AI Readiness.

## Entregáveis

### Parte 1 — Self-Service BI

[Visualizar dashboard no Looker Studio](https://datastudio.google.com/reporting/76e8c097-dfbd-4499-beb5-8c8496a97335)

### Parte 2 — Análise estatística

[Visualizar notebook](notebooks/finlend_default_hypothesis_analysis.ipynb)

### Parte 3 — Curadoria e validação de IA generativa

[Visualizar relatório](docs/ai_validation_report.md)

## Principais resultados

- A inadimplência aumentou 1,16 ponto percentual entre 2014 e 2015.
- A participação das grades D/E caiu 2,07 pontos percentuais.
- A deterioração ocorreu dentro de D/E e também nas demais grades.
- A hipótese de que a alta foi causada por maior participação de D/E não foi
  sustentada.
- A resposta do agente de IA foi classificada como `BLOCK` por conter erros
  materiais e inverter a direção do risco do segmento.

## Como avaliar esta entrega

1. Abra o dashboard do Looker Studio e teste os filtros.
2. Leia o resumo executivo do notebook da Parte 2.
3. Consulte a classificação e a análise de sensibilidade no relatório da Parte 3.
4. Para validar a reprodutibilidade, execute `python -m pytest -q`.

---

## Arquitetura de dados

O projeto segue uma arquitetura Medallion enxuta:

- **Bronze:** arquivos-fonte imutáveis e metadados de ingestão
- **Silver:** base loan-level limpa, com tipos padronizados e desfechos
  classificados
- **Gold:** coorte analítica governada e métricas oficiais usadas no dashboard,
  na análise estatística e na validação de IA

Testes automatizados reconciliam as três camadas.

A coorte analítica principal é:

- originações de **2014–2015**
- prazo contratual de **36 meses**
- apenas contratos com **desfecho conhecido**

Essa coorte é a fonte única de verdade para o Looker Studio, o notebook e o
relatório de validação de IA.

---

## Estrutura do repositório

```text
franq_ia_readiness_e_self_service_bi/
├── README.md
├── requirements.txt
├── pytest.ini
├── notebooks/
│   └── finlend_default_hypothesis_analysis.ipynb
├── docs/
│   ├── ai_validation_report.md
│   └── methodology.md
├── scripts/
│   ├── 01_build_bronze.py
│   ├── 02_build_silver.py
│   ├── 03_build_gold.py
│   └── 04_validate_ai_claims_across_cohorts.py
├── tests/
├── outputs/
│   ├── figures/
│   └── tables/
└── data/
    ├── bronze/
    ├── silver/
    └── gold/
```

Arquivos grandes e datasets regeneráveis foram propositalmente excluídos do Git.

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se necessário, registre o kernel do Jupyter:

```bash
python -m ipykernel install --user --name finlend --display-name "Python 3.12 — FinLend"
```

---

## Fonte de dados

Utilize o dataset público Lending Club no Kaggle:

[Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Coloque o arquivo de empréstimos aceitos em:

```text
data/bronze/raw/accepted_2007_to_2018Q4.csv.gz
```

O arquivo `data/bronze/raw/archive.zip` pode ser mantido localmente, mas não é
necessário para o pipeline e não é versionado.

---

## Reconstrução das camadas Medallion

Na raiz do projeto, com o ambiente virtual ativado:

```bash
python scripts/01_build_bronze.py
python scripts/02_build_silver.py
python scripts/03_build_gold.py
```

Principais saídas da Gold:

- `data/gold/analysis_cohort.parquet` — coorte do notebook
- `data/gold/dashboard_loans.csv` — fonte do Looker Studio
- `data/gold/quarterly_risk_metrics.csv`
- `data/gold/risk_by_grade.csv`
- `data/gold/risk_by_purpose.csv`
- `data/gold/ai_validation_metrics.json` — métricas oficiais da Parte 3

Agregados pequenos e manifests da Gold ficam no Git para auditoria. Extratos
loan-level em Parquet/CSV permanecem locais por tamanho.

---

## Validação de IA e testes

```bash
python scripts/04_validate_ai_claims_across_cohorts.py
python -m pytest -q
```

Resultado esperado da suíte:

```text
11 passed
```

A validação entre coortes grava:

```text
outputs/tables/ai_claims_cross_cohort_validation.csv
```

---

## Notas

- A Parte 1 consome `data/gold/dashboard_loans.csv` no Looker Studio.
- A Parte 2 lê `data/gold/analysis_cohort.parquet`.
- A Parte 3 valida a resposta do LLM contra as métricas Gold e verifica
  robustez em coortes maduras alternativas na Silver.
