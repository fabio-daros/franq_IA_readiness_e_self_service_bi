# Parte 3 — Curadoria e validação de IA generativa

Validação da resposta do agente de IA da FinLend à pergunta:

> *"Qual é o perfil de risco dos nossos clientes que tomam empréstimos para consolidação de dívidas?"*

**Classificação final: BLOCK**

Existe erro material e robusto na interpretação de risco do segmento. A resposta
não deve ser liberada para stakeholders sem correção.

---

## 1. Escopo e fonte de verdade

A validação usa a mesma coorte oficial da Gold e do dashboard:

| Item | Definição |
|---|---|
| Fonte | `data/gold/ai_validation_metrics.json` (derivada de `analysis_cohort.parquet`) |
| Originação | jan/2014 – dez/2015 |
| Prazo | 36 meses |
| População | Contratos com desfecho conhecido |
| Segmento | `purpose = debt_consolidation` |

Tolerâncias usadas na classificação:

- percentuais: ±0,25 ponto percentual;
- valores monetários arredondados: erro relativo ≤ 1%.

Script de sensibilidade entre coortes:

`scripts/04_validate_ai_claims_across_cohorts.py`

Saída:

`outputs/tables/ai_claims_cross_cohort_validation.csv`

### Arquitetura e rastreabilidade dos dados

A solução foi organizada em uma arquitetura Medallion:

- **Bronze:** preservação dos arquivos originais e metadados de ingestão;
- **Silver:** limpeza, padronização de tipos, classificação dos desfechos e
  aplicação das regras de qualidade;
- **Gold:** materialização da coorte analítica e das métricas governadas
  utilizadas no dashboard e na validação do agente.

A validação principal consome as métricas oficiais da camada Gold. A análise de
sensibilidade consulta a camada Silver com regras explícitas de coorte e
reconcilia o recorte principal com a Gold. Testes automatizados verificam
integridade, regras de domínio e consistência entre as camadas.

---

## 2. Validação quantitativa

Resposta alegada pelo LLM versus métricas Gold:

| Alegação | LLM | Gold | Status | Observação |
|---|---|---|---|---|
| Participação do segmento | 48,0% | 57,41% | FAIL | Diferença de 9,41 pp |
| Inadimplência do segmento | 12,3% | 15,22% | FAIL | Diferença de 2,92 pp |
| Inadimplência geral | 14,1% | 14,46% | FAIL | Diferença de 0,36 pp |
| Segmento abaixo da média | Abaixo | 0,76 pp acima | FAIL | Direção invertida |
| Ticket médio | US$ 15.200 | US$ 13.189,82 | FAIL | Erro relativo de 15,2% |
| Renda anual média | US$ 72.000 | US$ 72.397,73 | PASS | Dentro da tolerância de 1% |
| Participação B/C | 62,0% | 61,94% | PASS | Dentro da tolerância de 0,25 pp |
| Taxa de juros média | 13,8% | 11,95% | FAIL | Diferença de 1,85 pp |

A falha principal é a inversão da conclusão de risco:

| Métrica | Valor |
|---|---|
| Default do segmento | 15,2237% |
| Default geral | 14,4631% |
| Diferença | **+0,7606 pp** |

O segmento está **acima**, e não abaixo, da inadimplência geral.

A renda anual média (~US$ 72.398) é compatível com os US$ 72.000 alegados,
enquanto o ticket médio real é ~US$ 13.190. As duas métricas foram validadas
separadamente para evitar confusão semântica.

---

## 3. Avaliação qualitativa

Pontos positivos da resposta do agente:

- identifica corretamente consolidação de dívidas como segmento relevante;
- acerta, por arredondamento, a participação de grades B/C;
- produz uma narrativa fluida e comercialmente legível.

Falhas materiais:

1. **Erro direcional de risco** — afirma que o segmento está abaixo da média,
   quando está acima.
2. **Erros quantitativos relevantes** — participação, inadimplência do segmento,
   ticket e juros divergem além da tolerância.
3. **Possível confusão semântica** — renda e ticket podem ter sido misturados
   no raciocínio do modelo.
4. **Falta de escopo explícito** — a resposta não declara coorte, prazo nem se
   contratos sem desfecho foram excluídos.

Para um agente usado por gestores de crédito, o erro direcional é suficiente
para bloquear a liberação: a conclusão de negócio fica invertida.

---

## 4. Análise de sensibilidade entre coortes

Para verificar se as divergências eram consequência da escolha da coorte, as
afirmações foram reavaliadas em três populações maduras e no snapshot completo.
Em todos os escopos, a inadimplência de empréstimos para consolidação de dívidas
permaneceu acima da taxa geral da carteira, com diferenças entre +0,66 e +1,18
ponto percentual.

Resumo dos recortes:

| Escopo | Share | Seg. vs geral | Ticket | Renda | Direção |
|---|---:|---:|---:|---:|---|
| 2014–2015, 36m, resolvidos (Gold) | 57,41% | +0,76 pp | US$ 13.190 | US$ 72.398 | Acima |
| 2007–2015, 36m, resolvidos | 56,65% | +0,66 pp | US$ 13.064 | US$ 71.276 | Acima |
| 2007–2013, 60m, resolvidos | 63,74% | +0,91 pp | US$ 20.526 | US$ 77.272 | Acima |
| Snapshot completo* | 56,53% | +1,18 pp | US$ 15.968 | US$ 76.881 | Acima |

\*No snapshot completo, 21,16% (segmento) e 19,98% (carteira) referem-se à
**taxa de inadimplência entre contratos com desfecho conhecido no snapshot
completo**, não ao default sobre os 2,26 milhões de contratos. 40,37% da
carteira ainda estava sem desfecho final nesse recorte.

Alguns valores isolados podem se aproximar das alegações quando o escopo é
alterado, mas nenhuma das populações avaliadas reproduz simultaneamente as
métricas apresentadas pelo LLM. Além disso, a conclusão direcional sobre o
risco do segmento permanece incorreta em todos os recortes.

---

## 5. Proposta de melhoria

1. **Respostas com escopo obrigatório** — toda saída do agente deve declarar
   período, prazo, população (resolvidos vs snapshot) e definição de default.
2. **Grounding em métricas Gold** — o LLM deve consultar
   `ai_validation_metrics.json` / tabelas Gold em vez de estimar números.
3. **Checagem direcional automática** — bloquear respostas que invertam
   comparações segmentadas (acima/abaixo da carteira).
4. **Separação semântica de métricas** — ticket (`loan_amount`) e renda
   (`annual_income`) devem ter campos distintos e rótulos explícitos no prompt
   e na ferramenta de consulta.
5. **Recusa quando a evidência for ambígua** — se o agente não encontrar a
   métrica na Gold, deve dizer que não sabe em vez de inventar.

Exemplo de formato seguro de resposta:

> Na coorte 2014–2015 (36 meses, contratos resolvidos), consolidação de dívidas
> representa 57,4% da carteira. A inadimplência do segmento é 15,22%, **0,76 pp
> acima** da média geral (14,46%). Ticket médio: US$ 13.190. Renda média anual:
> US$ 72.398. Grades B/C: 61,9%. Juros médios: 11,95%.

---

## 6. Processo recorrente de validação

### Antes de liberar o agente

1. Definir um conjunto fixo de perguntas de negócio (golden set).
2. Calcular respostas oficiais na camada Gold.
3. Classificar cada resposta do LLM como `PASS`, `REVIEW` ou `BLOCK`.
4. Exigir `PASS` nas perguntas críticas de risco antes do go-live.

### Em produção

1. Amostrar semanalmente respostas do agente.
2. Reconciliar números contra a Gold com as mesmas tolerâncias.
3. Rodar sensibilidade entre coortes quando houver divergência relevante.
4. Registrar falhas direcionais como incidente de alto impacto.
5. Atualizar o golden set sempre que a política de crédito ou a coorte mudar.

### Critério de bloqueio

Qualquer resposta que:

- inverta a direção de risco de um segmento; ou
- erre métricas críticas além da tolerância; ou
- omita o escopo analítico em pergunta de risco

deve ser classificada como **BLOCK**.

---

## 7. Reprodutibilidade

A análise pode ser reproduzida com:

```bash
python scripts/04_validate_ai_claims_across_cohorts.py
python -m pytest -q
```

Resultado da suíte automatizada:

```text
11 passed
```

Essa separação evita que o LLM consulte diretamente dados brutos e reduz o risco
de produzir métricas com definições, denominadores ou populações inconsistentes.

---

## 8. Decisão

| Item | Resultado |
|---|---|
| Validação quantitativa | Múltiplas falhas materiais |
| Avaliação qualitativa | Narrativa fluida, mas conclusão de risco invertida |
| Sensibilidade entre coortes | Direção errada em todos os recortes |
| Liberação para stakeholders | **Não recomendada** |
| Status | **BLOCK** |

---

## Uso de ferramentas de IA

Foi utilizada uma ferramenta assistida por IA no Cursor para acelerar a
estruturação inicial do código, a organização do relatório e a revisão de
linguagem. O escopo analítico, as definições das métricas, os critérios de
classificação, a validação dos resultados e as conclusões finais foram
revisados pelo autor.
