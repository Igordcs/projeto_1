# Projeto 1 — Laboratório de Ciência de Dados (UFAPE)
**Análise Estatística e Qualidade de Dados da NBA (Temporada Regular 2025–2026): Boston Celtics vs. New Orleans Pelicans**

Sistema web interativo desenvolvido em **Flask + Pandas + SciPy + Statsmodels + Plotly** como substituto do relatório estático tradicional.

---

## Execução Rápida (Comando Único)

Para instalar as dependências e iniciar o servidor local em um único comando no terminal:

```bash
pip install -r requirements.txt && python main.py
```

*(Caso as dependências já estejam instaladas, basta executar `python main.py` e acessar `http://127.0.0.1:5000` no navegador).*

---

## Estrutura do Projeto

```text
projeto1/
├── app/
│   ├── __init__.py                 # Application Factory do Flask e filtros Jinja2
│   ├── data_loader.py              # Pipeline de limpeza e função única load_clean(team=None)
│   ├── routes.py                   # Blueprints por parte do projeto
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── preparacao.py           # Parte 1: Dicionário, Tratamentos, Especiais e Simulação MCAR/MAR
│   │   └── outliers.py             # Parte 2: Z-score, IQR, Grubbs, ECOD, k-NN, DBSCAN e LOF
│   ├── static/
│   │   ├── css/style.css           # Identidade visual responsiva
│   │   └── js/main.js              # Renderização Plotly e seletores dinâmicos
│   └── templates/
│       ├── base.html               # Layout base com menu lateral por seções
│       ├── index.html              # Painel executivo (Visão Geral)
│       ├── parte1/
│       │   ├── dicionario.html     # 1.1 Dicionário de Dados (Apêndice)
│       │   ├── tipos_tamanho.html  # 1.2 Tipos, Tamanho (2238×28 -> 150×34) e Sazonalidade
│       │   ├── tratamentos.html    # 1.3 Datas, Categóricas e Exclusão para ML/DL (Data Leakage)
│       │   ├── valores_especiais.html # 1.4 Zeros, Duplicados e Prorrogações (MIN != 240)
│       │   └── ausentes.html       # 1.5 Teoria MCAR/MAR/MNAR e Simulação (Little e Fairclough)
│       └── parte2/
│           └── outliers.html       # 2. Outliers univariados e jogos de prorrogação
├── data/
│   ├── raw/
│   │   └── stats_times_regular_final.csv  # Base bruta oficial (2238 linhas × 28 colunas, 30 times)
│   └── processed/
│       └── jogos_limpo.csv                # Recorte tratado (150 jogos × 34 colunas)
├── requirements.txt
├── main.py
└── README.md
```

---

## Convenções Metodológicas Fixas

- **Desvio padrão e variância amostrais:** Calculados com correção de Bessel (`ddof=1`).
- **Quartis ($Q_1, Q_3$) e IQR:** Interpolação linear padrão (`pandas`/`numpy`), utilizada de modo idêntico nas tabelas e nos boxplots.
- **Critério de Outliers (Tukey):** Limites definidos por $Q_1 - 1{,}5 \times \text{IQR}$ e $Q_3 + 1{,}5 \times \text{IQR}$.
- **Coeficiente de Variação (CV):** Expresso em percentual com `ddof=1`.
- **Fonte única de dados limpos:** Todas as análises consomem `load_clean(team=None)` em `app/data_loader.py` a partir de `data/processed/jogos_limpo.csv`.

