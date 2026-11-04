"""
Assimetria, coeficiente de variação e IQR por time e variável (Parte 4),
mais a síntese geral do sistema.

As três medidas são lidas juntas: o CV usa média e desvio, que a assimetria
desloca; o IQR é a alternativa robusta aos dois. Cada série é um time × uma
variável, como na Parte 2.

Convenções: desvio amostral (ddof = 1), quartis por interpolação linear do
pandas, cercas de Tukey (1,5×IQR), CV em percentual. Os boxplots recebem
Q1, mediana, Q3 e bigodes já calculados, porque o `quartilemethod="linear"`
do plotly.js interpola na posição p·n − 0,5 (Hazen) e não em (n − 1)·p, como
o pandas — com n = 74 ou 76 os quartis do gráfico mudariam.

As funções `quartis()` e `limites_outliers()` da Parte 3
(`app/analysis/descritiva.py`) são chamadas para conferir os valores quando
o módulo estiver no projeto.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from app.analysis.outliers import IQR_K, TEAM_SHORT, _br, _plotly_to_json
from app.analysis.outliers import _count_rows as _outlier_count_rows
from app.analysis.outliers import _get_catalog as _get_outlier_catalog
from app.data_loader import STUDY_VARIABLES, TEAMS, load_clean

try:
    from app.analysis import descritiva as _parte3
except ImportError:  # Parte 3 ainda não integrada ao repositório
    _parte3 = None

LIMIAR_MODERADA = 0.5
LIMIAR_FORTE = 1.0
CV_BAIXO = 15.0
CV_ALTO = 30.0
IQR_SOBRE_DP_NORMAL = 1.349
BAIXA_CONTAGEM = {"roubos", "tocos"}
HIST_BIN = {"pontos": 5.0, "rebotes": 2.0}

ESCALAS = {
    "bruta": "Escala original",
    "log1p": "Escala log1p — ln(1 + x)",
}

TEAM_COLORS = {"Boston Celtics": "#007A33", "New Orleans Pelicans": "#0C2340"}
TEAM_FILL = {
    "Boston Celtics": "rgba(0, 122, 51, 0.35)",
    "New Orleans Pelicans": "rgba(12, 35, 64, 0.30)",
}

_CATALOGO: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------

def _box_stats(x: np.ndarray) -> Dict[str, Any]:
    """Quartis lineares, cercas de Tukey e extremos dos bigodes."""
    s = pd.Series(x, dtype=float)
    q1 = float(s.quantile(0.25, interpolation="linear"))
    q2 = float(s.quantile(0.50, interpolation="linear"))
    q3 = float(s.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    inf = q1 - IQR_K * iqr
    sup = q3 + IQR_K * iqr
    dentro = x[(x >= inf) & (x <= sup)]
    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": iqr,
        "lim_inf": inf,
        "lim_sup": sup,
        "bigode_inf": float(dentro.min()),
        "bigode_sup": float(dentro.max()),
        "out_inf": [float(v) for v in x[x < inf]],
        "out_sup": [float(v) for v in x[x > sup]],
        "media": float(s.mean()),
    }


def _sentido(g1: float) -> str:
    if abs(g1) < 0.005:
        return "nula"
    return "à direita (positiva)" if g1 > 0 else "à esquerda (negativa)"


def _intensidade(g1: float) -> str:
    a = abs(g1)
    if a < LIMIAR_MODERADA:
        return "aproximadamente simétrica"
    if a < LIMIAR_FORTE:
        return "moderada"
    return "forte"


def _nivel_cv(cv: float) -> str:
    if cv < CV_BAIXO:
        return "baixo"
    if cv <= CV_ALTO:
        return "moderado"
    return "alto"


def _medidas(x: np.ndarray) -> Dict[str, Any]:
    n = len(x)
    s = pd.Series(x, dtype=float)
    box = _box_stats(x)
    media = float(s.mean())
    mediana = box["q2"]
    dp = float(s.std(ddof=1))
    g1 = float(stats.skew(x, bias=False))
    g1_p = float(stats.skewtest(x).pvalue)
    ses = float(np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
    g1_log = float(stats.skew(np.log1p(x), bias=False))
    mn, mx = float(s.min()), float(s.max())
    return {
        "n": n,
        **box,
        "media": media,
        "mediana": mediana,
        "dp": dp,
        "variancia": dp ** 2,
        "minimo": mn,
        "maximo": mx,
        "amplitude": mx - mn,
        "n_outliers": len(box["out_inf"]) + len(box["out_sup"]),
        "zeros": int((x == 0).sum()),
        "g1": g1,
        "g1_p": g1_p,
        "g1_ses": ses,
        "g1_log": g1_log,
        "sentido": _sentido(g1),
        "intensidade": _intensidade(g1),
        "bowley": (box["q3"] + box["q1"] - 2 * mediana) / box["iqr"] if box["iqr"] > 0 else float("nan"),
        "pearson2": 3 * (media - mediana) / dp if dp > 0 else float("nan"),
        "media_menos_mediana": media - mediana,
        "cv": dp / media * 100 if media > 0 else float("nan"),
        "cv_robusto": box["iqr"] / mediana * 100 if mediana > 0 else float("nan"),
        "cv_poisson": 100 / np.sqrt(media) if media > 0 else float("nan"),
        "dispersao_poisson": dp ** 2 / media if media > 0 else float("nan"),
        "mad": float(stats.median_abs_deviation(x, scale=1.0)),
        "iqr_sobre_dp": box["iqr"] / dp if dp > 0 else float("nan"),
        "iqr_sobre_amp": box["iqr"] / (mx - mn) if mx > mn else float("nan"),
        "q1_hazen": float(np.quantile(x, 0.25, method="hazen")),
        "q3_hazen": float(np.quantile(x, 0.75, method="hazen")),
    }


def _numeros(resultado: Any) -> List[float]:
    if isinstance(resultado, dict):
        valores: Iterable[Any] = resultado.values()
    elif isinstance(resultado, pd.Series):
        valores = resultado.tolist()
    elif isinstance(resultado, (list, tuple, np.ndarray)):
        valores = resultado
    else:
        valores = [resultado]
    saida = []
    for v in valores:
        try:
            saida.append(float(v))
        except (TypeError, ValueError):
            continue
    return saida


def _confere_parte3(x: np.ndarray, med: Dict[str, Any]) -> str:
    """Chama quartis() e limites_outliers() da Parte 3 e compara com o cálculo local."""
    if _parte3 is None:
        return "não integrada"
    serie = pd.Series(x, dtype=float)
    try:
        nums_q = _numeros(_parte3.quartis(serie))
        nums_l = _numeros(_parte3.limites_outliers(serie))
    except Exception:
        return "assinatura incompatível"

    def contem(nums: List[float], alvos: Sequence[float]) -> bool:
        return all(any(np.isclose(n, a) for n in nums) for a in alvos)

    ok = contem(nums_q, (med["q1"], med["q3"])) and contem(nums_l, (med["lim_inf"], med["lim_sup"]))
    return "confere" if ok else "diverge"


def _build_catalogo() -> Dict[Tuple[str, str], Dict[str, Any]]:
    frame = load_clean()
    outliers = _get_outlier_catalog()
    catalogo: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for team in TEAMS:
        sub = frame[frame["time"] == team].sort_values(["data_jogo", "id_jogo"])
        for variable in STUDY_VARIABLES:
            x = sub[variable].to_numpy(dtype=float)
            med = _medidas(x)
            p2 = outliers[(team, variable)]
            med.update(
                {
                    "time": team,
                    "time_short": TEAM_SHORT[team],
                    "variable": variable,
                    "label": STUDY_VARIABLES[variable]["short_label"],
                    "values": x,
                    "prorrogacao": sub["prorrogacao"].to_numpy(dtype=bool),
                    "parte2_confere": bool(
                        np.allclose(
                            [p2["q1"], p2["q3"], p2["lower"], p2["upper"]],
                            [med["q1"], med["q3"], med["lim_inf"], med["lim_sup"]],
                        )
                        and p2["counts"]["iqr"] == med["n_outliers"]
                    ),
                    "parte2_outliers": p2["counts"]["iqr"],
                    "parte3": _confere_parte3(x, med),
                }
            )
            catalogo[(team, variable)] = med
    return catalogo


def _get_catalogo() -> Dict[Tuple[str, str], Dict[str, Any]]:
    global _CATALOGO
    if _CATALOGO is None:
        _CATALOGO = _build_catalogo()
    return _CATALOGO


def _selected_teams(team: Optional[str]) -> List[str]:
    return [team] if team in TEAMS else list(TEAMS)


def _linhas(catalogo: Dict[Tuple[str, str], Dict[str, Any]], variable: str) -> List[Dict[str, Any]]:
    linhas = []
    for var in STUDY_VARIABLES:
        for team in TEAMS:
            row = dict(catalogo[(team, var)])
            row["selecionada"] = var == variable
            linhas.append(row)
    return linhas


def _pct(v: float) -> str:
    return f"{_br(v)}%"


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _box_traces(
    x: np.ndarray,
    team: str,
    pos: int,
    unidade: str,
    show_aux_legend: bool,
    xaxis: str = "x",
    yaxis: str = "y",
) -> List[go.BaseTraceType]:
    """Boxplot com estatísticas pré-calculadas (quartis lineares do pandas)."""
    b = _box_stats(x)
    color = TEAM_COLORS[team]
    traces: List[go.BaseTraceType] = [
        go.Box(
            x=[pos],
            q1=[b["q1"]],
            median=[b["q2"]],
            q3=[b["q3"]],
            lowerfence=[b["bigode_inf"]],
            upperfence=[b["bigode_sup"]],
            mean=[b["media"]],
            boxmean=True,
            name=team,
            legendgroup=team,
            marker_color=color,
            line_color=color,
            fillcolor=TEAM_FILL[team],
            width=0.5,
            xaxis=xaxis,
            yaxis=yaxis,
        )
    ]
    outs = b["out_inf"] + b["out_sup"]
    if outs:
        traces.append(
            go.Scatter(
                x=[pos] * len(outs),
                y=outs,
                mode="markers",
                name="Outliers (fora de 1,5×IQR)",
                legendgroup="outliers",
                showlegend=show_aux_legend,
                marker=dict(symbol="circle-open", size=9, color="#C8102E", line=dict(width=2)),
                hovertemplate=f"{team}<br>Outlier: %{{y:.2f}} {unidade}<extra></extra>",
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )
    return traces


def _team_axis(teams: Sequence[str]) -> Dict[str, Any]:
    return dict(
        tickmode="array",
        tickvals=list(range(len(teams))),
        ticktext=list(teams),
        range=[-0.6, len(teams) - 0.4],
    )


def _chart_distribuicao(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    variable: str,
    escala: str,
) -> str:
    meta = STUDY_VARIABLES[variable]
    log = escala == "log1p"
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.36],
        horizontal_spacing=0.1,
        subplot_titles=("Histograma (densidade) + KDE", "Boxplot (quartis lineares)"),
    )
    for i, team in enumerate(teams):
        raw = catalogo[(team, variable)]["values"]
        x = np.log1p(raw) if log else raw
        color = TEAM_COLORS[team]
        if log:
            xbins = None
            nbins = 16
        else:
            size = HIST_BIN.get(variable, 1.0)
            start = np.floor(x.min() / size) * size - (0.5 if size == 1.0 else 0.0)
            xbins = dict(start=float(start), end=float(x.max() + size), size=size)
            nbins = None
        fig.add_trace(
            go.Histogram(
                x=x,
                xbins=xbins,
                nbinsx=nbins,
                histnorm="probability density",
                name=f"{TEAM_SHORT[team]} — histograma",
                legendgroup=team,
                marker=dict(color=color, line=dict(color="#FFFFFF", width=0.5)),
                opacity=0.45,
                hovertemplate="Faixa: %{x}<br>Densidade: %{y:.4f}<extra>" + team + "</extra>",
            ),
            row=1,
            col=1,
        )
        if np.std(x) > 0:
            kde = stats.gaussian_kde(x)
            pad = (x.max() - x.min()) * 0.1
            grid = np.linspace(x.min() - pad, x.max() + pad, 200)
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=kde(grid),
                    mode="lines",
                    name=f"{TEAM_SHORT[team]} — KDE",
                    legendgroup=team,
                    line=dict(color=color, width=2.5),
                    hovertemplate="x = %{x:.2f}<br>Densidade KDE: %{y:.4f}<extra>" + team + "</extra>",
                ),
                row=1,
                col=1,
            )
        for trace in _box_traces(x, team, i, meta["unit"], i == 0, xaxis="x2", yaxis="y2"):
            trace.showlegend = trace.showlegend if isinstance(trace, go.Scatter) else False
            fig.add_trace(trace)

    eixo = f"ln(1 + {meta['short_label'].lower()})" if log else f"{meta['short_label']} ({meta['unit']})"
    fig.update_layout(
        title=f"Distribuição de {meta['label']} por time — {ESCALAS[escala].lower()}",
        barmode="overlay",
        height=500,
    )
    fig.update_xaxes(title_text=eixo, row=1, col=1)
    fig.update_yaxes(title_text=f"Densidade (por {'unidade de ln(1 + x)' if log else meta['unit']})", row=1, col=1)
    fig.update_xaxes(title_text="Equipe", row=1, col=2, **_team_axis(teams))
    fig.update_yaxes(title_text=eixo, row=1, col=2)
    _plotly_to_json(fig, top=95)
    # Legenda abaixo dos eixos: no topo ela cobriria os títulos dos dois painéis.
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(b=110),
        height=540,
    )
    return fig.to_json()


def _chart_barras_variaveis(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    campo: str,
    titulo: str,
    eixo_y: str,
    formato: str,
    faixas: Sequence[Tuple[float, float, str]] = (),
    linhas: Sequence[Tuple[float, str]] = (),
) -> str:
    fig = go.Figure()
    labels = [meta["short_label"] for meta in STUDY_VARIABLES.values()]
    for team in TEAMS:
        y = [catalogo[(team, var)][campo] for var in STUDY_VARIABLES]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=y,
                name=team,
                marker_color=TEAM_COLORS[team],
                text=[formato.format(v).replace(".", ",") for v in y],
                textposition="outside",
                hovertemplate=f"{team}<br>%{{x}}: %{{y:.2f}}<extra></extra>",
            )
        )
    for y0, y1, cor in faixas:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=cor, opacity=0.12, line_width=0, layer="below")
    for y, rotulo in linhas:
        fig.add_hline(
            y=y,
            line=dict(color="#64748B", dash="dash", width=1),
            annotation_text=rotulo,
            annotation_position="top left",
            annotation_font=dict(size=10, color="#475569"),
        )
    fig.update_layout(
        title=titulo,
        barmode="group",
        xaxis_title="Variável do estudo",
        yaxis_title=eixo_y,
        height=470,
    )
    return _plotly_to_json(fig, top=80)


def _chart_media_cv(catalogo: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    fig = go.Figure()
    for team in TEAMS:
        rows = [catalogo[(team, var)] for var in STUDY_VARIABLES]
        fig.add_trace(
            go.Scatter(
                x=[r["media"] for r in rows],
                y=[r["cv"] for r in rows],
                mode="markers+text",
                text=[r["label"] for r in rows],
                textposition="top center",
                name=team,
                marker=dict(size=12, color=TEAM_COLORS[team], line=dict(color="#FFFFFF", width=1)),
                hovertemplate="%{text}<br>Média: %{x:.2f}<br>CV: %{y:.2f}%<extra>" + team + "</extra>",
            )
        )
    grid = np.geomspace(3, 140, 120)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=100 / np.sqrt(grid),
            mode="lines",
            name="Referência Poisson: CV = 100/√média",
            line=dict(color="#94A3B8", dash="dot", width=2),
            hovertemplate="Média: %{x:.2f}<br>CV Poisson: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Média × CV por variável e time (escala log no eixo x)",
        xaxis=dict(title="Média por jogo (unidade da variável; escala log)", type="log"),
        yaxis_title="Coeficiente de variação (%)",
        height=500,
    )
    return _plotly_to_json(fig, top=80)


def _chart_iqr_box(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    variable: str,
) -> str:
    meta = STUDY_VARIABLES[variable]
    fig = go.Figure()
    for i, team in enumerate(teams):
        row = catalogo[(team, variable)]
        for trace in _box_traces(row["values"], team, i, meta["unit"], i == 0):
            fig.add_trace(trace)
        for chave, rotulo in (("lim_inf", "Q1 − 1,5×IQR"), ("lim_sup", "Q3 + 1,5×IQR")):
            fig.add_trace(
                go.Scatter(
                    x=[i - 0.38, i + 0.38],
                    y=[row[chave], row[chave]],
                    mode="lines",
                    name="Limites de Tukey (1,5×IQR)",
                    legendgroup="limites",
                    showlegend=(i == 0 and chave == "lim_inf"),
                    line=dict(color="#C8102E", dash="dash", width=1.5),
                    hovertemplate=f"{team}<br>{rotulo} = {row[chave]:.2f} {meta['unit']}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=i + 0.4,
                y=row[chave],
                text=f"{rotulo} = {_br(row[chave])}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="#C8102E"),
            )
        for chave, rotulo in (("q1", "Q1"), ("q3", "Q3")):
            fig.add_annotation(
                x=i - 0.27,
                y=row[chave],
                text=f"{rotulo} = {_br(row[chave])}",
                showarrow=False,
                xanchor="right",
                font=dict(size=10, color=TEAM_COLORS[team]),
            )
    fig.update_layout(
        title=f"Boxplot de {meta['label']} com os limites 1,5×IQR (quartis lineares do pandas)",
        xaxis=dict(title="Equipe", **_team_axis(teams)),
        yaxis_title=f"{meta['short_label']} ({meta['unit']})",
        height=520,
    )
    fig.update_xaxes(range=[-0.8, len(teams) - 0.05])
    return _plotly_to_json(fig, top=80)


def _chart_faixas(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    variable: str,
) -> str:
    meta = STUDY_VARIABLES[variable]
    fig = go.Figure()
    nomes = [TEAM_SHORT[t] for t in teams]
    rows = [catalogo[(t, variable)] for t in teams]
    fig.add_trace(
        go.Bar(
            y=nomes,
            x=[r["amplitude"] for r in rows],
            base=[r["minimo"] for r in rows],
            orientation="h",
            name="Amplitude (mín. → máx.)",
            marker_color="rgba(148, 163, 184, 0.45)",
            width=0.6,
            hovertemplate="%{y}<br>Mín. %{base:.2f} · amplitude %{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=nomes,
            x=[r["iqr"] for r in rows],
            base=[r["q1"] for r in rows],
            orientation="h",
            name="IQR (Q1 → Q3, 50% centrais)",
            marker_color=[TEAM_COLORS[t] for t in teams],
            width=0.3,
            hovertemplate="%{y}<br>Q1 %{base:.2f} · IQR %{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=nomes,
            x=[r["media"] for r in rows],
            mode="markers",
            name="Média ± 1 desvio (ddof = 1)",
            marker=dict(symbol="diamond", size=11, color="#F59E0B", line=dict(color="#0F172A", width=1)),
            error_x=dict(type="data", array=[r["dp"] for r in rows], color="#F59E0B", thickness=2, width=8),
            hovertemplate="%{y}<br>Média %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Amplitude, IQR e média ± desvio — {meta['label']}",
        barmode="overlay",
        xaxis_title=f"{meta['short_label']} ({meta['unit']})",
        yaxis_title="Equipe",
        height=360,
    )
    return _plotly_to_json(fig, top=80)


def _chart_outliers(catalogo: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    return _chart_barras_variaveis(
        catalogo,
        "n_outliers",
        "Outliers de Tukey (1,5×IQR) por variável e time",
        "Jogos fora das cercas (contagem)",
        "{:.0f}",
    )


# ---------------------------------------------------------------------------
# Textos
# ---------------------------------------------------------------------------

def _lista(itens: Sequence[str]) -> str:
    itens = list(itens)
    if not itens:
        return "nenhuma"
    if len(itens) == 1:
        return itens[0]
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def _serie_nome(row: Dict[str, Any]) -> str:
    return f"{row['label'].lower()} dos {row['time_short']}"


def _textos_assimetria(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    variable: str,
    escala: str,
) -> Dict[str, List[str]]:
    meta = STUDY_VARIABLES[variable]
    todas = list(catalogo.values())
    sel = [catalogo[(t, variable)] for t in teams]
    n_ref = sel[0]["n"]
    fator = np.sqrt(n_ref * (n_ref - 1)) / (n_ref - 2)

    formula = [
        "Usamos o coeficiente de Fisher-Pearson ajustado, G1 = √(n(n − 1)) / (n − 2) × m3 / m2^(3/2), "
        "com m2 e m3 como momentos centrais da amostra (scipy.stats.skew com bias=False). "
        "O g1 sem ajuste (m3 / m2^(3/2)) subestima a assimetria em amostras pequenas. "
        f"Com n = {n_ref}, o fator de correção vale {_br(fator, 4)}: a diferença é pequena, mas este é o estimador "
        "que pandas, Excel e SAS reportam, o que deixa os números comparáveis com qualquer conferência externa.",
        f"Leitura: o sinal dá a direção da cauda mais longa. Pela intensidade, |g1| < {_br(LIMIAR_MODERADA, 1)} é "
        f"aproximadamente simétrica, {_br(LIMIAR_MODERADA, 1)} ≤ |g1| < {_br(LIMIAR_FORTE, 1)} é moderada e "
        f"|g1| ≥ {_br(LIMIAR_FORTE, 1)} é forte. "
        f"Com n ≈ 75, o erro padrão de g1 é cerca de {_br(sel[0]['g1_ses'])}. Por isso um |g1| abaixo de "
        f"{_br(2 * sel[0]['g1_ses'])} não se distingue de zero, e o teste de D'Agostino (scipy.stats.skewtest) "
        "acompanha a tabela.",
    ]

    leitura = []
    for r in sel:
        signif = "significativa a 5%" if r["g1_p"] < 0.05 else "não significativa a 5%"
        leitura.append(
            f"{r['time']}: g1 = {_br(r['g1'])}, sinal {r['sentido'].split(' (')[0]}, intensidade {r['intensidade']} "
            f"(teste de D'Agostino p = {_br(r['g1_p'], 4)}, {signif}). "
            f"Média {_br(r['media'])} contra mediana {_br(r['mediana'])} {meta['unit']} "
            f"(diferença {_br(r['media_menos_mediana'])}); Bowley = {_br(r['bowley'])}."
        )
    if len(sel) == 2:
        a, b = sel
        mais = a if abs(a["g1"]) > abs(b["g1"]) else b
        menos = b if mais is a else a
        leitura.append(
            f"Entre os times, a série mais assimétrica em {meta['short_label'].lower()} é a dos {mais['time_short']} "
            f"(|g1| = {_br(abs(mais['g1']))} contra {_br(abs(menos['g1']))}). "
            + (
                "Os dois sinais coincidem: a cauda está do mesmo lado nas duas equipes."
                if np.sign(a["g1"]) == np.sign(b["g1"])
                else "Os sinais divergem: a cauda fica de lados opostos nas duas equipes."
            )
        )

    por_int: Dict[str, List[str]] = {"aproximadamente simétrica": [], "moderada": [], "forte": []}
    for r in todas:
        por_int[r["intensidade"]].append(_serie_nome(r))
    signif = [_serie_nome(r) for r in todas if r["g1_p"] < 0.05]
    positivas = sum(1 for r in todas if r["g1"] > 0)
    negativa = min(todas, key=lambda r: r["g1"])
    comparacao = [
        f"Das {len(todas)} séries, {len(por_int['aproximadamente simétrica'])} são aproximadamente simétricas, "
        f"{len(por_int['moderada'])} têm assimetria moderada ({_lista(por_int['moderada'])}) e "
        f"{len(por_int['forte'])} têm assimetria forte. {positivas} das {len(todas)} têm g1 positivo.",
        f"Pelo teste de D'Agostino, a assimetria é significativa a 5% em {len(signif)} série(s): {_lista(signif)}. "
        "Nas demais, o g1 observado está dentro da variação esperada de uma amostra de ~75 jogos vinda de "
        "uma distribuição simétrica.",
        "Contagens por jogo têm piso em zero e nenhum teto rígido, o que tende a gerar cauda à direita: "
        "um jogo com muitas bolas de 3 ou tocos fica bem acima da média, mas nenhum jogo fica muito abaixo de zero."
        + (
            f" O g1 mais negativo é o de {_serie_nome(negativa)} ({_br(negativa['g1'])}): ali a cauda vem dos jogos "
            f"com valores baixos, e a mediana ({_br(negativa['mediana'])}) fica acima da média ({_br(negativa['media'])})."
            if negativa["g1"] < 0
            else ""
        ),
    ]

    concordam = [r for r in todas if np.sign(r["media_menos_mediana"]) == np.sign(r["g1"]) or abs(r["media_menos_mediana"]) < 1e-9]
    discordam = [_serie_nome(r) for r in todas if r not in concordam]
    media_mediana = [
        "Na cauda à direita, a média é puxada para cima da mediana; na cauda à esquerda, para baixo. "
        f"O sinal de média − mediana concorda com o sinal de g1 em {len(concordam)} das {len(todas)} séries."
        + (
            f" As exceções ({_lista(discordam)}) vêm da discretização: com contagens inteiras, a mediana só assume "
            "valores inteiros ou meios, e a diferença média − mediana fica dominada por esse arredondamento "
            "quando g1 é pequeno."
            if discordam
            else ""
        ),
        "O 2º coeficiente de Pearson, 3(média − mediana)/s, e o coeficiente de Bowley, (Q3 + Q1 − 2·Q2)/IQR, "
        "medem a mesma coisa sem o terceiro momento. O Bowley usa só quartis, então não reage a um jogo extremo isolado: "
        "quando ele e o g1 discordam, a assimetria de g1 vem das caudas, não do miolo da distribuição.",
    ]

    zeros = [f"{_serie_nome(r)} ({r['zeros']} jogo{'s' if r['zeros'] > 1 else ''})" for r in todas if r["zeros"] > 0]
    transf = []
    for r in sel:
        inverte = np.sign(r["g1_log"]) != np.sign(r["g1"])
        if inverte and abs(r["g1_log"]) >= LIMIAR_MODERADA:
            efeito = (
                f"inverte a direção: a assimetria continua {_intensidade(r['g1_log'])}, "
                f"agora {_sentido(r['g1_log']).split(' (')[0]}"
            )
        elif abs(r["g1_log"]) < LIMIAR_MODERADA <= abs(r["g1"]):
            efeito = "deixa a série aproximadamente simétrica"
        elif abs(r["g1_log"]) < abs(r["g1"]):
            efeito = "reduz |g1|" + (", mas troca o sinal" if inverte else "")
        else:
            efeito = "aumenta |g1|"
        transf.append(
            f"{r['time_short']}: g1 passa de {_br(r['g1'])} (bruta) para {_br(r['g1_log'])} em log1p, ou seja, a transformação {efeito}."
        )
    transformacao = [
        "O logaritmo comprime a cauda direita e só ajuda quando a assimetria é positiva e relevante. "
        "Em séries já simétricas ou com cauda à esquerda, ele cria ou piora uma assimetria negativa.",
        *transf,
        "Usamos log1p(x) = ln(1 + x) e não ln(x) porque há jogos com zero, onde ln(0) não existe: "
        f"{_lista(zeros)}. O deslocamento de 1 é pequeno perto de pontos e rebotes, mas pesa nas contagens baixas, "
        "então em roubos e tocos o g1 transformado depende dessa escolha.",
    ]

    robustos = [
        "Onde a assimetria é moderada, a mediana e o IQR descrevem o jogo típico melhor do que média e desvio. "
        "Eles não dependem da cauda e continuam válidos sem transformar a escala, o que mantém a interpretação na "
        "unidade original (pontos, rebotes, …).",
        "Para as séries aproximadamente simétricas, média e mediana praticamente coincidem, e média ± desvio "
        "é um resumo adequado. A transformação log1p fica como recurso para modelos que exigem simetria, "
        "não para a descrição.",
    ]

    return {
        "formula": formula,
        "leitura": leitura,
        "comparacao": comparacao,
        "media_mediana": media_mediana,
        "transformacao": transformacao,
        "robustos": robustos,
    }


def _textos_cv(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    variable: str,
) -> Dict[str, List[str]]:
    meta = STUDY_VARIABLES[variable]
    todas = list(catalogo.values())
    sel = [catalogo[(t, variable)] for t in teams]
    menor_media = min(todas, key=lambda r: r["media"])

    validade = [
        "O CV = s / média × 100 (s com ddof = 1) mede o desvio em proporção ao nível da variável e deixa "
        "a dispersão sem unidade. Ele só faz sentido em escala de razão, com zero absoluto e valores não negativos. "
        "As sete variáveis são contagens por jogo, então a condição vale para todas. O CV não serve para escalas "
        "intervalares nem para variáveis que mudam de sinal (como o saldo de pontos).",
        "A média também precisa estar longe de zero. Quando ela é pequena, qualquer variação absoluta modesta vira um "
        f"CV grande. A menor média do estudo é a de {_serie_nome(menor_media)} ({_br(menor_media['media'])}), "
        "e é exatamente nessas séries que o CV fica inflado.",
        f"Faixas usadas: CV < {_br(CV_BAIXO, 0)}% baixo, {_br(CV_BAIXO, 0)}–{_br(CV_ALTO, 0)}% moderado e "
        f"> {_br(CV_ALTO, 0)}% alto. São faixas de referência, não um teste: elas servem para comparar séries entre si.",
    ]

    baixas = [catalogo[(t, v)] for v in BAIXA_CONTAGEM for t in TEAMS]
    medias = [r["media"] for r in baixas]
    poisson = [r["cv_poisson"] for r in baixas]
    inflado = [
        f"Roubos e tocos têm médias de {_br(min(medias))} a {_br(max(medias))} por jogo, e o CV dessas séries é o "
        "maior do estudo. Há duas razões: o denominador é pequeno, e a variável é uma contagem discreta com piso em "
        "zero, cuja variância cresce junto com a média. Num processo de Poisson, var = média e, portanto, "
        f"CV = 100/√média. Ou seja, só pela natureza de contagem, um CV de {_br(min(poisson), 0)}–"
        f"{_br(max(poisson), 0)}% é esperado nessas médias.",
    ]
    for r in sorted(baixas, key=lambda r: (r["variable"], r["time"])):
        inflado.append(
            f"{r['label']} dos {r['time_short']}: média {_br(r['media'])}, s = {_br(r['dp'])}, CV = {_pct(r['cv'])}. "
            f"Uma contagem de Poisson com a mesma média teria CV de {_pct(r['cv_poisson'])}. O índice de dispersão "
            f"s²/média = {_br(r['dispersao_poisson'])} "
            + ("indica sobredispersão" if r["dispersao_poisson"] > 1.1 else "fica perto de 1 (equidispersão)")
            + (f"; houve {r['zeros']} jogo(s) com zero." if r["zeros"] else ".")
        )
    inflado.append(
        "Logo, um CV alto em roubos e tocos não quer dizer que essas estatísticas sejam mais erráticas do que pontos "
        "em sentido prático. Grande parte vem da escala pequena. A comparação justa entre os times é dentro da mesma "
        "variável, ou contra a referência de Poisson, como no gráfico média × CV."
    )

    leitura = []
    for r in sel:
        leitura.append(
            f"{r['time']}: CV de {meta['short_label'].lower()} = {_pct(r['cv'])} ({_nivel_cv(r['cv'])}), "
            f"com média {_br(r['media'])} e s = {_br(r['dp'])} {meta['unit']}. Um CV baixo significa jogos parecidos "
            "entre si, isto é, uma produção previsível. Um CV alto indica muita oscilação de jogo para jogo em relação "
            "ao nível médio do time."
        )
    if len(sel) == 2:
        a, b = sorted(sel, key=lambda r: r["cv"])
        leitura.append(
            f"Em {meta['short_label'].lower()}, os {a['time_short']} são mais consistentes "
            f"(CV {_pct(a['cv'])} contra {_pct(b['cv'])}, diferença de {_br(b['cv'] - a['cv'])} p.p.)."
        )

    pts = catalogo[(TEAMS[0], "pontos")]
    blk = catalogo[(TEAMS[0], "tocos")]
    rank_dp = sorted(STUDY_VARIABLES, key=lambda v: -catalogo[(TEAMS[0], v)]["dp"])
    rank_cv = sorted(STUDY_VARIABLES, key=lambda v: -catalogo[(TEAMS[0], v)]["cv"])
    escalas = [
        f"O desvio bruto carrega a unidade e o nível da variável. Nos Celtics, s = {_br(pts['dp'])} pontos contra "
        f"s = {_br(blk['dp'])} tocos, e ainda assim pontos é a variável mais estável "
        f"(CV {_pct(pts['cv'])} contra {_pct(blk['cv'])}). Um desvio de {_br(pts['dp'])} sobre uma média de "
        f"{_br(pts['media'])} pontos é uma variação pequena; {_br(blk['dp'])} sobre {_br(blk['media'])} tocos é uma "
        "variação grande.",
        "Ordenando as variáveis dos Celtics por desvio: "
        f"{_lista([STUDY_VARIABLES[v]['short_label'] for v in rank_dp])}. Por CV: "
        f"{_lista([STUDY_VARIABLES[v]['short_label'] for v in rank_cv])}. "
        "As duas ordens quase se invertem, e por isso o desvio bruto não compara variáveis de escalas diferentes.",
    ]

    consistencia_rows = []
    vence: Dict[str, List[str]] = {t: [] for t in TEAMS}
    for var, m in STUDY_VARIABLES.items():
        a, b = sorted((catalogo[(t, var)] for t in TEAMS), key=lambda r: r["cv"])
        vence[a["time"]].append(m["short_label"].lower())
        consistencia_rows.append(
            {
                "label": m["short_label"],
                "consistente": a["time"],
                "irregular": b["time"],
                "cv_menor": a["cv"],
                "cv_maior": b["cv"],
                "diferenca": b["cv"] - a["cv"],
                "selecionada": var == variable,
            }
        )
    consistencia = [
        f"Os Celtics têm o menor CV em {len(vence[TEAMS[0]])} das 7 variáveis ({_lista(vence[TEAMS[0]])}); "
        f"os Pelicans, em {len(vence[TEAMS[1]])} ({_lista(vence[TEAMS[1]])}).",
        "Diferenças de poucos pontos percentuais entre os times não devem ser lidas como diferença real de "
        "consistência: com ~75 jogos, o próprio CV tem erro amostral de alguns pontos percentuais.",
    ]

    reb = [catalogo[(t, "rebotes")] for t in TEAMS]
    assim = [r for r in todas if r["intensidade"] != "aproximadamente simétrica"]
    quase_igual = abs(reb[0]["dp"] - reb[1]["dp"]) < 0.25
    confronto = [
        ("Mesmo desvio não significa mesmo CV. " if quase_igual else "")
        + f"Em rebotes, os desvios são {_br(reb[0]['dp'])} (Celtics) e {_br(reb[1]['dp'])} (Pelicans), "
        f"mas, como as médias são {_br(reb[0]['media'])} e {_br(reb[1]['media'])}, os CVs ficam "
        f"em {_pct(reb[0]['cv'])} e {_pct(reb[1]['cv'])}. O CV corrige o nível, o desvio não.",
        "O CV herda as fragilidades da média e do desvio. Em séries assimétricas, um ou dois jogos de cauda inflam s "
        "e deslocam a média. "
        + (
            f"As séries com assimetria moderada ou forte ({_lista([_serie_nome(r) for r in assim])}) são as que "
            "mais pedem a versão robusta."
            if assim
            else "Aqui todas as séries são aproximadamente simétricas, o que dá mais segurança ao CV."
        ),
    ]

    discord = []
    for var in STUDY_VARIABLES:
        a, b = (catalogo[(t, var)] for t in TEAMS)
        if np.sign(a["cv"] - b["cv"]) != np.sign(a["cv_robusto"] - b["cv_robusto"]):
            discord.append(STUDY_VARIABLES[var]["short_label"].lower())
    robusto = [
        "A alternativa robusta é o CV quartílico, IQR / mediana × 100. Ele troca média e desvio por mediana e "
        "amplitude interquartil, ignora os 25% de jogos de cada cauda e não é movido por outliers.",
        (
            f"O CV robusto inverte qual time é mais consistente em {len(discord)} variável(is): {_lista(discord)}. "
            "Nesses casos a vantagem do CV clássico depende das caudas, não do comportamento típico."
            if discord
            else "O CV robusto aponta o mesmo time mais consistente que o CV clássico em todas as variáveis, "
            "e a conclusão não depende das caudas."
        ),
        "Em contagens baixas, o CV robusto anda em degraus, porque Q1, Q3 e a mediana assumem poucos valores possíveis. "
        "Ele é estável diante de outliers, mas pouco sensível a diferenças pequenas.",
    ]

    return {
        "validade": validade,
        "inflado": inflado,
        "leitura": leitura,
        "escalas": escalas,
        "consistencia": consistencia,
        "confronto": confronto,
        "robusto": robusto,
        "consistencia_rows": consistencia_rows,
    }


def _leitura_curta(r: Dict[str, Any]) -> str:
    partes = []
    if r["intensidade"] == "aproximadamente simétrica":
        partes.append("aproximadamente simétrica: média ± s e mediana/IQR concordam")
    else:
        partes.append(f"assimetria {r['intensidade']} {r['sentido'].split(' (')[0]}: prefira mediana e IQR")
    if r["variable"] in BAIXA_CONTAGEM:
        partes.append("média baixa infla o CV")
    elif r["cv"] < CV_BAIXO:
        partes.append("produção estável")
    if r["n_outliers"]:
        partes.append(f"{r['n_outliers']} outlier(s) de Tukey")
    texto = "; ".join(partes)
    return texto[0].upper() + texto[1:] + "."


def _textos_iqr(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    variable: str,
) -> Dict[str, List[str]]:
    meta = STUDY_VARIABLES[variable]
    todas = list(catalogo.values())
    sel = [catalogo[(t, variable)] for t in teams]
    un = meta["unit"]

    leitura = []
    for r in sel:
        leitura.append(
            f"{r['time']}: Q1 = {_br(r['q1'])} e Q3 = {_br(r['q3'])} {un}. A metade central dos jogos "
            f"(do 25º ao 75º percentil) ficou numa faixa de {_br(r['iqr'])} {un}, com mediana {_br(r['mediana'])}. "
            "O IQR descreve o jogo típico, não o pior nem o melhor."
        )
    if len(sel) == 2:
        a, b = sorted(sel, key=lambda r: r["iqr"])
        if np.isclose(a["iqr"], b["iqr"]):
            leitura.append(f"Os dois times têm o mesmo IQR em {meta['short_label'].lower()} ({_br(a['iqr'])} {un}).")
        else:
            leitura.append(
                f"Os {a['time_short']} têm a faixa central mais estreita ({_br(a['iqr'])} contra {_br(b['iqr'])} {un}). "
                f"Comparando as faixas [Q1, Q3], as dos dois times "
                + ("se sobrepõem" if a["q1"] <= b["q3"] and b["q1"] <= a["q3"] else "não se sobrepõem")
                + ", então o nível típico "
                + ("é parecido." if a["q1"] <= b["q3"] and b["q1"] <= a["q3"] else "é claramente diferente.")
            )

    comparacao = []
    for r in sel:
        comparacao.append(
            f"{r['time_short']}: amplitude = {_br(r['amplitude'])} {un} (de {_br(r['minimo'])} a {_br(r['maximo'])}), "
            f"IQR = {_br(r['iqr'])} ({_pct(r['iqr_sobre_amp'] * 100)} da amplitude) e s = {_br(r['dp'])}. "
            f"A razão IQR/s = {_br(r['iqr_sobre_dp'])}; numa normal ela vale {_br(IQR_SOBRE_DP_NORMAL)}."
        )
    comparacao.append(
        "A amplitude depende só de dois jogos, o mínimo e o máximo, e cresce com o número de partidas. "
        "O desvio usa todos os jogos, mas pesa os afastamentos ao quadrado, o que dá muito peso às caudas. "
        "O IQR usa a ordem dos dados e ignora 25% de cada lado: é a medida que menos se move com uma partida atípica."
    )
    comparacao.append(
        "IQR/s abaixo de 1,35 indica caudas mais pesadas do que as de uma normal, com o desvio inflado em relação "
        "ao miolo. Acima de 1,35, o miolo é largo e as caudas são curtas. Em contagens baixas, a razão também salta "
        "porque os quartis caem em valores inteiros."
    )

    ok2 = sum(1 for r in todas if r["parte2_confere"])
    status3 = {r["parte3"] for r in todas}
    if status3 == {"não integrada"}:
        txt3 = (
            "A conferência com quartis() e limites_outliers() da Parte 3 (app/analysis/descritiva.py) fica pendente: "
            "o módulo ainda não está no projeto. Quando ele entrar, a coluna “Parte 3” passa a ser preenchida sozinha."
        )
    else:
        ok3 = sum(1 for r in todas if r["parte3"] == "confere")
        txt3 = (
            f"As funções quartis() e limites_outliers() da Parte 3 conferem com os valores daqui em "
            f"{ok3} das {len(todas)} séries."
        )
    difere = [r for r in todas if not (np.isclose(r["q1"], r["q1_hazen"]) and np.isclose(r["q3"], r["q3_hazen"]))]
    ex = catalogo[(TEAMS[0], "pontos")]
    limites = []
    for r in sel:
        limites.append(
            f"{r['time_short']}: limites {_br(r['q1'])} − 1,5×{_br(r['iqr'])} = {_br(r['lim_inf'])} e "
            f"{_br(r['q3'])} + 1,5×{_br(r['iqr'])} = {_br(r['lim_sup'])} {un}. Os bigodes param no último jogo dentro "
            f"dessas cercas ({_br(r['bigode_inf'])} e {_br(r['bigode_sup'])}), e {r['n_outliers']} jogo(s) ficam de fora."
        )
    limites += [
        f"Os quartis, os limites e a contagem de outliers conferem com o catálogo da Parte 2 (detecção de outliers) "
        f"em {ok2} das {len(todas)} séries. {txt3}",
        "O boxplot desta página usa esses mesmos Q1, Q3 e limites, que são passados prontos ao Plotly, porque o "
        "quartilemethod=\"linear\" do plotly.js interpola "
        "na posição p·n − 0,5 (método de Hazen) e não em (n − 1)·p, como o pandas. "
        f"Com isso, {len(difere)} das {len(todas)} séries teriam Q1 ou Q3 diferentes no gráfico automático. "
        f"Exemplo: nos pontos dos Celtics, Q1 = {_br(ex['q1'])} pelo pandas contra {_br(ex['q1_hazen'])} pelo plotly.js. "
        "Boxplots desenhados com o quartilemethod automático (como os das Partes 2 e 3) mostram, nessas séries, "
        "caixas levemente diferentes das tabelas. Os números de referência são os das tabelas.",
    ]

    times = []
    vence: Dict[str, List[str]] = {t: [] for t in TEAMS}
    empates = []
    for var, m in STUDY_VARIABLES.items():
        a, b = (catalogo[(t, var)] for t in TEAMS)
        if np.isclose(a["iqr"], b["iqr"]):
            empates.append(m["short_label"].lower())
        else:
            vence[(a if a["iqr"] < b["iqr"] else b)["time"]].append(m["short_label"].lower())
    times.append(
        f"Faixa central mais estreita, ou seja, jogo típico mais previsível: Celtics em {_lista(vence[TEAMS[0]])}; "
        f"Pelicans em {_lista(vence[TEAMS[1]])}"
        + (f"; empate em {_lista(empates)}." if empates else ".")
    )
    times.append(
        "Para comparar o IQR entre variáveis de escalas diferentes, vale o mesmo raciocínio do CV: divida pela "
        "mediana (CV robusto), ou o IQR de pontos sempre parecerá maior só pela escala."
    )

    articulacao = [
        "O quadro-resumo junta as três medidas da Parte 4. O IQR dá a largura da faixa típica, o CV dá a oscilação "
        "relativa ao nível e o g1 diz se média e desvio são confiáveis como resumo.",
        "Onde g1 é pequeno e não há outliers, IQR/s fica perto de 1,35 e CV clássico e robusto apontam no mesmo "
        "sentido. Nesses casos as medidas clássicas bastam. Onde há assimetria ou média baixa, o IQR e o CV robusto "
        "são o resumo a reportar.",
    ]

    return {
        "leitura": leitura,
        "comparacao": comparacao,
        "limites": limites,
        "times": times,
        "articulacao": articulacao,
    }


def _qualidade() -> Dict[str, Any]:
    df = load_clean()
    return {
        "n_total": len(df),
        "n_times": {t: int((df["time"] == t).sum()) for t in TEAMS},
        "inicio": df["data_jogo"].min().strftime("%d/%m/%Y"),
        "fim": df["data_jogo"].max().strftime("%d/%m/%Y"),
        "ausentes": int(df[list(STUDY_VARIABLES)].isna().sum().sum()),
        "duplicados": int(df.duplicated(subset=["time", "id_jogo"]).sum()),
        "prorrogacoes": int(df["prorrogacao"].sum()),
        "zeros": {v: int((df[v] == 0).sum()) for v in STUDY_VARIABLES},
    }


def _efeito_prorrogacao(catalogo: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Quanto cada medida muda ao retirar os jogos com prorrogação (variação relativa máxima)."""
    delta = {"media": 0.0, "mediana": 0.0, "dp": 0.0, "iqr": 0.0}
    ot_out = 0
    for r in catalogo.values():
        x, ot = r["values"], r["prorrogacao"]
        ot_out += int((ot & ((x < r["lim_inf"]) | (x > r["lim_sup"]))).sum())
        if not ot.any():
            continue
        reg = _medidas(x[~ot])
        for k in delta:
            if r[k] > 0:
                delta[k] = max(delta[k], abs(reg[k] - r[k]) / r[k] * 100)
    return {"delta": delta, "ot_outliers": ot_out}


def _textos_sintese(
    catalogo: Dict[Tuple[str, str], Dict[str, Any]],
    teams: Sequence[str],
    q: Dict[str, Any],
    uniao: Dict[Tuple[str, str], int],
    intersecao: Dict[Tuple[str, str], int],
) -> Dict[str, List[str]]:
    todas = list(catalogo.values())
    zeros = [f"{STUDY_VARIABLES[v]['short_label'].lower()} ({n})" for v, n in q["zeros"].items() if n]
    efeito = _efeito_prorrogacao(catalogo)
    dl = efeito["delta"]
    n_out = sum(r["n_outliers"] for r in todas)

    dados = [
        f"A base tratada tem {q['n_total']} jogos ({q['n_times'][TEAMS[0]]} dos Celtics e "
        f"{q['n_times'][TEAMS[1]]} dos Pelicans), de {q['inicio']} a {q['fim']}, recortados das 2238 linhas e 30 times "
        "da base bruta (Parte 1). Nas sete variáveis do estudo há "
        f"{q['ausentes']} valor(es) ausente(s) e {q['duplicados']} jogo(s) duplicado(s): nenhuma medida depende de imputação.",
        f"Os únicos zeros aparecem em {_lista(zeros)}. São zeros legítimos (jogos sem tocos ou sem roubos), não "
        "falhas de registro, mas impedem o logaritmo direto e inflam o CV das contagens baixas (Parte 4).",
        f"Há {q['prorrogacoes']} jogos com prorrogação (MIN ≠ 240), mantidos na base pela Parte 2 porque são jogos "
        f"reais da temporada. Eles respondem por {efeito['ot_outliers']} dos {n_out} outliers de Tukey das 14 séries. "
        "Retirá-los mudaria, no pior caso, a média em "
        f"{_br(dl['media'])}%, a mediana em {_br(dl['mediana'])}%, o desvio em {_br(dl['dp'])}% e o IQR em "
        f"{_br(dl['iqr'])}%. "
        + (
            "As medidas de centro quase não dependem dessa decisão; as de dispersão, um pouco mais."
            if max(dl["media"], dl["mediana"]) < 2
            else "A decisão pesa nas medidas de centro e deve ser lembrada ao comparar os times."
        ),
    ]

    centro = []
    distancia = {}
    for var, m in STUDY_VARIABLES.items():
        a, b = (catalogo[(t, var)] for t in TEAMS)
        iqr_medio = (a["iqr"] + b["iqr"]) / 2
        distancia[var] = (a["mediana"] - b["mediana"]) / iqr_medio if iqr_medio > 0 else 0.0
        if np.isclose(a["mediana"], b["mediana"]):
            frase = f"mesma mediana ({_br(a['mediana'])})"
        else:
            maior = a if a["mediana"] > b["mediana"] else b
            menor = b if maior is a else a
            frase = f"{maior['time_short']} à frente ({_br(maior['mediana'])} contra {_br(menor['mediana'])})"
        centro.append(f"{m['short_label']}: {frase}.")
    grandes = sorted((v for v in distancia if abs(distancia[v]) >= 0.4), key=lambda v: -abs(distancia[v]))
    pequenas = [STUDY_VARIABLES[v]["short_label"].lower() for v in distancia if abs(distancia[v]) < 0.2]
    centro_txt = [
        "Centro (Parte 3, mediana por jogo): " + " ".join(centro),
        "Medindo a diferença de medianas em unidades do IQR médio dos dois times, os perfis mais se separam em "
        + _lista(
            [
                f"{STUDY_VARIABLES[v]['short_label'].lower()} "
                f"({TEAM_SHORT[TEAMS[0] if distancia[v] > 0 else TEAMS[1]]}, {_br(abs(distancia[v]))} IQR)"
                for v in grandes
            ]
        )
        + ". "
        + (
            f"Em {_lista(pequenas)}, a diferença fica abaixo de 0,2 IQR, pequena perto da variação de jogo para jogo."
            if pequenas
            else ""
        ),
    ]

    est = sorted(todas, key=lambda r: r["cv"])
    forma = [r for r in todas if r["intensidade"] != "aproximadamente simétrica"]
    out_total = {t: sum(catalogo[(t, v)]["n_outliers"] for v in STUDY_VARIABLES) for t in TEAMS}
    nivel_var: Dict[str, List[str]] = {"baixo": [], "moderado": [], "alto": []}
    for var, m in STUDY_VARIABLES.items():
        pior = max(catalogo[(t, var)]["cv"] for t in TEAMS)
        nivel_var[_nivel_cv(pior)].append(m["short_label"].lower())
    fortes = [r for r in forma if r["intensidade"] == "forte"]
    excecoes = [f"{_serie_nome(r)} (g1 = {_br(r['g1'])})" for r in forma]
    dispersao = [
        f"Dispersão (Parte 4): a série mais estável é a de {_serie_nome(est[0])} (CV {_pct(est[0]['cv'])}) e a mais "
        f"volátil é a de {_serie_nome(est[-1])} (CV {_pct(est[-1]['cv'])}). O ranking segue de perto o nível da variável. "
        f"Pelo maior CV entre os dois times, o CV é baixo em {_lista(nivel_var['baixo'])}, moderado em "
        f"{_lista(nivel_var['moderado'])} e alto em {_lista(nivel_var['alto'])}. Nas contagens baixas, grande parte "
        "disso é efeito de escala (referência de Poisson).",
        f"Forma: {len(todas) - len(forma)} das {len(todas)} séries são aproximadamente simétricas. "
        + (
            f"As exceções são {_lista(excecoes)}, "
            + (f"das quais {len(fortes)} com assimetria forte. " if fortes else "todas com assimetria moderada. ")
            + "Por isso média e mediana ficam próximas na maior parte do estudo."
            if forma
            else "Nenhuma série tem assimetria relevante."
        ),
        f"Outliers (Partes 2 e 3): pela regra de Tukey, os Celtics somam {out_total[TEAMS[0]]} e os Pelicans "
        f"{out_total[TEAMS[1]]} jogos fora das cercas nas sete variáveis. "
        f"Juntando os sete métodos da Parte 2, a união chega a {sum(uniao.values())} sinalizações, mas só "
        f"{sum(intersecao.values())} jogo(s) são marcados por todos os métodos aplicáveis. A falta de consenso e a "
        "ausência de valores impossíveis indicam jogos raros, não erros de registro.",
    ]

    perfis = []
    for t in teams:
        outro = TEAMS[1] if t == TEAMS[0] else TEAMS[0]
        mais = [
            STUDY_VARIABLES[v]["short_label"].lower()
            for v in STUDY_VARIABLES
            if catalogo[(t, v)]["mediana"] > catalogo[(outro, v)]["mediana"]
        ]
        estavel = [
            STUDY_VARIABLES[v]["short_label"].lower()
            for v in STUDY_VARIABLES
            if catalogo[(t, v)]["cv"] < catalogo[(outro, v)]["cv"]
        ]
        perfis.append(
            f"{t}: mediana maior que a do rival em {_lista(mais)}; mais consistente (menor CV) em {_lista(estavel)}."
        )

    classicas = [
        m["short_label"].lower()
        for v, m in STUDY_VARIABLES.items()
        if v not in BAIXA_CONTAGEM
        and all(catalogo[(t, v)]["intensidade"] == "aproximadamente simétrica" for t in TEAMS)
    ]
    conclusao = [
        "Para descrever o jogo típico, a combinação mediana + IQR é a que menos depende das decisões de qualidade "
        "de dados (prorrogação, zeros, outliers) e serve a todas as variáveis. "
        f"Média ± desvio é adequada em {_lista(classicas)}, que são aproximadamente simétricas nos dois times e "
        "têm média longe de zero. Para roubos e tocos, CV e log1p precisam vir acompanhados da ressalva de "
        "contagem baixa.",
        "Limitações: uma única temporada e ~75 jogos por time (erro amostral de alguns pontos em CV e de cerca de "
        "0,28 em g1); totais por jogo sem ajuste por ritmo ou posse; jogos com prorrogação mantidos; e adversários "
        "diferentes para cada time. As diferenças pequenas entre Celtics e Pelicans devem ser lidas como tendência, "
        "não como diferença estabelecida.",
    ]

    return {
        "dados": dados,
        "centro": centro_txt,
        "dispersao": dispersao,
        "perfis": perfis,
        "conclusao": conclusao,
    }


# ---------------------------------------------------------------------------
# Funções públicas (uma por página)
# ---------------------------------------------------------------------------

def _base(team: Optional[str], variable: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[str], str]:
    if variable not in STUDY_VARIABLES:
        variable = "pontos"
    return _get_catalogo(), _selected_teams(team), variable


def get_assimetria_analysis(
    team: Optional[str] = None,
    variable: str = "pontos",
    escala: str = "bruta",
) -> Dict[str, Any]:
    """Assimetria (g1 ajustado) por variável e time, com histograma, KDE e boxplot."""
    catalogo, teams, variable = _base(team, variable)
    if escala not in ESCALAS:
        escala = "bruta"
    sel = [catalogo[(t, variable)] for t in teams]
    todas = list(catalogo.values())
    return {
        "variable": variable,
        "var_meta": STUDY_VARIABLES[variable],
        "escala": escala,
        "escalas": ESCALAS,
        "rows": _linhas(catalogo, variable),
        "sel": sel,
        "kpis": {
            "simetricas": sum(1 for r in todas if r["intensidade"] == "aproximadamente simétrica"),
            "moderadas": sum(1 for r in todas if r["intensidade"] == "moderada"),
            "fortes": sum(1 for r in todas if r["intensidade"] == "forte"),
            "significativas": sum(1 for r in todas if r["g1_p"] < 0.05),
            "n_series": len(todas),
        },
        "texts": _textos_assimetria(catalogo, teams, variable, escala),
        "chart_dist_json": _chart_distribuicao(catalogo, teams, variable, escala),
        "chart_g1_json": _chart_barras_variaveis(
            catalogo,
            "g1",
            "Assimetria g1 (Fisher-Pearson ajustado) por variável e time",
            "g1 (adimensional)",
            "{:.2f}",
            faixas=[(-LIMIAR_MODERADA, LIMIAR_MODERADA, "#22C55E")],
            linhas=[
                (LIMIAR_MODERADA, "+0,5 moderada"),
                (-LIMIAR_MODERADA, "−0,5 moderada"),
                (LIMIAR_FORTE, "+1 forte"),
                (-LIMIAR_FORTE, "−1 forte"),
            ],
        ),
    }


def get_cv_analysis(team: Optional[str] = None, variable: str = "pontos") -> Dict[str, Any]:
    """Coeficiente de variação (%) por variável e time, com a alternativa robusta."""
    catalogo, teams, variable = _base(team, variable)
    texts = _textos_cv(catalogo, teams, variable)
    rows = _linhas(catalogo, variable)
    for row in rows:
        row["nivel_cv"] = _nivel_cv(row["cv"])
    todas = list(catalogo.values())
    est = min(todas, key=lambda r: r["cv"])
    vol = max(todas, key=lambda r: r["cv"])
    return {
        "variable": variable,
        "var_meta": STUDY_VARIABLES[variable],
        "rows": rows,
        "sel": [catalogo[(t, variable)] for t in teams],
        "consistencia_rows": texts.pop("consistencia_rows"),
        "kpis": {"estavel": est, "volatil": vol},
        "texts": texts,
        "chart_cv_json": _chart_barras_variaveis(
            catalogo,
            "cv",
            "Coeficiente de variação (%) por variável e time (desvio com ddof = 1)",
            "CV (%)",
            "{:.1f}",
            linhas=[(CV_BAIXO, "15% (baixo/moderado)"), (CV_ALTO, "30% (moderado/alto)")],
        ),
        "chart_media_cv_json": _chart_media_cv(catalogo),
    }


def get_iqr_analysis(team: Optional[str] = None, variable: str = "pontos") -> Dict[str, Any]:
    """Q1, Q3, IQR e limites de Tukey, conferidos com as Partes 2 e 3."""
    catalogo, teams, variable = _base(team, variable)
    rows = _linhas(catalogo, variable)
    for row in rows:
        row["leitura_curta"] = _leitura_curta(row)
    lado_a_lado = []
    for var, m in STUDY_VARIABLES.items():
        a, b = (catalogo[(t, var)] for t in TEAMS)
        lado_a_lado.append({"label": m["short_label"], "unit": m["unit"], "a": a, "b": b, "selecionada": var == variable})
    todas = list(catalogo.values())
    return {
        "variable": variable,
        "var_meta": STUDY_VARIABLES[variable],
        "rows": rows,
        "lado_a_lado": lado_a_lado,
        "sel": [catalogo[(t, variable)] for t in teams],
        "parte3_integrada": _parte3 is not None,
        "kpis": {
            "parte2_ok": sum(1 for r in todas if r["parte2_confere"]),
            "n_series": len(todas),
            "n_outliers": sum(r["n_outliers"] for r in todas),
            "hazen_difere": sum(
                1 for r in todas if not (np.isclose(r["q1"], r["q1_hazen"]) and np.isclose(r["q3"], r["q3_hazen"]))
            ),
        },
        "texts": _textos_iqr(catalogo, teams, variable),
        "chart_box_json": _chart_iqr_box(catalogo, teams, variable),
        "chart_faixas_json": _chart_faixas(catalogo, teams, variable),
    }


def get_sintese_analysis(team: Optional[str] = None) -> Dict[str, Any]:
    """Síntese geral: qualidade dos dados, centro, dispersão, forma e outliers."""
    catalogo, teams, _ = _base(team, "pontos")
    q = _qualidade()
    out_catalog = _get_outlier_catalog()
    count_rows = _outlier_count_rows(out_catalog, "pontos")
    uniao = {(row["time"], row["variavel"]): row["uniao"] for row in count_rows}
    intersecao = {(row["time"], row["variavel"]): row["intersecao"] for row in count_rows}
    rows = []
    for var in STUDY_VARIABLES:
        for t in teams:
            r = dict(catalogo[(t, var)])
            r["uniao"] = uniao[(t, var)]
            r["leitura_curta"] = _leitura_curta(r)
            rows.append(r)
    return {
        "rows": rows,
        "qualidade": q,
        "texts": _textos_sintese(catalogo, teams, q, uniao, intersecao),
        "chart_outliers_json": _chart_outliers(catalogo),
    }
