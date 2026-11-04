"""
Detecção univariada de outliers por time e por variável (Parte 2).

Os seis métodos rodam em separado em cada série (um time × uma variável).
Celtics e Pelicans não entram na mesma conta: misturar os dois criaria
"outlier" só pela diferença de nível entre as equipes. ECOD, k-NN, DBSCAN
e LOF ficam em uma dimensão de propósito — o enunciado pede a tabela
variável × método × time e deixa métodos de alta dimensionalidade de fora.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyod.models.ecod import ECOD
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from app.data_loader import STUDY_VARIABLES, TEAMS, load_clean

Z_THRESHOLD = 3.0
IQR_K = 1.5
GRUBBS_ALPHA = 0.05
GRUBBS_MAX_FRAC = 0.10
ECOD_CONTAMINATION = 0.05
KNN_K = 5
KNN_PERCENTILE = 95.0
DBSCAN_MIN_SAMPLES = 5
DBSCAN_K = DBSCAN_MIN_SAMPLES - 1
LOF_NEIGHBORS = 10
LOF_CONTAMINATION = 0.05
DISCRETE_CAUTION = {"roubos", "tocos"}

DETECTORS: List[Dict[str, str]] = [
    {"id": "zscore", "short": "Z-score", "label": "Z-score (|z| > 3)"},
    {"id": "iqr", "short": "IQR", "label": "IQR / Boxplot (1,5×IQR)"},
    {"id": "grubbs", "short": "Grubbs", "label": "Grubbs (α = 0,05, iterativo)"},
    {"id": "ecod", "short": "ECOD", "label": "ECOD (contamination = 0,05)"},
    {"id": "knn", "short": "k-NN", "label": "k-NN (k = 5, percentil 95)"},
    {"id": "dbscan", "short": "DBSCAN", "label": "DBSCAN (eps pelo cotovelo, min_samples = 5)"},
    {"id": "lof", "short": "LOF", "label": "LOF (10 vizinhos, contamination = 0,05)"},
]
DETECTOR_IDS: List[str] = [item["id"] for item in DETECTORS]
METHOD_CHOICES: List[Dict[str, str]] = [
    {"id": "todos", "short": "Comparar todos", "label": "Comparar todos os métodos"},
    *DETECTORS,
]
DETECTOR_SHORT = {item["id"]: item["short"] for item in DETECTORS}
DETECTOR_LABEL = {item["id"]: item["label"] for item in DETECTORS}

TEAM_SHORT = {
    "Boston Celtics": "Celtics",
    "New Orleans Pelicans": "Pelicans",
}

_CATALOG: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None


def _br(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _br_p(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.0001:
        return "< 0,0001"
    return _br(value, 4)


def _as_count(value: float) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _plotly_to_json(fig: go.Figure, top: int = 70) -> str:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1E293B"),
        title_font=dict(size=15, color="#0F172A"),
        margin=dict(l=60, r=30, t=top, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return fig.to_json()


def _theory_blocks(selected_method: str) -> List[Dict[str, Any]]:
    blocks = [
        {
            "id": "zscore",
            "titulo": "1. Z-score com limiar 3",
            "texto": (
                "O escore padrão é z = (x − média) / s, com s amostral (ddof = 1), "
                "o mesmo desvio usado no resto do projeto. A regra |z| > 3 marca o jogo "
                "que está a mais de três desvios da média do próprio time. Sob normalidade, "
                "essa cauda cobre cerca de 0,27% das observações: em n ≈ 75, o esperado é "
                "menos de um jogo por série. O critério é frágil quando o próprio outlier "
                "infla a média e o desvio, e não se sustenta se a série é assimétrica."
            ),
        },
        {
            "id": "iqr",
            "titulo": "2. IQR e cercas do boxplot (Tukey)",
            "texto": (
                "Q1 e Q3 saem da interpolação linear, o padrão do pandas e do numpy, "
                "a mesma regra dos quartis e do boxplot do projeto. IQR = Q3 − Q1 e as "
                "cercas são Q1 − 1,5×IQR e Q3 + 1,5×IQR. Ponto sobre a cerca não é outlier; "
                "só o que cai estritamente fora. A regra não usa a média, então um extremo "
                "não alarga o próprio critério. Em roubos e tocos o IQR costuma ser estreito: "
                "a cerca fica perto da caixa e mais jogos são marcados."
            ),
        },
        {
            "id": "grubbs",
            "titulo": "3. Teste de Grubbs iterativo",
            "texto": (
                "O teste de Grubbs pergunta se o ponto mais distante da média é incompatível "
                "com uma amostra normal. A estatística é G = max|x − média| / s, comparada "
                "ao valor crítico da t de Student com n − 2 graus de liberdade e nível "
                "α / (2n), α = 0,05, teste bicaudal. O SciPy não traz uma função pronta de "
                "Grubbs: o crítico é calculado com scipy.stats.t. O procedimento remove o "
                "extremo, recalcula média e desvio e testa de novo, até G não superar o "
                "crítico ou atingir o teto de 10% de n. Cada passo usa α = 0,05 sem correção "
                "de multiplicidade, então a taxa de falso positivo da sequência passa de 5%. "
                "Se o Shapiro-Wilk rejeita a normalidade, ou se a variável é contagem discreta "
                "com empates (roubos e tocos), o resultado fica indicativo: a conta é feita, "
                "mas a hipótese que justifica o valor crítico não se sustenta. Empates no "
                "extremo são desfeitos um a um, na ordem da série."
            ),
        },
        {
            "id": "ecod",
            "titulo": "4. ECOD",
            "texto": (
                "O ECOD (Empirical Cumulative Distribution-based Outlier Detection, Li et al., 2022) "
                "entra pela implementação do PyOD. Em uma dimensão, o escore cresce quando o jogo "
                "cai numa cauda da distribuição empírica, à esquerda ou à direita, sem assumir "
                "normalidade. O limiar usa contamination = 0,05: aproximadamente os 5% maiores "
                "escores da série são marcados. É uma regra de taxa. Ela aponta extremos mesmo "
                "quando a série é regular e o Z-score não marca ninguém. Como roda em uma variável "
                "só, não enxerga a combinação de pontos, rebotes e assistências no mesmo jogo."
            ),
        },
        {
            "id": "knn",
            "titulo": "5. k-NN para anomalias",
            "texto": (
                "A distância de cada jogo é a média das distâncias euclidianas aos k = 5 vizinhos "
                "mais próximos, excluindo o próprio ponto (sklearn.neighbors.NearestNeighbors). "
                "k = 5 é uma vizinhança local estável em amostras de cerca de 75 jogos: pequena "
                "o bastante para não engolir a cauda, grande o bastante para não reagir a um "
                "único empate. O limiar é o percentil 95 dessa distância; marca-se quem fica "
                "estritamente acima dele. Em contagem discreta há um efeito colateral: se vários "
                "jogos empatam num extremo, a distância entre eles é zero e o grupo deixa de "
                "parecer anômalo, mesmo longe do restante do time."
            ),
        },
        {
            "id": "dbscan",
            "titulo": "6. DBSCAN",
            "texto": (
                "O DBSCAN forma regiões densas e trata como ruído (rótulo −1) quem não alcança "
                "min_samples = 5 pontos dentro do raio eps, contando o próprio ponto. "
                "min_samples = 5 alinha a vizinhança ao k-NN. eps é o cotovelo da curva de "
                "k-distância com k = 4 (distância ao 4º vizinho, já ordenada): o ponto da curva "
                "mais afastado da corda entre o primeiro e o último valor. Em contagem inteira "
                "a curva é uma escada; o eps adotado aparece no gráfico. Se o cotovelo cair em "
                "zero, usa-se a menor distância positiva, porque o sklearn exige eps > 0. "
                "Valores raros, típicos de tocos altos, tendem a sair como ruído."
            ),
        },
        {
            "id": "lof",
            "titulo": "7. LOF",
            "texto": (
                "O Local Outlier Factor compara a densidade local do jogo com a dos vizinhos. "
                "Usa-se sklearn.neighbors.LocalOutlierFactor com n_neighbors = 10 (cerca de 13% "
                "de uma amostra de 75 jogos), o bastante para estabilizar a densidade sem deixar "
                "de ser local, e contamination = 0,05, de modo que os cerca de 5% com menor "
                "densidade relativa sejam marcados. Um jogo no meio de uma nuvem densa fica com "
                "LOF perto de 1; um jogo numa região rarefeita fica acima disso. Também é uma "
                "regra de taxa e, nesta página, também é univariada. Em contagens com muitos "
                "empates a distância de alcançabilidade zera no miolo da distribuição e o "
                "sklearn devolve escores sentinela, da ordem de 1e9, que derrubam o limiar "
                "interno. Esses escores são descartados e os 5% são recalculados só com os "
                "escores finitos."
            ),
        },
    ]
    for block in blocks:
        block["selecionado"] = selected_method in {block["id"], "todos"}
    return blocks


def _parameters() -> List[Dict[str, str]]:
    return [
        {
            "nome": "Z-score",
            "valor": "|z| > 3, desvio amostral (ddof = 1)",
            "justificativa": "Limiar clássico de cauda. Com n ≈ 75, sob normalidade, o esperado é menos de um jogo por série.",
        },
        {
            "nome": "Quartis e IQR",
            "valor": "Interpolação linear; cercas em 1,5×IQR",
            "justificativa": "Convenção fixa do projeto, a mesma do boxplot. Ponto sobre a cerca permanece dentro.",
        },
        {
            "nome": "Grubbs",
            "valor": "Bicaudal, α = 0,05, teto de 10% de n",
            "justificativa": "Valor crítico via scipy.stats.t. A iteração não corrige a multiplicidade; Shapiro-Wilk acompanha a hipótese de normalidade.",
        },
        {
            "nome": "ECOD",
            "valor": "contamination = 0,05",
            "justificativa": "Cauda de 5% do escore empírico. Regra de taxa, não um teste de significância.",
        },
        {
            "nome": "k-NN",
            "valor": "k = 5; limiar no percentil 95 da distância média",
            "justificativa": "Vizinhança local estável em n ≈ 75. A comparação é estrita: distância acima do percentil.",
        },
        {
            "nome": "DBSCAN",
            "valor": "min_samples = 5; eps = cotovelo da k-distância (k = 4)",
            "justificativa": "min_samples acompanha o k do k-NN. O cotovelo é calculado em cada série e mostrado no gráfico.",
        },
        {
            "nome": "LOF",
            "valor": "n_neighbors = 10; contamination = 0,05",
            "justificativa": "Dez vizinhos (~13% da amostra) equilibram localidade e estabilidade. A taxa de 5% alinha a leitura ao ECOD e é reaplicada só nos escores finitos quando o empate gera sentinela.",
        },
    ]


def _shapiro(x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(x) < 3 or np.nanstd(x) == 0:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.shapiro(x)
    return float(result.statistic), float(result.pvalue)


def _grubbs(x: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], bool, str]:
    n_orig = len(x)
    flagged = np.zeros(n_orig, dtype=bool)
    steps: List[Dict[str, Any]] = []
    if n_orig < 3:
        return flagged, steps, False, "Amostra menor que 3: o teste de Grubbs não se aplica."
    if float(np.std(x, ddof=1)) == 0:
        return flagged, steps, False, "Desvio zero: G não se define."

    max_out = int(np.floor(GRUBBS_MAX_FRAC * n_orig))
    remaining = np.ones(n_orig, dtype=bool)
    passo = 0
    while int(flagged.sum()) < max_out:
        passo += 1
        idx = np.flatnonzero(remaining)
        sample = x[idx]
        mu = float(sample.mean())
        sigma = float(sample.std(ddof=1))
        if sigma == 0 or not np.isfinite(sigma):
            break
        dev = np.abs(sample - mu)
        local = int(np.argmax(dev))
        g_stat = float(dev[local] / sigma)
        n_ties = int(np.sum(np.isclose(dev, dev[local])))
        m = int(len(idx))
        t_crit = float(stats.t.ppf(1 - GRUBBS_ALPHA / (2 * m), m - 2))
        g_crit = ((m - 1) / np.sqrt(m)) * np.sqrt((t_crit ** 2) / (m - 2 + t_crit ** 2))
        remove = bool(g_stat > g_crit)
        steps.append(
            {
                "passo": passo,
                "n": m,
                "G": g_stat,
                "Gcrit": float(g_crit),
                "valor": float(sample[local]),
                "empates": n_ties,
                "removido": remove,
            }
        )
        if not remove:
            break
        flagged[int(idx[local])] = True
        remaining[int(idx[local])] = False
        if passo > max_out + 1:
            break
    return flagged, steps, True, ""


def _ecod_mask(x: np.ndarray) -> Tuple[np.ndarray, bool, str]:
    n = len(x)
    empty = np.zeros(n, dtype=bool)
    if float(np.std(x, ddof=1)) == 0:
        return empty, False, "Desvio zero: o escore ECOD não separa caudas."
    try:
        model = ECOD(contamination=ECOD_CONTAMINATION)
        model.fit(x.reshape(-1, 1))
        labels = np.asarray(model.labels_).astype(int)
        if len(labels) != n:
            return empty, False, "ECOD devolveu um vetor de tamanho inesperado."
        return labels == 1, True, ""
    except Exception:
        return empty, False, "ECOD não convergiu nesta série."


def _knn_mask(x: np.ndarray) -> Tuple[np.ndarray, Optional[float], int, bool, str]:
    n = len(x)
    empty = np.zeros(n, dtype=bool)
    if n <= KNN_K:
        return empty, None, 0, False, "Amostra pequena demais para k = 5."
    model = NearestNeighbors(n_neighbors=KNN_K + 1, metric="euclidean")
    model.fit(x.reshape(-1, 1))
    distances, _ = model.kneighbors(x.reshape(-1, 1))
    mean_dist = distances[:, 1:].mean(axis=1)
    threshold = float(np.quantile(mean_dist, KNN_PERCENTILE / 100.0))
    mask = mean_dist > threshold
    tied = int(np.sum(np.isclose(mean_dist, threshold)))
    return mask, threshold, tied, True, ""


def _knee_eps(sorted_dist: np.ndarray) -> Tuple[Optional[float], Optional[int]]:
    y = np.asarray(sorted_dist, dtype=float)
    n = len(y)
    if n < 3:
        eps = float(y[-1]) if n else None
        return eps, n
    x = np.linspace(0.0, 1.0, n)
    span = float(y.max() - y.min())
    y_norm = (y - y.min()) / span if span > 0 else np.zeros(n)
    y0 = float(y_norm[0])
    y1 = float(y_norm[-1])
    cross = x * (y1 - y0) - (y_norm - y0)
    dist_line = np.abs(cross) / (np.hypot(1.0, y1 - y0) + 1e-12)
    dist_line[0] = -1.0
    dist_line[-1] = -1.0
    knee = int(np.argmax(dist_line))
    eps = float(y[knee])
    if eps <= 0:
        positive = y[y > 0]
        if len(positive) == 0:
            return None, knee + 1
        eps = float(positive[0])
    return eps, knee + 1


def _dbscan_mask(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    n = len(x)
    empty = np.zeros(n, dtype=bool)
    meta: Dict[str, Any] = {
        "eps": None,
        "knee_rank": None,
        "kdist": [],
        "clusters": None,
        "noise": None,
    }
    if n <= DBSCAN_K:
        return empty, meta, False, "Amostra pequena demais para a curva de k-distância."
    model = NearestNeighbors(n_neighbors=DBSCAN_K + 1, metric="euclidean")
    model.fit(x.reshape(-1, 1))
    distances, _ = model.kneighbors(x.reshape(-1, 1))
    kdist = np.sort(distances[:, -1])
    eps, knee_rank = _knee_eps(kdist)
    meta["kdist"] = [float(v) for v in kdist]
    meta["knee_rank"] = knee_rank
    meta["eps"] = eps
    if eps is None or eps <= 0:
        return empty, meta, False, "eps não positivo: o DBSCAN não roda nesta série."
    labels = DBSCAN(eps=float(eps), min_samples=DBSCAN_MIN_SAMPLES).fit_predict(x.reshape(-1, 1))
    mask = labels == -1
    n_clusters = len({int(lab) for lab in labels if int(lab) >= 0})
    meta["clusters"] = n_clusters
    meta["noise"] = int(mask.sum())
    return mask, meta, True, ""


def _lof_mask(x: np.ndarray) -> Tuple[np.ndarray, int, bool, str]:
    """LOF com limiar de 5% sobre escores finitos.

    Empates zeraram a distância de alcançabilidade em parte das contagens e o
    sklearn passou a gravar escores sentinela (~1e9) justamente no miolo da
    série. O offset de contamination acompanha esse sentinela e a previsão
    oficial marca zero outliers. O escore sentinela é descartado; o percentil
    95 é reaplicado só nos escores finitos.
    """
    n = len(x)
    empty = np.zeros(n, dtype=bool)
    if n <= LOF_NEIGHBORS:
        return empty, 0, False, "Amostra pequena demais para 10 vizinhos."
    if float(np.std(x, ddof=1)) == 0:
        return empty, 0, False, "Desvio zero: a densidade local não se diferencia."
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = LocalOutlierFactor(
            n_neighbors=LOF_NEIGHBORS,
            contamination=LOF_CONTAMINATION,
        )
        model.fit(x.reshape(-1, 1))
    scores = -np.asarray(model.negative_outlier_factor_, dtype=float)
    finite = np.isfinite(scores) & (scores < 1e6)
    n_sentinel = int((~finite).sum())
    if int(finite.sum()) < max(5, LOF_NEIGHBORS):
        return empty, n_sentinel, False, "Escores LOF finitos insuficientes por causa dos empates."
    threshold = float(np.quantile(scores[finite], 1.0 - LOF_CONTAMINATION))
    mask = finite & (scores > threshold)
    reason = ""
    if n_sentinel:
        reason = (
            "Empates zeraram a distância de alcançabilidade no miolo da série. "
            "Escores sentinela do sklearn foram descartados e o limiar de 5% "
            "foi reaplicado só nos escores finitos."
        )
    return mask, n_sentinel, True, reason


def _regulation_fences(x: np.ndarray, minutes: np.ndarray) -> Dict[str, Optional[float]]:
    regular = x[minutes == 240]
    out: Dict[str, Optional[float]] = {
        "lower": None,
        "upper": None,
        "z_low": None,
        "z_high": None,
    }
    if len(regular) < 4:
        return out
    series = pd.Series(regular)
    q1 = float(series.quantile(0.25, interpolation="linear"))
    q3 = float(series.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    out["lower"] = q1 - IQR_K * iqr
    out["upper"] = q3 + IQR_K * iqr
    sigma = float(np.std(regular, ddof=1))
    mu = float(np.mean(regular))
    if sigma > 0:
        out["z_low"] = mu - Z_THRESHOLD * sigma
        out["z_high"] = mu + Z_THRESHOLD * sigma
    return out


def _analyze_series(
    frame: pd.DataFrame,
    team: str,
    variable: str,
) -> Dict[str, Any]:
    x = frame[variable].to_numpy(dtype=float)
    minutes = frame["minutos"].to_numpy(dtype=float)
    n = len(x)
    series = pd.Series(x)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    q1 = float(series.quantile(0.25, interpolation="linear"))
    q3 = float(series.quantile(0.75, interpolation="linear"))
    iqr = float(q3 - q1)
    lower = q1 - IQR_K * iqr
    upper = q3 + IQR_K * iqr
    if std > 0:
        z = (x - mean) / std
        z_mask = np.abs(z) > Z_THRESHOLD
        z_ok, z_reason = True, ""
    else:
        z = np.zeros(n, dtype=float)
        z_mask = np.zeros(n, dtype=bool)
        z_ok, z_reason = False, "Desvio zero: o Z-score não se define."

    iqr_mask = (x < lower) | (x > upper)
    grubbs_mask, grubbs_steps, grubbs_ok, grubbs_reason = _grubbs(x)
    ecod_mask, ecod_ok, ecod_reason = _ecod_mask(x)
    knn_mask, knn_threshold, knn_ties, knn_ok, knn_reason = _knn_mask(x)
    dbscan_mask, dbscan_meta, dbscan_ok, dbscan_reason = _dbscan_mask(x)
    lof_mask, lof_sentinels, lof_ok, lof_reason = _lof_mask(x)

    shapiro_w, shapiro_p = _shapiro(x)
    _, tie_counts = np.unique(x, return_counts=True)
    max_tie_share = float(tie_counts.max() / n) if n else 0.0
    normality_rejected = bool(shapiro_p is not None and shapiro_p < GRUBBS_ALPHA)
    grubbs_indicativo = bool(
        normality_rejected or variable in DISCRETE_CAUTION or max_tie_share >= 0.20
    )

    masks = {
        "zscore": z_mask,
        "iqr": iqr_mask,
        "grubbs": grubbs_mask,
        "ecod": ecod_mask,
        "knn": knn_mask,
        "dbscan": dbscan_mask,
        "lof": lof_mask,
    }
    applicable = {
        "zscore": z_ok,
        "iqr": True,
        "grubbs": grubbs_ok,
        "ecod": ecod_ok,
        "knn": knn_ok,
        "dbscan": dbscan_ok,
        "lof": lof_ok,
    }
    reasons = {
        "zscore": z_reason,
        "iqr": "",
        "grubbs": grubbs_reason,
        "ecod": ecod_reason,
        "knn": knn_reason,
        "dbscan": dbscan_reason,
        "lof": lof_reason,
    }
    counts: Dict[str, Optional[int]] = {}
    for det in DETECTOR_IDS:
        counts[det] = int(masks[det].sum()) if applicable[det] else None

    considered = [det for det in DETECTOR_IDS if applicable[det]]
    if considered:
        stacked = np.vstack([masks[det] for det in considered])
        n_hit = stacked.sum(axis=0).astype(int)
    else:
        stacked = np.zeros((0, n), dtype=bool)
        n_hit = np.zeros(n, dtype=int)

    dates = pd.to_datetime(frame["data_jogo"])
    return {
        "time": team,
        "variable": variable,
        "n": n,
        "values": x,
        "z": z,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "dates_br": [d.strftime("%d/%m/%Y") for d in dates],
        "adversario": frame["adversario"].astype(str).tolist(),
        "confronto": frame["confronto"].astype(str).tolist(),
        "minutos": [int(v) for v in minutes],
        "prorrogacao": [int(v) for v in frame["prorrogacao"].tolist()],
        "mean": mean,
        "std": std,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "normality_rejected": normality_rejected,
        "grubbs_indicativo": grubbs_indicativo,
        "max_tie_share": max_tie_share,
        "n_unique": int(len(tie_counts)),
        "masks": masks,
        "applicable": applicable,
        "reasons": reasons,
        "counts": counts,
        "considered": considered,
        "n_hit": n_hit,
        "stacked": stacked,
        "grubbs_steps": grubbs_steps,
        "knn_threshold": knn_threshold,
        "knn_ties": knn_ties,
        "lof_sentinels": lof_sentinels,
        "dbscan": dbscan_meta,
        "regulation": _regulation_fences(x, minutes),
    }


def _build_catalog() -> Dict[Tuple[str, str], Dict[str, Any]]:
    frame = load_clean()
    catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for team in TEAMS:
        team_frame = (
            frame[frame["time"] == team]
            .sort_values(["data_jogo", "id_jogo"])
            .reset_index(drop=True)
        )
        for variable in STUDY_VARIABLES:
            catalog[(team, variable)] = _analyze_series(team_frame, team, variable)
    return catalog


def _get_catalog() -> Dict[Tuple[str, str], Dict[str, Any]]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG


def _selected_teams(team: Optional[str]) -> List[str]:
    if team is None:
        return list(TEAMS)
    if team in TEAMS:
        return [team]
    return list(TEAMS)


def _hit_methods(series: Dict[str, Any], index: int) -> List[str]:
    if series["stacked"].size == 0:
        return []
    column = series["stacked"][:, index]
    return [det for det, flag in zip(series["considered"], column) if bool(flag)]


def _side_and_distance(series: Dict[str, Any], index: int) -> Tuple[str, float]:
    value = float(series["values"][index])
    if value > series["upper"]:
        return "superior", value - series["upper"]
    if value < series["lower"]:
        return "inferior", series["lower"] - value
    return "dentro das cercas", 0.0


def _count_rows(catalog: Dict[Tuple[str, str], Dict[str, Any]], variable: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for team in TEAMS:
        for var_key, meta in STUDY_VARIABLES.items():
            series = catalog[(team, var_key)]
            rows.append(
                {
                    "variavel": var_key,
                    "label": meta["short_label"],
                    "time": team,
                    "time_short": TEAM_SHORT[team],
                    "n": series["n"],
                    "counts": series["counts"],
                    "uniao": int((series["n_hit"] > 0).sum()),
                    "intersecao": int((series["n_hit"] == len(series["considered"])).sum())
                    if series["considered"]
                    else 0,
                    "shapiro_p_txt": _br_p(series["shapiro_p"]),
                    "grubbs_indicativo": series["grubbs_indicativo"],
                    "selecionada": var_key == variable,
                }
            )
    return rows


def _games_for(
    series: Dict[str, Any],
    method: str,
    unit: str,
) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    for index in range(series["n"]):
        methods = _hit_methods(series, index)
        if method == "todos":
            keep = bool(methods)
        else:
            keep = method in methods
        if not keep:
            continue
        side, distance = _side_and_distance(series, index)
        games.append(
            {
                "time": series["time"],
                "time_short": TEAM_SHORT[series["time"]],
                "data": series["dates_br"][index],
                "data_iso": series["dates"][index],
                "adversario": series["adversario"][index],
                "confronto": series["confronto"][index],
                "valor": _as_count(series["values"][index]),
                "unidade": unit,
                "minutos": series["minutos"][index],
                "prorrogacao": bool(series["prorrogacao"][index]),
                "metodos": ", ".join(DETECTOR_SHORT[item] for item in methods),
                "n_metodos": len(methods),
                "z": float(series["z"][index]),
                "dist_cerca": float(distance),
                "lado": side,
            }
        )
    games.sort(key=lambda game: (-game["n_metodos"], -abs(game["z"]), game["data_iso"]))
    return games


def _agreement(series: Dict[str, Any]) -> Dict[str, Any]:
    n_methods = len(series["considered"])
    buckets = []
    for k in range(1, n_methods + 1):
        indexes = np.flatnonzero(series["n_hit"] == k)
        buckets.append({"k": k, "jogos": int(len(indexes))})

    isolated: Dict[str, int] = {det: 0 for det in series["considered"]}
    if series["stacked"].size:
        for index in np.flatnonzero(series["n_hit"] == 1):
            hit = _hit_methods(series, int(index))
            if hit:
                isolated[hit[0]] += 1
    isolated_rows = [
        {"metodo": DETECTOR_SHORT[det], "jogos": count}
        for det, count in isolated.items()
        if count > 0
    ]
    isolated_rows.sort(key=lambda row: (-row["jogos"], row["metodo"]))

    intersection_games = []
    if n_methods:
        for index in np.flatnonzero(series["n_hit"] == n_methods):
            intersection_games.append(
                {
                    "data": series["dates_br"][int(index)],
                    "adversario": series["adversario"][int(index)],
                    "valor": _as_count(series["values"][int(index)]),
                    "prorrogacao": bool(series["prorrogacao"][int(index)]),
                }
            )
    return {
        "time": series["time"],
        "time_short": TEAM_SHORT[series["time"]],
        "n": series["n"],
        "n_metodos": n_methods,
        "uniao": int((series["n_hit"] > 0).sum()),
        "intersecao": len(intersection_games),
        "por_quantidade": buckets,
        "isolados": isolated_rows,
        "intersecao_jogos": intersection_games,
    }


def _stat_row(series: Dict[str, Any]) -> Dict[str, Any]:
    z_low = series["mean"] - Z_THRESHOLD * series["std"] if series["std"] > 0 else None
    z_high = series["mean"] + Z_THRESHOLD * series["std"] if series["std"] > 0 else None
    if series["normality_rejected"]:
        normality = "Rejeita normalidade (p < 0,05)"
    elif series["shapiro_p"] is None:
        normality = "Não calculado"
    else:
        normality = "Não rejeita normalidade (p ≥ 0,05)"
    return {
        "time": series["time"],
        "time_short": TEAM_SHORT[series["time"]],
        "n": series["n"],
        "media": series["mean"],
        "desvio": series["std"],
        "q1": series["q1"],
        "q3": series["q3"],
        "iqr": series["iqr"],
        "cerca_inf": series["lower"],
        "cerca_sup": series["upper"],
        "z_inf": z_low,
        "z_sup": z_high,
        "shapiro_w": series["shapiro_w"],
        "shapiro_p_txt": _br_p(series["shapiro_p"]),
        "normalidade": normality,
        "grubbs_indicativo": series["grubbs_indicativo"],
        "grubbs_removidos": series["counts"]["grubbs"],
        "eps": series["dbscan"]["eps"],
        "dbscan_clusters": series["dbscan"]["clusters"],
        "dbscan_ruido": series["dbscan"]["noise"],
        "knn_limiar": series["knn_threshold"],
        "knn_empates": series["knn_ties"],
        "lof_sentinels": series["lof_sentinels"],
        "max_tie_pct": 100.0 * series["max_tie_share"],
        "n_unique": series["n_unique"],
        "aplicaveis": series["applicable"],
        "motivos": series["reasons"],
    }


def _ot_tables(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    variable: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    for team in TEAMS:
        series = catalog[(team, variable)]
        for index in range(series["n"]):
            if not series["prorrogacao"][index]:
                continue
            value = float(series["values"][index])
            minutes = series["minutos"][index]
            value_48 = value * 240.0 / minutes if minutes else None
            reg = series["regulation"]
            outside_iqr_48 = False
            outside_z_48 = False
            if value_48 is not None and reg["lower"] is not None and reg["upper"] is not None:
                outside_iqr_48 = value_48 < reg["lower"] or value_48 > reg["upper"]
            if value_48 is not None and reg["z_low"] is not None and reg["z_high"] is not None:
                outside_z_48 = value_48 < reg["z_low"] or value_48 > reg["z_high"]
            methods = _hit_methods(series, index)
            selected_rows.append(
                {
                    "time": team,
                    "time_short": TEAM_SHORT[team],
                    "data": series["dates_br"][index],
                    "adversario": series["adversario"][index],
                    "confronto": series["confronto"][index],
                    "minutos": minutes,
                    "valor": _as_count(value),
                    "valor_48": value_48,
                    "metodos": ", ".join(DETECTOR_SHORT[item] for item in methods) or "—",
                    "n_metodos": len(methods),
                    "fora_iqr": "iqr" in methods,
                    "fora_z": "zscore" in methods,
                    "fora_iqr_48": outside_iqr_48,
                    "fora_z_48": outside_z_48,
                }
            )
            counts = {}
            for var_key in STUDY_VARIABLES:
                other = catalog[(team, var_key)]
                # O mesmo jogo ocupa a mesma linha nos dois catálogos do time,
                # porque a ordenação é data + id_jogo.
                counts[var_key] = int(other["n_hit"][index])
            matrix_rows.append(
                {
                    "time": team,
                    "time_short": TEAM_SHORT[team],
                    "data": series["dates_br"][index],
                    "adversario": series["adversario"][index],
                    "minutos": minutes,
                    "counts": counts,
                }
            )
    selected_rows.sort(key=lambda row: (row["time"], row["data"]))
    matrix_rows.sort(key=lambda row: (row["time"], row["data"]))
    return selected_rows, matrix_rows


def _join_games(games: Sequence[Dict[str, Any]]) -> str:
    if not games:
        return ""
    parts = [
        f"{game['data']} contra {game['adversario']} ({game['valor']})"
        for game in games
    ]
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts[:-1]) + " e " + parts[-1]


def _build_texts(
    variable: str,
    method: str,
    series_list: Sequence[Dict[str, Any]],
    agreements: Sequence[Dict[str, Any]],
    ot_selected: Sequence[Dict[str, Any]],
    ot_matrix: Sequence[Dict[str, Any]],
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, List[str]]:
    meta = STUDY_VARIABLES[variable]
    label = meta["short_label"]
    unit = meta["unit"]
    method_label = DETECTOR_LABEL.get(method, "a união dos seis métodos")
    scope = " e ".join(TEAM_SHORT[series["time"]] for series in series_list)

    leitura = []
    for series in series_list:
        counts_txt = ", ".join(
            f"{DETECTOR_SHORT[det]} {series['counts'][det] if series['counts'][det] is not None else '—'}"
            for det in DETECTOR_IDS
        )
        union = int((series["n_hit"] > 0).sum())
        inter = int((series["n_hit"] == len(series["considered"])).sum()) if series["considered"] else 0
        leitura.append(
            f"{TEAM_SHORT[series['time']]} em {label} (n = {series['n']}): "
            f"{counts_txt}. A união marca {union} jogos e a interseção dos métodos aplicáveis marca {inter}."
        )
    numeric_counts = []
    for series in series_list:
        for det in DETECTOR_IDS:
            value = series["counts"][det]
            if value is not None:
                numeric_counts.append((value, DETECTOR_SHORT[det], TEAM_SHORT[series["time"]]))
    if numeric_counts:
        lowest = min(numeric_counts, key=lambda item: item[0])
        highest = max(numeric_counts, key=lambda item: item[0])
        leitura.append(
            f"No recorte {scope}, o menor contingente é {lowest[1]} nos {lowest[2]} "
            f"({lowest[0]} jogos) e o maior é {highest[1]} nos {highest[2]} ({highest[0]} jogos). "
            f"A lista de partidas abaixo segue {method_label.lower()}."
        )

    normalidade = []
    for series in series_list:
        name = TEAM_SHORT[series["time"]]
        if series["shapiro_p"] is None:
            normalidade.append(f"{name}: Shapiro-Wilk não calculado (série degenerada).")
            continue
        decision = (
            "Rejeita a normalidade a 5%."
            if series["normality_rejected"]
            else "Não rejeita a normalidade a 5%."
        )
        extra = ""
        if variable in DISCRETE_CAUTION:
            extra = (
                f" {label} é contagem discreta, com {series['n_unique']} valores distintos "
                f"e o valor mais repetido em {_br(100 * series['max_tie_share'])}% dos jogos. "
                "O teste de Grubbs assume normalidade contínua; aqui ele fica indicativo."
            )
        elif series["grubbs_indicativo"]:
            extra = (
                " O Grubbs fica indicativo: ou a normalidade foi rejeitada, ou há empate forte "
                f"(valor mais repetido em {_br(100 * series['max_tie_share'])}% dos jogos)."
            )
        else:
            extra = (
                " A hipótese formal do Grubbs não foi rejeitada pelo Shapiro-Wilk, mas a iteração "
                "sem correção de multiplicidade continua valendo: cada passo usa α = 0,05."
            )
        removed = series["counts"]["grubbs"]
        if removed is None:
            grubbs_sentence = "Grubbs não se aplica nesta série."
        elif removed == 0:
            grubbs_sentence = "Grubbs não remove nenhum jogo até o teto de 10% de n."
        elif removed == 1:
            grubbs_sentence = "Grubbs remove 1 jogo até o teto de 10% de n."
        else:
            grubbs_sentence = f"Grubbs remove {removed} jogos até o teto de 10% de n."
        normalidade.append(
            f"{name}: Shapiro-Wilk W = {_br(series['shapiro_w'], 4)}, p = {_br_p(series['shapiro_p'])}. "
            f"{decision} {grubbs_sentence}{extra}"
        )

    concordancia = []
    for agreement in agreements:
        name = agreement["time_short"]
        if agreement["intersecao_jogos"]:
            n_inter = agreement["intersecao"]
            jogo_txt = "1 jogo" if n_inter == 1 else f"{n_inter} jogos"
            inter_txt = (
                f"A interseção reúne {jogo_txt}: "
                f"{_join_games(agreement['intersecao_jogos'])}."
            )
        else:
            inter_txt = "Nenhum jogo é marcado pelos seis métodos ao mesmo tempo."
        isolated_txt = (
            "Métodos isolados (único a marcar o jogo): "
            + ", ".join(f"{row['metodo']} ({row['jogos']})" for row in agreement["isolados"])
            + "."
            if agreement["isolados"]
            else "Nenhum método marca um jogo sozinho."
        )
        por_k = ", ".join(
            f"{row['k']} método(s) → {row['jogos']} jogos" for row in agreement["por_quantidade"]
        )
        concordancia.append(f"{name}: {inter_txt} {isolated_txt} Distribuição: {por_k}.")

    rate_methods = ("ecod", "knn", "lof")
    rate_bits = []
    for series in series_list:
        bits = []
        for det in rate_methods:
            count = series["counts"][det]
            if count is not None:
                bits.append(f"{DETECTOR_SHORT[det]} {count}")
        if bits:
            rate_bits.append(f"{TEAM_SHORT[series['time']]} ({', '.join(bits)})")
    if rate_bits:
        concordancia.append(
            "ECOD, k-NN no percentil 95 e LOF com contamination = 0,05 são regras de taxa: "
            "apontam cerca de 5% da série (por volta de 4 jogos em n ≈ 75) mesmo quando "
            "Z-score e Grubbs não marcam ninguém. A concordância entre eles pode ser só "
            "“os valores mais extremos”, não um consenso de anomalia rara. "
            + " ".join(rate_bits)
            + "."
        )
        sentinels = [
            f"{TEAM_SHORT[series['time']]} ({series['lof_sentinels']} escores)"
            for series in series_list
            if series["lof_sentinels"]
        ]
        if sentinels:
            concordancia.append(
                "No LOF, empates de contagem zeraram a distância de alcançabilidade no miolo "
                "da série e o sklearn gravou escores sentinela. O limiar oficial de contamination "
                "acompanha esse sentinela e passaria a marcar zero jogos. Esses escores foram "
                "descartados e os 5% foram recalculados só com os escores finitos. Séries afetadas: "
                + ", ".join(sentinels)
                + "."
            )
    concordancia.append(
        "O IQR reage à largura da caixa; Z-score e Grubbs reagem à média e ao desvio e exigem "
        "afastamento forte. O DBSCAN reage a vazio de vizinhança. O k-NN pode deixar passar "
        "um extremo empado, porque jogos com o mesmo valor são vizinhos à distância zero. "
        "Em uma dimensão, padronizar a variável antes de k-NN, DBSCAN, LOF ou ECOD não muda "
        "quem é marcado: esses critérios dependem de ordem ou de razões de distância. "
        "O Z-score entra como regra própria, com limiar absoluto de 3 desvios, e não como "
        "pré-processamento dos outros."
    )

    ot_flagged_var = [row for row in ot_selected if row["n_metodos"] > 0]
    ot_still = [row for row in ot_selected if row["fora_iqr_48"] or row["fora_z_48"]]
    any_var = []
    for row in ot_matrix:
        marked = [STUDY_VARIABLES[key]["short_label"] for key, count in row["counts"].items() if count > 0]
        if marked:
            any_var.append(
                f"{row['data']} ({row['time_short']}, {row['minutos']} min) em {', '.join(marked)}"
            )
    prorrogacao = [
        (
            "A Parte 1 já isolou 6 jogos com minutos diferentes de 240: cinco dos Pelicans "
            "com 265 minutos-jogador e um dos Celtics com 290. São uma ou duas prorrogações "
            "(cada uma soma 25 minutos-jogador, 5 atletas × 5 minutos). O total de pontos, "
            "rebotes ou faltas sobe porque o jogo durou mais, não porque a coleta quebrou."
        )
    ]
    if ot_flagged_var:
        described = "; ".join(
            f"{row['data']} ({row['time_short']} contra {row['adversario']}, {row['minutos']} min, "
            f"{row['valor']} {unit}, métodos: {row['metodos']})"
            for row in ot_flagged_var
        )
        prorrogacao.append(
            f"Na variável {label}, {len(ot_flagged_var)} dessas prorrogações entram na união dos métodos: {described}."
        )
    else:
        prorrogacao.append(
            f"Na variável {label}, nenhuma das 6 prorrogações entra na união dos seis métodos. "
            "O tempo extra não bastou, nesta estatística, para atravessar as regras."
        )
    if ot_still:
        still_txt = "; ".join(
            "{} ({}): {} {} em ritmo de 48 min".format(
                row["data"], row["time_short"], _br(row["valor_48"]), unit
            )
            for row in ot_still
        )
        prorrogacao.append(
            "Reescalando o total por 240/MIN e comparando com as cercas calculadas só nos jogos "
            f"de 240 minutos, ainda ficam fora do IQR ou do |z| = 3 regulamentar: {still_txt}. "
            "O ritmo do jogo, e não apenas a duração, é atípico. Mesmo assim não é erro de registro."
        )
    else:
        prorrogacao.append(
            "Reescalando o total por 240/MIN, nenhum desses jogos permanece fora das cercas de Tukey "
            "nem do |z| = 3 calculados só nos jogos de 240 minutos, na variável selecionada. "
            "O extremo bruto, quando existe, é explicado pela duração."
        )
    if any_var:
        prorrogacao.append(
            "Olhando as sete variáveis, as prorrogações marcadas por ao menos um método são: "
            + "; ".join(any_var)
            + "."
        )
    else:
        prorrogacao.append(
            "Nenhuma das 6 prorrogações é marcada em qualquer das sete variáveis."
        )

    zero_notes = []
    if variable in DISCRETE_CAUTION:
        for series in series_list:
            zero_idx = np.flatnonzero(series["values"] == 0)
            if len(zero_idx) == 0:
                continue
            flagged_zeros = [int(i) for i in zero_idx if series["n_hit"][int(i)] > 0]
            zero_notes.append(
                f"{TEAM_SHORT[series['time']]} tem {len(zero_idx)} jogo(s) com {label.lower()} = 0; "
                f"{len(flagged_zeros)} deles entram na união dos métodos."
            )

    recomendacao = [
        (
            "Decisão: manter os seis jogos de prorrogação em jogos_limpo.csv. "
            "MIN = 265 ou 290 é contexto de overtime, já auditado na Parte 1, e a flag "
            "prorrogacao fica disponível para as análises seguintes. Remover esses jogos "
            "apagaria partidas reais e encolheria a cauda de um jeito artificial."
        ),
        (
            "A regra descritiva desta disciplina continua sendo a cerca de Tukey (1,5×IQR), "
            "com a mesma interpolação linear do boxplot. Z-score, Grubbs, ECOD, k-NN, DBSCAN "
            "e LOF confrontam essa regra; não a substituem em silêncio. Quando só os métodos "
            "de taxa fixa marcam um jogo, ele é um extremo relativo da série, não um ponto "
            "fora da cerca."
        ),
        (
            "Não reescrever a base. Winsorizar, trocar a média pela mediana ou analisar por "
            "48 minutos pode fazer sentido num modelo posterior sensível à média. A Parte 3 "
            "descreve o jogo como aconteceu, em totais brutos. Identificar o outlier já é o "
            "tratamento desta etapa."
        ),
    ]
    if zero_notes:
        recomendacao.insert(
            1,
            "Não remover zeros de roubos ou tocos. " + " ".join(zero_notes)
            + " A Parte 1 mostrou que zero nessas colunas é evento de jogo, não código de ausência.",
        )
    elif variable in DISCRETE_CAUTION:
        recomendacao.insert(
            1,
            f"Não remover zeros de {label.lower()} se aparecerem em outro recorte: "
            "são eventos de jogo, não ausência disfarçada.",
        )

    univariada = [
        (
            "Os seis detectores rodam por time e por variável, em uma dimensão. Celtics e "
            "Pelicans não compartilham a mesma série: a diferença de nível entre as equipes "
            "não é outlier de ninguém. Métodos de alta dimensionalidade ficam de fora, como "
            "pede o enunciado."
        ),
        (
            "O custo é perder a combinação entre variáveis. Um jogo comum em pontos e extremo "
            "em tocos não aparece como anomalia conjunta. ECOD, k-NN, DBSCAN e LOF, do jeito "
            "que estão nesta página, respondem à pergunta “este número é estranho na história "
            "deste time nesta estatística?”, não “este jogo é estranho no conjunto das sete estatísticas?”."
        ),
    ]
    return {
        "leitura": leitura,
        "normalidade": normalidade,
        "concordancia": concordancia,
        "prorrogacao": prorrogacao,
        "recomendacao": recomendacao,
        "univariada": univariada,
    }


def _chart_scatter(
    series_list: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    method: str,
) -> str:
    titles = [
        f"{series['time']} (n = {series['n']})"
        for series in series_list
    ]
    fig = make_subplots(
        rows=len(series_list),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.14 if len(series_list) > 1 else 0.08,
        subplot_titles=titles,
    )
    method_name = "união dos métodos" if method == "todos" else DETECTOR_SHORT[method]
    legend_drawn = {
        "regular": False,
        "outlier": False,
        "ot": False,
        "ot_outlier": False,
        "mean": False,
        "z": False,
        "fence": False,
    }
    for row, series in enumerate(series_list, start=1):
        x = series["dates"]
        y = series["values"]
        groups = {"regular": [], "outlier": [], "ot": [], "ot_outlier": []}
        for index in range(series["n"]):
            methods = _hit_methods(series, index)
            flagged = bool(methods) if method == "todos" else method in methods
            overtime = bool(series["prorrogacao"][index])
            if overtime and flagged:
                key = "ot_outlier"
            elif overtime:
                key = "ot"
            elif flagged:
                key = "outlier"
            else:
                key = "regular"
            groups[key].append(index)

        styles = {
            "regular": ("Jogos dentro da regra selecionada", "#64748B", "circle", 8),
            "outlier": (f"Outlier ({method_name})", "#C8102E", "circle", 11),
            "ot": ("Prorrogação não sinalizada", "#D97706", "star", 13),
            "ot_outlier": ("Prorrogação sinalizada", "#C8102E", "star", 15),
        }
        for key, indexes in groups.items():
            if not indexes:
                continue
            name, color, symbol, size = styles[key]
            custom = [
                [
                    series["adversario"][i],
                    float(series["z"][i]),
                    series["minutos"][i],
                    _hit_methods(series, i) and ", ".join(DETECTOR_SHORT[m] for m in _hit_methods(series, i)) or "nenhum",
                ]
                for i in indexes
            ]
            fig.add_trace(
                go.Scatter(
                    x=[x[i] for i in indexes],
                    y=[float(y[i]) for i in indexes],
                    mode="markers",
                    name=name,
                    legendgroup=key,
                    showlegend=not legend_drawn[key],
                    marker=dict(color=color, symbol=symbol, size=size, line=dict(width=0)),
                    customdata=custom,
                    hovertemplate=(
                        "Data: %{x}<br>Adversário: %{customdata[0]}"
                        f"<br>{meta['short_label']}: %{{y}} {meta['unit']}"
                        "<br>z: %{customdata[1]:.2f}<br>Minutos: %{customdata[2]}"
                        "<br>Métodos: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            legend_drawn[key] = True

        x0, x1 = x[0], x[-1]
        line_specs = [
            ("mean", series["mean"], "Média do time", "#007A33", "dash"),
            ("z", series["mean"] - Z_THRESHOLD * series["std"], "Limite |z| = 3", "#64748B", "dot"),
            ("z", series["mean"] + Z_THRESHOLD * series["std"], "Limite |z| = 3", "#64748B", "dot"),
            ("fence", series["lower"], "Cerca de Tukey (1,5×IQR)", "#C8102E", "dash"),
            ("fence", series["upper"], "Cerca de Tukey (1,5×IQR)", "#C8102E", "dash"),
        ]
        if series["std"] <= 0:
            line_specs = [item for item in line_specs if item[0] != "z"]
        for key, y_line, name, color, dash in line_specs:
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y_line, y_line],
                    mode="lines",
                    name=name,
                    legendgroup=key,
                    showlegend=not legend_drawn[key],
                    line=dict(color=color, dash=dash, width=1.5),
                    hovertemplate=f"{name}: {y_line:.2f} {meta['unit']}<extra></extra>",
                ),
                row=row,
                col=1,
            )
            legend_drawn[key] = True
        fig.update_yaxes(
            title_text=f"{meta['short_label']} ({meta['unit']})",
            row=row,
            col=1,
        )
        fig.update_xaxes(title_text="Data do jogo", tickformat="%d/%m/%Y", row=row, col=1)

    fig.update_layout(
        title=(
            f"{meta['label']} por jogo — média, limites |z| = 3 e cercas de Tukey. "
            f"Destaque: {method_name}."
        ),
        height=430 * len(series_list),
        hovermode="closest",
    )
    return _plotly_to_json(fig, top=90)


def _chart_box(series_list: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    fig = go.Figure()
    colors = {"Boston Celtics": "#007A33", "New Orleans Pelicans": "#0C2340"}
    for series in series_list:
        color = colors[series["time"]]
        fig.add_trace(
            go.Box(
                y=[float(v) for v in series["values"]],
                name=series["time"],
                marker_color=color,
                line_color=color,
                boxmean=True,
                boxpoints="outliers",
                quartilemethod="linear",
                hovertemplate=(
                    f"{series['time']}<br>{meta['short_label']}: %{{y}} {meta['unit']}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=(
            f"Boxplot de {meta['label']} por time "
            "(quartis por interpolação linear, cercas de 1,5×IQR)"
        ),
        yaxis_title=f"{meta['short_label']} ({meta['unit']})",
        xaxis_title="Equipe",
        height=480,
        showlegend=True,
    )
    return _plotly_to_json(fig)


def _chart_bars(catalog: Dict[Tuple[str, str], Dict[str, Any]], variable: str, meta: Dict[str, Any]) -> str:
    fig = go.Figure()
    colors = {"Boston Celtics": "#007A33", "New Orleans Pelicans": "#C8102E"}
    x_labels = [item["short"] for item in DETECTORS]
    for team in TEAMS:
        series = catalog[(team, variable)]
        y_values = [0 if series["counts"][det] is None else series["counts"][det] for det in DETECTOR_IDS]
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_values,
                name=team,
                marker_color=colors[team],
                text=[str(v) for v in y_values],
                textposition="outside",
                hovertemplate=(
                    f"{team}<br>Método: %{{x}}<br>Jogos sinalizados: %{{y}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Jogos sinalizados em {meta['label']} por método — Celtics e Pelicans",
        barmode="group",
        xaxis_title="Método de detecção",
        yaxis_title="Jogos sinalizados (contagem)",
        height=480,
    )
    return _plotly_to_json(fig, top=80)


def _chart_kdist(series_list: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    fig = go.Figure()
    colors = {"Boston Celtics": "#007A33", "New Orleans Pelicans": "#0C2340"}
    for series in series_list:
        kdist = series["dbscan"]["kdist"]
        if not kdist:
            continue
        color = colors[series["time"]]
        ranks = list(range(1, len(kdist) + 1))
        fig.add_trace(
            go.Scatter(
                x=ranks,
                y=kdist,
                mode="lines+markers",
                name=f"{TEAM_SHORT[series['time']]} — distância ao 4º vizinho",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                hovertemplate=(
                    "Ordem: %{x}<br>Distância: %{y:.2f} "
                    f"{meta['unit']}<extra>{series['time']}</extra>"
                ),
            )
        )
        eps = series["dbscan"]["eps"]
        knee = series["dbscan"]["knee_rank"]
        if eps is not None and knee is not None:
            fig.add_trace(
                go.Scatter(
                    x=[knee],
                    y=[eps],
                    mode="markers",
                    name=f"Cotovelo {TEAM_SHORT[series['time']]} (eps = {_br(eps)} {meta['unit']})",
                    marker=dict(symbol="diamond", size=14, color=color, line=dict(color="#0F172A", width=1)),
                    hovertemplate=(
                        f"eps = {eps:.2f} {meta['unit']}<br>Posição do cotovelo: {knee}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        title=(
            f"Curva de k-distância para o eps do DBSCAN — {meta['label']} "
            f"(k = {DBSCAN_K}, min_samples = {DBSCAN_MIN_SAMPLES})"
        ),
        xaxis_title="Jogos ordenados pela distância ao 4º vizinho",
        yaxis_title=f"Distância ao 4º vizinho ({meta['unit']})",
        height=460,
    )
    return _plotly_to_json(fig, top=80)


def get_outliers_analysis(
    team: Optional[str] = None,
    variable: str = "pontos",
    method: str = "todos",
) -> Dict[str, Any]:
    """Monta tabelas, textos e gráficos da página de outliers."""
    if variable not in STUDY_VARIABLES:
        variable = "pontos"
    if method not in {item["id"] for item in METHOD_CHOICES}:
        method = "todos"

    catalog = _get_catalog()
    teams = _selected_teams(team)
    meta = STUDY_VARIABLES[variable]
    series_list = [catalog[(name, variable)] for name in teams]
    games = []
    for series in series_list:
        games.extend(_games_for(series, method, meta["unit"]))
    agreements = [_agreement(series) for series in series_list]
    ot_selected, ot_matrix = _ot_tables(catalog, variable)
    stat_rows = [_stat_row(series) for series in series_list]
    grubbs_blocks = [
        {
            "time": series["time"],
            "indicativo": series["grubbs_indicativo"],
            "aplicavel": series["applicable"]["grubbs"],
            "motivo": series["reasons"]["grubbs"],
            "steps": series["grubbs_steps"],
        }
        for series in series_list
    ]

    n_inter = sum(row["intersecao"] for row in agreements)
    n_ot_listed = sum(1 for game in games if game["prorrogacao"])
    shapiro_rejects = sum(1 for series in series_list if series["normality_rejected"])
    if not series_list:
        shapiro_label = "—"
    elif shapiro_rejects == 0:
        shapiro_label = "Não rejeita"
    elif shapiro_rejects == len(series_list):
        shapiro_label = "Rejeita"
    else:
        shapiro_label = f"Rejeita em {shapiro_rejects} de {len(series_list)}"

    return {
        "variable": variable,
        "var_meta": meta,
        "method": method,
        "method_label": "Comparar todos os métodos" if method == "todos" else DETECTOR_LABEL[method],
        "method_choices": METHOD_CHOICES,
        "parameters": _parameters(),
        "theory": _theory_blocks(method),
        "count_rows": _count_rows(catalog, variable),
        "stat_rows": stat_rows,
        "grubbs_blocks": grubbs_blocks,
        "flagged_games": games,
        "agreements": agreements,
        "ot_selected": ot_selected,
        "ot_matrix": ot_matrix,
        "variable_order": [
            {"key": key, "label": value["short_label"]} for key, value in STUDY_VARIABLES.items()
        ],
        "kpis": {
            "n_sinalizados": len(games),
            "n_intersecao": n_inter,
            "n_ot_sinalizados": n_ot_listed,
            "n_series": len(series_list),
            "shapiro_label": shapiro_label,
            "shapiro_rejects": shapiro_rejects,
        },
        "texts": _build_texts(
            variable, method, series_list, agreements, ot_selected, ot_matrix, catalog
        ),
        "chart_scatter_json": _chart_scatter(series_list, meta, method),
        "chart_box_json": _chart_box(series_list, meta),
        "chart_bars_json": _chart_bars(catalog, variable, meta),
        "chart_kdist_json": _chart_kdist(series_list, meta),
    }
