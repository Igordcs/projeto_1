"""
Módulo analítico da Parte 3: Estatística descritiva univariada, boxplot anotado
e gráficos de barras comparativos (com padronização Z-score pela referência da liga).

Exporta as funções reutilizáveis para a Parte 4:
- `quartis()`
- `limites_outliers()`
- `resumo_descritivo()`
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from app.data_loader import STUDY_VARIABLES, TEAMS, load_clean, load_raw

TEAM_COLORS = {
    "Boston Celtics": "#007A33",
    "New Orleans Pelicans": "#C8102E",
    "Pelicans Navy": "#0C2340",
    "Liga NBA (30 times)": "#64748B",
}

TEAM_SHORT = {
    "Boston Celtics": "Celtics",
    "New Orleans Pelicans": "Pelicans",
}

MEASURE_CHOICES = [
    {"id": "media", "short": "Média", "label": "Média Amostral (x̄)"},
    {"id": "mediana", "short": "Mediana", "label": "Mediana (Q2)"},
    {"id": "ambas", "short": "Média e Mediana", "label": "Comparar Média e Mediana"},
]


class _StatsDict(dict):
    """Dicionário que também permite acesso por atributo (ex.: obj.q1 ou obj['q1'])."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


_LEAGUE_REF_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def _br(value: Optional[float], digits: int = 2) -> str:
    """Formata número float no padrão brasileiro (vírgula decimal)."""
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _as_num(value: float) -> Union[int, float]:
    """Converte para int se o float representar um inteiro exato."""
    val = float(value)
    if np.isfinite(val) and val.is_integer():
        return int(val)
    return round(val, 4)


def _extract_series(
    dados: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    coluna: Optional[str] = None,
) -> pd.Series:
    """Extrai uma `pd.Series` numérica limpa a partir de diferentes formatos de entrada."""
    if isinstance(dados, pd.DataFrame):
        if coluna is None:
            num_cols = [
                c for c in dados.columns if pd.api.types.is_numeric_dtype(dados[c])
            ]
            if not num_cols:
                raise ValueError("DataFrame não possui colunas numéricas para cálculo.")
            coluna = num_cols[0]
        return pd.Series(dados[coluna], dtype=float).dropna().reset_index(drop=True)
    if isinstance(dados, pd.Series):
        return dados.astype(float).dropna().reset_index(drop=True)
    return pd.Series(np.asarray(dados, dtype=float)).dropna().reset_index(drop=True)


def quartis(
    dados: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]],
    coluna: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calcula Q1 (25%), Mediana/Q2 (50%), Q3 (75%) e IQR = Q3 - Q1
    usando interpolação linear (convenção fixa do projeto).

    Parâmetros:
        dados: Série, array 1D ou DataFrame.
        coluna: Nome da coluna caso `dados` seja um DataFrame.

    Retorna:
        Dicionário com chaves `q1`, `q2`, `mediana`, `q3`, `iqr` (e aliases maiúsculos).
    """
    s = _extract_series(dados, coluna)
    if s.empty:
        return _StatsDict(
            q1=np.nan,
            q2=np.nan,
            mediana=np.nan,
            q3=np.nan,
            iqr=np.nan,
            Q1=np.nan,
            Q2=np.nan,
            Q3=np.nan,
            IQR=np.nan,
        )

    q1 = float(s.quantile(0.25, interpolation="linear"))
    q2 = float(s.quantile(0.50, interpolation="linear"))
    q3 = float(s.quantile(0.75, interpolation="linear"))
    iqr = float(q3 - q1)

    return _StatsDict(
        q1=q1,
        q2=q2,
        mediana=q2,
        q3=q3,
        iqr=iqr,
        Q1=q1,
        Q2=q2,
        Q3=q3,
        IQR=iqr,
    )


def limites_outliers(
    dados: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float], Dict[str, Any]],
    coluna: Optional[str] = None,
    k: float = 1.5,
) -> Dict[str, Any]:
    """
    Calcula as cercas de Tukey (Q1 - k*IQR e Q3 + k*IQR, com k=1,5 por padrão),
    os limites adjacentes dos bigodes e identifica observações outliers.

    Parâmetros:
        dados: Série, array 1D, DataFrame ou dicionário retornado por `quartis()`.
        coluna: Nome da coluna caso `dados` seja um DataFrame.
        k: Fator multiplicador do IQR (padrão = 1.5, critério de Tukey).

    Retorna:
        Dicionário com `q1`, `mediana`, `q3`, `iqr`, `limite_inferior`, `limite_superior`,
        `cerca_inferior`, `cerca_superior`, `bigode_inferior`, `bigode_superior`,
        `n_outliers`, `n_outliers_inf`, `n_outliers_sup`, `outliers_inferiores`,
        `outliers_superiores`, `outliers_valores` e `mask_outliers`.
    """
    if isinstance(dados, dict) and "q1" in dados and "q3" in dados:
        q1 = float(dados["q1"])
        q2 = float(dados.get("mediana", dados.get("q2", np.nan)))
        q3 = float(dados["q3"])
        iqr = float(dados.get("iqr", q3 - q1))
        cerca_inf = float(q1 - k * iqr)
        cerca_sup = float(q3 + k * iqr)
        return _StatsDict(
            q1=q1,
            q2=q2,
            mediana=q2,
            q3=q3,
            iqr=iqr,
            k=float(k),
            limite_inferior=cerca_inf,
            limite_superior=cerca_sup,
            cerca_inferior=cerca_inf,
            cerca_superior=cerca_sup,
            lower=cerca_inf,
            upper=cerca_sup,
            bigode_inferior=cerca_inf,
            bigode_superior=cerca_sup,
            n_outliers=0,
            n_outliers_inf=0,
            n_outliers_sup=0,
            outliers_inferiores=[],
            outliers_superiores=[],
            outliers_valores=[],
            mask_outliers=np.array([], dtype=bool),
        )

    s = _extract_series(dados, coluna)
    q = quartis(s)
    q1, q2, q3, iqr = q["q1"], q["q2"], q["q3"], q["iqr"]
    cerca_inf = float(q1 - k * iqr)
    cerca_sup = float(q3 + k * iqr)

    arr = s.to_numpy(dtype=float)
    mask_inf = arr < cerca_inf
    mask_sup = arr > cerca_sup
    mask_out = mask_inf | mask_sup
    inside = arr[~mask_out]

    bigode_inf = float(np.min(inside)) if inside.size > 0 else float(np.min(arr))
    bigode_sup = float(np.max(inside)) if inside.size > 0 else float(np.max(arr))

    out_inf = [_as_num(v) for v in arr[mask_inf]]
    out_sup = [_as_num(v) for v in arr[mask_sup]]
    out_all = [_as_num(v) for v in arr[mask_out]]

    return _StatsDict(
        q1=q1,
        q2=q2,
        mediana=q2,
        q3=q3,
        iqr=iqr,
        k=float(k),
        limite_inferior=cerca_inf,
        limite_superior=cerca_sup,
        cerca_inferior=cerca_inf,
        cerca_superior=cerca_sup,
        lower=cerca_inf,
        upper=cerca_sup,
        bigode_inferior=bigode_inf,
        bigode_superior=bigode_sup,
        n_outliers=int(mask_out.sum()),
        n_outliers_inf=int(mask_inf.sum()),
        n_outliers_sup=int(mask_sup.sum()),
        outliers_inferiores=out_inf,
        outliers_superiores=out_sup,
        outliers_valores=out_all,
        mask_inferior=mask_inf,
        mask_superior=mask_sup,
        mask_outliers=mask_out,
    )


def get_league_reference() -> Dict[str, Dict[str, float]]:
    """
    Calcula e armazena em cache as medidas de referência da liga inteira
    (30 times, 2.238 jogos em `stats_times_regular_final.csv`) para as 7 variáveis.
    """
    global _LEAGUE_REF_CACHE
    if _LEAGUE_REF_CACHE is not None:
        return _LEAGUE_REF_CACHE

    df_raw = load_raw()
    ref: Dict[str, Dict[str, float]] = {}
    for var_pt, meta in STUDY_VARIABLES.items():
        orig_col = meta["original"]
        s = df_raw[orig_col].astype(float).dropna()
        q = quartis(s)
        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1))
        var_val = float(s.var(ddof=1))
        ref[var_pt] = {
            "variavel": var_pt,
            "original": orig_col,
            "n": int(len(s)),
            "media": mean_val,
            "dp": std_val,
            "variancia": var_val,
            "cv": float(std_val / mean_val * 100.0) if mean_val != 0 else 0.0,
            "mediana": q["mediana"],
            "q1": q["q1"],
            "q3": q["q3"],
            "iqr": q["iqr"],
            "minimo": float(s.min()),
            "maximo": float(s.max()),
            "amplitude": float(s.max() - s.min()),
        }

    _LEAGUE_REF_CACHE = ref
    return ref


def _compute_mode_details(
    s: pd.Series, var_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calcula moda discreta exata (tratando séries unimodais, bimodais ou multimodais)
    e, para `pontos` (PTS — quase contínua), também calcula a classe modal por faixas
    de 5 pontos e o pico da densidade por núcleo gaussiano (KDE).
    """
    vc = s.value_counts()
    if vc.empty:
        return {
            "modas": [],
            "moda_principal": np.nan,
            "moda_freq": 0,
            "moda_pct": 0.0,
            "moda_tipo": "Sem moda",
            "moda_texto": "—",
            "moda_curta": "—",
            "is_multimodal": False,
            "classe_modal_pts": None,
            "classe_modal_freq": None,
            "classe_modal_pct": None,
            "kde_pico_pts": None,
            "criterio_explicacao": "",
        }

    max_freq = int(vc.max())
    raw_modes = sorted(vc[vc == max_freq].index.tolist())
    modes = [_as_num(m) for m in raw_modes]
    n = len(s)
    pct = float(max_freq / n * 100.0) if n else 0.0

    if max_freq == 1:
        moda_tipo = "Amodal (todos os valores únicos)"
    elif len(modes) == 1:
        moda_tipo = "Unimodal"
    elif len(modes) == 2:
        moda_tipo = "Bimodal"
    elif len(modes) == 3:
        moda_tipo = "Trimodal"
    else:
        moda_tipo = f"Multimodal ({len(modes)} modas)"

    modes_str_list = [
        str(m) if isinstance(m, int) else _br(float(m), 2) for m in modes
    ]
    if len(modes_str_list) == 1:
        modes_joined = modes_str_list[0]
    else:
        modes_joined = ", ".join(modes_str_list[:-1]) + " e " + modes_str_list[-1]

    moda_texto = f"{modes_joined} ({moda_tipo}; {max_freq} jogos = {_br(pct, 1)}%)"
    moda_curta = f"{modes_joined} ({max_freq}x)"

    # Para PTS (quase contínua), adicionar classe modal de 5 pontos e pico KDE
    classe_modal_pts = None
    classe_modal_freq = None
    classe_modal_pct = None
    kde_pico_pts = None
    criterio_explicacao = (
        f"Moda empírica discreta obtida pela frequência máxima absoluta (f = {max_freq} em n = {n} jogos; "
        f"classificação: {moda_tipo.lower()})."
    )

    if var_name == "pontos" and n >= 5:
        bins = pd.interval_range(start=80, end=155, freq=5, closed="left")
        cut = pd.cut(s, bins=bins)
        vc_bin = cut.value_counts()
        if not vc_bin.empty:
            top_interval = vc_bin.index[0]
            classe_modal_freq = int(vc_bin.iloc[0])
            classe_modal_pct = float(classe_modal_freq / n * 100.0)
            classe_modal_pts = f"[{int(top_interval.left)} ; {int(top_interval.right)}) pts"
        try:
            kde = stats.gaussian_kde(s.to_numpy(dtype=float))
            grid = np.linspace(float(s.min()), float(s.max()), 500)
            kde_pico_pts = float(grid[int(np.argmax(kde(grid)))])
        except Exception:
            kde_pico_pts = float(s.median())

        criterio_explicacao = (
            f"Em Pontos (PTS), a variável é discreta de suporte amplo ({int(s.nunique())} valores distintos em "
            f"n = {n} jogos, amplitude de {int(s.max() - s.min())} pts), comportando-se como quase contínua. "
            f"Reporta-se a moda discreta exata ({modes_joined} pts, f = {max_freq} jogos) e, complementarmente, "
            f"a classe modal agrupada em faixas de 5 pontos ({classe_modal_pts}, com {classe_modal_freq} jogos = "
            f"{_br(classe_modal_pct, 1)}%) e a moda de densidade contínua KDE ({_br(kde_pico_pts, 2)} pts)."
        )

    return {
        "modas": modes,
        "moda_principal": float(np.mean(raw_modes)),
        "moda_freq": max_freq,
        "moda_pct": pct,
        "moda_tipo": moda_tipo,
        "moda_texto": moda_texto,
        "moda_curta": moda_curta,
        "modes_joined": modes_joined,
        "is_multimodal": len(modes) > 1,
        "n_unique": int(s.nunique()),
        "classe_modal_pts": classe_modal_pts,
        "classe_modal_freq": classe_modal_freq,
        "classe_modal_pct": classe_modal_pct,
        "kde_pico_pts": kde_pico_pts,
        "criterio_explicacao": criterio_explicacao,
    }


def _summarize_single(
    df_or_series: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]],
    variavel: Optional[str] = None,
    team_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Constrói o resumo descritivo completo para uma única série (1 variável × 1 recorte)."""
    has_game_cols = False
    df_sub: Optional[pd.DataFrame] = None

    if isinstance(df_or_series, pd.DataFrame):
        df_sub = df_or_series.copy()
        if variavel is None:
            num_cols = [
                c for c in df_sub.columns if pd.api.types.is_numeric_dtype(df_sub[c])
            ]
            variavel = num_cols[0]
        s = df_sub[variavel].astype(float).reset_index(drop=True)
        has_game_cols = {"data_jogo", "adversario"}.issubset(df_sub.columns)
        if team_label is None and "time" in df_sub.columns:
            unique_teams = df_sub["time"].dropna().unique().tolist()
            team_label = (
                unique_teams[0] if len(unique_teams) == 1 else "Ambos os Times"
            )
    else:
        s = _extract_series(df_or_series)
        if team_label is None:
            team_label = "Amostra"

    var_key = variavel or "serie"
    meta = STUDY_VARIABLES.get(
        var_key,
        {
            "original": var_key.upper(),
            "label": var_key,
            "short_label": var_key,
            "unit": "unid.",
            "type": "Numérica",
        },
    )

    n = int(len(s))
    mean_val = float(s.mean()) if n else np.nan
    std_val = float(s.std(ddof=1)) if n > 1 else 0.0
    var_val = float(s.var(ddof=1)) if n > 1 else 0.0
    cv_val = float(std_val / mean_val * 100.0) if (n > 1 and mean_val != 0) else 0.0
    min_val = _as_num(float(s.min())) if n else np.nan
    max_val = _as_num(float(s.max())) if n else np.nan
    amp_val = _as_num(float(s.max() - s.min())) if n else np.nan
    skew_val = float(s.skew()) if n > 2 else 0.0
    kurt_val = float(s.kurt()) if n > 3 else 0.0

    q_info = quartis(s)
    out_info = limites_outliers(s, k=1.5)
    mode_info = _compute_mode_details(s, var_name=var_key)

    # Identificar jogos outliers (data/adversário/valor) quando há colunas de partida
    outlier_games: List[Dict[str, Any]] = []
    if has_game_cols and df_sub is not None:
        df_reset = df_sub.reset_index(drop=True)
        mask_out = out_info["mask_outliers"]
        for idx in np.flatnonzero(mask_out):
            row = df_reset.iloc[int(idx)]
            val = _as_num(float(row[var_key]))
            dt_val = row["data_jogo"]
            dt_str = (
                dt_val.strftime("%d/%m/%Y")
                if hasattr(dt_val, "strftime")
                else pd.to_datetime(dt_val).strftime("%d/%m/%Y")
            )
            side = "superior" if float(val) > out_info["cerca_superior"] else "inferior"
            side_arrow = "↑" if side == "superior" else "↓"
            mins = int(row["minutos"]) if "minutos" in row else 240
            is_ot = bool(row["prorrogacao"]) if "prorrogacao" in row else (mins != 240)
            t_name = str(row["time"]) if "time" in row else (team_label or "")
            t_short = TEAM_SHORT.get(t_name, t_name)
            adv = str(row["adversario"])
            outlier_games.append(
                {
                    "indice": int(idx),
                    "data": dt_str,
                    "time": t_name,
                    "time_short": t_short,
                    "adversario": adv,
                    "confronto": str(row.get("confronto", f"vs. {adv}")),
                    "valor": val,
                    "tipo": side,
                    "seta": side_arrow,
                    "minutos": mins,
                    "prorrogacao": is_ot,
                    "resumo": f"{dt_str} vs {adv} ({val} {meta['unit']}{', OT' if is_ot else ''})",
                    "resumo_curto": f"{dt_str} vs {adv} ({val}{side_arrow})",
                }
            )
    else:
        for v in out_info["outliers_valores"]:
            side = "superior" if float(v) > out_info["cerca_superior"] else "inferior"
            side_arrow = "↑" if side == "superior" else "↓"
            outlier_games.append(
                {
                    "indice": -1,
                    "data": "—",
                    "time": team_label or "",
                    "time_short": TEAM_SHORT.get(team_label or "", team_label or ""),
                    "adversario": "—",
                    "confronto": "—",
                    "valor": v,
                    "tipo": side,
                    "seta": side_arrow,
                    "minutos": 240,
                    "prorrogacao": False,
                    "resumo": f"{v} ({side})",
                    "resumo_curto": f"{v}{side_arrow}",
                }
            )

    outliers_texto = (
        "; ".join(g["resumo_curto"] for g in outlier_games)
        if outlier_games
        else "Nenhum (0)"
    )

    # Padronização Z-score usando a referência da liga toda (30 times, 2238 linhas)
    league_ref = get_league_reference().get(var_key)
    if league_ref and league_ref["dp"] > 0:
        liga_media = league_ref["media"]
        liga_dp = league_ref["dp"]
        liga_mediana = league_ref["mediana"]
        z_media_liga = float((mean_val - liga_media) / liga_dp)
        z_mediana_liga = float((q_info["mediana"] - liga_media) / liga_dp)
    else:
        liga_media = mean_val
        liga_dp = std_val if std_val > 0 else 1.0
        liga_mediana = q_info["mediana"]
        z_media_liga = 0.0
        z_mediana_liga = 0.0

    t_short = TEAM_SHORT.get(team_label or "", team_label or "Ambos")

    return _StatsDict(
        variavel=var_key,
        original=meta["original"],
        label=meta["label"],
        short_label=meta["short_label"],
        unidade=meta["unit"],
        unit=meta["unit"],
        time=team_label,
        time_short=t_short,
        n=n,
        media=mean_val,
        mediana=q_info["mediana"],
        q1=q_info["q1"],
        q2=q_info["q2"],
        q3=q_info["q3"],
        iqr=q_info["iqr"],
        modas=mode_info["modas"],
        moda_principal=mode_info["moda_principal"],
        moda_freq=mode_info["moda_freq"],
        moda_pct=mode_info["moda_pct"],
        moda_tipo=mode_info["moda_tipo"],
        moda_texto=mode_info["moda_texto"],
        moda_curta=mode_info["moda_curta"],
        modes_joined=mode_info["modes_joined"],
        is_multimodal=mode_info["is_multimodal"],
        n_unique=mode_info["n_unique"],
        classe_modal_pts=mode_info["classe_modal_pts"],
        classe_modal_freq=mode_info["classe_modal_freq"],
        classe_modal_pct=mode_info["classe_modal_pct"],
        kde_pico_pts=mode_info["kde_pico_pts"],
        moda_criterio=mode_info["criterio_explicacao"],
        minimo=min_val,
        min=min_val,
        maximo=max_val,
        max=max_val,
        amplitude=amp_val,
        desvio_padrao=std_val,
        dp=std_val,
        std=std_val,
        variancia=var_val,
        var=var_val,
        cv=cv_val,
        cv_pct=cv_val,
        assimetria=skew_val,
        skewness=skew_val,
        curtose=kurt_val,
        kurtosis=kurt_val,
        limite_inferior=out_info["limite_inferior"],
        limite_superior=out_info["limite_superior"],
        cerca_inferior=out_info["cerca_inferior"],
        cerca_superior=out_info["cerca_superior"],
        bigode_inferior=out_info["bigode_inferior"],
        bigode_superior=out_info["bigode_superior"],
        n_outliers=out_info["n_outliers"],
        n_outliers_inf=out_info["n_outliers_inf"],
        n_outliers_sup=out_info["n_outliers_sup"],
        outliers=outlier_games,
        outliers_valores=out_info["outliers_valores"],
        outliers_texto=outliers_texto,
        liga_media=liga_media,
        liga_dp=liga_dp,
        liga_mediana=liga_mediana,
        z_media_liga=z_media_liga,
        z_mediana_liga=z_mediana_liga,
        valores=s.to_numpy(dtype=float).tolist(),
    )


def resumo_descritivo(
    dados: Optional[
        Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float], str]
    ] = None,
    variavel: Optional[str] = None,
    team: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Função reutilizável de estatística descritiva univariada (exportada para a Parte 4).

    Modos de chamada suportados:
    1. `resumo_descritivo(serie_1d)` -> retorna `dict` com o resumo da série.
    2. `resumo_descritivo(df, variavel="pontos", team="Boston Celtics")` -> retorna `dict` da série.
    3. `resumo_descritivo(variavel="pontos", team="Boston Celtics")` -> carrega `load_clean` e retorna `dict`.
    4. `resumo_descritivo(team=None, variavel=None)` -> retorna `list[dict]` com os resumos das 7 variáveis
       para cada equipe do recorte (`Boston Celtics` e `New Orleans Pelicans`).
    """
    # Caso `dados` seja string, pode ser o nome da variável ou o nome do time
    if isinstance(dados, str):
        token = dados.strip()
        if token in STUDY_VARIABLES:
            variavel = token
            dados = None
        else:
            team = token
            dados = None

    # Caso `dados` seja uma Série/array 1D direto
    if dados is not None and not isinstance(dados, pd.DataFrame):
        return _summarize_single(dados, variavel=variavel, team_label=team)

    # Caso `dados` seja um DataFrame fornecido pelo chamador
    if isinstance(dados, pd.DataFrame):
        df = dados.copy()
        if (
            team
            and team.lower() not in {"todos", "all", "ambos"}
            and "time" in df.columns
        ):
            df = df[df["time"] == team].copy()
        if variavel is not None:
            if "time" in df.columns and df["time"].nunique() > 1 and team is None:
                return [
                    _summarize_single(
                        df[df["time"] == t], variavel=variavel, team_label=t
                    )
                    for t in TEAMS
                    if (df["time"] == t).any()
                ]
            return _summarize_single(df, variavel=variavel, team_label=team)

        # Sem variável específica: retorna lista para as 7 variáveis por time presente no DataFrame
        results: List[Dict[str, Any]] = []
        teams_present = (
            [t for t in TEAMS if (df["time"] == t).any()]
            if "time" in df.columns
            else [team or "Amostra"]
        )
        for var_key in STUDY_VARIABLES:
            if var_key not in df.columns:
                continue
            for t_name in teams_present:
                sub = df[df["time"] == t_name] if "time" in df.columns else df
                results.append(
                    _summarize_single(sub, variavel=var_key, team_label=t_name)
                )
        return results

    # Caso `dados is None`: carregar base limpa via `load_clean`
    df_all = load_clean()
    selected_teams = (
        list(TEAMS)
        if (team is None or str(team).lower() in {"todos", "all", "ambos"})
        else [team]
    )

    if variavel is not None and len(selected_teams) == 1:
        t_name = selected_teams[0]
        sub = df_all[df_all["time"] == t_name].sort_values(["data_jogo", "id_jogo"])
        return _summarize_single(sub, variavel=variavel, team_label=t_name)

    vars_to_run = (
        [variavel] if (variavel in STUDY_VARIABLES) else list(STUDY_VARIABLES.keys())
    )
    summaries: List[Dict[str, Any]] = []
    for var_key in vars_to_run:
        for t_name in selected_teams:
            sub = df_all[df_all["time"] == t_name].sort_values(["data_jogo", "id_jogo"])
            summaries.append(
                _summarize_single(sub, variavel=var_key, team_label=t_name)
            )
    return summaries


def _plotly_to_json(fig: go.Figure, top: int = 75) -> str:
    """Serializa figura Plotly com identidade visual padronizada do sistema."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1E293B"),
        title_font=dict(size=15, color="#0F172A"),
        margin=dict(l=60, r=35, t=top, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    return json.dumps(fig.to_plotly_json(), default=str)


def _stagger_y(
    entries: List[Dict[str, Any]], min_gap: float
) -> List[Dict[str, Any]]:
    """
    Ajusta coordenadas verticais `y_text` de anotações próximas para evitar colisão visual
    de rótulos no boxplot, mantendo a seta apontada para a coordenada exata `y_target`.
    """
    if not entries:
        return []
    ordered = sorted(entries, key=lambda item: float(item["y"]))
    for i, item in enumerate(ordered):
        if i == 0:
            item["y_text"] = float(item["y"])
        else:
            prev_y = ordered[i - 1]["y_text"]
            item["y_text"] = max(float(item["y"]), prev_y + min_gap)

    # Passagem reversa leve para centralizar o bloco caso tenha subido demais
    shift = (ordered[-1]["y_text"] - float(ordered[-1]["y"])) * 0.35
    if shift > 0 and len(ordered) > 2:
        for item in ordered:
            item["y_text"] -= shift
        for i in range(1, len(ordered)):
            if ordered[i]["y_text"] - ordered[i - 1]["y_text"] < min_gap:
                ordered[i]["y_text"] = ordered[i - 1]["y_text"] + min_gap
    return ordered


def _build_annotated_boxplot(
    summaries_var: Sequence[Dict[str, Any]],
    var_meta: Dict[str, str],
) -> str:
    """
    Constrói o Boxplot (Plotly) por time para a variável selecionada, anotando
    explicitamente:
    - Outliers superiores e inferiores (com data e adversário)
    - Limites adjacentes dos bigodes e Cercas de Tukey (Q1 - 1,5*IQR e Q3 + 1,5*IQR)
    - Q3, Média, Mediana (Q2), Moda(s) e Q1
    Com todos os elementos identificados na legenda interativa.
    """
    n_cols = len(summaries_var)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.10 if n_cols > 1 else 0.05,
        subplot_titles=[
            f"{s['time']} (n = {s['n']} jogos)" for s in summaries_var
        ],
    )

    # Calcular escala vertical global para espaçamento anti-colisão das anotações
    all_y: List[float] = []
    for s in summaries_var:
        all_y.extend(s["valores"])
        all_y.extend([s["cerca_inferior"], s["cerca_superior"]])
    y_min_global = min(all_y) if all_y else 0.0
    y_max_global = max(all_y) if all_y else 100.0
    y_span = max(y_max_global - y_min_global, 5.0)
    min_gap = y_span * 0.058

    legend_shown = {
        "mean": False,
        "median": False,
        "mode": False,
        "quartiles": False,
        "whiskers": False,
        "fences": False,
        "out_sup": False,
        "out_inf": False,
    }

    for col_idx, s in enumerate(summaries_var, start=1):
        team_name = s["time"]
        box_color = (
            TEAM_COLORS["Boston Celtics"]
            if team_name == "Boston Celtics"
            else TEAM_COLORS["Pelicans Navy"]
        )
        fill_color = (
            "rgba(0, 122, 51, 0.18)"
            if team_name == "Boston Celtics"
            else "rgba(12, 35, 64, 0.18)"
        )

        # Filtrar valores internos às cercas para o traço principal e sobrepor outliers separados
        vals = np.asarray(s["valores"], dtype=float)
        cerca_inf = float(s["cerca_inferior"])
        cerca_sup = float(s["cerca_superior"])

        # 1. Traço Boxplot oficial (quartis lineares idênticos à tabela)
        fig.add_trace(
            go.Box(
                x=[0.0] * len(vals),
                y=vals.tolist(),
                name=f"Caixa e Bigodes — {s['time_short']}",
                marker_color=box_color,
                line=dict(color=box_color, width=2.2),
                fillcolor=fill_color,
                boxmean=False,
                boxpoints=False,
                quartilemethod="linear",
                width=0.34,
                hoverinfo="y+name",
            ),
            row=1,
            col=col_idx,
        )

        # 2. Cercas de Tukey (Q1 - 1,5*IQR e Q3 + 1,5*IQR) como linhas tracejadas
        for fence_val, fence_label in [
            (cerca_sup, "Cerca Superior (Q3 + 1,5×IQR)"),
            (cerca_inf, "Cerca Inferior (Q1 − 1,5×IQR)"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[-0.24, 0.24],
                    y=[fence_val, fence_val],
                    mode="lines",
                    name="Cercas de Tukey (Q1 − 1,5×IQR ; Q3 + 1,5×IQR)",
                    legendgroup="fences",
                    showlegend=not legend_shown["fences"],
                    line=dict(color="#DC2626", width=1.8, dash="dot"),
                    hovertemplate=f"<b>{s['time_short']} — {fence_label}</b>: %{{y:.2f}} {var_meta['unit']}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )
            legend_shown["fences"] = True

        # 3. Limites Adjacentes dos Bigodes (Mín/Máx dentro das cercas)
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[s["bigode_inferior"], s["bigode_superior"]],
                mode="markers",
                name="Limites dos Bigodes (Mín/Máx não-outlier)",
                legendgroup="whiskers",
                showlegend=not legend_shown["whiskers"],
                marker=dict(
                    symbol="line-ew",
                    size=14,
                    color="#334155",
                    line=dict(width=2.5, color="#334155"),
                ),
                hovertemplate=(
                    f"<b>{s['time_short']} — Limite do Bigode</b>: %{{y:.2f}} {var_meta['unit']}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        legend_shown["whiskers"] = True

        # 4. Quartis Q1 e Q3
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[s["q1"], s["q3"]],
                mode="markers",
                name="Quartis Q1 (25%) e Q3 (75%)",
                legendgroup="quartiles",
                showlegend=not legend_shown["quartiles"],
                marker=dict(
                    symbol="circle",
                    size=9,
                    color=box_color,
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                hovertemplate=(
                    f"<b>{s['time_short']} — Quartil</b>: %{{y:.2f}} {var_meta['unit']}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        legend_shown["quartiles"] = True

        # 5. Mediana (Q2)
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[s["mediana"]],
                mode="markers",
                name="Mediana (Q2)",
                legendgroup="median",
                showlegend=not legend_shown["median"],
                marker=dict(
                    symbol="square",
                    size=10,
                    color="#0F172A",
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                hovertemplate=(
                    f"<b>{s['time_short']} — Mediana (Q2)</b>: {s['mediana']:.2f} {var_meta['unit']}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        legend_shown["median"] = True

        # 6. Média Amostral
        fig.add_trace(
            go.Scatter(
                x=[-0.05],
                y=[s["media"]],
                mode="markers",
                name="Média Amostral (x̄)",
                legendgroup="mean",
                showlegend=not legend_shown["mean"],
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="#D97706",
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                hovertemplate=(
                    f"<b>{s['time_short']} — Média (x̄)</b>: {s['media']:.2f} {var_meta['unit']}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        legend_shown["mean"] = True

        # 7. Moda(s) (plota todas as modas caso bimodal/multimodal)
        modas_y = [float(m) for m in s["modas"]]
        if modas_y:
            fig.add_trace(
                go.Scatter(
                    x=[0.05] * len(modas_y),
                    y=modas_y,
                    mode="markers",
                    name="Moda (Mo — Unimodal/Bimodal)",
                    legendgroup="mode",
                    showlegend=not legend_shown["mode"],
                    marker=dict(
                        symbol="star",
                        size=13,
                        color="#7C3AED",
                        line=dict(width=1, color="#FFFFFF"),
                    ),
                    hovertemplate=(
                        f"<b>{s['time_short']} — Moda ({s['moda_tipo']})</b>: %{{y}} {var_meta['unit']} "
                        f"(f = {s['moda_freq']} jogos)<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            legend_shown["mode"] = True

        # 8. Outliers Superiores e Inferiores na Legenda e no Gráfico
        outliers_sup = [g for g in s["outliers"] if g["tipo"] == "superior"]
        outliers_inf = [g for g in s["outliers"] if g["tipo"] == "inferior"]

        if outliers_sup:
            x_sup = np.linspace(-0.06, 0.06, len(outliers_sup)).tolist() if len(outliers_sup) > 1 else [0.0]
            fig.add_trace(
                go.Scatter(
                    x=x_sup,
                    y=[float(g["valor"]) for g in outliers_sup],
                    mode="markers",
                    name="Outlier Superior (> Q3 + 1,5×IQR)",
                    legendgroup="out_sup",
                    showlegend=not legend_shown["out_sup"],
                    marker=dict(
                        symbol="triangle-up",
                        size=13,
                        color="#DC2626",
                        line=dict(width=1.5, color="#7F1D1D"),
                    ),
                    customdata=[[g["data"], g["adversario"], g["confronto"], g["minutos"]] for g in outliers_sup],
                    hovertemplate=(
                        f"<b>Outlier Superior — {s['time_short']}</b><br>"
                        "Data: %{customdata[0]} (%{customdata[2]})<br>"
                        f"{var_meta['short_label']}: <b>%{{y}} {var_meta['unit']}</b><br>"
                        "Minutos: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            legend_shown["out_sup"] = True

        if outliers_inf:
            x_inf = np.linspace(-0.06, 0.06, len(outliers_inf)).tolist() if len(outliers_inf) > 1 else [0.0]
            fig.add_trace(
                go.Scatter(
                    x=x_inf,
                    y=[float(g["valor"]) for g in outliers_inf],
                    mode="markers",
                    name="Outlier Inferior (< Q1 − 1,5×IQR)",
                    legendgroup="out_inf",
                    showlegend=not legend_shown["out_inf"],
                    marker=dict(
                        symbol="triangle-down",
                        size=13,
                        color="#2563EB",
                        line=dict(width=1.5, color="#1E3A8A"),
                    ),
                    customdata=[[g["data"], g["adversario"], g["confronto"], g["minutos"]] for g in outliers_inf],
                    hovertemplate=(
                        f"<b>Outlier Inferior — {s['time_short']}</b><br>"
                        "Data: %{customdata[0]} (%{customdata[2]})<br>"
                        f"{var_meta['short_label']}: <b>%{{y}} {var_meta['unit']}</b><br>"
                        "Minutos: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            legend_shown["out_inf"] = True

        # --- Anotações à direita da caixa: Cerca Sup, Bigode Sup, Q3, Mediana, Q1, Bigode Inf, Cerca Inf ---
        right_notes = [
            {
                "y": cerca_sup,
                "x_target": 0.22,
                "text": f"<b>Cerca Sup.</b>: {_br(cerca_sup)}",
                "color": "#B91C1C",
                "bg": "#FEF2F2",
            },
            {
                "y": float(s["bigode_superior"]),
                "x_target": 0.06,
                "text": f"<b>Lim. Sup. (Bigode)</b>: {_br(s['bigode_superior'], 0)}",
                "color": "#334155",
                "bg": "#F8FAFC",
            },
            {
                "y": float(s["q3"]),
                "x_target": 0.17,
                "text": f"<b>Q3 (75%)</b>: {_br(s['q3'])}",
                "color": box_color,
                "bg": "#F0FDF4" if team_name == "Boston Celtics" else "#EFF6FF",
            },
            {
                "y": float(s["mediana"]),
                "x_target": 0.17,
                "text": f"<b>Mediana (Q2)</b>: {_br(s['mediana'])}",
                "color": "#0F172A",
                "bg": "#F1F5F9",
            },
            {
                "y": float(s["q1"]),
                "x_target": 0.17,
                "text": f"<b>Q1 (25%)</b>: {_br(s['q1'])}",
                "color": box_color,
                "bg": "#F0FDF4" if team_name == "Boston Celtics" else "#EFF6FF",
            },
            {
                "y": float(s["bigode_inferior"]),
                "x_target": 0.06,
                "text": f"<b>Lim. Inf. (Bigode)</b>: {_br(s['bigode_inferior'], 0)}",
                "color": "#334155",
                "bg": "#F8FAFC",
            },
            {
                "y": cerca_inf,
                "x_target": 0.22,
                "text": f"<b>Cerca Inf.</b>: {_br(cerca_inf)}",
                "color": "#B91C1C",
                "bg": "#FEF2F2",
            },
        ]
        right_staggered = _stagger_y(right_notes, min_gap=min_gap)
        xref = "x" if col_idx == 1 else f"x{col_idx}"
        yref = "y" if col_idx == 1 else f"y{col_idx}"

        for item in right_staggered:
            fig.add_annotation(
                x=item["x_target"],
                y=item["y"],
                ax=0.36,
                ay=item["y_text"],
                xref=xref,
                yref=yref,
                axref=xref,
                ayref=yref,
                text=item["text"],
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1.2,
                arrowcolor=item["color"],
                xanchor="left",
                align="left",
                font=dict(size=10.5, color=item["color"]),
                bgcolor=item["bg"],
                bordercolor=item["color"],
                borderwidth=1,
                borderpad=2,
            )

        # --- Anotações à esquerda da caixa: Média, Moda(s) e Outliers ---
        left_notes = [
            {
                "y": float(s["media"]),
                "x_target": -0.05,
                "text": f"<b>Média (x̄)</b>: {_br(s['media'])}",
                "color": "#B45309",
                "bg": "#FFFBEB",
            },
            {
                "y": float(s["moda_principal"]),
                "x_target": 0.05,
                "text": f"<b>Moda</b>: {s['modes_joined']} ({s['moda_freq']}x)",
                "color": "#6D28D9",
                "bg": "#F5F3FF",
            },
        ]

        if outliers_sup:
            max_out_y = max(float(g["valor"]) for g in outliers_sup)
            out_sup_txt = "<br>".join(
                f"• {g['valor']} ({g['data'][:5]} vs {g['adversario']})"
                for g in outliers_sup
            )
            left_notes.append(
                {
                    "y": max_out_y,
                    "x_target": 0.0,
                    "text": f"<b>Outlier Sup. ({len(outliers_sup)})</b>:<br>{out_sup_txt}",
                    "color": "#DC2626",
                    "bg": "#FEF2F2",
                }
            )

        if outliers_inf:
            min_out_y = min(float(g["valor"]) for g in outliers_inf)
            out_inf_txt = "<br>".join(
                f"• {g['valor']} ({g['data'][:5]} vs {g['adversario']})"
                for g in outliers_inf
            )
            left_notes.append(
                {
                    "y": min_out_y,
                    "x_target": 0.0,
                    "text": f"<b>Outlier Inf. ({len(outliers_inf)})</b>:<br>{out_inf_txt}",
                    "color": "#1D4ED8",
                    "bg": "#EFF6FF",
                }
            )

        left_staggered = _stagger_y(left_notes, min_gap=min_gap * 1.15)
        for item in left_staggered:
            fig.add_annotation(
                x=item["x_target"],
                y=item["y"],
                ax=-0.34,
                ay=item["y_text"],
                xref=xref,
                yref=yref,
                axref=xref,
                ayref=yref,
                text=item["text"],
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1.2,
                arrowcolor=item["color"],
                xanchor="right",
                align="right",
                font=dict(size=10.5, color=item["color"]),
                bgcolor=item["bg"],
                bordercolor=item["color"],
                borderwidth=1,
                borderpad=2,
            )

        fig.update_xaxes(
            range=[-1.05, 1.05],
            showticklabels=False,
            zeroline=False,
            showgrid=False,
            title_text=f"{s['time']} (IQR = {_br(s['iqr'])} {var_meta['unit']})",
            row=1,
            col=col_idx,
        )

    fig.update_yaxes(
        title_text=f"{var_meta['label']} ({var_meta['unit']})",
        range=[y_min_global - y_span * 0.14, y_max_global + y_span * 0.14],
        row=1,
        col=1,
    )
    fig.update_layout(
        title=(
            f"Boxplot Anotado de {var_meta['label']} — Quartis (Q1, Mediana, Q3), Média, Moda, "
            "Limites e Outliers (1,5×IQR)"
        ),
        height=560,
        showlegend=True,
    )
    return _plotly_to_json(fig, top=95)


def _build_zscore_bars_chart(
    all_summaries: Sequence[Dict[str, Any]],
    selected_var: str,
    medida: str,
) -> str:
    """
    Constrói o Gráfico de Barras Comparativo Padronizado via Z-score da Liga Toda
    (30 times, 2.238 linhas) para comparar as 7 variáveis de escalas diferentes
    no mesmo gráfico entre Boston Celtics e New Orleans Pelicans.
    """
    fig = go.Figure()

    var_keys = list(STUDY_VARIABLES.keys())
    x_labels = [
        f"<b>{STUDY_VARIABLES[k]['short_label']}</b>"
        if k == selected_var
        else STUDY_VARIABLES[k]["short_label"]
        for k in var_keys
    ]

    by_team_var = {(s["time"], s["variavel"]): s for s in all_summaries}

    traces_spec = []
    if medida in {"media", "ambas"}:
        traces_spec.append(("Boston Celtics", "media", "Celtics — Z(Média)", "#007A33"))
        traces_spec.append(
            ("New Orleans Pelicans", "media", "Pelicans — Z(Média)", "#C8102E")
        )
    if medida in {"mediana", "ambas"}:
        c_bos = "#10B981" if medida == "ambas" else "#007A33"
        c_nop = "#0C2340" if medida == "ambas" else "#C8102E"
        traces_spec.append(("Boston Celtics", "mediana", "Celtics — Z(Mediana)", c_bos))
        traces_spec.append(
            ("New Orleans Pelicans", "mediana", "Pelicans — Z(Mediana)", c_nop)
        )

    for t_name, m_kind, trace_name, color in traces_spec:
        z_vals = []
        text_labels = []
        custom_rows = []
        for k in var_keys:
            s = by_team_var[(t_name, k)]
            z_val = s["z_media_liga"] if m_kind == "media" else s["z_mediana_liga"]
            raw_val = s["media"] if m_kind == "media" else s["mediana"]
            z_vals.append(z_val)
            text_labels.append(f"{z_val:+.2f}z".replace(".", ","))
            custom_rows.append(
                [
                    s["label"],
                    _br(raw_val, 2),
                    s["unit"],
                    _br(s["liga_media"], 2),
                    _br(s["liga_dp"], 2),
                    _br(z_val, 3),
                ]
            )

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=z_vals,
                name=trace_name,
                marker_color=color,
                text=text_labels,
                textposition="outside",
                customdata=custom_rows,
                hovertemplate=(
                    f"<b>{trace_name}</b><br>"
                    "Variável: %{customdata[0]}<br>"
                    f"Valor bruto ({'Média' if m_kind == 'media' else 'Mediana'}): <b>%{{customdata[1]}} %{{customdata[2]}}</b><br>"
                    "Referência Liga (30 times, n=2238): μ = %{customdata[3]} | σ = %{customdata[4]}<br>"
                    "Escore padronizado: <b>z = %{customdata[5]} σ</b><extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#0F172A",
        line_width=1.8,
        annotation_text="Média de Referência da Liga NBA (30 times, N = 2.238 jogos): z = 0,00",
        annotation_position="top left",
    )

    medida_titulo = {
        "media": "Média Amostral Padronizada: z = (x̄_time − μ_liga) / σ_liga",
        "mediana": "Mediana Padronizada: z = (Mediana_time − μ_liga) / σ_liga",
        "ambas": "Média e Mediana Padronizadas pela Liga (30 times, N = 2.238)",
    }[medida]

    fig.update_layout(
        title=(
            f"Comparação Conjunta das 7 Variáveis na Mesma Escala (Z-score da Liga) — {medida_titulo}"
        ),
        xaxis_title="Variáveis do Estudo (destaque em negrito na variável selecionada)",
        yaxis_title="Escore Padronizado Z (desvios-padrão da liga NBA, adimensional)",
        barmode="group",
        yaxis=dict(range=[-0.85, 0.85]),
        height=470,
    )
    return _plotly_to_json(fig, top=85)


def _build_raw_bars_chart(
    all_summaries: Sequence[Dict[str, Any]],
    selected_var: str,
    medida: str,
) -> str:
    """
    Constrói o Gráfico de Barras Comparativo em Unidades Originais:
    - Painel 1: Comparação direta Celtics × Pelicans × Liga NBA na variável selecionada
      (mostrando Média ± DP e Mediana com Q1–Q3).
    - Painel 2: As 7 variáveis na medida selecionada em unidades brutas, evidenciando a
      diferença de ordem de grandeza (PTS ~ 115 vs BLK ~ 5) que motiva o uso do Z-score.
    """
    var_meta = STUDY_VARIABLES[selected_var]
    by_team_var = {(s["time"], s["variavel"]): s for s in all_summaries}
    s_bos = by_team_var[("Boston Celtics", selected_var)]
    s_nop = by_team_var[("New Orleans Pelicans", selected_var)]
    league_ref = get_league_reference()[selected_var]

    med_label = (
        "Média"
        if medida == "media"
        else ("Mediana" if medida == "mediana" else "Média")
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Foco em {var_meta['short_label']} ({var_meta['unit']}): Média vs. Mediana",
            f"As 7 Variáveis em Unidades Brutas ({med_label}) — Escalas Distintas",
        ),
        horizontal_spacing=0.14,
    )

    # Painel 1: Celtics vs Pelicans vs Liga na variável selecionada (Média e Mediana)
    groups_x = ["Boston Celtics (n=74)", "New Orleans Pelicans (n=76)", "Liga NBA (30 times, N=2238)"]
    means_y = [s_bos["media"], s_nop["media"], league_ref["media"]]
    stds_y = [s_bos["dp"], s_nop["dp"], league_ref["dp"]]
    medians_y = [s_bos["mediana"], s_nop["mediana"], league_ref["mediana"]]

    fig.add_trace(
        go.Bar(
            x=groups_x,
            y=means_y,
            name=f"Média ({var_meta['unit']})",
            marker_color="#007A33",
            text=[_br(v, 2) for v in means_y],
            textposition="auto",
            error_y=dict(type="data", array=stds_y, visible=True, color="#1E293B"),
            hovertemplate=(
                "<b>%{x}</b><br>Média: <b>%{y:.2f} "
                + var_meta["unit"]
                + "</b> (±1 DP = %{error_y.array:.2f})<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=groups_x,
            y=medians_y,
            name=f"Mediana ({var_meta['unit']})",
            marker_color="#0C2340",
            text=[_br(v, 2) for v in medians_y],
            textposition="auto",
            hovertemplate=(
                "<b>%{x}</b><br>Mediana (Q2): <b>%{y:.2f} "
                + var_meta["unit"]
                + "</b><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Painel 2: Todas as 7 variáveis em unidades brutas
    var_keys = list(STUDY_VARIABLES.keys())
    x_vars = [STUDY_VARIABLES[k]["short_label"] for k in var_keys]
    key_field = "mediana" if medida == "mediana" else "media"

    bos_raw = [by_team_var[("Boston Celtics", k)][key_field] for k in var_keys]
    nop_raw = [by_team_var[("New Orleans Pelicans", k)][key_field] for k in var_keys]

    fig.add_trace(
        go.Bar(
            x=x_vars,
            y=bos_raw,
            name=f"Celtics — {med_label} Bruta",
            marker_color="#007A33",
            text=[_br(v, 1) for v in bos_raw],
            textposition="outside",
            hovertemplate="<b>Boston Celtics</b><br>%{x}: <b>%{y:.2f}</b><extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=x_vars,
            y=nop_raw,
            name=f"Pelicans — {med_label} Bruta",
            marker_color="#C8102E",
            text=[_br(v, 1) for v in nop_raw],
            textposition="outside",
            hovertemplate="<b>New Orleans Pelicans</b><br>%{x}: <b>%{y:.2f}</b><extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Recorte Comparado", row=1, col=1)
    fig.update_yaxes(title_text=f"{var_meta['short_label']} ({var_meta['unit']})", row=1, col=1)
    fig.update_xaxes(title_text="7 Variáveis do Estudo", row=1, col=2)
    fig.update_yaxes(
        title_text="Valor em Unidade Nativa (Escalas Heterogêneas)",
        range=[0, max(bos_raw + nop_raw) * 1.18],
        row=1,
        col=2,
    )
    fig.update_layout(
        title=(
            f"Comparação em Unidades Originais: Detalhe de {var_meta['label']} (±1 DP) "
            "vs. Disparidade de Escala Bruta entre as 7 Variáveis"
        ),
        barmode="group",
        height=450,
    )
    return _plotly_to_json(fig, top=85)


def _build_variable_interpretations(
    by_team_var: Dict[Tuple[str, str], Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Constrói a interpretação textual crítica detalhada para cada uma das 7 variáveis
    comparando Boston Celtics × New Orleans Pelicans e posicionando-os frente à liga.
    """
    interps: Dict[str, Dict[str, Any]] = {}

    for var_key, meta in STUDY_VARIABLES.items():
        bos = by_team_var[("Boston Celtics", var_key)]
        nop = by_team_var[("New Orleans Pelicans", var_key)]
        diff_mean = bos["media"] - nop["media"]
        diff_med = bos["mediana"] - nop["mediana"]

        if var_key == "pontos":
            contexto_tatico = (
                f"Os dois ataques apresentam médias globais muito próximas e alinhadas ao padrão da NBA "
                f"(Liga = {_br(bos['liga_media'])} pts): o Pelicans tem média ligeiramente superior "
                f"({_br(nop['media'])} vs. {_br(bos['media'])} pts; z = {_br(nop['z_media_liga'], 2)} vs. "
                f"{_br(bos['z_media_liga'], 2)}) e mediana maior ({_br(nop['mediana'])} vs. {_br(bos['mediana'])} pts). "
                f"Contudo, o Celtics exibe maior amplitude ({bos['amplitude']} pts, de {bos['minimo']} a {bos['maximo']} pts) "
                f"e maior variância ({_br(bos['variancia'])} vs. {_br(nop['variancia'])} pts²), impulsionadas por três "
                f"explosões ofensivas acima da cerca superior de Tukey ({_br(bos['cerca_superior'])} pts): 146 pts contra "
                f"WAS (04/12/2025) e LAC (03/01/2026) e 148 pts contra BKN (27/02/2026), todas em tempo normal (240 min)."
            )
            criterio_moda = (
                f"Por variar de 84 a 148 pontos com {bos['n_unique']} valores distintos no Celtics e {nop['n_unique']} "
                f"no Pelicans, Pontos (PTS) comporta-se como variável quase contínua. No Celtics, a moda discreta bruta é "
                f"bimodal em 111 e 120 pts (apenas 4 jogos cada = 5,4% da amostra); já a classe modal em intervalos de 5 pts "
                f"é {bos['classe_modal_pts']} ({bos['classe_modal_freq']} jogos = {_br(bos['classe_modal_pct'], 1)}%) e o pico "
                f"de densidade KDE fica em {_br(bos['kde_pico_pts'])} pts, cravado na mediana (113,50 pts). No Pelicans, "
                f"a moda discreta (116 pts, 7 jogos), a mediana (116,00 pts), a classe modal {nop['classe_modal_pts']} "
                f"({nop['classe_modal_freq']} jogos = {_br(nop['classe_modal_pct'], 1)}%) e o KDE ({_br(nop['kde_pico_pts'])} pts) "
                f"convergem no mesmo ponto."
            )
        elif var_key == "rebotes":
            contexto_tatico = (
                f"O Boston Celtics domina claramente a tábua de rebotes em relação ao Pelicans: média de "
                f"{_br(bos['media'])} vs. {_br(nop['media'])} rebotes (+{_br(diff_mean)} reb/jogo) e mediana de "
                f"{_br(bos['mediana'])} vs. {_br(nop['mediana'])} rebotes (+{_br(diff_med)} reb). Na escala padronizada da liga "
                f"(μ = {_br(bos['liga_media'])}), o Celtics opera em z = +{_br(bos['z_media_liga'], 2)} σ (acima da média da NBA), "
                f"enquanto o Pelicans fica exatamente sobre a média da liga na média (z = {_br(nop['z_media_liga'], 2)} σ) e "
                f"abaixo na mediana (z = {_br(nop['z_mediana_liga'], 2)} σ). Ambos têm desvios padrões quase idênticos "
                f"(6,66 rebotes), mas o Pelicans registra 1 outlier inferior severo: apenas 26 rebotes contra o Orlando Magic "
                f"(11/01/2026), abaixo da cerca inferior de {_br(nop['cerca_inferior'])} rebotes."
            )
            criterio_moda = (
                f"O Celtics é unimodal em 44 rebotes (7 jogos = 9,5%), enquanto o Pelicans apresenta distribuição "
                f"bimodal em 40 e 42 rebotes (9 jogos cada = 11,8%), refletindo que a concentração típica do Pelicans "
                f"está 2 a 4 rebotes abaixo de sua própria média (43,78), puxada para cima por partidas com prorrogação."
            )
        elif var_key == "assistencias":
            contexto_tatico = (
                f"Ambas as equipes operam abaixo da média da NBA em assistências (Liga = {_br(bos['liga_media'])} ast): "
                f"o Celtics tem média de {_br(bos['media'])} ast (z = {_br(bos['z_media_liga'], 2)} σ) e mediana de "
                f"{_br(bos['mediana'])} ast, enquanto o Pelicans registra média de {_br(nop['media'])} ast "
                f"(z = {_br(nop['z_media_liga'], 2)} σ) e mediana de {_br(nop['mediana'])} ast. Apesar da média ligeiramente menor "
                f"(reflexo de posses decididas em isolamentos e bolas de 3 em transição), o teto criativo do Celtics gerou "
                f"2 outliers superiores pelo critério de Tukey (> {_br(bos['cerca_superior'])} ast): 36 ast contra GSW (19/02/2026) "
                f"e 38 ast contra BKN (27/02/2026). O Pelicans não teve outliers (intervalo de 16 a 37 ast)."
            )
            criterio_moda = (
                f"Ambas as séries individuais são unimodais: moda de 24 assistências no Celtics (9 jogos = 12,2%, idêntica "
                f"à mediana de 24,00) e moda de 27 assistências no Pelicans (9 jogos = 11,8%, ligeiramente acima da mediana de 25,00)."
            )
        elif var_key == "roubos":
            contexto_tatico = (
                f"Roubos de bola (STL) separam nitidamente a postura defensiva das duas franquias: o Pelicans pressiona "
                f"muito mais as linhas de passe, com média de {_br(nop['media'])} vs. {_br(bos['media'])} roubos do Celtics "
                f"(mediana de {_br(nop['mediana'])} vs. {_br(bos['mediana'])} roubos; moda de 10 vs. 7 roubos). "
                f"Em termos padronizados, o Pelicans supera a liga (z = +{_br(nop['z_media_liga'], 2)} σ na média e "
                f"+{_br(nop['z_mediana_liga'], 2)} σ na mediana), enquanto o Celtics contém sem arriscar botes "
                f"(z = {_br(bos['z_media_liga'], 2)} σ). Como a caixa do Celtics é estreita (Q1=6, Q3=9, IQR=3), "
                f"surgem 3 outliers: 1 superior (14 roubos vs WAS em 04/12/2025) e 2 inferiores (1 roubo vs CHA em 04/03/2026 "
                f"e 0 roubos vs ATL em 27/03/2026)."
            )
            criterio_moda = (
                f"Por ser contagem discreta de baixa magnitude, a concentração inteira é forte e unimodal: "
                f"7 roubos no Celtics (13 jogos = 17,6%, igual à mediana) e 10 roubos no Pelicans (11 jogos = 14,5%). "
                f"O Coeficiente de Variação é elevado (38,21% no Celtics e 33,36% no Pelicans) por conta da média próxima de zero."
            )
        elif var_key == "tocos":
            contexto_tatico = (
                f"Em tocos (BLK), ambos os times superam levemente a média da NBA (Liga = {_br(bos['liga_media'])} tocos): "
                f"Celtics com média de {_br(bos['media'])} (z = +{_br(bos['z_media_liga'], 2)} σ) e Pelicans com "
                f"{_br(nop['media'])} (z = +{_br(nop['z_media_liga'], 2)} σ), empatados com mediana de 5,00 tocos. "
                f"A diferença crítica reside na dispersão: a variância do Pelicans ({_br(nop['variancia'])} tocos², DP = {_br(nop['dp'])}, "
                f"CV = {_br(nop['cv'])}%) é 64% maior que a do Celtics ({_br(bos['variancia'])} tocos², DP = {_br(bos['dp'])}, "
                f"CV = {_br(bos['cv'])}%), variando de 0 a 12 tocos sem ultrapassar a cerca superior de 13,00."
            )
            criterio_moda = (
                f"Séries unimodais discretas: moda de 4 tocos no Celtics (13 jogos = 17,6%) e 3 tocos no Pelicans "
                f"(12 jogos = 15,8%). No Pelicans, Moda (3) < Mediana (5,00) < Média (5,18) caracteriza assimetria positiva "
                f"(skew = +{_br(nop['assimetria'])}) causada por partidas de proteção de aro intensa (10 a 12 tocos)."
            )
        elif var_key == "faltas":
            contexto_tatico = (
                f"As duas equipes cometem menos faltas na média que a liga (Liga = {_br(bos['liga_media'])} faltas) e dividem "
                f"exatamente a mesma mediana (20,00 faltas). Porém, o Celtics tem média menor ({_br(bos['media'])} vs. "
                f"{_br(nop['media'])}; z = {_br(bos['z_media_liga'], 2)} σ) e maior variância ({_br(bos['variancia'])} vs. "
                f"{_br(nop['variancia'])}) devido a uma cauda esquerda pronunciada: 3 partidas de disciplina defensiva extrema "
                f"foram sinalizadas como outliers inferiores (< {_br(bos['cerca_inferior'])} faltas) — 6 faltas vs GSW (19/02/2026), "
                f"7 vs IND (12/01/2026) e 9 vs CHI (24/01/2026)."
            )
            criterio_moda = (
                f"Unimodal em ambos os times, mas em posições opostas: no Celtics a moda é 20 faltas (14 jogos = 18,9%, "
                f"coincidindo com a mediana de 20,00, enquanto a média cai para 19,15 pelos outliers inferiores); no Pelicans "
                f"a moda é 16 faltas (10 jogos = 13,2%)."
            )
        else:  # bolas3_convertidas
            contexto_tatico = (
                f"Bolas de 3 convertidas (FG3M) é a variável de maior contraste estrutural entre as duas franquias em todo o estudo: "
                f"o Boston Celtics converte em média {_br(bos['media'])} cestas de 3 por jogo (mediana {_br(bos['mediana'])}, "
                f"Q1 = 13,00, Q3 = 18,00), situando-se meio desvio-padrão acima da NBA (z = +{_br(bos['z_media_liga'], 2)} σ). "
                f"O New Orleans Pelicans converte apenas {_br(nop['media'])} por jogo (mediana {_br(nop['mediana'])}, "
                f"Q1 = 8,75, Q3 = 13,00), operando em z = {_br(nop['z_media_liga'], 2)} σ — uma distância superior a 1,02 desvios-padrão "
                f"da liga entre os dois times! Note que o Q3 do Pelicans (13,00 cestas) equivale exatamente ao Q1 e à moda do Celtics (13,00). "
                f"Para o Pelicans atingir 24 bolas de 3 (mesmo máximo do Celtics), foi necessária uma partida com prorrogação "
                f"(265 min contra MIL em 04/02/2026), que saltou como outlier superior isolado (> {_br(nop['cerca_superior'])})."
            )
            criterio_moda = (
                f"O Celtics é unimodal em 13 bolas de 3 (12 jogos = 16,2%). O Pelicans é bimodal em 7 e 10 bolas de 3 "
                f"(9 jogos cada = 11,8%), confirmando que seu volume habitual de perímetro orbita um patamar estruturalmente "
                f"inferior ao do Celtics."
            )

        interps[var_key] = {
            "variavel": var_key,
            "label": meta["label"],
            "short_label": meta["short_label"],
            "contexto_tatico": contexto_tatico,
            "criterio_moda": criterio_moda,
            "diff_media": _br(diff_mean, 2),
            "diff_mediana": _br(diff_med, 2),
            "diff_z_media": _br(bos["z_media_liga"] - nop["z_media_liga"], 2),
        }

    return interps


def get_descritiva_analysis(
    team: Optional[str] = None,
    variable: str = "pontos",
    medida: str = "media",
) -> Dict[str, Any]:
    """
    Monta todos os dados, tabelas, gráficos Plotly anotados e interpretações críticas
    da página de Estatística Descritiva (Parte 3).
    """
    if variable not in STUDY_VARIABLES:
        variable = "pontos"
    if medida not in {"media", "mediana", "ambas"}:
        medida = "media"

    var_meta = STUDY_VARIABLES[variable]

    # Resumos completos das 14 séries (7 variáveis × 2 times)
    all_summaries: List[Dict[str, Any]] = resumo_descritivo(team=None, variavel=None)  # type: ignore[assignment]
    by_team_var = {(s["time"], s["variavel"]): s for s in all_summaries}

    # Filtrar conforme o seletor de time para a tabela principal e para o boxplot
    if team and team in TEAMS:
        selected_teams = [team]
    else:
        selected_teams = list(TEAMS)

    table_rows = [s for s in all_summaries if s["time"] in selected_teams]
    summaries_selected_var = [by_team_var[(t, variable)] for t in selected_teams]

    # Comparativo direto Celtics x Pelicans x Liga na variável selecionada
    s_bos = by_team_var[("Boston Celtics", variable)]
    s_nop = by_team_var[("New Orleans Pelicans", variable)]
    league_ref_all = get_league_reference()
    l_ref = league_ref_all[variable]

    # Tabela de padronização Z-score das 7 variáveis
    zscore_table = []
    for k, meta in STUDY_VARIABLES.items():
        b = by_team_var[("Boston Celtics", k)]
        p = by_team_var[("New Orleans Pelicans", k)]
        lr = league_ref_all[k]
        zscore_table.append(
            {
                "variavel": k,
                "label": meta["short_label"],
                "full_label": meta["label"],
                "unit": meta["unit"],
                "selecionada": k == variable,
                "liga_media": lr["media"],
                "liga_dp": lr["dp"],
                "liga_mediana": lr["mediana"],
                "bos_media": b["media"],
                "bos_mediana": b["mediana"],
                "bos_z_media": b["z_media_liga"],
                "bos_z_mediana": b["z_mediana_liga"],
                "nop_media": p["media"],
                "nop_mediana": p["mediana"],
                "nop_z_media": p["z_media_liga"],
                "nop_z_mediana": p["z_mediana_liga"],
                "delta_z_media": b["z_media_liga"] - p["z_media_liga"],
                "delta_z_mediana": b["z_mediana_liga"] - p["z_mediana_liga"],
            }
        )

    # Tabela de modas e multimodalidade das 7 variáveis
    modes_table = []
    for k, meta in STUDY_VARIABLES.items():
        b = by_team_var[("Boston Celtics", k)]
        p = by_team_var[("New Orleans Pelicans", k)]
        modes_table.append(
            {
                "variavel": k,
                "label": meta["label"],
                "short_label": meta["short_label"],
                "unit": meta["unit"],
                "selecionada": k == variable,
                "bos_modas": b["modes_joined"],
                "bos_tipo": b["moda_tipo"],
                "bos_freq": b["moda_freq"],
                "bos_pct": b["moda_pct"],
                "bos_multimodal": b["is_multimodal"],
                "nop_modas": p["modes_joined"],
                "nop_tipo": p["moda_tipo"],
                "nop_freq": p["moda_freq"],
                "nop_pct": p["moda_pct"],
                "nop_multimodal": p["is_multimodal"],
            }
        )

    interpretations = _build_variable_interpretations(by_team_var)

    chart_boxplot_json = _build_annotated_boxplot(summaries_selected_var, var_meta)
    chart_bars_zscore_json = _build_zscore_bars_chart(all_summaries, variable, medida)
    chart_bars_raw_json = _build_raw_bars_chart(all_summaries, variable, medida)

    return {
        "variable": variable,
        "var_meta": var_meta,
        "medida": medida,
        "measure_choices": MEASURE_CHOICES,
        "selected_teams": selected_teams,
        "table_rows": table_rows,
        "all_summaries": all_summaries,
        "summaries_selected_var": summaries_selected_var,
        "bos": s_bos,
        "nop": s_nop,
        "liga": l_ref,
        "zscore_table": zscore_table,
        "modes_table": modes_table,
        "interpretation_selected": interpretations[variable],
        "interpretations_all": [interpretations[k] for k in STUDY_VARIABLES],
        "chart_boxplot_json": chart_boxplot_json,
        "chart_bars_zscore_json": chart_bars_zscore_json,
        "chart_bars_raw_json": chart_bars_raw_json,
    }

