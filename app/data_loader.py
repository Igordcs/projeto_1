"""
Módulo central de carregamento e preparação dos dados limpos.
Garante que todas as partes do sistema consumam exclusivamente `load_clean(team=None)`
a partir de `data/processed/jogos_limpo.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATHS = [
    BASE_DIR / "data" / "raw" / "stats_times_regular_final.csv",
    BASE_DIR / "data" / "stats_times_regular_final.csv",
]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "jogos_limpo.csv"

TEAMS = ("Boston Celtics", "New Orleans Pelicans")
TEAM_ABBR_MAP = {
    "BOS": "Boston Celtics",
    "CELTICS": "Boston Celtics",
    "BOSTON CELTICS": "Boston Celtics",
    "NOP": "New Orleans Pelicans",
    "PELICANS": "New Orleans Pelicans",
    "NEW ORLEANS PELICANS": "New Orleans Pelicans",
}

COLUMN_RENAME_MAP = {
    "Team_ID": "id_time",
    "Game_ID": "id_jogo",
    "GAME_DATE": "data_jogo",
    "MATCHUP": "confronto",
    "WL": "vitoria",
    "W": "vitorias_acum",
    "L": "derrotas_acum",
    "W_PCT": "aproveitamento_acum",
    "MIN": "minutos",
    "FGM": "arremessos_convertidos",
    "FGA": "arremessos_tentados",
    "FG_PCT": "pct_arremessos",
    "FG3M": "bolas3_convertidas",
    "FG3A": "bolas3_tentadas",
    "FG3_PCT": "pct_bolas3",
    "FTM": "lances_livres_convertidos",
    "FTA": "lances_livres_tentados",
    "FT_PCT": "pct_lances_livres",
    "OREB": "rebotes_ofensivos",
    "DREB": "rebotes_defensivos",
    "REB": "rebotes",
    "AST": "assistencias",
    "STL": "roubos",
    "BLK": "tocos",
    "TOV": "erros",
    "PF": "faltas",
    "PTS": "pontos",
    "TEAM_NAME": "time",
}

STUDY_VARIABLES = {
    "pontos": {
        "original": "PTS",
        "label": "Pontos (PTS)",
        "short_label": "Pontos",
        "unit": "pontos",
        "type": "Numérica Discreta (tratada como contínua em larga escala)",
    },
    "rebotes": {
        "original": "REB",
        "label": "Rebotes Totais (REB)",
        "short_label": "Rebotes",
        "unit": "rebotes",
        "type": "Numérica Discreta",
    },
    "assistencias": {
        "original": "AST",
        "label": "Assistências (AST)",
        "short_label": "Assistências",
        "unit": "assistências",
        "type": "Numérica Discreta",
    },
    "roubos": {
        "original": "STL",
        "label": "Roubos de Bola (STL)",
        "short_label": "Roubos",
        "unit": "roubos",
        "type": "Numérica Discreta",
    },
    "tocos": {
        "original": "BLK",
        "label": "Tocos / Bloqueios (BLK)",
        "short_label": "Tocos",
        "unit": "tocos",
        "type": "Numérica Discreta",
    },
    "faltas": {
        "original": "PF",
        "label": "Faltas Pessoais (PF)",
        "short_label": "Faltas",
        "unit": "faltas",
        "type": "Numérica Discreta",
    },
    "bolas3_convertidas": {
        "original": "FG3M",
        "label": "Bolas de 3 Convertidas (FG3M)",
        "short_label": "Bolas de 3",
        "unit": "cestas de 3 pts",
        "type": "Numérica Discreta",
    },
}


def get_raw_path() -> Path:
    """Localiza o arquivo CSV bruto na estrutura do projeto."""
    for path in RAW_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Arquivo bruto não encontrado em nenhum dos caminhos esperados: {RAW_PATHS}"
    )


def load_raw() -> pd.DataFrame:
    """Carrega a base bruta integral (2238 linhas x 28 colunas, 30 times)."""
    raw_path = get_raw_path()
    return pd.read_csv(raw_path)


def _extract_mandante(matchup: str) -> int:
    """Retorna 1 se o jogo foi em casa ('vs.') ou 0 se visitante ('@')."""
    return 1 if "vs." in str(matchup) else 0


def _extract_adversario(matchup: str) -> str:
    """Extrai a sigla de 3 letras do time adversário a partir de MATCHUP."""
    text = str(matchup).strip()
    if "vs." in text:
        return text.split("vs.")[-1].strip()
    if "@" in text:
        return text.split("@")[-1].strip()
    parts = text.split()
    return parts[-1] if parts else ""


def build_clean_dataframe(save: bool = True) -> pd.DataFrame:
    """
    Executa o pipeline completo de preparação da Parte 1 sobre o recorte
    Boston Celtics e New Orleans Pelicans (150 jogos) e salva em
    `data/processed/jogos_limpo.csv`.
    """
    df_raw = load_raw()

    # 1. Filtrar os 2 times do grupo
    df = df_raw[df_raw["TEAM_NAME"].isin(TEAMS)].copy()

    # 2. Renomear as 28 colunas originais para português
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # 3. Converter data_jogo para datetime e ordenar cronologicamente por time
    df["data_jogo"] = pd.to_datetime(df["data_jogo"], format="%Y-%m-%d")
    df = df.sort_values(["time", "data_jogo", "id_jogo"]).reset_index(drop=True)

    # 4. Criar variáveis temporais derivadas da data
    df["mes"] = df["data_jogo"].dt.month
    estreia_por_time = df.groupby("time")["data_jogo"].transform("min")
    df["dias_desde_estreia"] = (df["data_jogo"] - estreia_por_time).dt.days
    df["numero_jogo"] = df.groupby("time").cumcount() + 1

    # 5. Codificar categóricas (MATCHUP -> mandante binária e adversario sigla)
    df["mandante"] = df["confronto"].apply(_extract_mandante).astype(int)
    df["adversario"] = df["confronto"].apply(_extract_adversario)

    # 6. Flag de prorrogação (minutos > 240)
    df["prorrogacao"] = (df["minutos"] != 240).astype(int)

    # Garantir que vitoria é int binário (0/1)
    df["vitoria"] = df["vitoria"].astype(int)

    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = df.copy()
        df_to_save["data_jogo"] = df_to_save["data_jogo"].dt.strftime("%Y-%m-%d")
        df_to_save.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")

    return df


def load_clean(team: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega os dados limpos (`data/processed/jogos_limpo.csv`).
    Se o arquivo processado ainda não existir, executa `build_clean_dataframe(save=True)`.

    Parâmetros:
        team (str | None): Nome ou sigla do time ("Boston Celtics", "New Orleans Pelicans",
                           "BOS", "NOP"). Se None ou "todos"/"all", retorna ambos os times (150 jogos).
    """
    if not PROCESSED_PATH.exists():
        df = build_clean_dataframe(save=True)
    else:
        df = pd.read_csv(PROCESSED_PATH, parse_dates=["data_jogo"])

    if team is not None:
        key = str(team).strip()
        if key and key.lower() not in {"todos", "all", "ambos", "none"}:
            normalized = TEAM_ABBR_MAP.get(key.upper(), key)
            df = df[df["time"] == normalized].copy().reset_index(drop=True)

    return df

