"""
Definição de rotas e Blueprints do sistema Flask.
"""

from __future__ import annotations

from flask import Blueprint, render_template, request

from app.analysis.descritiva import MEASURE_CHOICES, get_descritiva_analysis
from app.analysis.dispersao import (
    ESCALAS,
    get_assimetria_analysis,
    get_cv_analysis,
    get_iqr_analysis,
    get_sintese_analysis,
)
from app.analysis.outliers import METHOD_CHOICES, get_outliers_analysis
from app.analysis.preparacao import (
    get_data_dictionary,
    get_missing_values_analysis,
    get_overview_summary,
    get_special_values_analysis,
    get_treatments_analysis,
    get_types_and_size_analysis,
)

main_bp = Blueprint("main", __name__)
parte1_bp = Blueprint("parte1", __name__)
parte2_bp = Blueprint("parte2", __name__)
parte3_bp = Blueprint("parte3", __name__)
parte4_bp = Blueprint("parte4", __name__)


def _get_selected_team() -> str:
    team = request.args.get("team", "todos").strip()
    if team not in {"todos", "Boston Celtics", "New Orleans Pelicans"}:
        return "todos"
    return team


def _get_selected_method() -> str:
    method = request.args.get("method", "todos").strip()
    valid = {item["id"] for item in METHOD_CHOICES}
    return method if method in valid else "todos"


def _get_selected_measure() -> str:
    medida = request.args.get("medida", "media").strip()
    valid = {item["id"] for item in MEASURE_CHOICES}
    return medida if medida in valid else "media"


def _get_selected_variable() -> str:
    var = request.args.get("var", "pontos").strip()
    valid = {
        "pontos",
        "rebotes",
        "assistencias",
        "roubos",
        "tocos",
        "faltas",
        "bolas3_convertidas",
    }
    return var if var in valid else "pontos"


def _get_selected_scale() -> str:
    escala = request.args.get("escala", "bruta").strip()
    return escala if escala in ESCALAS else "bruta"


@main_bp.route("/")
def index():
    """Página inicial: seleção do time e visualização direta dos dados e jogos."""
    team = _get_selected_team()
    overview = get_overview_summary(team=None if team == "todos" else team)
    return render_template(
        "index.html",
        overview=overview,
        selected_team=team,
        active_page="home",
    )


@parte1_bp.route("/dicionario")
@parte1_bp.route("/parte1/dicionario")
def dicionario():
    """(i) Dicionário de Dados."""
    team = _get_selected_team()
    role = request.args.get("papel", "todas").strip()
    data = get_data_dictionary(
        team=None if team == "todos" else team,
        role_filter=role,
    )
    return render_template(
        "parte1/dicionario.html",
        data=data,
        selected_team=team,
        selected_role=role,
        active_page="dicionario",
    )


@parte1_bp.route("/tipos-tamanho")
@parte1_bp.route("/parte1/tipos-tamanho")
def tipos_tamanho():
    """Tipos de Dados, Tamanho da Base e Cobertura Mensal."""
    team = _get_selected_team()
    var = _get_selected_variable()
    data = get_types_and_size_analysis(
        team=None if team == "todos" else team,
        variable=var,
    )
    return render_template(
        "parte1/tipos_tamanho.html",
        data=data,
        selected_team=team,
        selected_var=var,
        active_page="tipos_tamanho",
    )


@parte1_bp.route("/tratamentos")
@parte1_bp.route("/parte1/tratamentos")
def tratamentos():
    """Preparação dos Dados: Datas, Categóricas e Seleção de Colunas."""
    team = _get_selected_team()
    var = _get_selected_variable()
    data = get_treatments_analysis(
        team=None if team == "todos" else team,
        variable=var,
    )
    return render_template(
        "parte1/tratamentos.html",
        data=data,
        selected_team=team,
        selected_var=var,
        active_page="tratamentos",
    )


@parte1_bp.route("/valores-especiais")
@parte1_bp.route("/parte1/valores-especiais")
def valores_especiais():
    """(v) Qualidade dos Dados: Valores Especiais e Inconsistências."""
    team = _get_selected_team()
    var = _get_selected_variable()
    data = get_special_values_analysis(
        team=None if team == "todos" else team,
        variable=var,
    )
    return render_template(
        "parte1/valores_especiais.html",
        data=data,
        selected_team=team,
        selected_var=var,
        active_page="valores_especiais",
    )


@parte1_bp.route("/ausentes")
@parte1_bp.route("/parte1/ausentes")
def ausentes():
    """(v) Qualidade dos Dados: Valores Ausentes (MCAR, MAR, MNAR)."""
    team = _get_selected_team()
    try:
        rate = float(request.args.get("taxa", "0.10"))
    except ValueError:
        rate = 0.10
    try:
        seed = int(request.args.get("seed", "42"))
    except ValueError:
        seed = 42

    data = get_missing_values_analysis(
        team=None if team == "todos" else team,
        missing_rate=rate,
        seed=seed,
    )
    return render_template(
        "parte1/ausentes.html",
        data=data,
        selected_team=team,
        selected_rate=f"{rate:.2f}",
        selected_seed=seed,
        active_page="ausentes",
    )


@parte2_bp.route("/outliers")
@parte2_bp.route("/parte2/outliers")
def outliers():
    """Qualidade dos dados: outliers e inconsistências pelos seis métodos."""
    team = _get_selected_team()
    var = _get_selected_variable()
    method = _get_selected_method()
    data = get_outliers_analysis(
        team=None if team == "todos" else team,
        variable=var,
        method=method,
    )
    return render_template(
        "parte2/outliers.html",
        data=data,
        selected_team=team,
        selected_var=var,
        selected_method=method,
        active_page="outliers",
    )


@parte3_bp.route("/estatistica-descritiva")
@parte3_bp.route("/parte3/descritiva")
@parte3_bp.route("/descritiva")
def descritiva():
    """(ii) e (iii) Estatística descritiva univariada, boxplot anotado e barras comparativas."""
    team = _get_selected_team()
    var = _get_selected_variable()
    medida = _get_selected_measure()
    data = get_descritiva_analysis(
        team=None if team == "todos" else team,
        variable=var,
        medida=medida,
    )
    return render_template(
        "parte3/descritiva.html",
        data=data,
        selected_team=team,
        selected_var=var,
        selected_medida=medida,
        active_page="descritiva",
    )


@parte4_bp.route("/assimetria")
@parte4_bp.route("/parte4/assimetria")
def assimetria():
    """Assimetria (Fisher-Pearson ajustado) por variável e time."""
    team = _get_selected_team()
    var = _get_selected_variable()
    escala = _get_selected_scale()
    data = get_assimetria_analysis(
        team=None if team == "todos" else team,
        variable=var,
        escala=escala,
    )
    return render_template(
        "parte4/assimetria.html",
        data=data,
        selected_team=team,
        selected_var=var,
        selected_escala=escala,
        active_page="assimetria",
    )


@parte4_bp.route("/cv")
@parte4_bp.route("/parte4/cv")
def cv():
    """Coeficiente de variação (%) por variável e time."""
    team = _get_selected_team()
    var = _get_selected_variable()
    data = get_cv_analysis(team=None if team == "todos" else team, variable=var)
    return render_template(
        "parte4/cv.html",
        data=data,
        selected_team=team,
        selected_var=var,
        active_page="cv",
    )


@parte4_bp.route("/iqr")
@parte4_bp.route("/parte4/iqr")
def iqr():
    """Amplitude interquartil, limites de Tukey e conferência com o boxplot."""
    team = _get_selected_team()
    var = _get_selected_variable()
    data = get_iqr_analysis(team=None if team == "todos" else team, variable=var)
    return render_template(
        "parte4/iqr.html",
        data=data,
        selected_team=team,
        selected_var=var,
        active_page="iqr",
    )


@parte4_bp.route("/sintese")
@parte4_bp.route("/parte4/sintese")
def sintese():
    """Síntese e interpretação geral do sistema."""
    team = _get_selected_team()
    data = get_sintese_analysis(team=None if team == "todos" else team)
    return render_template(
        "parte4/sintese.html",
        data=data,
        selected_team=team,
        active_page="sintese",
    )
