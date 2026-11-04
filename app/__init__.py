"""
Inicialização da aplicação web Flask — Análise Estatística NBA (Boston Celtics & New Orleans Pelicans).
"""

from __future__ import annotations

from flask import Flask

from app.data_loader import PROCESSED_PATH, STUDY_VARIABLES, TEAMS, build_clean_dataframe


def create_app() -> Flask:
    """Application Factory do sistema Flask."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "ufape-lab-cd-projeto1-celtics-pelicans-2026"

    if not PROCESSED_PATH.exists():
        build_clean_dataframe(save=True)

    from app.routes import main_bp, parte1_bp, parte2_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(parte1_bp)
    app.register_blueprint(parte2_bp)

    @app.context_processor
    def inject_globals():
        return {
            "STUDY_VARIABLES": STUDY_VARIABLES,
            "TEAMS": TEAMS,
        }

    @app.template_filter("num2")
    def format_num2(value):
        try:
            return f"{float(value):.2f}".replace(".", ",")
        except (ValueError, TypeError):
            return str(value)

    @app.template_filter("num4")
    def format_num4(value):
        try:
            return f"{float(value):.4f}".replace(".", ",")
        except (ValueError, TypeError):
            return str(value)

    return app

