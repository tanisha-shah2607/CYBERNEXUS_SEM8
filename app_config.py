"""
Application configuration. Only stdlib imports - never import from this project here.
"""
import os
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


class Settings:
    def __init__(self):
        self.db_path = PROJECT_ROOT / "cyber_bi.db"
        self.csv_seed_path = PROJECT_ROOT / "cyber_data.csv"
        self.reports_dir = PROJECT_ROOT / "generated_reports"
        self.secret_key = os.getenv("APP_SECRET_KEY") or secrets.token_urlsafe(48)
        self.session_ttl_hours = 12
        self.bcrypt_rounds = 12
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.scheduler_enabled = True
        self.scheduler_poll_seconds = 30
        self.default_admin_email = "admin@cyberbi.local"
        self.default_admin_password = "admin123"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}?check_same_thread=False"


settings = Settings()