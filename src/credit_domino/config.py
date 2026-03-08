from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres (individual fields or DATABASE_URL)
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "credit_domino"
    postgres_user: str = "dev"
    postgres_password: str = "dev"
    database_url: str | None = None  # Render / Heroku standard

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "devpassword123"

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_db: str = "credit_domino"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    port: int | None = None  # Render injects PORT
    log_level: str = "info"

    @property
    def postgres_dsn(self) -> str:
        if self.database_url:
            # Render provides postgres:// but SQLAlchemy 2.x requires postgresql://
            url = self.database_url
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            return url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def effective_port(self) -> int:
        """Render injects PORT; fall back to api_port for local dev."""
        return self.port if self.port is not None else self.api_port

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
