def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "credit_domino")
    monkeypatch.setenv("POSTGRES_USER", "dev")
    monkeypatch.setenv("POSTGRES_PASSWORD", "dev")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "devpassword123")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("CLICKHOUSE_HOST", "localhost")

    from credit_domino.config import Settings

    s = Settings()
    assert s.postgres_host == "localhost"
    assert s.neo4j_uri == "bolt://localhost:7687"
    assert "postgresql" in s.postgres_dsn
    assert s.clickhouse_host == "localhost"
    assert s.clickhouse_port == 8123


def test_postgres_dsn_format(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

    from credit_domino.config import Settings

    s = Settings()
    assert s.postgres_dsn == "postgresql://testuser:testpass@db.example.com:5433/testdb"
