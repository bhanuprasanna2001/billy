FROM apache/airflow:3.1.7-python3.13

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

USER airflow

# Install credit_domino package with dbt extras into Airflow's Python env
COPY --chown=airflow:root pyproject.toml /tmp/credit-domino/
COPY --chown=airflow:root src/ /tmp/credit-domino/src/
RUN pip install --no-cache-dir "/tmp/credit-domino[dbt]" && \
    rm -rf /tmp/credit-domino
