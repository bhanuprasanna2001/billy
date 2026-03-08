#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# Credit Domino Lab – End-to-End Demo Walkthrough
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

step() { echo -e "\n${CYAN}──── $1 ────${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo -e "${YELLOW}ℹ $1${NC}"; }

# ── 0. Prerequisites ───────────────────────────────────────────
step "0. Checking prerequisites"
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "ERROR: python not found"; exit 1; }
ok "docker and python available"

# ── 1. Start infrastructure ────────────────────────────────────
step "1. Starting infrastructure (docker compose)"
docker compose up -d
info "Waiting for services to become healthy..."
sleep 15

# Health check core services
curl -sf http://localhost:5001/health >/dev/null && ok "MLflow healthy" || info "MLflow still starting..."

# ── 2. Prepare data ────────────────────────────────────────────
step "2. Preparing credit data (CSV → Postgres + graph features)"
python scripts/prepare_credit_data.py
ok "Data loaded into Postgres (customers, relationships, graph_features)"

# ── 3. dbt build ───────────────────────────────────────────────
step "3. Running dbt (staging + marts)"
cd dbt && dbt build --profiles-dir . --project-dir . && cd ..
ok "dbt models materialized and tests passed"

# ── 4. Load Neo4j graph ────────────────────────────────────────
step "4. Loading graph into Neo4j"
python -m credit_domino.graph.loader
ok "Neo4j graph loaded"

# ── 5. Train & register model ──────────────────────────────────
step "5. Training XGBoost model and registering in MLflow"
python scripts/train_model.py
ok "Model trained, evaluated, and registered as @champion"

# ── 6. Start scoring API ───────────────────────────────────────
step "6. Starting FastAPI scoring API"
# Kill any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1
uvicorn credit_domino.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
sleep 8

# Health check
curl -sf http://localhost:8000/health | python -m json.tool
ok "API is live"

# ── 7. Score a customer ─────────────────────────────────────────
step "7. Scoring a sample customer"
curl -sf -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_0",
    "person_age": 35,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": 0,
    "cb_person_cred_hist_length": 8
  }' | python -m json.tool
ok "Customer scored (graph features auto-looked up from Postgres)"

# ── 8. Run domino simulation ───────────────────────────────────
step "8. Running domino simulation"
curl -s -X POST http://localhost:8000/simulate-domino \
  -H "Content-Type: application/json" \
  -d '{
    "trigger_customer_id": "CUST_0",
    "initial_shock": 1.0,
    "decay": 0.6,
    "threshold": 0.3,
    "max_hops": 3,
    "n_customers": 500
  }' | python -m json.tool
ok "Domino cascade simulated"

# ── 9. Cleanup ──────────────────────────────────────────────────
step "9. Stopping API"
kill $API_PID 2>/dev/null || true
ok "API stopped"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Demo complete! All components working end-to-end.${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "
${YELLOW}Service URLs:${NC}
  Scoring API:   http://localhost:8000/docs
  MLflow UI:     http://localhost:5001
  Airflow UI:    http://localhost:8080
  Neo4j Browser: http://localhost:7474
  Grafana:       http://localhost:3000  (admin/admin)
  Prometheus:    http://localhost:9090

${YELLOW}Next steps:${NC}
  • Launch dashboard:  streamlit run src/credit_domino/dashboard/app.py
  • Start monitoring:  make up PROFILE=monitoring
  • Trigger DAG:       open http://localhost:8080 and enable credit_domino_pipeline
"
