{{ config(
    materialized='incremental',
    unique_key='scoring_event_id',
    on_schema_change='fail'
) }}

select
    scoring_event_id,
    customer_id,
    risk_score,
    decision_band,
    requested_amount,
    top_factors,
    scored_at,
    current_timestamp as _dbt_ingested_at
from {{ source('credit_domino', 'raw_scoring_events') }}

{% if is_incremental() %}
    where scored_at > (select max(scored_at) from {{ this }})
{% endif %}
