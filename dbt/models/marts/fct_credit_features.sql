{{ config(materialized='table') }}

select
    c.customer_id,
    c.person_age,
    c.person_income,
    c.person_home_ownership,
    c.person_emp_length,
    c.loan_intent,
    c.loan_grade,
    c.loan_amnt,
    c.loan_int_rate,
    c.loan_status,
    c.loan_percent_income,
    c.cb_person_default_on_file,
    c.cb_person_cred_hist_length,
    c.is_recent_default,
    {{ interest_rate_risk_band('c.loan_int_rate') }} as interest_rate_risk_band,
    g.degree,
    g.in_degree,
    g.out_degree,
    g.norm_in_degree,
    g.norm_out_degree,
    g.pagerank,
    g.distance_to_prior_default,
    g.clustering_coefficient,
    g.neighbor_default_frac,
    g.neighbor_default_frac_2hop,
    current_timestamp as _dbt_loaded_at
from {{ ref('stg_customers') }} c
left join {{ ref('stg_graph_features') }} g
    on c.customer_id = g.customer_id
