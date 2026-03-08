{{ config(materialized='view') }}

select
    customer_id,
    person_age::integer as person_age,
    person_income::float as person_income,
    person_home_ownership,
    person_emp_length::float as person_emp_length,
    loan_intent,
    loan_grade,
    loan_amnt::float as loan_amnt,
    loan_int_rate::float as loan_int_rate,
    loan_status::integer as loan_status,
    loan_percent_income::float as loan_percent_income,
    cb_person_default_on_file::integer as cb_person_default_on_file,
    cb_person_cred_hist_length::integer as cb_person_cred_hist_length,
    is_recent_default::boolean as is_recent_default
from {{ source('credit_domino', 'customers') }}
where person_age > 18
  and person_income > 0
  and loan_amnt > 0
