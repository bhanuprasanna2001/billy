{{ config(materialized='view') }}

select
    src_customer_id,
    dst_customer_id,
    edge_type,
    case
        when edge_type in ('co-borrower', 'guarantor', 'employer', 'loan') then edge_type
        else 'unknown'
    end as validated_edge_type
from {{ source('credit_domino', 'relationships') }}
where src_customer_id != dst_customer_id
