{{ config(materialized='view') }}

select
    customer_id,
    degree::integer as degree,
    in_degree::integer as in_degree,
    out_degree::integer as out_degree,
    norm_in_degree::float as norm_in_degree,
    norm_out_degree::float as norm_out_degree,
    pagerank::float as pagerank,
    distance_to_prior_default::integer as distance_to_prior_default,
    clustering_coefficient::float as clustering_coefficient,
    neighbor_default_frac::float as neighbor_default_frac,
    neighbor_default_frac_2hop::float as neighbor_default_frac_2hop
from {{ source('credit_domino', 'graph_features') }}
