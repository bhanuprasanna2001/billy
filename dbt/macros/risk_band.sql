{% macro interest_rate_risk_band(rate_column) %}
    case
        when {{ rate_column }} > 15.0 then 'high'
        when {{ rate_column }} > 10.0 then 'medium'
        else 'low'
    end
{% endmacro %}
