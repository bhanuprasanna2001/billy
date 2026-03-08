import json


def test_structured_logger_outputs_json(capsys):
    from credit_domino.logging import get_logger

    logger = get_logger("test")
    logger.info("score_computed", customer_id="CUST_1", risk_score=0.73)

    captured = capsys.readouterr()
    line = captured.out.strip().split("\n")[-1]
    data = json.loads(line)
    assert data["customer_id"] == "CUST_1"
    assert data["event"] == "score_computed"
    assert "timestamp" in data
    assert data["level"] == "info"
