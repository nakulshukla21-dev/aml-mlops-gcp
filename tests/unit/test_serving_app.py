"""Unit tests for FastAPI serving routes."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.party_resolver import SenderNotAuthorizedError


class FakeScoringService:
    profile = "dev"
    project_id = "test-project"

    def __init__(self) -> None:
        self.vertex = MagicMock()
        self.vertex.endpoint_resource_name = "projects/test/locations/us-central1/endpoints/123"

    def score_payment(self, payload):
        if payload.customer_id == 9999:
            raise SenderNotAuthorizedError("Account does not belong to customer.")
        return {
            "transaction_id": payload.transaction_id,
            "fraud_score": 0.87,
            "is_fraud_predicted": True,
            "risk_level": "High",
            "resolved_parties": {
                "sender_account_id": payload.sender_account_id,
                "sender_counterparty_account_id": None,
                "receiver_account_id": None,
                "receiver_counterparty_account_id": "CPAKY00001",
                "receiver_created": False,
            },
            "features_used": {"amount": payload.amount},
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    def validate_transaction(self, payload):
        return {
            "transaction_id": payload.transaction_id,
            "parity": {"matched": True, "max_feature_delta": 0.0, "mismatched_columns": []},
            "features_used": {"amount": payload.amount},
        }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(service=FakeScoringService()))


def _score_payload() -> dict:
    return {
        "transaction_id": "TXN-NEW-001",
        "timestamp": "2024-11-15T14:30:00Z",
        "customer_id": 1001,
        "sender_account_id": "BAUS0000100",
        "amount": 9500.0,
        "transaction_currency": "USD",
        "transaction_type": "transfer",
        "channel": "wire",
        "payment_sender_country": "US",
        "payment_receiver_country": "KY",
        "settlement_currency": "USD",
        "settlement_amount": 9500.0,
        "settlement_status": "pending",
        "receiver": {
            "type": "external",
            "beneficiary_name": "Global Trade Solutions",
            "country": "KY",
            "account_reference": "EXT-12345",
        },
        "log_prediction": False,
    }


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["profile"] == "dev"


def test_score_endpoint_returns_prediction(client: TestClient):
    response = client.post("/score", json=_score_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TXN-NEW-001"
    assert body["fraud_score"] == 0.87
    assert body["risk_level"] == "High"


def test_score_endpoint_maps_sender_auth_error_to_403(client: TestClient):
    payload = _score_payload()
    payload["customer_id"] = 9999
    response = client.post("/score", json=payload)
    assert response.status_code == 403


def test_validate_endpoint(client: TestClient):
    payload = {
        "transaction_id": "TXN-05BE16B0E90E",
        "timestamp": "2024-06-15T12:00:00Z",
        "sender_account_id": "BAUS0000100",
        "receiver_counterparty_account_id": "CPAKY00001",
        "amount": 100.0,
        "transaction_currency": "USD",
        "transaction_type": "payment",
        "channel": "wire",
        "payment_sender_country": "US",
        "payment_receiver_country": "KY",
        "settlement_currency": "USD",
        "settlement_amount": 100.0,
        "settlement_status": "settled",
    }
    response = client.post("/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["parity"]["matched"] is True
