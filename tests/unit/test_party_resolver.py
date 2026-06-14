"""Unit tests for party resolution."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.serving.party_resolver import (
    DimensionTables,
    InvalidPaymentRequestError,
    PartyResolver,
    SenderNotAuthorizedError,
)
from src.serving.schemas import PaymentScoreRequest, ReceiverInstruction


@pytest.fixture
def dimensions() -> DimensionTables:
    return DimensionTables(
        dim_account=pd.DataFrame(
            [
                {
                    "account_id": "BAUS0000100",
                    "customer_id": 1001,
                    "status": "active",
                    "opened_date": date(2020, 1, 1),
                },
                {
                    "account_id": "BAUS0000200",
                    "customer_id": 1002,
                    "status": "active",
                    "opened_date": date(2019, 6, 1),
                },
            ]
        ),
        dim_customer=pd.DataFrame(
            [
                {"customer_id": 1001, "legal_name": "Alice Example"},
                {"customer_id": 1002, "legal_name": "Bob Example"},
            ]
        ),
        dim_counterparty=pd.DataFrame(
            [
                {
                    "counterparty_id": 5001,
                    "entity_type": "business",
                    "counterparty_role": "wire_beneficiary",
                    "legal_name": "Global Trade Solutions",
                    "dba_name": None,
                    "country": "KY",
                    "is_shell_entity": True,
                    "first_seen_date": date(2023, 1, 1),
                }
            ]
        ),
        dim_counterparty_account=pd.DataFrame(
            [
                {
                    "counterparty_account_id": "CPAKY00001",
                    "counterparty_id": 5001,
                    "account_label": "EXT-12345",
                    "country": "KY",
                    "first_seen_date": date(2023, 1, 1),
                }
            ]
        ),
    )


def _payment_request(**overrides) -> PaymentScoreRequest:
    base = {
        "transaction_id": "TXN-NEW-001",
        "timestamp": datetime(2024, 11, 15, 14, 30, tzinfo=timezone.utc),
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
        "receiver": ReceiverInstruction(
            type="external",
            beneficiary_name="Global Trade Solutions",
            country="KY",
            account_reference="EXT-12345",
        ),
    }
    base.update(overrides)
    return PaymentScoreRequest(**base)


def test_resolve_external_receiver_matches_existing_counterparty(dimensions: DimensionTables):
    resolver = PartyResolver(dimensions)
    resolved = resolver.resolve_payment(_payment_request())

    assert resolved.sender_account_id == "BAUS0000100"
    assert resolved.receiver_counterparty_account_id == "CPAKY00001"
    assert resolved.receiver_account_id is None
    assert resolved.resolved_parties is not None
    assert resolved.resolved_parties.receiver_created is False


def test_resolve_external_receiver_creates_stub_for_new_beneficiary(dimensions: DimensionTables):
    resolver = PartyResolver(dimensions)
    request = _payment_request(
        receiver=ReceiverInstruction(
            type="external",
            beneficiary_name="New Beneficiary LLC",
            country="VG",
            account_reference="EXT-NEW-99",
        )
    )
    resolved = resolver.resolve_payment(request)

    assert resolved.receiver_counterparty_account_id is not None
    assert resolved.receiver_counterparty_account_id.startswith("CPAVG")
    assert resolved.resolved_parties is not None
    assert resolved.resolved_parties.receiver_created is True
    assert len(resolver.stub_counterparty_accounts) == 1


def test_resolve_bank_receiver(dimensions: DimensionTables):
    resolver = PartyResolver(dimensions)
    request = _payment_request(
        receiver=ReceiverInstruction(type="bank", account_id="BAUS0000200"),
        payment_receiver_country="US",
    )
    resolved = resolver.resolve_payment(request)

    assert resolved.receiver_account_id == "BAUS0000200"
    assert resolved.receiver_counterparty_account_id is None


def test_sender_must_belong_to_customer(dimensions: DimensionTables):
    resolver = PartyResolver(dimensions)
    request = _payment_request(customer_id=1002, sender_account_id="BAUS0000100")

    with pytest.raises(SenderNotAuthorizedError, match="does not belong"):
        resolver.resolve_payment(request)


def test_unknown_sender_account_rejected(dimensions: DimensionTables):
    resolver = PartyResolver(dimensions)
    request = _payment_request(sender_account_id="BAUS9999999")

    with pytest.raises(SenderNotAuthorizedError, match="Unknown sender account"):
        resolver.resolve_payment(request)


def test_bank_receiver_requires_account_id(dimensions: DimensionTables):
    resolver = PartyResolver(dimensions)
    request = _payment_request(receiver=ReceiverInstruction(type="bank"))

    with pytest.raises(InvalidPaymentRequestError, match="account_id is required"):
        resolver.resolve_payment(request)
