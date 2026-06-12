"""Shared fixtures for unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.apply_noise import NoiseConfig, TransactionNoiseApplicator


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Minimal transaction frame matching the raw v2 CSV schema."""
    base_ts = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    return pd.DataFrame(
        [
            {
                "transaction_id": "txn-001",
                "timestamp": base_ts,
                "sender_account_id": "BAUS0000100",
                "sender_counterparty_account_id": None,
                "receiver_account_id": None,
                "receiver_counterparty_account_id": "CPAUS00001",
                "amount": 100.0,
                "transaction_currency": "USD",
                "transaction_type": "payment",
                "channel": "wire",
                "channel_indicator": "Online",
                "terminal_id": None,
                "atm_id": None,
                "merchant_city": None,
                "merchant_state": None,
                "merchant_country": None,
                "merchant_legal_name": "Global Trade Solutions",
                "merchant_dba_name": None,
                "pos_entry_mode": None,
                "payment_reference": "REF-001",
                "memo": "Invoice payment for consulting services",
                "payment_sender_country": "US",
                "payment_receiver_country": "GB",
                "settlement_currency": "USD",
                "settlement_amount": 100.0,
                "fx_rate": 1.0,
                "settlement_date": base_ts.date(),
                "settlement_status": "settled",
                "clearing_system": "SWIFT",
                "correspondent_bic": "CHASUS33",
                "typology": "legitimate",
                "is_fraud": False,
            },
            {
                "transaction_id": "txn-002",
                "timestamp": base_ts.replace(hour=13),
                "sender_account_id": "BAUS0000200",
                "sender_counterparty_account_id": None,
                "receiver_account_id": "BAUS0000300",
                "receiver_counterparty_account_id": None,
                "amount": 250.0,
                "transaction_currency": "USD",
                "transaction_type": "transfer",
                "channel": "ach",
                "channel_indicator": "Phone",
                "terminal_id": None,
                "atm_id": None,
                "merchant_city": None,
                "merchant_state": None,
                "merchant_country": None,
                "merchant_legal_name": None,
                "merchant_dba_name": None,
                "pos_entry_mode": None,
                "payment_reference": None,
                "memo": "Payroll",
                "payment_sender_country": "US",
                "payment_receiver_country": "US",
                "settlement_currency": "USD",
                "settlement_amount": 250.0,
                "fx_rate": 1.0,
                "settlement_date": base_ts.date(),
                "settlement_status": "settled",
                "clearing_system": None,
                "correspondent_bic": None,
                "typology": "smurfing",
                "is_fraud": True,
            },
        ]
    )


@pytest.fixture
def noise_applicator() -> TransactionNoiseApplicator:
    config = NoiseConfig(
        enabled=True,
        label_flip_rate=0.5,
        missing_optional_field_rate=0.0,
        dirty_enum_rate=1.0,
        memo_truncate_rate=1.0,
        timestamp_jitter_rate=1.0,
        amount_jitter_rate=1.0,
        external_receiver_metadata_rate=1.0,
        merchant_name_typo_rate=0.0,
    )
    return TransactionNoiseApplicator(config, seed=42)
