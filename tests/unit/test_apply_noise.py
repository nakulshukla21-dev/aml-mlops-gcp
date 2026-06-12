"""Unit tests for transaction noise application."""

from __future__ import annotations

import pandas as pd

from src.apply_noise import NoiseConfig, TransactionNoiseApplicator


def test_noise_disabled_returns_unchanged_copy(sample_transactions):
    config = NoiseConfig(enabled=False)
    applicator = TransactionNoiseApplicator(config, seed=42)

    result = applicator.apply(sample_transactions)

    pd.testing.assert_frame_equal(result, sample_transactions)


def test_label_flips_match_configured_rate(sample_transactions):
    config = NoiseConfig(enabled=True, label_flip_rate=0.5)
    applicator = TransactionNoiseApplicator(config, seed=7)

    original_labels = sample_transactions["is_fraud"].tolist()
    result = applicator.apply(sample_transactions)

    assert applicator.stats["label_flips"] == 1
    assert result["is_fraud"].tolist() != original_labels


def test_high_noise_rates_modify_data(sample_transactions, noise_applicator):
    original_channel = sample_transactions.loc[0, "channel"]
    original_memo = sample_transactions.loc[0, "memo"]

    result = noise_applicator.apply(sample_transactions)

    assert result is not sample_transactions
    assert noise_applicator.stats["dirty_enums"] > 0
    assert noise_applicator.stats["memo_truncated"] > 0
    assert noise_applicator.stats["timestamp_jittered"] > 0
    assert noise_applicator.stats["amount_jittered"] > 0
    assert result.loc[0, "channel"] != original_channel or result.loc[0, "memo"] != original_memo


def test_dirty_country_codes_applied_to_payment_countries(sample_transactions):
    config = NoiseConfig(
        enabled=True,
        label_flip_rate=0.0,
        missing_optional_field_rate=0.0,
        dirty_enum_rate=1.0,
        memo_truncate_rate=0.0,
        timestamp_jitter_rate=0.0,
        amount_jitter_rate=0.0,
    )
    applicator = TransactionNoiseApplicator(config, seed=42)
    result = applicator.apply(sample_transactions)

    assert applicator.stats["dirty_enums"] > 0
    assert result.loc[0, "payment_sender_country"] in {"US", "USA", "us", "U.S."}
