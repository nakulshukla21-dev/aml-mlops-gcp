"""
Apply controlled noise to synthetic transactions for more realistic model training.

Noise is applied after clean generation so pipeline tests can still use noise.enabled: false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OPTIONAL_NULLABLE_COLUMNS = [
    "channel_indicator",
    "terminal_id",
    "atm_id",
    "merchant_city",
    "merchant_state",
    "merchant_country",
    "merchant_legal_name",
    "merchant_dba_name",
    "pos_entry_mode",
    "payment_reference",
    "memo",
    "fx_rate",
    "settlement_date",
    "clearing_system",
    "correspondent_bic",
    "sender_account_age_days",
    "receiver_account_age_days",
    "receiver_is_shell_company",
]

DIRTY_CHANNEL = {
    "wire": ["WIRE", "Wire", "wire transfer", "Wire Transfer"],
    "ach": ["ACH", "ach", "ACH Transfer"],
    "card": ["CARD", "Card", "card"],
    "internal": ["INTERNAL", "Internal", "internal transfer"],
}

DIRTY_TRANSACTION_TYPE = {
    "payment": ["PAYMENT", "Payment", "payment "],
    "transfer": ["TRANSFER", "Transfer", "xfer"],
    "withdrawal": ["WITHDRAWAL", "Withdrawal"],
    "deposit": ["DEPOSIT", "Deposit"],
}

DIRTY_COUNTRY = {
    "US": ["USA", "us", "U.S."],
    "GB": ["UK", "gb", "U.K."],
    "DE": ["de", "Deutschland"],
    "FR": ["fr", "FRA"],
}

DIRTY_CHANNEL_INDICATOR = {
    "Online": ["ONLINE", "online", "Web"],
    "In-Store": ["IN-STORE", "in store", "InStore"],
    "Mobile App": ["MOBILE APP", "mobile", "Mobile"],
    "Phone": ["PHONE", "phone"],
    "ATM": ["atm", "ATM "],
}


@dataclass
class NoiseConfig:
    enabled: bool = True
    label_flip_rate: float = 0.008
    missing_optional_field_rate: float = 0.04
    dirty_enum_rate: float = 0.03
    memo_truncate_rate: float = 0.05
    timestamp_jitter_rate: float = 0.03
    timestamp_jitter_max_hours: int = 48
    amount_jitter_rate: float = 0.02
    external_receiver_metadata_rate: float = 0.80
    merchant_name_typo_rate: float = 0.04

    @classmethod
    def from_dict(cls, raw: dict | None) -> NoiseConfig:
        if not raw:
            return cls(enabled=False)
        return cls(
            enabled=raw.get("enabled", True),
            label_flip_rate=raw.get("label_flip_rate", 0.008),
            missing_optional_field_rate=raw.get("missing_optional_field_rate", 0.04),
            dirty_enum_rate=raw.get("dirty_enum_rate", 0.03),
            memo_truncate_rate=raw.get("memo_truncate_rate", 0.05),
            timestamp_jitter_rate=raw.get("timestamp_jitter_rate", 0.03),
            timestamp_jitter_max_hours=raw.get("timestamp_jitter_max_hours", 48),
            amount_jitter_rate=raw.get("amount_jitter_rate", 0.02),
            external_receiver_metadata_rate=raw.get("external_receiver_metadata_rate", 0.80),
            merchant_name_typo_rate=raw.get("merchant_name_typo_rate", 0.04),
        )


def _maybe(rng: np.random.Generator, rate: float) -> bool:
    return rng.random() < rate


def _dirty_value(rng: np.random.Generator, value: str, mapping: dict[str, list[str]]) -> str:
    variants = mapping.get(value)
    if not variants:
        return value
    return str(rng.choice(variants))


def _typo_name(rng: np.random.Generator, name: str) -> str:
    if len(name) < 4:
        return name
    idx = int(rng.integers(1, len(name) - 1))
    chars = list(name)
    chars[idx] = str(rng.choice(list("abcdefghijklmnopqrstuvwxyz")))
    return "".join(chars)


class TransactionNoiseApplicator:
    def __init__(self, config: NoiseConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return df

        out = df.copy()
        stats: dict[str, int] = {}

        stats["label_flips"] = self._apply_label_noise(out)
        stats["external_receiver_nulled"] = self._null_external_receiver_metadata(out)
        stats["missing_injections"] = self._inject_missing_values(out)
        stats["dirty_enums"] = self._dirty_enums(out)
        stats["memo_truncated"] = self._truncate_memos(out)
        stats["timestamp_jittered"] = self._jitter_timestamps(out)
        stats["amount_jittered"] = self._jitter_amounts(out)
        stats["merchant_typos"] = self._typo_merchant_names(out)

        self.stats = stats
        return out

    def _apply_label_noise(self, df: pd.DataFrame) -> int:
        n_flip = int(round(len(df) * self.config.label_flip_rate))
        if n_flip == 0:
            return 0
        indices = self.rng.choice(len(df), n_flip, replace=False)
        df.loc[indices, "is_fraud"] = ~df.loc[indices, "is_fraud"].astype(bool)
        return n_flip

    def _null_external_receiver_metadata(self, df: pd.DataFrame) -> int:
        """Banks rarely know receiver age/shell status for external counterparties."""
        cross_border = df["sender_country"] != df["receiver_country"]
        mask = cross_border | (self.rng.random(len(df)) < 0.35)
        apply_mask = mask & (self.rng.random(len(df)) < self.config.external_receiver_metadata_rate)
        df.loc[apply_mask, "receiver_account_age_days"] = pd.NA
        if "receiver_is_shell_company" in df.columns:
            df["receiver_is_shell_company"] = df["receiver_is_shell_company"].astype(object)
            df.loc[apply_mask, "receiver_is_shell_company"] = None
        return int(apply_mask.sum())

    def _inject_missing_values(self, df: pd.DataFrame) -> int:
        count = 0
        for column in OPTIONAL_NULLABLE_COLUMNS:
            if column not in df.columns:
                continue
            eligible = df[column].notna()
            n = eligible.sum()
            if n == 0:
                continue
            drop_mask = eligible & (
                self.rng.random(len(df)) < self.config.missing_optional_field_rate
            )
            count += int(drop_mask.sum())
            df.loc[drop_mask, column] = pd.NA
        return count

    def _dirty_enums(self, df: pd.DataFrame) -> int:
        count = 0
        for idx in df.index:
            if not _maybe(self.rng, self.config.dirty_enum_rate):
                continue
            count += 1
            if _maybe(self.rng, 0.4):
                df.at[idx, "channel"] = _dirty_value(
                    self.rng, str(df.at[idx, "channel"]), DIRTY_CHANNEL
                )
            if _maybe(self.rng, 0.35):
                df.at[idx, "transaction_type"] = _dirty_value(
                    self.rng, str(df.at[idx, "transaction_type"]), DIRTY_TRANSACTION_TYPE
                )
            if _maybe(self.rng, 0.35):
                for col in ("sender_country", "receiver_country"):
                    df.at[idx, col] = _dirty_value(
                        self.rng, str(df.at[idx, col]), DIRTY_COUNTRY
                    )
            indicator = df.at[idx, "channel_indicator"]
            if pd.notna(indicator) and _maybe(self.rng, 0.3):
                df.at[idx, "channel_indicator"] = _dirty_value(
                    self.rng, str(indicator), DIRTY_CHANNEL_INDICATOR
                )
        return count

    def _truncate_memos(self, df: pd.DataFrame) -> int:
        count = 0
        for idx in df.index:
            memo = df.at[idx, "memo"]
            if not isinstance(memo, str) or len(memo) <= 20:
                continue
            if not _maybe(self.rng, self.config.memo_truncate_rate):
                continue
            cut = int(self.rng.integers(8, min(40, len(memo))))
            df.at[idx, "memo"] = memo[:cut]
            count += 1
        return count

    def _jitter_timestamps(self, df: pd.DataFrame) -> int:
        count = 0
        max_seconds = self.config.timestamp_jitter_max_hours * 3600
        for idx in df.index:
            if not _maybe(self.rng, self.config.timestamp_jitter_rate):
                continue
            offset = int(self.rng.integers(-max_seconds, max_seconds))
            df.at[idx, "timestamp"] = pd.to_datetime(df.at[idx, "timestamp"]) + pd.Timedelta(
                seconds=offset
            )
            count += 1
        return count

    def _jitter_amounts(self, df: pd.DataFrame) -> int:
        count = 0
        for idx in df.index:
            if not _maybe(self.rng, self.config.amount_jitter_rate):
                continue
            amount = float(df.at[idx, "amount"])
            noise = float(self.rng.uniform(-0.02, 0.02))
            df.at[idx, "amount"] = round(max(0.01, amount * (1 + noise)), 2)
            if "settlement_amount" in df.columns and pd.notna(df.at[idx, "settlement_amount"]):
                settlement = float(df.at[idx, "settlement_amount"])
                df.at[idx, "settlement_amount"] = round(max(0.01, settlement * (1 + noise)), 2)
            count += 1
        return count

    def _typo_merchant_names(self, df: pd.DataFrame) -> int:
        count = 0
        for idx in df.index:
            if not _maybe(self.rng, self.config.merchant_name_typo_rate):
                continue
            for col in ("merchant_legal_name", "merchant_dba_name"):
                value = df.at[idx, col]
                if isinstance(value, str):
                    df.at[idx, col] = _typo_name(self.rng, value)
                    count += 1
        return count


def print_noise_stats(applicator: TransactionNoiseApplicator) -> None:
    if not applicator.config.enabled:
        print("Noise: disabled")
        return
    print("Noise applied:")
    for key, value in applicator.stats.items():
        print(f"  {key}: {value:,}")
