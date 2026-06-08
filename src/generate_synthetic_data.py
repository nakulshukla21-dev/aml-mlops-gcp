"""
Generate synthetic AML transaction data with realistic fraud typologies.

Typologies modeled:
  - smurfing: many sub-threshold payments into one account
  - layering: multi-hop transfers through shell companies
  - round_tripping: outbound cross-border flow returning to origin entity
  - funnel_account: high fan-in followed by a large outbound transfer
  - legitimate: baseline non-fraud activity

typology is included for evaluation; strip before model training.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.apply_noise import NoiseConfig, TransactionNoiseApplicator, print_noise_stats
from src.config import (
    PROJECT_ROOT,
    add_config_arguments,
    default_data_path,
    load_config_from_args,
)
from src.reference_data import (
    CONFUSING_DBA_NAMES,
    COUNTRY_CURRENCY,
    FRAUD_MEMOS,
    FX_TO_USD,
    GENERIC_SHELL_NAMES,
    LEGIT_MEMOS,
    MERCHANT_LOCATIONS,
    NAME_STEMS,
    NAME_SUFFIXES,
    VAGUE_MEMOS,
)

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "raw_transactions.json"
POST_LOAD_COLUMNS = frozenset({"ingested_at"})

COUNTRIES = list(COUNTRY_CURRENCY.keys())
HIGH_RISK_COUNTRIES = {"KY", "VG", "PA", "AE"}

TRANSACTION_TYPES = ["payment", "transfer", "withdrawal", "deposit"]
CHANNELS = ["wire", "ach", "card", "internal"]

SWIFT_BICS = {
    "US": "CHASUS33",
    "GB": "BARCGB22",
    "DE": "DEUTDEFF",
    "FR": "BNPAFRPP",
    "CA": "ROYCCAT2",
    "AU": "CTBAAU2S",
    "SG": "DBSSSGSG",
    "HK": "HSBCHKHH",
    "AE": "NBADAEAA",
    "KY": "KYCBKYKY",
    "VG": "VPBVVGVG",
    "PA": "BAGEPAPA",
    "CH": "UBSWCHZH",
    "JP": "BOTKJPJT",
    "IN": "HDFCINBB",
    "BR": "ITAUBRSP",
    "MX": "BNMXMXMM",
}


@dataclass
class Account:
    account_id: str
    country: str
    legal_name: str
    dba_name: str | None
    opened_at: datetime
    is_shell_company: bool = False

    def age_days(self, at: datetime) -> int:
        return max(0, (at - self.opened_at).days)


class SyntheticAMLGenerator:
    def __init__(
        self,
        n_transactions: int,
        fraud_rate: float,
        start_date: datetime,
        end_date: datetime,
        seed: int = 42,
    ) -> None:
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.start_date = start_date
        self.end_date = end_date
        self.rng = np.random.default_rng(seed)
        self.accounts: dict[str, Account] = {}
        self._init_account_pool()

    def _country_weights(self) -> np.ndarray:
        weights = np.array([3 if c == "US" else 1 for c in COUNTRIES], dtype=float)
        return weights / weights.sum()

    def _merchant_location(self, country: str) -> tuple[str, str]:
        city, state = self.rng.choice(MERCHANT_LOCATIONS[country])
        return str(city), str(state)

    def _format_legal_name(self, stem: str, suffix: str) -> str:
        style = str(self.rng.choice(["standard", "upper", "comma", "period"]))
        base = f"{stem} {suffix}"
        if style == "upper":
            return base.upper()
        if style == "comma":
            return f"{stem}, {suffix}"
        if style == "period" and suffix in {"Inc", "Corp", "Ltd"}:
            return f"{stem} {suffix}."
        return base

    def _assign_business_names(self, is_shell: bool) -> tuple[str, str | None]:
        if is_shell:
            legal_name = str(self.rng.choice(GENERIC_SHELL_NAMES))
            dba_name = (
                str(self.rng.choice(CONFUSING_DBA_NAMES))
                if self.rng.random() < 0.55
                else None
            )
            return legal_name, dba_name

        stem = str(self.rng.choice(NAME_STEMS))
        suffix = str(self.rng.choice(NAME_SUFFIXES))
        legal_name = self._format_legal_name(stem, suffix)
        if self.rng.random() < 0.04:
            legal_name = legal_name.replace(stem, f"{stem} Trading")

        dba_name = None
        if self.rng.random() < 0.38:
            dba_name = str(self.rng.choice(CONFUSING_DBA_NAMES))
        return legal_name, dba_name

    def _init_account_pool(self) -> None:
        n_accounts = max(8000, int(self.n_transactions * 0.08))
        span_days = (self.end_date - self.start_date).days

        for i in range(n_accounts):
            country = str(self.rng.choice(COUNTRIES, p=self._country_weights()))
            opened_offset = int(self.rng.integers(0, max(span_days, 1)))
            opened_at = self.start_date + timedelta(days=opened_offset)
            is_shell = country in HIGH_RISK_COUNTRIES and self.rng.random() < 0.35
            legal_name, dba_name = self._assign_business_names(is_shell)
            account_id = f"ACC{country}{i:05d}"
            self.accounts[account_id] = Account(
                account_id=account_id,
                country=country,
                legal_name=legal_name,
                dba_name=dba_name,
                opened_at=opened_at,
                is_shell_company=is_shell,
            )

    def _random_timestamp(self) -> datetime:
        span_seconds = int((self.end_date - self.start_date).total_seconds())
        offset = int(self.rng.integers(0, max(span_seconds, 1)))
        return self.start_date + timedelta(seconds=offset)

    def _pick_account_pair(
        self,
        *,
        same_country_bias: float = 0.0,
        shell_bias: float = 0.0,
    ) -> tuple[Account, Account]:
        if self.rng.random() < same_country_bias:
            country = str(self.rng.choice(COUNTRIES, p=self._country_weights()))
            country_accounts = [a for a in self.accounts.values() if a.country == country]
            if len(country_accounts) >= 2:
                pair = self.rng.choice(country_accounts, 2, replace=False)
                return pair[0], pair[1]

        sender, receiver = self._pick_accounts(2, shell_bias=shell_bias)
        while sender.account_id == receiver.account_id:
            sender, receiver = self._pick_accounts(2, shell_bias=shell_bias)
        return sender, receiver

    def _pick_accounts(self, n: int, *, shell_bias: float = 0.0) -> list[Account]:
        ids = list(self.accounts.keys())
        if shell_bias > 0:
            shell_ids = [a.account_id for a in self.accounts.values() if a.is_shell_company]
            if shell_ids and self.rng.random() < shell_bias:
                chosen = [self.accounts[self.rng.choice(shell_ids)]]
                remaining = n - 1
                if remaining > 0:
                    chosen.extend(self.accounts[i] for i in self.rng.choice(ids, remaining, replace=False))
                return chosen
        picked = self.rng.choice(ids, n, replace=n > len(ids))
        return [self.accounts[i] for i in picked]

    def _new_shell_account(self, country: str, opened_at: datetime) -> Account:
        legal_name, dba_name = self._assign_business_names(is_shell=True)
        account_id = f"ACC{country}S{uuid.uuid4().hex[:6].upper()}"
        account = Account(
            account_id=account_id,
            country=country,
            legal_name=legal_name,
            dba_name=dba_name,
            opened_at=opened_at,
            is_shell_company=True,
        )
        self.accounts[account_id] = account
        return account

    def _pos_entry_mode(self, channel_indicator: str) -> str | None:
        if channel_indicator == "In-Store":
            return str(
                self.rng.choice(
                    ["Chip/EMV", "Contactless/Tap", "Magstripe", "Manually Keyed"],
                    p=[0.50, 0.30, 0.12, 0.08],
                )
            )
        if channel_indicator == "ATM":
            return str(
                self.rng.choice(
                    ["Chip/EMV", "Contactless/Tap", "Magstripe"],
                    p=[0.82, 0.13, 0.05],
                )
            )
        if channel_indicator in {"Online", "Mobile App"}:
            return str(
                self.rng.choice(
                    ["Manually Keyed", "Contactless/Tap", "Chip/EMV"],
                    p=[0.72, 0.22, 0.06],
                )
            )
        if channel_indicator == "Phone":
            return "Manually Keyed"
        return None

    def _payment_channel_context(
        self,
        rail: str,
        sender: Account,
        receiver: Account,
        transaction_type: str,
        is_fraud: bool,
    ) -> dict:
        empty = {
            "channel_indicator": None,
            "terminal_id": None,
            "atm_id": None,
            "merchant_city": None,
            "merchant_state": None,
            "merchant_country": None,
            "pos_entry_mode": None,
        }

        if rail == "wire":
            empty["channel_indicator"] = str(self.rng.choice(["Online", "Phone"], p=[0.65, 0.35]))
            return empty

        if rail == "ach":
            empty["channel_indicator"] = str(
                self.rng.choice(["Online", "Mobile App", "Phone"], p=[0.50, 0.35, 0.15])
            )
            return empty

        if rail == "internal":
            empty["channel_indicator"] = "Online"
            return empty

        if rail != "card":
            return empty

        if transaction_type == "withdrawal":
            channel_indicator = "ATM"
        else:
            channel_indicator = str(
                self.rng.choice(
                    ["In-Store", "Online", "Mobile App", "ATM", "Phone"],
                    p=[0.36, 0.28, 0.20, 0.11, 0.05],
                )
            )

        merchant_country = receiver.country
        if is_fraud and self.rng.random() < 0.30:
            merchant_country = str(self.rng.choice(list(HIGH_RISK_COUNTRIES)))

        if channel_indicator == "ATM":
            merchant_country = sender.country
            if is_fraud and self.rng.random() < 0.25:
                merchant_country = str(self.rng.choice(COUNTRIES))

        merchant_city, merchant_state = self._merchant_location(merchant_country)
        terminal_id = None
        atm_id = None

        if channel_indicator == "In-Store":
            terminal_id = f"TERM{merchant_country}{uuid.uuid4().hex[:8].upper()}"
        elif channel_indicator == "ATM":
            atm_id = f"ATM{merchant_country}{uuid.uuid4().hex[:6].upper()}"

        return {
            "channel_indicator": channel_indicator,
            "terminal_id": terminal_id,
            "atm_id": atm_id,
            "merchant_city": merchant_city,
            "merchant_state": merchant_state,
            "merchant_country": merchant_country,
            "pos_entry_mode": self._pos_entry_mode(channel_indicator),
        }

    def _payment_reference(self, rail: str) -> str | None:
        fill_rates = {"wire": 0.88, "ach": 0.72, "card": 0.42, "internal": 0.18}
        if self.rng.random() > fill_rates.get(rail, 0.0):
            return None
        ref_type = str(self.rng.choice(["invoice", "po", "wire"], p=[0.45, 0.30, 0.25]))
        if ref_type == "invoice":
            return f"INV-2024-{int(self.rng.integers(1000, 9999))}"
        if ref_type == "po":
            return f"PO-{int(self.rng.integers(100000, 999999))}"
        return f"WIRE-{uuid.uuid4().hex[:8].upper()}"

    def _memo(
        self,
        rail: str,
        typology: str,
        is_fraud: bool,
        payment_reference: str | None,
    ) -> str | None:
        fill_rates = {"wire": 0.92, "ach": 0.78, "card": 0.52, "internal": 0.12}
        if self.rng.random() > fill_rates.get(rail, 0.0):
            return None

        if self.rng.random() < 0.08:
            return str(self.rng.choice(VAGUE_MEMOS))

        if is_fraud and self.rng.random() < 0.65:
            return str(self.rng.choice(FRAUD_MEMOS))

        template = str(self.rng.choice(LEGIT_MEMOS))
        ref = payment_reference or f"{int(self.rng.integers(1000, 9999))}"
        return template.format(
            ref=ref,
            week=int(self.rng.integers(1, 53)),
            month=str(self.rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])),
            q=int(self.rng.integers(1, 5)),
        )

    def _merchant_names(
        self,
        rail: str,
        receiver: Account,
        has_merchant_geo: bool,
    ) -> tuple[str | None, str | None]:
        if rail == "internal":
            return None, None
        if rail == "card" and not has_merchant_geo:
            return None, None
        return receiver.legal_name, receiver.dba_name

    def _settlement_lag_days(self, channel: str, cross_border: bool) -> int:
        if channel == "internal":
            return 0
        if channel == "card":
            return int(self.rng.integers(0, 2))
        if channel == "ach":
            return int(self.rng.integers(1, 3))
        if cross_border:
            return int(self.rng.integers(1, 4))
        return int(self.rng.integers(0, 2))

    def _clearing_system(self, channel: str, sender_country: str, receiver_country: str) -> str:
        if channel == "internal":
            return "INTERNAL"
        if channel == "ach":
            return "ACH"
        if channel == "card":
            return "CARD_NETWORK"
        if sender_country != receiver_country:
            return "SWIFT"
        if sender_country in {"DE", "FR"} and receiver_country in {"DE", "FR"}:
            return "SEPA"
        if sender_country == "GB" and receiver_country == "GB":
            return "CHAPS"
        if sender_country == "US" and receiver_country == "US":
            return "FEDWIRE"
        return "SWIFT"

    def _settlement_fields(
        self,
        amount: float,
        sender: Account,
        receiver: Account,
        timestamp: datetime,
        channel: str,
        is_fraud: bool,
    ) -> dict:
        transaction_currency = COUNTRY_CURRENCY[sender.country]
        receiver_currency = COUNTRY_CURRENCY[receiver.country]
        cross_border = sender.country != receiver.country

        if cross_border and channel == "wire":
            settlement_currency = "USD"
        else:
            settlement_currency = receiver_currency

        if transaction_currency == settlement_currency:
            fx_rate = 1.0
            settlement_amount = round(amount, 2)
        else:
            source_usd = amount / FX_TO_USD[transaction_currency]
            fx_rate = round(FX_TO_USD[transaction_currency] / FX_TO_USD[settlement_currency], 6)
            settlement_amount = round(source_usd * FX_TO_USD[settlement_currency], 2)

        lag_days = self._settlement_lag_days(channel, cross_border)
        settlement_date = (timestamp + timedelta(days=lag_days)).date().isoformat()

        if is_fraud and self.rng.random() < 0.08:
            settlement_status = str(self.rng.choice(["pending", "failed"], p=[0.7, 0.3]))
        elif self.rng.random() < 0.02:
            settlement_status = str(self.rng.choice(["pending", "returned"], p=[0.6, 0.4]))
        else:
            settlement_status = "settled"

        clearing_system = self._clearing_system(channel, sender.country, receiver.country)
        correspondent_bic = None
        if clearing_system == "SWIFT":
            correspondent_bic = SWIFT_BICS.get(receiver.country)

        return {
            "settlement_currency": settlement_currency,
            "settlement_amount": settlement_amount,
            "fx_rate": fx_rate,
            "settlement_date": settlement_date,
            "settlement_status": settlement_status,
            "clearing_system": clearing_system,
            "correspondent_bic": correspondent_bic,
        }

    def _make_row(
        self,
        sender: Account,
        receiver: Account,
        amount: float,
        timestamp: datetime,
        transaction_type: str,
        channel: str,
        typology: str,
        is_fraud: bool,
    ) -> dict:
        payment_context = self._payment_channel_context(
            channel, sender, receiver, transaction_type, is_fraud
        )
        settlement = self._settlement_fields(
            amount, sender, receiver, timestamp, channel, is_fraud
        )
        has_merchant_geo = payment_context.get("merchant_country") is not None
        merchant_legal_name, merchant_dba_name = self._merchant_names(
            channel, receiver, has_merchant_geo
        )
        if merchant_legal_name is None and channel in {"wire", "ach"}:
            merchant_legal_name, merchant_dba_name = receiver.legal_name, receiver.dba_name
        payment_reference = self._payment_reference(channel)
        memo = self._memo(channel, typology, is_fraud, payment_reference)
        return {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "timestamp": timestamp.isoformat(),
            "sender_account": sender.account_id,
            "receiver_account": receiver.account_id,
            "amount": round(float(amount), 2),
            "transaction_currency": COUNTRY_CURRENCY[sender.country],
            "transaction_type": transaction_type,
            "channel": channel,
            **payment_context,
            "merchant_legal_name": merchant_legal_name,
            "merchant_dba_name": merchant_dba_name,
            "payment_reference": payment_reference,
            "memo": memo,
            "sender_country": sender.country,
            "receiver_country": receiver.country,
            **settlement,
            "sender_account_age_days": sender.age_days(timestamp),
            "receiver_account_age_days": receiver.age_days(timestamp),
            "receiver_is_shell_company": receiver.is_shell_company,
            "typology": typology,
            "is_fraud": is_fraud,
        }

    def _generate_legitimate(self, n: int) -> list[dict]:
        rows: list[dict] = []
        for _ in range(n):
            sender, receiver = self._pick_account_pair(same_country_bias=0.82)
            timestamp = self._random_timestamp()
            amount = float(self.rng.lognormal(mean=7.0, sigma=1.1))
            amount = min(max(amount, 5.0), 250_000.0)
            rows.append(
                self._make_row(
                    sender=sender,
                    receiver=receiver,
                    amount=amount,
                    timestamp=timestamp,
                    transaction_type=str(self.rng.choice(TRANSACTION_TYPES)),
                    channel=str(self.rng.choice(CHANNELS, p=[0.35, 0.4, 0.15, 0.1])),
                    typology="legitimate",
                    is_fraud=False,
                )
            )
        return rows

    def _generate_smurfing(self, n_clusters: int) -> list[dict]:
        rows: list[dict] = []
        txs_per_cluster = 12

        for _ in range(n_clusters):
            receiver = self._pick_accounts(1)[0]
            cluster_start = self._random_timestamp()
            senders = self._pick_accounts(txs_per_cluster)

            for i, sender in enumerate(senders):
                timestamp = cluster_start + timedelta(hours=int(i * self.rng.integers(1, 4)))
                amount = float(self.rng.uniform(8_500, 9_950))
                rows.append(
                    self._make_row(
                        sender=sender,
                        receiver=receiver,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type="deposit",
                        channel="ach",
                        typology="smurfing",
                        is_fraud=True,
                    )
                )
        return rows

    def _generate_layering(self, n_chains: int) -> list[dict]:
        rows: list[dict] = []
        hops = 4

        for _ in range(n_chains):
            origin, destination = self._pick_accounts(2)
            while origin.account_id == destination.account_id:
                origin, destination = self._pick_accounts(2)

            chain_start = self._random_timestamp()
            current = origin
            shell_country = str(self.rng.choice(list(HIGH_RISK_COUNTRIES)))

            for hop in range(hops):
                if hop < hops - 1:
                    next_account = self._new_shell_account(
                        shell_country,
                        chain_start - timedelta(days=int(self.rng.integers(30, 180))),
                    )
                else:
                    next_account = destination

                timestamp = chain_start + timedelta(hours=hop * 6)
                amount = float(self.rng.uniform(20_000, 150_000))
                rows.append(
                    self._make_row(
                        sender=current,
                        receiver=next_account,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type="transfer",
                        channel="wire",
                        typology="layering",
                        is_fraud=True,
                    )
                )
                current = next_account
        return rows

    def _generate_round_tripping(self, n_patterns: int) -> list[dict]:
        rows: list[dict] = []

        for _ in range(n_patterns):
            entity = self._pick_accounts(1)[0]
            offshore = self._new_shell_account(
                str(self.rng.choice(list(HIGH_RISK_COUNTRIES))),
                self._random_timestamp() - timedelta(days=90),
            )
            start = self._random_timestamp()
            outbound_amount = float(self.rng.uniform(50_000, 300_000))

            rows.append(
                self._make_row(
                    sender=entity,
                    receiver=offshore,
                    amount=outbound_amount,
                    timestamp=start,
                    transaction_type="transfer",
                    channel="wire",
                    typology="round_tripping",
                    is_fraud=True,
                )
            )
            rows.append(
                self._make_row(
                    sender=offshore,
                    receiver=entity,
                    amount=outbound_amount * float(self.rng.uniform(0.92, 0.99)),
                    timestamp=start + timedelta(days=int(self.rng.integers(5, 21))),
                    transaction_type="transfer",
                    channel="wire",
                    typology="round_tripping",
                    is_fraud=True,
                )
            )
        return rows

    def _generate_funnel_accounts(self, n_funnels: int) -> list[dict]:
        rows: list[dict] = []

        for _ in range(n_funnels):
            funnel = self._pick_accounts(1)[0]
            inbound_count = 15
            funnel_start = self._random_timestamp()
            inbound_total = 0.0

            for i in range(inbound_count):
                sender = self._pick_accounts(1)[0]
                amount = float(self.rng.uniform(1_000, 8_000))
                inbound_total += amount
                rows.append(
                    self._make_row(
                        sender=sender,
                        receiver=funnel,
                        amount=amount,
                        timestamp=funnel_start + timedelta(hours=i * 2),
                        transaction_type="payment",
                        channel="ach",
                        typology="funnel_account",
                        is_fraud=True,
                    )
                )

            destination = self._pick_accounts(1, shell_bias=0.8)[0]
            rows.append(
                self._make_row(
                    sender=funnel,
                    receiver=destination,
                    amount=inbound_total * float(self.rng.uniform(0.85, 0.95)),
                    timestamp=funnel_start + timedelta(hours=inbound_count * 2 + 6),
                    transaction_type="transfer",
                    channel="wire",
                    typology="funnel_account",
                    is_fraud=True,
                )
            )
        return rows

    def generate(self) -> pd.DataFrame:
        target_fraud = int(round(self.n_transactions * self.fraud_rate))
        fraud_rows: list[dict] = []

        smurf_clusters = max(1, int(target_fraud * 0.15 / 12))
        layering_chains = max(1, int(target_fraud * 0.25 / 4))
        round_trip_patterns = max(1, int(target_fraud * 0.20 / 2))
        funnel_patterns = max(1, int(target_fraud * 0.20 / 16))

        fraud_rows.extend(self._generate_smurfing(smurf_clusters))
        fraud_rows.extend(self._generate_layering(layering_chains))
        fraud_rows.extend(self._generate_round_tripping(round_trip_patterns))
        fraud_rows.extend(self._generate_funnel_accounts(funnel_patterns))

        if len(fraud_rows) > target_fraud:
            indices = self.rng.choice(len(fraud_rows), target_fraud, replace=False)
            fraud_rows = [fraud_rows[i] for i in sorted(indices)]
        elif len(fraud_rows) < target_fraud:
            extra = target_fraud - len(fraud_rows)
            fraud_rows.extend(self._generate_smurfing(max(1, extra // 12)))

        legitimate_count = self.n_transactions - len(fraud_rows)
        rows = fraud_rows + self._generate_legitimate(legitimate_count)
        self.rng.shuffle(rows)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)


def csv_column_order() -> list[str]:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        fields = json.load(f)
    return [field["name"] for field in fields if field["name"] not in POST_LOAD_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AML transactions.")
    add_config_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to data/<output_filename> from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    output = args.output or default_data_path(config)

    print(f"Profile: {profile} ({config_path.name})")

    gen_cfg = config["data_generation"]
    start = datetime.fromisoformat(gen_cfg["start_date"])
    end = datetime.fromisoformat(gen_cfg["end_date"]) + timedelta(days=1) - timedelta(seconds=1)

    generator = SyntheticAMLGenerator(
        n_transactions=gen_cfg["n_transactions"],
        fraud_rate=gen_cfg["fraud_rate"],
        start_date=start,
        end_date=end,
        seed=gen_cfg.get("random_seed", 42),
    )
    df = generator.generate()

    noise_cfg = NoiseConfig.from_dict(config.get("noise"))
    noise_applicator = TransactionNoiseApplicator(
        noise_cfg,
        seed=gen_cfg.get("random_seed", 42) + 1,
    )
    df = noise_applicator.apply(df)
    print_noise_stats(noise_applicator)

    for col in ("sender_account_age_days", "receiver_account_age_days"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[csv_column_order()]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    fraud_count = int(df["is_fraud"].sum())
    print(f"Wrote {len(df):,} transactions to {output}")
    print(f"Fraud rate: {fraud_count / len(df):.2%} ({fraud_count:,} fraudulent)")
    print("Typology distribution:")
    print(df.groupby(["typology", "is_fraud"]).size().to_string())
    print(f"Cross-border share: {(df['sender_country'] != df['receiver_country']).mean():.1%}")
    print("Channel indicator distribution:")
    print(df["channel_indicator"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
