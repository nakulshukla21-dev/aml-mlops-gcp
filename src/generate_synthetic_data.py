"""
Generate synthetic AML data: dimension tables + v2 transactions.

Bank clients (dim_customer / dim_account) are separate from external
counterparties (dim_counterparty / dim_counterparty_account).

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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.apply_noise import NoiseConfig, TransactionNoiseApplicator, print_noise_stats
from src.config import (
    PROJECT_ROOT,
    add_config_arguments,
    default_data_path,
    load_config_from_args,
)
from src.dimension_data import (
    REF_NAICS,
    build_ref_country_df,
    build_ref_naics_df,
    build_ref_product_df,
    build_ref_state_df,
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

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "raw_transactions_v2.json"
POST_LOAD_COLUMNS = frozenset({"ingested_at"})

COUNTRIES = list(COUNTRY_CURRENCY.keys())
HIGH_RISK_COUNTRIES = {"KY", "VG", "PA", "AE"}

TRANSACTION_TYPES = ["payment", "transfer", "withdrawal", "deposit"]
CHANNELS = ["wire", "ach", "card", "internal"]

CIP_STATUSES = ["complete", "pending", "edd"]
RISK_RATINGS = ["low", "medium", "high"]
COUNTERPARTY_ROLES = ["wire_beneficiary", "merchant", "payer", "shell_entity", "other"]

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
class Customer:
    customer_id: int
    entity_type: str
    legal_name: str
    dba_name: str | None
    residence_country: str
    residence_state: str | None
    date_of_birth: date | None
    incorporation_date: date | None
    citizenship_country: str | None
    onboarding_date: date
    cip_status: str
    risk_rating: str
    is_pep: bool | None
    is_sanctioned: bool | None
    is_shell_entity: bool | None
    naics_code: str | None


@dataclass
class BankAccount:
    account_id: str
    customer_id: int
    product_code: str
    country: str
    opened_at: datetime
    currency: str
    status: str = "active"

    def age_days(self, at: datetime) -> int:
        return max(0, (at - self.opened_at).days)


@dataclass
class Counterparty:
    counterparty_id: int
    entity_type: str | None
    counterparty_role: str
    legal_name: str
    dba_name: str | None
    country: str | None
    is_shell_entity: bool | None
    first_seen_date: date | None


@dataclass
class CounterpartyAccount:
    counterparty_account_id: str
    counterparty_id: int
    country: str | None
    first_seen_date: date | None
    account_label: str | None = None


@dataclass
class TxnLeg:
    kind: Literal["bank", "counterparty"]
    bank: BankAccount | None = None
    customer: Customer | None = None
    cp_account: CounterpartyAccount | None = None
    counterparty: Counterparty | None = None

    @property
    def country(self) -> str:
        if self.kind == "bank" and self.bank is not None:
            return self.bank.country
        if self.cp_account is not None and self.cp_account.country:
            return self.cp_account.country
        if self.counterparty is not None and self.counterparty.country:
            return self.counterparty.country
        return "US"

    @property
    def legal_name(self) -> str:
        if self.kind == "bank" and self.customer is not None:
            return self.customer.legal_name
        if self.counterparty is not None:
            return self.counterparty.legal_name
        return "Unknown"

    @property
    def dba_name(self) -> str | None:
        if self.kind == "bank" and self.customer is not None:
            return self.customer.dba_name
        if self.counterparty is not None:
            return self.counterparty.dba_name
        return None

    @property
    def is_shell(self) -> bool:
        if self.kind == "bank" and self.customer is not None:
            return bool(self.customer.is_shell_entity)
        if self.counterparty is not None:
            return bool(self.counterparty.is_shell_entity)
        return False


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

        self.customers: dict[int, Customer] = {}
        self.bank_accounts: dict[str, BankAccount] = {}
        self.counterparties: dict[int, Counterparty] = {}
        self.counterparty_accounts: dict[str, CounterpartyAccount] = {}
        self.beneficial_owners: list[dict] = []

        self._next_customer_id = 1
        self._next_counterparty_id = 1
        self._next_bo_id = 1

        self._init_customer_pool()

    def _country_weights(self) -> np.ndarray:
        weights = np.array([3 if c == "US" else 1 for c in COUNTRIES], dtype=float)
        return weights / weights.sum()

    def _random_date(self, *, min_offset_days: int = 0, max_offset_days: int | None = None) -> date:
        span_days = (self.end_date - self.start_date).days
        upper = span_days if max_offset_days is None else min(max_offset_days, span_days)
        offset = int(self.rng.integers(min_offset_days, max(min_offset_days + 1, upper + 1)))
        return (self.start_date + timedelta(days=offset)).date()

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

    def _individual_name(self) -> str:
        first = str(self.rng.choice(["James", "Maria", "Wei", "Priya", "Carlos", "Emma", "Noah", "Ava"]))
        last = str(self.rng.choice(["Smith", "Garcia", "Chen", "Patel", "Johnson", "Brown", "Kim", "Singh"]))
        return f"{first} {last}"

    def _product_for_customer(self, customer: Customer) -> str:
        if customer.entity_type == "individual":
            return str(self.rng.choice(["DDA01", "SAV01", "MMDA01"], p=[0.55, 0.30, 0.15]))
        return str(self.rng.choice(["DDA02", "DDA01", "LOC01"], p=[0.55, 0.30, 0.15]))

    def _create_customer(self, *, force_business: bool | None = None, is_shell: bool = False) -> Customer:
        entity_type = (
            "business"
            if force_business or self.rng.random() < 0.40
            else "individual"
        )
        country = str(self.rng.choice(COUNTRIES, p=self._country_weights()))
        state = None
        if country in MERCHANT_LOCATIONS:
            state = str(self.rng.choice(MERCHANT_LOCATIONS[country])[1])

        if entity_type == "individual":
            legal_name = self._individual_name()
            dba_name = None
            date_of_birth = self._random_date(max_offset_days=365 * 50)
            incorporation_date = None
            naics_code = None
            is_shell_entity = None
        else:
            legal_name, dba_name = self._assign_business_names(is_shell)
            date_of_birth = None
            incorporation_date = self._random_date(max_offset_days=365 * 20)
            naics_code = str(self.rng.choice([code for code, *_ in REF_NAICS]))
            is_shell_entity = is_shell if is_shell else (self.rng.random() < 0.01)

        customer_id = self._next_customer_id
        self._next_customer_id += 1

        customer = Customer(
            customer_id=customer_id,
            entity_type=entity_type,
            legal_name=legal_name,
            dba_name=dba_name,
            residence_country=country,
            residence_state=state,
            date_of_birth=date_of_birth,
            incorporation_date=incorporation_date,
            citizenship_country=country if entity_type == "individual" else None,
            onboarding_date=self._random_date(max_offset_days=(self.end_date - self.start_date).days),
            cip_status=str(self.rng.choice(CIP_STATUSES, p=[0.85, 0.10, 0.05])),
            risk_rating=str(self.rng.choice(RISK_RATINGS, p=[0.70, 0.22, 0.08])),
            is_pep=bool(self.rng.random() < 0.01) if entity_type == "individual" else None,
            is_sanctioned=bool(self.rng.random() < 0.002),
            is_shell_entity=is_shell_entity,
            naics_code=naics_code,
        )
        self.customers[customer_id] = customer
        return customer

    def _create_bank_account(self, customer: Customer, opened_at: datetime) -> BankAccount:
        product_code = self._product_for_customer(customer)
        account_id = f"BA{customer.residence_country}{customer.customer_id:05d}{len(self.bank_accounts):02d}"
        account = BankAccount(
            account_id=account_id,
            customer_id=customer.customer_id,
            product_code=product_code,
            country=customer.residence_country,
            opened_at=opened_at,
            currency=COUNTRY_CURRENCY[customer.residence_country],
        )
        self.bank_accounts[account_id] = account
        return account

    def _init_customer_pool(self) -> None:
        n_customers = max(3000, int(self.n_transactions * 0.03))
        span_days = (self.end_date - self.start_date).days

        for _ in range(n_customers):
            customer = self._create_customer()
            opened_offset = int(self.rng.integers(0, max(span_days, 1)))
            opened_at = self.start_date + timedelta(days=opened_offset)
            n_accounts = int(self.rng.integers(1, 4))
            for _ in range(n_accounts):
                self._create_bank_account(customer, opened_at)

        self._generate_beneficial_owners()

    def _generate_beneficial_owners(self) -> None:
        business_customers = [c for c in self.customers.values() if c.entity_type == "business"]
        for business in business_customers:
            if self.rng.random() > 0.35:
                continue
            owner_count = int(self.rng.integers(1, 3))
            remaining_pct = 100.0
            for _ in range(owner_count):
                if remaining_pct <= 0:
                    break
                ownership_pct = float(self.rng.uniform(10, min(60, remaining_pct)))
                remaining_pct -= ownership_pct
                owner_is_bank_client = self.rng.random() < 0.55
                owner_customer_id = None
                owner_counterparty_id = None
                if owner_is_bank_client:
                    owner = self._create_customer(force_business=False)
                    owner_customer_id = owner.customer_id
                else:
                    cp, _ = self._new_counterparty(
                        role="other",
                        country=business.residence_country,
                        is_shell=False,
                        first_seen=self._random_date(),
                    )
                    owner_counterparty_id = cp.counterparty_id

                self.beneficial_owners.append(
                    {
                        "beneficial_owner_id": self._next_bo_id,
                        "business_customer_id": business.customer_id,
                        "owner_customer_id": owner_customer_id,
                        "owner_counterparty_id": owner_counterparty_id,
                        "owner_is_bank_client": owner_is_bank_client,
                        "ownership_pct": round(ownership_pct, 2),
                        "control_type": str(self.rng.choice(["ownership", "voting", "other_control"])),
                        "is_pep": bool(self.rng.random() < 0.02),
                        "effective_from": business.onboarding_date.isoformat(),
                        "effective_to": None,
                    }
                )
                self._next_bo_id += 1

    def _new_counterparty(
        self,
        *,
        role: str,
        country: str,
        is_shell: bool,
        first_seen: date,
    ) -> tuple[Counterparty, CounterpartyAccount]:
        legal_name, dba_name = self._assign_business_names(is_shell)
        counterparty_id = self._next_counterparty_id
        self._next_counterparty_id += 1

        counterparty = Counterparty(
            counterparty_id=counterparty_id,
            entity_type="business" if is_shell or self.rng.random() < 0.7 else "unknown",
            counterparty_role=role,
            legal_name=legal_name,
            dba_name=dba_name,
            country=country,
            is_shell_entity=is_shell,
            first_seen_date=first_seen,
        )
        self.counterparties[counterparty_id] = counterparty

        account_id = f"CPA{country}{counterparty_id:05d}"
        cp_account = CounterpartyAccount(
            counterparty_account_id=account_id,
            counterparty_id=counterparty_id,
            country=country,
            first_seen_date=first_seen,
            account_label=f"EXT-{uuid.uuid4().hex[:8].upper()}",
        )
        self.counterparty_accounts[account_id] = cp_account
        return counterparty, cp_account

    def _bank_leg(self, account: BankAccount) -> TxnLeg:
        return TxnLeg(
            kind="bank",
            bank=account,
            customer=self.customers[account.customer_id],
        )

    def _counterparty_leg(self, cp_account: CounterpartyAccount) -> TxnLeg:
        return TxnLeg(
            kind="counterparty",
            cp_account=cp_account,
            counterparty=self.counterparties[cp_account.counterparty_id],
        )

    def _new_shell_counterparty(self, country: str, opened_at: datetime) -> CounterpartyAccount:
        _, cp_account = self._new_counterparty(
            role="shell_entity",
            country=country,
            is_shell=True,
            first_seen=opened_at.date(),
        )
        return cp_account

    def _merchant_counterparty(self, country: str) -> CounterpartyAccount:
        existing = [
            acct
            for acct in self.counterparty_accounts.values()
            if self.counterparties[acct.counterparty_id].counterparty_role == "merchant"
            and acct.country == country
        ]
        if existing and self.rng.random() < 0.6:
            return self.rng.choice(existing)

        _, cp_account = self._new_counterparty(
            role="merchant",
            country=country,
            is_shell=False,
            first_seen=self._random_date(),
        )
        return cp_account

    def _random_timestamp(self) -> datetime:
        span_seconds = int((self.end_date - self.start_date).total_seconds())
        offset = int(self.rng.integers(0, max(span_seconds, 1)))
        return self.start_date + timedelta(seconds=offset)

    def _pick_bank_accounts(self, n: int) -> list[BankAccount]:
        ids = list(self.bank_accounts.keys())
        picked = self.rng.choice(ids, n, replace=n > len(ids))
        return [self.bank_accounts[i] for i in picked]

    def _pick_bank_account_pair(
        self,
        *,
        same_country_bias: float = 0.0,
    ) -> tuple[BankAccount, BankAccount]:
        if self.rng.random() < same_country_bias:
            country = str(self.rng.choice(COUNTRIES, p=self._country_weights()))
            country_accounts = [a for a in self.bank_accounts.values() if a.country == country]
            if len(country_accounts) >= 2:
                pair = self.rng.choice(country_accounts, 2, replace=False)
                return pair[0], pair[1]

        sender, receiver = self._pick_bank_accounts(2)
        while sender.account_id == receiver.account_id:
            sender, receiver = self._pick_bank_accounts(2)
        return sender, receiver

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
        sender: TxnLeg,
        receiver: TxnLeg,
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
        receiver: TxnLeg,
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
        sender: TxnLeg,
        receiver: TxnLeg,
        timestamp: datetime,
        channel: str,
        is_fraud: bool,
    ) -> dict:
        transaction_currency = COUNTRY_CURRENCY[sender.country]
        receiver_currency = COUNTRY_CURRENCY.get(receiver.country, transaction_currency)
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

    def _account_columns(self, sender: TxnLeg, receiver: TxnLeg) -> dict:
        return {
            "sender_account_id": sender.bank.account_id if sender.kind == "bank" and sender.bank else None,
            "sender_counterparty_account_id": (
                sender.cp_account.counterparty_account_id
                if sender.kind == "counterparty" and sender.cp_account
                else None
            ),
            "receiver_account_id": (
                receiver.bank.account_id if receiver.kind == "bank" and receiver.bank else None
            ),
            "receiver_counterparty_account_id": (
                receiver.cp_account.counterparty_account_id
                if receiver.kind == "counterparty" and receiver.cp_account
                else None
            ),
            "payment_sender_country": sender.country,
            "payment_receiver_country": receiver.country,
        }

    def _make_row(
        self,
        sender: TxnLeg,
        receiver: TxnLeg,
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
            **self._account_columns(sender, receiver),
            "amount": round(float(amount), 2),
            "transaction_currency": COUNTRY_CURRENCY[sender.country],
            "transaction_type": transaction_type,
            "channel": channel,
            **payment_context,
            "merchant_legal_name": merchant_legal_name,
            "merchant_dba_name": merchant_dba_name,
            "payment_reference": payment_reference,
            "memo": memo,
            **settlement,
            "typology": typology,
            "is_fraud": is_fraud,
        }

    def _maybe_card_receiver(self, channel: str, sender_account: BankAccount) -> TxnLeg:
        if channel == "card" and self.rng.random() < 0.85:
            merchant_country = sender_account.country
            if self.rng.random() < 0.12:
                merchant_country = str(self.rng.choice(COUNTRIES, p=self._country_weights()))
            cp_account = self._merchant_counterparty(merchant_country)
            return self._counterparty_leg(cp_account)
        receiver_account = self._pick_bank_accounts(1)[0]
        while receiver_account.account_id == sender_account.account_id:
            receiver_account = self._pick_bank_accounts(1)[0]
        return self._bank_leg(receiver_account)

    def _generate_legitimate(self, n: int) -> list[dict]:
        rows: list[dict] = []
        for _ in range(n):
            sender_account, receiver_account = self._pick_bank_account_pair(same_country_bias=0.82)
            channel = str(self.rng.choice(CHANNELS, p=[0.35, 0.4, 0.15, 0.1]))
            sender = self._bank_leg(sender_account)
            receiver = (
                self._maybe_card_receiver(channel, sender_account)
                if channel == "card"
                else self._bank_leg(receiver_account)
            )
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
                    channel=channel,
                    typology="legitimate",
                    is_fraud=False,
                )
            )
        return rows

    def _generate_smurfing(self, n_clusters: int) -> list[dict]:
        rows: list[dict] = []
        txs_per_cluster = 12

        for _ in range(n_clusters):
            receiver_account = self._pick_bank_accounts(1)[0]
            receiver = self._bank_leg(receiver_account)
            cluster_start = self._random_timestamp()
            senders = self._pick_bank_accounts(txs_per_cluster)

            for i, sender_account in enumerate(senders):
                timestamp = cluster_start + timedelta(hours=int(i * self.rng.integers(1, 4)))
                amount = float(self.rng.uniform(8_500, 9_950))
                rows.append(
                    self._make_row(
                        sender=self._bank_leg(sender_account),
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
            origin_account, destination_account = self._pick_bank_account_pair()
            chain_start = self._random_timestamp()
            current = self._bank_leg(origin_account)
            shell_country = str(self.rng.choice(list(HIGH_RISK_COUNTRIES)))

            for hop in range(hops):
                if hop < hops - 1:
                    cp_account = self._new_shell_counterparty(
                        shell_country,
                        chain_start - timedelta(days=int(self.rng.integers(30, 180))),
                    )
                    next_leg = self._counterparty_leg(cp_account)
                else:
                    next_leg = self._bank_leg(destination_account)

                timestamp = chain_start + timedelta(hours=hop * 6)
                amount = float(self.rng.uniform(20_000, 150_000))
                rows.append(
                    self._make_row(
                        sender=current,
                        receiver=next_leg,
                        amount=amount,
                        timestamp=timestamp,
                        transaction_type="transfer",
                        channel="wire",
                        typology="layering",
                        is_fraud=True,
                    )
                )
                current = next_leg
        return rows

    def _generate_round_tripping(self, n_patterns: int) -> list[dict]:
        rows: list[dict] = []

        for _ in range(n_patterns):
            entity_account = self._pick_bank_accounts(1)[0]
            entity = self._bank_leg(entity_account)
            offshore = self._counterparty_leg(
                self._new_shell_counterparty(
                    str(self.rng.choice(list(HIGH_RISK_COUNTRIES))),
                    self._random_timestamp() - timedelta(days=90),
                )
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
            funnel_account = self._pick_bank_accounts(1)[0]
            funnel = self._bank_leg(funnel_account)
            inbound_count = 15
            funnel_start = self._random_timestamp()
            inbound_total = 0.0

            for i in range(inbound_count):
                sender_account = self._pick_bank_accounts(1)[0]
                amount = float(self.rng.uniform(1_000, 8_000))
                inbound_total += amount
                rows.append(
                    self._make_row(
                        sender=self._bank_leg(sender_account),
                        receiver=funnel,
                        amount=amount,
                        timestamp=funnel_start + timedelta(hours=i * 2),
                        transaction_type="payment",
                        channel="ach",
                        typology="funnel_account",
                        is_fraud=True,
                    )
                )

            shell_country = str(self.rng.choice(list(HIGH_RISK_COUNTRIES)))
            destination = self._counterparty_leg(
                self._new_shell_counterparty(shell_country, funnel_start)
            )
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

    def customers_df(self) -> pd.DataFrame:
        rows = []
        for customer in self.customers.values():
            rows.append(
                {
                    "customer_id": customer.customer_id,
                    "entity_type": customer.entity_type,
                    "legal_name": customer.legal_name,
                    "dba_name": customer.dba_name,
                    "date_of_birth": (
                        customer.date_of_birth.isoformat() if customer.date_of_birth else None
                    ),
                    "incorporation_date": (
                        customer.incorporation_date.isoformat()
                        if customer.incorporation_date
                        else None
                    ),
                    "residence_country": customer.residence_country,
                    "residence_state": customer.residence_state,
                    "citizenship_country": customer.citizenship_country,
                    "onboarding_date": customer.onboarding_date.isoformat(),
                    "cip_status": customer.cip_status,
                    "risk_rating": customer.risk_rating,
                    "is_pep": customer.is_pep,
                    "is_sanctioned": customer.is_sanctioned,
                    "is_shell_entity": customer.is_shell_entity,
                    "naics_code": customer.naics_code,
                    "last_review_date": None,
                    "created_at": None,
                    "updated_at": None,
                }
            )
        return pd.DataFrame(rows)

    def counterparties_df(self) -> pd.DataFrame:
        rows = []
        for cp in self.counterparties.values():
            rows.append(
                {
                    "counterparty_id": cp.counterparty_id,
                    "entity_type": cp.entity_type,
                    "counterparty_role": cp.counterparty_role,
                    "legal_name": cp.legal_name,
                    "dba_name": cp.dba_name,
                    "country": cp.country,
                    "is_shell_entity": cp.is_shell_entity,
                    "is_high_risk_jurisdiction": cp.country in HIGH_RISK_COUNTRIES,
                    "first_seen_date": (
                        cp.first_seen_date.isoformat() if cp.first_seen_date else None
                    ),
                    "created_at": None,
                    "updated_at": None,
                }
            )
        return pd.DataFrame(rows)

    def bank_accounts_df(self) -> pd.DataFrame:
        rows = []
        for account in self.bank_accounts.values():
            rows.append(
                {
                    "account_id": account.account_id,
                    "customer_id": account.customer_id,
                    "product_code": account.product_code,
                    "status": account.status,
                    "opened_date": account.opened_at.date().isoformat(),
                    "closed_date": None,
                    "currency": account.currency,
                    "current_balance": round(float(self.rng.lognormal(8, 1.5)), 2),
                    "available_balance": None,
                    "is_primary": bool(self.rng.random() < 0.6),
                    "domicile_country": account.country,
                    "created_at": None,
                    "updated_at": None,
                }
            )
        return pd.DataFrame(rows)

    def counterparty_accounts_df(self) -> pd.DataFrame:
        rows = []
        for account in self.counterparty_accounts.values():
            rows.append(
                {
                    "counterparty_account_id": account.counterparty_account_id,
                    "counterparty_id": account.counterparty_id,
                    "account_label": account.account_label,
                    "country": account.country,
                    "first_seen_date": (
                        account.first_seen_date.isoformat() if account.first_seen_date else None
                    ),
                    "created_at": None,
                    "updated_at": None,
                }
            )
        return pd.DataFrame(rows)

    def beneficial_owners_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.beneficial_owners)


def schema_column_order(schema_path: Path, *, exclude: frozenset[str] = POST_LOAD_COLUMNS) -> list[str]:
    with schema_path.open(encoding="utf-8") as f:
        fields = json.load(f)
    return [field["name"] for field in fields if field["name"] not in exclude]


def csv_column_order() -> list[str]:
    return schema_column_order(SCHEMA_PATH)


def dimensions_output_dir(config: dict) -> Path:
    rel = config.get("storage", {}).get("dimensions_dir", "dimensions")
    return PROJECT_ROOT / "data" / rel


def write_dimension_tables(generator: SyntheticAMLGenerator, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "ref_country.csv": build_ref_country_df(),
        "ref_state.csv": build_ref_state_df(),
        "ref_naics.csv": build_ref_naics_df(),
        "ref_product.csv": build_ref_product_df(),
        "dim_customer.csv": generator.customers_df(),
        "dim_counterparty.csv": generator.counterparties_df(),
        "dim_account.csv": generator.bank_accounts_df(),
        "dim_counterparty_account.csv": generator.counterparty_accounts_df(),
        "beneficial_owner.csv": generator.beneficial_owners_df(),
    }
    paths: dict[str, Path] = {}
    for filename, frame in tables.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        paths[filename] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AML transactions and dimensions.")
    add_config_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to data/<output_filename> from config.",
    )
    parser.add_argument(
        "--dimensions-dir",
        type=Path,
        default=None,
        help="Directory for dimension CSVs. Defaults to data/<dimensions_dir> from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config_from_args(args)
    profile = config.get("profile", args.profile)
    output = args.output or default_data_path(config)
    dim_dir = args.dimensions_dir or dimensions_output_dir(config)

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

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[csv_column_order()]

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    dim_paths = write_dimension_tables(generator, dim_dir)

    fraud_count = int(df["is_fraud"].sum())
    print(f"Wrote {len(df):,} transactions to {output}")
    print(f"Wrote {len(dim_paths)} dimension tables to {dim_dir}")
    print(
        f"Dimensions: {len(generator.customers):,} customers, "
        f"{len(generator.bank_accounts):,} bank accounts, "
        f"{len(generator.counterparties):,} counterparties"
    )
    print(f"Fraud rate: {fraud_count / len(df):.2%} ({fraud_count:,} fraudulent)")
    print("Typology distribution:")
    print(df.groupby(["typology", "is_fraud"]).size().to_string())
    print(
        "Cross-border share: "
        f"{(df['payment_sender_country'] != df['payment_receiver_country']).mean():.1%}"
    )
    print("Channel indicator distribution:")
    print(df["channel_indicator"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
