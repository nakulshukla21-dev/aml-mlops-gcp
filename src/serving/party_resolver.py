"""Resolve authenticated payment instructions to canonical v2 transaction rows."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd

from src.serving.schemas import (
    PaymentScoreRequest,
    RawTransactionV2,
    ReceiverInstruction,
    ResolvedParties,
)


class PartyResolutionError(Exception):
    """Base error for party resolution failures."""


class SenderNotAuthorizedError(PartyResolutionError):
    """Sender account is missing, inactive, or not owned by the customer."""


class InvalidPaymentRequestError(PartyResolutionError):
    """Payment instruction failed structural validation."""


@dataclass
class DimensionTables:
    dim_account: pd.DataFrame
    dim_customer: pd.DataFrame
    dim_counterparty: pd.DataFrame
    dim_counterparty_account: pd.DataFrame


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return cleaned


def _stub_counterparty_id(reference: str, existing_ids: set[int]) -> int:
    seed = int(hashlib.sha256(reference.encode()).hexdigest()[:8], 16)
    candidate = 900_000_000 + (seed % 99_000_000)
    while candidate in existing_ids:
        candidate += 1
    return candidate


def _stub_counterparty_account_id(country: str | None, reference: str) -> str:
    country_code = (country or "XX")[:2].upper()
    digest = hashlib.sha256(reference.encode()).hexdigest()[:8].upper()
    return f"CPA{country_code}{digest}"


class PartyResolver:
    def __init__(self, dimensions: DimensionTables) -> None:
        self.dimensions = dimensions
        self._stub_counterparties: list[dict] = []
        self._stub_accounts: list[dict] = []

    @property
    def stub_counterparties(self) -> pd.DataFrame:
        if not self._stub_counterparties:
            return pd.DataFrame(
                columns=[
                    "counterparty_id",
                    "entity_type",
                    "counterparty_role",
                    "legal_name",
                    "dba_name",
                    "country",
                    "is_shell_entity",
                    "first_seen_date",
                ]
            )
        return pd.DataFrame(self._stub_counterparties)

    @property
    def stub_counterparty_accounts(self) -> pd.DataFrame:
        if not self._stub_accounts:
            return pd.DataFrame(
                columns=[
                    "counterparty_account_id",
                    "counterparty_id",
                    "account_label",
                    "country",
                    "first_seen_date",
                ]
            )
        return pd.DataFrame(self._stub_accounts)

    def resolve_payment(self, request: PaymentScoreRequest) -> RawTransactionV2:
        self._validate_sender(request.customer_id, request.sender_account_id)
        receiver_account_id, receiver_counterparty_account_id, receiver_created = self._resolve_receiver(
            request
        )

        parties = ResolvedParties(
            sender_account_id=request.sender_account_id,
            sender_counterparty_account_id=None,
            receiver_account_id=receiver_account_id,
            receiver_counterparty_account_id=receiver_counterparty_account_id,
            receiver_created=receiver_created,
        )

        payment_receiver_country = request.payment_receiver_country
        if receiver_created and request.receiver.country and not payment_receiver_country:
            payment_receiver_country = request.receiver.country

        return RawTransactionV2(
            transaction_id=request.transaction_id,
            timestamp=request.timestamp,
            sender_account_id=request.sender_account_id,
            sender_counterparty_account_id=None,
            receiver_account_id=receiver_account_id,
            receiver_counterparty_account_id=receiver_counterparty_account_id,
            amount=request.amount,
            transaction_currency=request.transaction_currency,
            transaction_type=request.transaction_type,
            channel=request.channel,
            channel_indicator=request.channel_indicator,
            terminal_id=request.terminal_id,
            atm_id=request.atm_id,
            merchant_city=request.merchant_city,
            merchant_state=request.merchant_state,
            merchant_country=request.merchant_country,
            merchant_legal_name=request.merchant_legal_name,
            merchant_dba_name=request.merchant_dba_name,
            pos_entry_mode=request.pos_entry_mode,
            payment_reference=request.payment_reference,
            memo=request.memo,
            payment_sender_country=request.payment_sender_country,
            payment_receiver_country=payment_receiver_country,
            settlement_currency=request.settlement_currency,
            settlement_amount=request.settlement_amount,
            fx_rate=request.fx_rate,
            settlement_date=request.settlement_date,
            settlement_status=request.settlement_status,
            clearing_system=request.clearing_system,
            correspondent_bic=request.correspondent_bic,
            resolved_parties=parties,
        )

    def _validate_sender(self, customer_id: int, sender_account_id: str) -> None:
        accounts = self.dimensions.dim_account
        match = accounts.loc[accounts["account_id"] == sender_account_id]
        if match.empty:
            raise SenderNotAuthorizedError(f"Unknown sender account: {sender_account_id}")

        row = match.iloc[0]
        if int(row["customer_id"]) != int(customer_id):
            raise SenderNotAuthorizedError(
                f"Account {sender_account_id} does not belong to customer {customer_id}."
            )
        status = str(row.get("status", "active")).lower()
        if status != "active":
            raise SenderNotAuthorizedError(f"Sender account {sender_account_id} is not active.")

    def _resolve_receiver(
        self, request: PaymentScoreRequest
    ) -> tuple[str | None, str | None, bool]:
        receiver = request.receiver
        if receiver.type == "bank":
            return self._resolve_bank_receiver(receiver)
        if receiver.type == "external":
            return self._resolve_external_receiver(receiver, request.timestamp.date())
        if receiver.type == "merchant":
            return self._resolve_merchant_receiver(request)
        raise InvalidPaymentRequestError(f"Unsupported receiver type: {receiver.type}")

    def _resolve_bank_receiver(self, receiver: ReceiverInstruction) -> tuple[str | None, str | None, bool]:
        if not receiver.account_id:
            raise InvalidPaymentRequestError("receiver.account_id is required for bank transfers.")
        accounts = self.dimensions.dim_account
        if accounts.loc[accounts["account_id"] == receiver.account_id].empty:
            raise InvalidPaymentRequestError(f"Unknown receiver bank account: {receiver.account_id}")
        return receiver.account_id, None, False

    def _resolve_external_receiver(
        self,
        receiver: ReceiverInstruction,
        first_seen_date,
    ) -> tuple[str | None, str | None, bool]:
        if not receiver.beneficiary_name and not receiver.account_reference:
            raise InvalidPaymentRequestError(
                "External receiver requires beneficiary_name or account_reference."
            )

        matched = self._match_counterparty_account(
            account_reference=receiver.account_reference,
            legal_name=receiver.beneficiary_name,
            dba_name=receiver.dba_name,
            country=receiver.country,
        )
        if matched is not None:
            return None, matched, False

        reference = receiver.account_reference or receiver.beneficiary_name or "unknown"
        account_id = _stub_counterparty_account_id(receiver.country, reference)
        existing_ids = set(self.dimensions.dim_counterparty["counterparty_id"].astype(int).tolist())
        existing_ids.update(int(row["counterparty_id"]) for row in self._stub_counterparties)
        counterparty_id = _stub_counterparty_id(reference, existing_ids)

        self._stub_counterparties.append(
            {
                "counterparty_id": counterparty_id,
                "entity_type": receiver.entity_type or "unknown",
                "counterparty_role": "wire_beneficiary",
                "legal_name": receiver.beneficiary_name or reference,
                "dba_name": receiver.dba_name,
                "country": receiver.country,
                "is_shell_entity": None,
                "first_seen_date": first_seen_date,
            }
        )
        self._stub_accounts.append(
            {
                "counterparty_account_id": account_id,
                "counterparty_id": counterparty_id,
                "account_label": receiver.account_reference,
                "country": receiver.country,
                "first_seen_date": first_seen_date,
            }
        )
        return None, account_id, True

    def _resolve_merchant_receiver(
        self, request: PaymentScoreRequest
    ) -> tuple[str | None, str | None, bool]:
        legal_name = request.merchant_legal_name
        dba_name = request.merchant_dba_name
        country = request.merchant_country
        if not legal_name and not dba_name:
            raise InvalidPaymentRequestError("Merchant receiver requires merchant_legal_name or merchant_dba_name.")

        matched = self._match_counterparty_account(
            account_reference=None,
            legal_name=legal_name,
            dba_name=dba_name,
            country=country,
            role="merchant",
        )
        if matched is not None:
            return None, matched, False

        reference = legal_name or dba_name or "merchant"
        account_id = _stub_counterparty_account_id(country, reference)
        existing_ids = set(self.dimensions.dim_counterparty["counterparty_id"].astype(int).tolist())
        existing_ids.update(int(row["counterparty_id"]) for row in self._stub_counterparties)
        counterparty_id = _stub_counterparty_id(reference, existing_ids)
        first_seen = request.timestamp.date()

        self._stub_counterparties.append(
            {
                "counterparty_id": counterparty_id,
                "entity_type": "business",
                "counterparty_role": "merchant",
                "legal_name": legal_name or dba_name,
                "dba_name": dba_name,
                "country": country,
                "is_shell_entity": None,
                "first_seen_date": first_seen,
            }
        )
        self._stub_accounts.append(
            {
                "counterparty_account_id": account_id,
                "counterparty_id": counterparty_id,
                "account_label": None,
                "country": country,
                "first_seen_date": first_seen,
            }
        )
        return None, account_id, True

    def _match_counterparty_account(
        self,
        *,
        account_reference: str | None,
        legal_name: str | None,
        dba_name: str | None,
        country: str | None,
        role: str | None = None,
    ) -> str | None:
        accounts = pd.concat(
            [self.dimensions.dim_counterparty_account, self.stub_counterparty_accounts],
            ignore_index=True,
        )
        counterparties = pd.concat(
            [self.dimensions.dim_counterparty, self.stub_counterparties],
            ignore_index=True,
        )
        if accounts.empty or counterparties.empty:
            return None

        joined = accounts.merge(counterparties, on="counterparty_id", how="left", suffixes=("_acct", "_cp"))

        if account_reference:
            ref_norm = account_reference.strip().lower()
            label_matches = joined[
                joined["account_label"].fillna("").str.strip().str.lower() == ref_norm
            ]
            if not label_matches.empty:
                return str(label_matches.iloc[0]["counterparty_account_id"])

        target_name = _normalize_name(legal_name)
        target_dba = _normalize_name(dba_name)
        if not target_name and not target_dba:
            return None

        for _, row in joined.iterrows():
            if role and str(row.get("counterparty_role")) != role:
                continue
            if country and row.get("country_acct") and str(row["country_acct"]) != country:
                if row.get("country_cp") and str(row["country_cp"]) != country:
                    continue
            row_name = _normalize_name(row.get("legal_name"))
            row_dba = _normalize_name(row.get("dba_name"))
            if target_name and target_name in {row_name, row_dba}:
                return str(row["counterparty_account_id"])
            if target_dba and target_dba in {row_name, row_dba}:
                return str(row["counterparty_account_id"])
        return None
