"""Request/response shapes for serving (dataclasses until FastAPI layer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Literal


ReceiverType = Literal["bank", "external", "merchant"]


@dataclass
class ReceiverInstruction:
    type: ReceiverType
    account_id: str | None = None
    beneficiary_name: str | None = None
    dba_name: str | None = None
    country: str | None = None
    account_reference: str | None = None
    entity_type: str | None = None


@dataclass
class PaymentScoreRequest:
    transaction_id: str
    timestamp: datetime
    customer_id: int
    sender_account_id: str
    amount: float
    transaction_currency: str
    transaction_type: str
    channel: str
    payment_sender_country: str
    payment_receiver_country: str
    settlement_currency: str
    settlement_amount: float
    settlement_status: str
    receiver: ReceiverInstruction
    fx_rate: float | None = 1.0
    settlement_date: date | None = None
    channel_indicator: str | None = None
    terminal_id: str | None = None
    atm_id: str | None = None
    merchant_city: str | None = None
    merchant_state: str | None = None
    merchant_country: str | None = None
    merchant_legal_name: str | None = None
    merchant_dba_name: str | None = None
    pos_entry_mode: str | None = None
    payment_reference: str | None = None
    memo: str | None = None
    clearing_system: str | None = None
    correspondent_bic: str | None = None


@dataclass
class ResolvedParties:
    sender_account_id: str | None
    sender_counterparty_account_id: str | None
    receiver_account_id: str | None
    receiver_counterparty_account_id: str | None
    receiver_created: bool = False


@dataclass
class RawTransactionV2:
    """Canonical v2 transaction row used by score_features.sql."""

    transaction_id: str
    timestamp: datetime
    amount: float
    transaction_currency: str
    transaction_type: str
    channel: str
    payment_sender_country: str
    payment_receiver_country: str
    settlement_currency: str
    settlement_amount: float
    settlement_status: str
    sender_account_id: str | None = None
    sender_counterparty_account_id: str | None = None
    receiver_account_id: str | None = None
    receiver_counterparty_account_id: str | None = None
    fx_rate: float | None = 1.0
    settlement_date: date | None = None
    channel_indicator: str | None = None
    terminal_id: str | None = None
    atm_id: str | None = None
    merchant_city: str | None = None
    merchant_state: str | None = None
    merchant_country: str | None = None
    merchant_legal_name: str | None = None
    merchant_dba_name: str | None = None
    pos_entry_mode: str | None = None
    payment_reference: str | None = None
    memo: str | None = None
    clearing_system: str | None = None
    correspondent_bic: str | None = None
    resolved_parties: ResolvedParties | None = field(default=None, repr=False)

    def validate_xor_legs(self) -> None:
        sender_set = int(self.sender_account_id is not None) + int(
            self.sender_counterparty_account_id is not None
        )
        receiver_set = int(self.receiver_account_id is not None) + int(
            self.receiver_counterparty_account_id is not None
        )
        if sender_set != 1:
            raise ValueError("Exactly one sender account leg must be set.")
        if receiver_set != 1:
            raise ValueError("Exactly one receiver account leg must be set.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawTransactionV2:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {key: data[key] for key in data if key in allowed and key != "resolved_parties"}
        if "timestamp" in payload and isinstance(payload["timestamp"], str):
            payload["timestamp"] = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
        if "settlement_date" in payload and isinstance(payload["settlement_date"], str):
            payload["settlement_date"] = date.fromisoformat(payload["settlement_date"])
        return cls(**payload)

    def to_param_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("resolved_parties", None)
        return data
