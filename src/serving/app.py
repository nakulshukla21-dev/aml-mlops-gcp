"""FastAPI application for online AML transaction scoring."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from pydantic import BaseModel, Field

from src.automl_utils import artifact_path, load_run_artifact
from src.config import load_config, resolve_config_path
from src.prediction_logging import log_online_prediction
from src.serving.dimensions import load_dimension_tables
from src.serving.feature_builder import (
    compare_feature_rows,
    fetch_offline_features,
    model_feature_columns,
    score_features,
)
from src.serving.party_resolver import (
    InvalidPaymentRequestError,
    PartyResolver,
    SenderNotAuthorizedError,
)
from src.serving.schemas import PaymentScoreRequest, RawTransactionV2, ReceiverInstruction
from src.serving.vertex_client import VertexScorer


class ReceiverPayload(BaseModel):
    type: Literal["bank", "external", "merchant"]
    account_id: str | None = None
    beneficiary_name: str | None = None
    dba_name: str | None = None
    country: str | None = None
    account_reference: str | None = None
    entity_type: str | None = None


class ScorePaymentPayload(BaseModel):
    transaction_id: str
    timestamp: datetime
    customer_id: int
    sender_account_id: str
    amount: float = Field(gt=0)
    transaction_currency: str
    transaction_type: str
    channel: str
    payment_sender_country: str
    payment_receiver_country: str
    settlement_currency: str
    settlement_amount: float
    settlement_status: str
    receiver: ReceiverPayload
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
    log_prediction: bool = True


class ValidatePayload(BaseModel):
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
    run_model: bool = False


class ScoringService:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.profile = config.get("profile", "dev")
        self.project_id = config["gcp"]["project_id"]
        self.bq_client = bigquery.Client(project=self.project_id)
        self.resolver = PartyResolver(load_dimension_tables(self.bq_client, config))
        self.vertex = VertexScorer(config)
        self.model_display_name = self._load_model_display_name()

    def _load_model_display_name(self) -> str | None:
        path = artifact_path(self.config)
        if path.exists():
            return load_run_artifact(path).get("model_display_name")
        return None

    def score_payment(self, payload: ScorePaymentPayload) -> dict[str, Any]:
        request = PaymentScoreRequest(
            transaction_id=payload.transaction_id,
            timestamp=_ensure_utc(payload.timestamp),
            customer_id=payload.customer_id,
            sender_account_id=payload.sender_account_id,
            amount=payload.amount,
            transaction_currency=payload.transaction_currency,
            transaction_type=payload.transaction_type,
            channel=payload.channel,
            payment_sender_country=payload.payment_sender_country,
            payment_receiver_country=payload.payment_receiver_country,
            settlement_currency=payload.settlement_currency,
            settlement_amount=payload.settlement_amount,
            settlement_status=payload.settlement_status,
            receiver=ReceiverInstruction(**payload.receiver.model_dump()),
            fx_rate=payload.fx_rate,
            settlement_date=payload.settlement_date,
            channel_indicator=payload.channel_indicator,
            terminal_id=payload.terminal_id,
            atm_id=payload.atm_id,
            merchant_city=payload.merchant_city,
            merchant_state=payload.merchant_state,
            merchant_country=payload.merchant_country,
            merchant_legal_name=payload.merchant_legal_name,
            merchant_dba_name=payload.merchant_dba_name,
            pos_entry_mode=payload.pos_entry_mode,
            payment_reference=payload.payment_reference,
            memo=payload.memo,
            clearing_system=payload.clearing_system,
            correspondent_bic=payload.correspondent_bic,
        )
        txn = self.resolver.resolve_payment(request)
        features = score_features(self.bq_client, self.config, txn)
        prediction = self.vertex.predict(features)
        if payload.log_prediction:
            log_online_prediction(
                self.bq_client,
                self.config,
                transaction_id=payload.transaction_id,
                predicted_is_fraud=prediction["predicted_is_fraud"],
                fraud_score=prediction["fraud_score"],
                model_resource_name=prediction["model_resource_name"],
                endpoint_resource_name=prediction["endpoint_resource_name"],
                model_display_name=self.model_display_name,
            )
        return {
            "transaction_id": payload.transaction_id,
            "fraud_score": prediction["fraud_score"],
            "is_fraud_predicted": prediction["predicted_is_fraud"],
            "risk_level": prediction["risk_level"],
            "resolved_parties": asdict(txn.resolved_parties) if txn.resolved_parties else None,
            "features_used": prediction["features_used"],
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    def validate_transaction(self, payload: ValidatePayload) -> dict[str, Any]:
        txn = RawTransactionV2.from_dict(payload.model_dump(exclude={"run_model"}))
        txn.validate_xor_legs()
        online = score_features(self.bq_client, self.config, txn)
        offline = fetch_offline_features(self.bq_client, self.config, txn.transaction_id)
        compare_keys = model_feature_columns(self.config, list(online.keys()))
        parity = (
            compare_feature_rows(online, offline, compare_columns=compare_keys)
            if offline is not None
            else {
                "matched": False,
                "max_feature_delta": None,
                "mismatched_columns": [],
                "compared_columns": compare_keys,
                "offline_row_missing": True,
            }
        )
        response: dict[str, Any] = {
            "transaction_id": txn.transaction_id,
            "parity": parity,
            "features_used": {key: online.get(key) for key in compare_keys},
        }
        if payload.run_model:
            prediction = self.vertex.predict(online)
            response["online_score"] = prediction["fraud_score"]
            response["is_fraud_predicted"] = prediction["predicted_is_fraud"]
            response["risk_level"] = prediction["risk_level"]
        return response


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _load_runtime_config() -> dict:
    from pathlib import Path

    profile = os.getenv("AML_PROFILE", "dev")
    config_path = os.getenv("AML_CONFIG_PATH")
    path = Path(config_path) if config_path else resolve_config_path(profile)
    return load_config(path)


def create_app(service: ScoringService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            app.state.service = ScoringService(_load_runtime_config())
        yield

    app = FastAPI(
        title="AML Fraud Scoring API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if service is not None:
        app.state.service = service

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        svc: ScoringService = request.app.state.service
        return {
            "status": "ok",
            "profile": svc.profile,
            "project_id": svc.project_id,
            "endpoint_resource_name": svc.vertex.endpoint_resource_name,
        }

    @app.post("/score")
    def score_payment(payload: ScorePaymentPayload, request: Request) -> dict[str, Any]:
        svc: ScoringService = request.app.state.service
        try:
            return svc.score_payment(payload)
        except SenderNotAuthorizedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (InvalidPaymentRequestError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GoogleAPIError as exc:
            raise HTTPException(status_code=503, detail=f"Dependency unavailable: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/validate")
    def validate_transaction(payload: ValidatePayload, request: Request) -> dict[str, Any]:
        svc: ScoringService = request.app.state.service
        try:
            return svc.validate_transaction(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GoogleAPIError as exc:
            raise HTTPException(status_code=503, detail=f"Dependency unavailable: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
