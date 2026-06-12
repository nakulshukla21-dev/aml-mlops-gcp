"""Static reference data for dimension tables."""

from __future__ import annotations

import pandas as pd

from src.reference_data import COUNTRY_CURRENCY, MERCHANT_LOCATIONS

HIGH_RISK_COUNTRIES = {"KY", "VG", "PA", "AE"}

COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "CA": "Canada",
    "AU": "Australia",
    "SG": "Singapore",
    "HK": "Hong Kong",
    "AE": "United Arab Emirates",
    "KY": "Cayman Islands",
    "VG": "British Virgin Islands",
    "PA": "Panama",
    "CH": "Switzerland",
    "JP": "Japan",
    "IN": "India",
    "BR": "Brazil",
    "MX": "Mexico",
}

ISO_ALPHA3 = {
    "US": "USA",
    "GB": "GBR",
    "DE": "DEU",
    "FR": "FRA",
    "CA": "CAN",
    "AU": "AUS",
    "SG": "SGP",
    "HK": "HKG",
    "AE": "ARE",
    "KY": "CYM",
    "VG": "VGB",
    "PA": "PAN",
    "CH": "CHE",
    "JP": "JPN",
    "IN": "IND",
    "BR": "BRA",
    "MX": "MEX",
}

REF_PRODUCTS = [
    ("DDA01", "Consumer Checking", "deposit", "retail"),
    ("DDA02", "Business Checking", "deposit", "commercial"),
    ("SAV01", "Consumer Savings", "deposit", "retail"),
    ("MMDA01", "Money Market", "deposit", "retail"),
    ("LOC01", "Line of Credit", "lending", "commercial"),
]

REF_NAICS = [
    ("522110", "Commercial Banking", "finance", "low"),
    ("541211", "Offices of Certified Public Accountants", "professional", "low"),
    ("445110", "Supermarkets and Grocery Stores", "retail", "low"),
    ("531210", "Real Estate Agents", "real_estate", "medium"),
    ("523110", "Investment Banking", "finance", "medium"),
    ("423430", "Computer Equipment Merchant Wholesalers", "wholesale", "medium"),
    ("813410", "Civic and Social Organizations", "nonprofit", "low"),
    ("561499", "All Other Business Support Services", "services", "medium"),
]


def build_ref_country_df() -> pd.DataFrame:
    rows = []
    for code, currency in COUNTRY_CURRENCY.items():
        risk = "high" if code in HIGH_RISK_COUNTRIES else ("medium" if code in {"AE", "IN", "MX", "BR"} else "low")
        rows.append(
            {
                "country_code": code,
                "country_name": COUNTRY_NAMES[code],
                "iso_alpha3": ISO_ALPHA3.get(code),
                "risk_tier": risk,
                "is_fatf_grey": code in HIGH_RISK_COUNTRIES,
                "default_currency": currency,
            }
        )
    return pd.DataFrame(rows)


def build_ref_state_df() -> pd.DataFrame:
    rows = []
    for country_code, locations in MERCHANT_LOCATIONS.items():
        seen: set[str] = set()
        for _city, state_code in locations:
            if state_code in seen:
                continue
            seen.add(state_code)
            rows.append(
                {
                    "country_code": country_code,
                    "state_code": state_code,
                    "state_name": state_code,
                }
            )
    return pd.DataFrame(rows)


def build_ref_naics_df() -> pd.DataFrame:
    return pd.DataFrame(
        REF_NAICS,
        columns=["naics_code", "naics_title", "sector", "aml_risk_tier"],
    )


def build_ref_product_df() -> pd.DataFrame:
    return pd.DataFrame(
        REF_PRODUCTS,
        columns=["product_code", "product_name", "product_type", "product_category"],
    )
