"""Country-level reference data for geo, currency, and payment-channel enums."""

from __future__ import annotations

# ISO country -> default currency
COUNTRY_CURRENCY: dict[str, str] = {
    "US": "USD",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "CA": "CAD",
    "AU": "AUD",
    "SG": "SGD",
    "HK": "HKD",
    "AE": "AED",
    "KY": "USD",
    "VG": "USD",
    "PA": "USD",
    "CH": "CHF",
    "JP": "JPY",
    "IN": "INR",
    "BR": "BRL",
    "MX": "MXN",
}

# Representative cities per country (city, region, postal prefix)
GEO_REGIONS: dict[str, list[tuple[str, str, str]]] = {
    "US": [
        ("New York", "NY", "100"),
        ("Los Angeles", "CA", "900"),
        ("Chicago", "IL", "606"),
        ("Miami", "FL", "331"),
        ("Houston", "TX", "770"),
    ],
    "GB": [("London", "ENG", "EC1"), ("Manchester", "ENG", "M1"), ("Edinburgh", "SCT", "EH1")],
    "DE": [("Frankfurt", "HE", "603"), ("Berlin", "BE", "101"), ("Munich", "BY", "803")],
    "FR": [("Paris", "IDF", "750"), ("Lyon", "ARA", "690"), ("Marseille", "PAC", "130")],
    "CA": [("Toronto", "ON", "M5H"), ("Vancouver", "BC", "V6B"), ("Montreal", "QC", "H2Y")],
    "AU": [("Sydney", "NSW", "2000"), ("Melbourne", "VIC", "3000"), ("Perth", "WA", "6000")],
    "SG": [("Singapore", "SG", "018"), ("Singapore", "SG", "238")],
    "HK": [("Hong Kong", "HK", "999"), ("Kowloon", "HK", "999")],
    "AE": [("Dubai", "DU", "000"), ("Abu Dhabi", "AZ", "000")],
    "KY": [("George Town", "GT", "KY1")],
    "VG": [("Road Town", "RT", "VG1")],
    "PA": [("Panama City", "8", "0801"), ("Colon", "3", "0301")],
    "CH": [("Zurich", "ZH", "8001"), ("Geneva", "GE", "1201")],
    "JP": [("Tokyo", "13", "100"), ("Osaka", "27", "530")],
    "IN": [("Mumbai", "MH", "400"), ("Delhi", "DL", "110"), ("Bengaluru", "KA", "560")],
    "BR": [("Sao Paulo", "SP", "010"), ("Rio de Janeiro", "RJ", "200")],
    "MX": [("Mexico City", "CMX", "010"), ("Monterrey", "NL", "640")],
}

# Rough FX rates vs USD for cross-currency settlement simulation
FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "GBP": 0.79,
    "EUR": 0.92,
    "CAD": 1.36,
    "AUD": 1.52,
    "SGD": 1.34,
    "HKD": 7.82,
    "AED": 3.67,
    "CHF": 0.88,
    "JPY": 149.0,
    "INR": 83.0,
    "BRL": 4.95,
    "MXN": 17.1,
}

# How the customer initiated the transaction (distinct from settlement rail: wire/ach/card/internal).
CHANNEL_INDICATORS = ["Online", "In-Store", "Mobile App", "Phone", "ATM"]

POS_ENTRY_MODES = ["Chip/EMV", "Contactless/Tap", "Magstripe", "Manually Keyed"]
