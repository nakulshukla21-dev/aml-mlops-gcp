"""Static lookup tables for synthetic transaction generation."""

from __future__ import annotations

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

# Merchant / ATM city and state by country (used for card-present transactions only).
MERCHANT_LOCATIONS: dict[str, list[tuple[str, str]]] = {
    "US": [("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Miami", "FL"), ("Houston", "TX")],
    "GB": [("London", "ENG"), ("Manchester", "ENG"), ("Edinburgh", "SCT")],
    "DE": [("Frankfurt", "HE"), ("Berlin", "BE"), ("Munich", "BY")],
    "FR": [("Paris", "IDF"), ("Lyon", "ARA"), ("Marseille", "PAC")],
    "CA": [("Toronto", "ON"), ("Vancouver", "BC"), ("Montreal", "QC")],
    "AU": [("Sydney", "NSW"), ("Melbourne", "VIC"), ("Perth", "WA")],
    "SG": [("Singapore", "SG")],
    "HK": [("Hong Kong", "HK"), ("Kowloon", "HK")],
    "AE": [("Dubai", "DU"), ("Abu Dhabi", "AZ")],
    "KY": [("George Town", "GT")],
    "VG": [("Road Town", "RT")],
    "PA": [("Panama City", "8"), ("Colon", "3")],
    "CH": [("Zurich", "ZH"), ("Geneva", "GE")],
    "JP": [("Tokyo", "13"), ("Osaka", "27")],
    "IN": [("Mumbai", "MH"), ("Delhi", "DL"), ("Bengaluru", "KA")],
    "BR": [("Sao Paulo", "SP"), ("Rio de Janeiro", "RJ")],
    "MX": [("Mexico City", "CMX"), ("Monterrey", "NL")],
}

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

NAME_STEMS = [
    "Acme", "Summit", "Pacific", "Metro", "Atlas", "Horizon", "Sterling", "Nova",
    "Bridgeport", "Westfield", "Oakwood", "Pinnacle", "Riverstone", "Northgate", "BlueRock",
]

NAME_SUFFIXES = ["LLC", "Inc", "Ltd", "Corp", "Group", "Holdings", "Partners"]

GENERIC_SHELL_NAMES = [
    "Global Trade Solutions",
    "International Holdings Group",
    "Capital Partners LLC",
    "Universal Trading Company",
    "Continental Services Ltd",
    "Premier Investment Corp",
    "Apex Commerce Group",
    "Meridian Asset Holdings",
]

CONFUSING_DBA_NAMES = [
    "Quick Mart",
    "City Electronics",
    "Family Pharmacy",
    "Main Street Cafe",
    "Auto Repair Center",
    "Travel Services",
    "Tech Supplies",
    "Home Goods",
]

LEGIT_MEMOS = [
    "Invoice {ref}",
    "PO {ref}",
    "Vendor payment - office supplies",
    "Payroll week {week}",
    "Rent payment - {month}",
    "Utility bill {ref}",
    "Insurance premium Q{q}",
    "Contractor fee - project {ref}",
    "Subscription renewal",
    "Refund adjustment {ref}",
]

FRAUD_MEMOS = [
    "consulting fees",
    "loan repayment",
    "gift",
    "reimbursement",
    "business expenses",
    "commission payment",
    "personal transfer",
    "urgent payment",
    "do not convert",
    "investment return",
]

VAGUE_MEMOS = ["payment", "transfer", "N/A", ".", "", "TXN", "funds", "services"]
