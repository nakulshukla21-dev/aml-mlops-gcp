"""Business names and payment narrative templates for synthetic transactions."""

from __future__ import annotations

NAME_STEMS = [
    "Acme",
    "Summit",
    "Pacific",
    "Metro",
    "Atlas",
    "Horizon",
    "Sterling",
    "Nova",
    "Bridgeport",
    "Westfield",
    "Oakwood",
    "Pinnacle",
    "Riverstone",
    "Northgate",
    "BlueRock",
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
