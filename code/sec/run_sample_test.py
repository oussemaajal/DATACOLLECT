"""
End-to-end pipeline test with synthetic 8-K press release data.

Generates realistic sample exhibits for ~20 filings from well-known companies,
then runs parse_guidance.py and build_guidance_panel.py to validate the full
pipeline. Use this when SEC EDGAR is unreachable (e.g. proxy restrictions).

Usage:
    python run_sample_test.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_INTERMEDIATE, DATA_CLEAN, OUTPUT_DIR
from utils import log

RAW_DIR = DATA_RAW / "guidance"

# ── Synthetic 8-K press release exhibits ─────────────────────────────────
# Modeled on real earnings press releases with management guidance language

SAMPLE_EXHIBITS = [
    {
        "cik": "320193",
        "company_name": "APPLE INC",
        "accession_number": "0000320193-25-000010",
        "date_filed": "2025-01-30",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000010/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Apple Reports First Quarter Results
Revenue of $124.3 Billion, Up 4 Percent Year Over Year

CUPERTINO, California — January 30, 2025 — Apple today announced financial results for its fiscal 2025 first quarter ended December 28, 2024.

"We are thrilled to report a record revenue quarter," said Tim Cook, Apple's CEO.

The Company expects second quarter fiscal 2025 revenue of $89 billion to $93 billion. Apple expects gross margin of 46 percent to 47 percent. The Company anticipates diluted earnings per share of $1.55 to $1.65 for the second quarter.

For the full fiscal year 2025, Apple expects revenue of $410 billion to $420 billion and diluted EPS of $7.10 to $7.30.

Apple reported quarterly revenue of $124.3 billion, net income of $36.3 billion, and diluted earnings per share of $2.40.
""",
    },
    {
        "cik": "789019",
        "company_name": "MICROSOFT CORP",
        "accession_number": "0000789019-25-000015",
        "date_filed": "2025-01-28",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/789019/000078901925000015/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Microsoft Cloud Strength Drives Second Quarter Results
Revenue of $69.6 Billion, Up 12 Percent Year Over Year

REDMOND, Wash. — January 28, 2025 — Microsoft Corp. today announced the following results for the quarter ended December 31, 2024.

Revenue was $69.6 billion and increased 12%. Operating income was $30.6 billion.

For the third quarter of fiscal year 2025, the Company expects revenue of $68.0 billion to $69.2 billion. Microsoft expects operating income of $29.5 billion to $30.5 billion. The company projects diluted earnings per share of $3.22 to $3.32. Microsoft anticipates full-year fiscal 2025 revenue of $268 billion to $272 billion.

Free cash flow for the quarter was $18.9 billion.
""",
    },
    {
        "cik": "1018724",
        "company_name": "AMAZON COM INC",
        "accession_number": "0001018724-25-000020",
        "date_filed": "2025-02-06",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1018724/000101872425000020/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Amazon.com Announces Fourth Quarter Results
Net Sales of $187.8 Billion

SEATTLE — February 6, 2025 — Amazon.com, Inc. today announced financial results for its fourth quarter ended December 31, 2024.

Net sales increased 10% to $187.8 billion. Operating income increased to $21.2 billion.

"For the first quarter 2025, we expect net sales of $151 billion to $155.5 billion, or growth of 5% to 9%," said Andrew Jassy, Amazon President and CEO. "Operating income is expected to be between $14.0 billion and $18.0 billion for Q1 2025."

Amazon anticipates full-year 2025 revenue of $650 billion to $670 billion and capital expenditures of approximately $100 billion.

Net income for Q4 was $20.0 billion, or diluted earnings per share of $1.86.
""",
    },
    {
        "cik": "1652044",
        "company_name": "ALPHABET INC",
        "accession_number": "0001652044-25-000012",
        "date_filed": "2025-02-04",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000012/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Alphabet Announces Fourth Quarter and Fiscal Year 2024 Results

MOUNTAIN VIEW, Calif. — February 4, 2025 — Alphabet Inc. today announced financial results for the quarter and fiscal year ended December 31, 2024.

"Our Q4 revenue of $96.5 billion reflects strong momentum across the business," said Sundar Pichai, CEO of Alphabet.

Revenue grew 12% year over year to $96.5 billion. Operating income was $30.7 billion, representing an operating margin of 31.8%.

For the first quarter of fiscal 2025, the Company expects revenue of $89 billion to $91 billion. Alphabet projects capital expenditures of approximately $75 billion for the full year 2025. The company anticipates an effective tax rate of approximately 15% for fiscal year 2025.

Net income was $26.5 billion, and diluted earnings per share was $2.12.
""",
    },
    {
        "cik": "1326801",
        "company_name": "META PLATFORMS INC",
        "accession_number": "0001326801-25-000008",
        "date_filed": "2025-01-29",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1326801/000132680125000008/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Meta Reports Fourth Quarter and Full Year 2024 Results

MENLO PARK, Calif. — January 29, 2025 — Meta Platforms, Inc. today reported financial results for the quarter and year ended December 31, 2024.

Revenue was $48.4 billion, up 21% year over year. Operating income was $23.4 billion.

For the first quarter of 2025, we expect total revenue to be in the range of $39.5 billion to $41.8 billion. Meta anticipates full-year 2025 total expenses of $114 billion to $119 billion. The company expects capital expenditures of $60 billion to $65 billion for the full year 2025.

Meta reaffirms its commitment to investing heavily in AI infrastructure. For the full year 2025, Meta expects revenue of $195 billion to $205 billion.

Net income was $20.8 billion. Diluted earnings per share was $8.02.
""",
    },
    {
        "cik": "1045810",
        "company_name": "NVIDIA CORP",
        "accession_number": "0001045810-25-000025",
        "date_filed": "2025-02-26",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000025/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """NVIDIA Announces Financial Results for Fourth Quarter and Fiscal 2025
Record Revenue of $39.3 Billion, Up 78% From a Year Ago

SANTA CLARA, Calif. — February 26, 2025 — NVIDIA today reported record revenue of $39.3 billion for the fourth quarter ended January 26, 2025, up 12% from the previous quarter and up 78% from a year ago.

"AI is transforming every industry," said Jensen Huang, founder and CEO of NVIDIA.

GAAP earnings per diluted share was $0.89. Revenue from the Data Center segment was $35.6 billion.

For the first quarter of fiscal 2026, the Company expects revenue of $43.0 billion, plus or minus 2%. NVIDIA expects GAAP gross margins of 70.6% and non-GAAP gross margins of 71.0%. The company projects operating expenses of approximately $5.2 billion.

For full fiscal year 2026, NVIDIA expects revenue of approximately $200 billion and anticipates diluted earnings per share of $4.20 to $4.50.
""",
    },
    {
        "cik": "21344",
        "company_name": "COCA COLA CO",
        "accession_number": "0000021344-25-000009",
        "date_filed": "2025-02-11",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/21344/000002134425000009/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """The Coca-Cola Company Reports Fourth Quarter and Full Year 2024 Results
Full Year Revenue of $47.1 Billion

ATLANTA — February 11, 2025 — The Coca-Cola Company today reported fourth quarter and full year 2024 results.

Net revenues grew 3% to $11.5 billion. Organic revenues grew 14%. Operating income grew 5% to $2.7 billion.

For full year 2025, the Company expects organic revenue growth of 5% to 6%. Coca-Cola expects comparable earnings per share growth of 2% to 3%, translating to comparable EPS of $2.93 to $2.96. The Company anticipates free cash flow of approximately $9.5 billion for fiscal 2025.

The Company reaffirms its long-term growth algorithm and expects operating margin improvement of 50 to 100 basis points.

Net income attributable to shareowners was $1.7 billion, or $0.39 per share.
""",
    },
    {
        "cik": "78003",
        "company_name": "PFIZER INC",
        "accession_number": "0000078003-25-000014",
        "date_filed": "2025-01-28",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/78003/000007800325000014/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Pfizer Reports Fourth Quarter and Full-Year 2024 Results
Provides 2025 Financial Guidance

NEW YORK — January 28, 2025 — Pfizer Inc. today reported financial results for fourth quarter and full-year 2024.

Fourth-quarter 2024 revenues of $17.8 billion. Reported diluted EPS of $0.37.

For full-year 2025, Pfizer provides the following guidance:
- Revenue of $61.0 billion to $64.0 billion
- Adjusted diluted EPS of $2.80 to $3.00
- Adjusted EBITDA of $22.0 billion to $24.0 billion
- Capital expenditures of $3.5 billion to $4.0 billion

The company expects first quarter 2025 revenue of $14.5 billion to $15.5 billion and diluted EPS of $0.62 to $0.72.

Pfizer raises its dividend by 2.4% to $0.43 per share.
""",
    },
    {
        "cik": "93410",
        "company_name": "CHEVRON CORP",
        "accession_number": "0000093410-25-000011",
        "date_filed": "2025-01-31",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/93410/000009341025000011/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Chevron Reports Fourth Quarter and Full-Year 2024 Earnings

SAN RAMON, Calif. — January 31, 2025 — Chevron Corporation today reported earnings of $3.2 billion for fourth quarter 2024.

Revenue was $47.6 billion. Worldwide net oil-equivalent production was 3.35 million barrels per day.

For 2025, Chevron expects capital expenditures of $14.5 billion to $15.5 billion. The company anticipates full-year 2025 production growth of 4% to 7%. Chevron projects free cash flow of $18 billion to $20 billion at $70/barrel Brent.

Earnings per diluted share were $1.72. Cash flow from operations was $8.2 billion.
""",
    },
    {
        "cik": "70858",
        "company_name": "BANK OF AMERICA CORP",
        "accession_number": "0000070858-25-000016",
        "date_filed": "2025-01-15",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/70858/000007085825000016/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Bank of America Reports Fourth Quarter 2024 Financial Results
Net Income of $6.7 Billion, EPS of $0.82

CHARLOTTE, N.C. — January 15, 2025 — Bank of America Corporation today reported net income of $6.7 billion, or $0.82 per diluted share, for the fourth quarter of 2024.

Revenue, net of interest expense, was $25.3 billion. Net interest income was $14.4 billion.

For the first quarter of 2025, Bank of America expects net interest income of $14.5 billion to $14.7 billion. The company projects full-year 2025 revenue of $104 billion to $108 billion. Management anticipates an effective tax rate of approximately 10% for fiscal year 2025.

The company expects operating expenses of approximately $65 billion for full-year 2025.
""",
    },
    {
        "cik": "1067983",
        "company_name": "BERKSHIRE HATHAWAY INC",
        "accession_number": "0001067983-25-000005",
        "date_filed": "2025-02-22",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1067983/000106798325000005/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Berkshire Hathaway Reports Fourth Quarter and Full Year 2024 Earnings

OMAHA, NE — February 22, 2025 — Berkshire Hathaway Inc. today reported fourth quarter earnings.

Operating earnings were $14.5 billion for the quarter. Insurance underwriting income was $3.4 billion. BNSF Railway revenue was $5.9 billion.

Net income attributable to Berkshire Hathaway shareholders was $19.7 billion, or diluted earnings per share of $13,503.

Note: Berkshire Hathaway does not provide forward-looking earnings guidance.
""",
    },
    {
        "cik": "1318605",
        "company_name": "TESLA INC",
        "accession_number": "0001318605-25-000018",
        "date_filed": "2025-01-29",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1318605/000131860525000018/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Tesla Fourth Quarter 2024 Update

AUSTIN, Texas — January 29, 2025 — Tesla today released its Q4 and Full Year 2024 Update.

Revenue for Q4 was $25.7 billion. GAAP net income was $2.3 billion. Diluted EPS was $0.73.

For fiscal year 2025, Tesla expects vehicle deliveries to grow 20% to 30%. The company projects revenue of $110 billion to $120 billion. Tesla anticipates capital expenditures of $11 billion to $13 billion. The company expects gross margin of 18% to 20% and operating income of $10 billion to $14 billion for the full year.

Free cash flow was $2.0 billion in Q4. Energy storage deployments grew 244% year over year.
""",
    },
    {
        "cik": "1800",
        "company_name": "ABBOTT LABORATORIES",
        "accession_number": "0000001800-25-000007",
        "date_filed": "2025-01-22",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1800/000000180025000007/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Abbott Reports Fourth-Quarter 2024 Results
Issues 2025 EPS Guidance

ABBOTT PARK, Ill. — January 22, 2025 — Abbott today announced fourth-quarter 2024 results.

Reported sales were $10.97 billion. Organic sales growth was 7.2%, excluding COVID testing.

Abbott issues full-year 2025 guidance for diluted earnings per share of $5.05 to $5.25 on a GAAP basis and $4.64 to $4.84 on an adjusted basis. The company expects organic sales growth of 7.5% to 8.5%, translating to revenue of approximately $44 billion to $45 billion.

For the first quarter of 2025, Abbott projects diluted EPS of $1.05 to $1.10 and revenue of $10.5 billion to $10.8 billion.

Diluted EPS from continuing operations was $1.34 on a reported basis.
""",
    },
    {
        "cik": "732717",
        "company_name": "AT&T INC",
        "accession_number": "0000732717-25-000022",
        "date_filed": "2025-01-22",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/732717/000073271725000022/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,7.01,9.01",
        "text": """AT&T Reports Fourth-Quarter and Full-Year 2024 Results
Provides 2025 Guidance

DALLAS — January 22, 2025 — AT&T Inc. reported fourth-quarter and full-year 2024 results.

Total revenues were $32.3 billion. Adjusted EBITDA was $11.5 billion. Free cash flow was $4.8 billion.

AT&T provides 2025 guidance:
The company expects adjusted EBITDA of $44.5 billion to $45.5 billion for fiscal 2025. AT&T projects free cash flow of $16 billion to $17 billion. The company expects adjusted diluted earnings per share of $1.97 to $2.07 for full-year 2025.

For the first quarter 2025, AT&T anticipates revenue of $30.5 billion to $31.0 billion.

Diluted EPS was $0.54 on a reported basis and $0.54 on an adjusted basis.
""",
    },
    {
        "cik": "200406",
        "company_name": "JOHNSON & JOHNSON",
        "accession_number": "0000200406-25-000013",
        "date_filed": "2025-01-21",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/200406/000020040625000013/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Johnson & Johnson Reports 2024 Fourth-Quarter and Full Year Results
Provides Full-Year 2025 Guidance

NEW BRUNSWICK, N.J. — January 21, 2025 — Johnson & Johnson today announced results for the fourth quarter and full year of 2024.

Worldwide sales were $22.5 billion for the fourth quarter. Net earnings were $3.5 billion and diluted EPS was $1.41.

For full-year 2025, Johnson & Johnson expects:
- Operational sales growth of 3.0% to 4.0%, resulting in total revenue of $91.5 billion to $92.5 billion
- Adjusted diluted EPS of $10.75 to $10.95
- Reported diluted earnings per share of $8.90 to $9.10

The company reaffirms its long-term guidance framework and expects operating income of $24.0 billion to $25.0 billion.

Free cash flow for the full year 2024 was $18.0 billion.
""",
    },
    {
        "cik": "51143",
        "company_name": "INTL BUSINESS MACHINES CORP",
        "accession_number": "0000051143-25-000019",
        "date_filed": "2025-01-29",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/51143/000005114325000019/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """IBM Reports Fourth-Quarter 2024 Results
Revenue of $17.6 Billion, Up 1%

ARMONK, N.Y. — January 29, 2025 — IBM today announced fourth-quarter 2024 earnings results.

Revenue was $17.6 billion. Software revenue grew 10% to $7.9 billion. Net income was $3.0 billion.

For fiscal year 2025, IBM expects revenue growth of 5% at constant currency. The company projects free cash flow of approximately $13.5 billion. IBM anticipates adjusted EBITDA of $17.0 billion to $17.5 billion. The company expects diluted earnings per share of $10.10 to $10.30 on an operating basis.

For the first quarter of 2025, IBM expects revenue of $14.5 billion to $15.0 billion.

Diluted earnings per share from continuing operations was $3.24.
""",
    },
    {
        "cik": "1730168",
        "company_name": "UBER TECHNOLOGIES INC",
        "accession_number": "0001730168-25-000011",
        "date_filed": "2025-02-05",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1730168/000173016825000011/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Uber Announces Results for Fourth Quarter and Full Year 2024

SAN FRANCISCO — February 5, 2025 — Uber Technologies, Inc. today announced financial results for the quarter and full year ended December 31, 2024.

Gross Bookings grew 18% year-over-year to $44.2 billion. Revenue of $11.8 billion. Adjusted EBITDA of $1.84 billion.

For the first quarter of 2025, Uber expects gross bookings of $42.0 billion to $43.5 billion. The company anticipates adjusted EBITDA of $1.79 billion to $1.89 billion for Q1 2025.

Uber projects full-year 2025 adjusted EBITDA of $8.2 billion to $8.8 billion and free cash flow of $6.5 billion to $7.0 billion.

Net income attributable to Uber was $6.9 billion. Diluted earnings per share was $3.21.
""",
    },
    {
        "cik": "1551152",
        "company_name": "SERVICENOW INC",
        "accession_number": "0001551152-25-000008",
        "date_filed": "2025-01-29",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1551152/000155115225000008/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """ServiceNow Reports Fourth Quarter and Full Year 2024 Results

SANTA CLARA, Calif. — January 29, 2025 — ServiceNow today announced fourth quarter and full year 2024 financial results.

Subscription revenues were $2.87 billion, up 21% year over year. Total revenues were $2.96 billion.

For the first quarter of 2025, ServiceNow expects subscription revenues of $3.015 billion to $3.025 billion. For full year 2025, the company expects subscription revenues of $13.1 billion to $13.15 billion, representing 20% growth. ServiceNow projects operating margin of 30.5% for the full year and free cash flow margin of approximately 33%.

The company expects diluted earnings per share of $16.50 to $17.00 for full fiscal year 2025.

Net income was $722 million, and diluted net income per share was $3.46.
""",
    },
    {
        "cik": "1467373",
        "company_name": "CROWDSTRIKE HOLDINGS INC",
        "accession_number": "0001467373-25-000006",
        "date_filed": "2025-03-04",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1467373/000146737325000006/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """CrowdStrike Reports Fourth Quarter and Fiscal Year 2025 Results

AUSTIN, Texas — March 4, 2025 — CrowdStrike Holdings, Inc. today announced financial results for the fourth quarter of fiscal year 2025 ended January 31, 2025.

Total revenue was $1.06 billion, up 29% year over year. Subscription revenue was $1.01 billion, up 31%.

For the first quarter of fiscal year 2026, CrowdStrike expects revenue of $1.09 billion to $1.10 billion. The company projects non-GAAP diluted earnings per share of $0.95 to $0.98 for Q1. For full fiscal year 2026, CrowdStrike expects total revenue of $4.74 billion to $4.81 billion and non-GAAP EPS of $4.10 to $4.25.

Non-GAAP net income was $296.3 million and non-GAAP diluted net income per share was $1.18. Free cash flow was $316.4 million.
""",
    },
    {
        "cik": "1364742",
        "company_name": "PALO ALTO NETWORKS INC",
        "accession_number": "0001364742-25-000010",
        "date_filed": "2025-02-20",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/1364742/000136474225000010/ex991.htm",
        "exhibit_filename": "ex991.htm",
        "exhibit_description": "EX-99.1 Press Release",
        "items": "2.02,9.01",
        "text": """Palo Alto Networks Reports Fiscal Second Quarter 2025 Results

SANTA CLARA, Calif. — February 20, 2025 — Palo Alto Networks, Inc. today announced financial results for its fiscal second quarter 2025 ended January 31, 2025.

Total revenue grew 14% year over year to $2.26 billion. Next-Generation Security ARR grew 37%.

For the fiscal third quarter 2025, Palo Alto Networks expects total revenue in the range of $2.26 billion to $2.29 billion and diluted non-GAAP net income per share of $0.80 to $0.82.

For full fiscal year 2025, the company raises its guidance and now expects revenue of $9.14 billion to $9.19 billion, up from prior guidance of $9.10 billion to $9.15 billion. Palo Alto Networks raises diluted non-GAAP EPS guidance to $3.27 to $3.30 from prior $3.22 to $3.25. The company expects adjusted EBITDA of $3.2 billion to $3.3 billion.

Non-GAAP net income was $577.2 million, or $1.57 per diluted share.
""",
    },
]


def create_sample_exhibits() -> pd.DataFrame:
    """Create sample exhibits DataFrame mimicking the output of download_8k_exhibits.py."""
    records = []
    for ex in SAMPLE_EXHIBITS:
        rec = dict(ex)
        rec["text_length"] = len(rec["text"])
        records.append(rec)
    return pd.DataFrame(records)


def main():
    log.info("=" * 60)
    log.info("Running pipeline end-to-end with sample data (FY 2025)")
    log.info("=" * 60)

    # --- Step 1 (simulated): Create sample 8-K index ---
    log.info("\n--- Step 1: Creating sample 8-K index ---")
    index_data = []
    for ex in SAMPLE_EXHIBITS:
        index_data.append({
            "cik": ex["cik"],
            "company_name": ex["company_name"],
            "form_type": ex["form_type"],
            "date_filed": ex["date_filed"],
            "filename": f"edgar/data/{ex['cik']}/{ex['accession_number'].replace('-', '')}/filing.txt",
            "accession_number": ex["accession_number"],
            "year": 2025,
            "quarter": int(ex["date_filed"].split("-")[1]) // 4 + 1,
            "cik_padded": ex["cik"].zfill(10),
        })
    index_df = pd.DataFrame(index_data)
    index_df["date_filed"] = pd.to_datetime(index_df["date_filed"])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / "eight_k_index_2025_2025.parquet"
    index_df.to_parquet(index_path, index=False)
    log.info(f"  Saved sample index ({len(index_df)} filings) -> {index_path}")

    # --- Step 2 (simulated): Create sample exhibits ---
    log.info("\n--- Step 2: Creating sample exhibits ---")
    exhibits_df = create_sample_exhibits()
    exhibits_path = RAW_DIR / "eight_k_exhibits.parquet"
    exhibits_df.to_parquet(exhibits_path, index=False)
    log.info(f"  Saved sample exhibits ({len(exhibits_df)} records) -> {exhibits_path}")

    # --- Step 3: Parse guidance ---
    log.info("\n--- Step 3: Parsing guidance from exhibits ---")
    from parse_guidance import extract_guidance_from_text

    all_guidance = []
    n_with_guidance = 0

    for _, row in exhibits_df.iterrows():
        filing_meta = {
            "cik": row["cik"],
            "company_name": row["company_name"],
            "accession_number": row["accession_number"],
            "date_filed": row["date_filed"],
            "form_type": row["form_type"],
            "exhibit_url": row["exhibit_url"],
        }
        records = extract_guidance_from_text(row["text"], filing_meta)
        if records:
            all_guidance.extend(records)
            n_with_guidance += 1

    log.info(f"  {len(exhibits_df)} exhibits parsed")
    log.info(f"  {n_with_guidance} contained extractable guidance")
    log.info(f"  {len(all_guidance)} total guidance records")

    if not all_guidance:
        log.error("No guidance records extracted! Check regex patterns.")
        sys.exit(1)

    guidance_df = pd.DataFrame(all_guidance)
    log.info(f"  Metrics: {guidance_df['metric'].value_counts().to_dict()}")
    log.info(f"  Actions: {guidance_df['action'].value_counts().to_dict()}")

    DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    parsed_path = DATA_INTERMEDIATE / "guidance_parsed.parquet"
    guidance_df.to_parquet(parsed_path, index=False)
    log.info(f"  Saved parsed guidance -> {parsed_path}")

    # --- Step 4: Build panel ---
    log.info("\n--- Step 4: Building guidance panel ---")

    # Since we can't fetch company_tickers.json from SEC, build a local mapping
    ticker_map = {
        "320193": "AAPL", "789019": "MSFT", "1018724": "AMZN",
        "1652044": "GOOGL", "1326801": "META", "1045810": "NVDA",
        "21344": "KO", "78003": "PFE", "93410": "CVX",
        "70858": "BAC", "1067983": "BRK-B", "1318605": "TSLA",
        "1800": "ABT", "732717": "T", "200406": "JNJ",
        "51143": "IBM", "1730168": "UBER", "1551152": "NOW",
        "1467373": "CRWD", "1364742": "PANW",
    }

    guidance_df["cik_padded"] = guidance_df["cik"].astype(str).str.zfill(10)
    guidance_df["ticker"] = guidance_df["cik"].astype(str).map(ticker_map)
    guidance_df["company_name_sec"] = guidance_df["company_name"]
    guidance_df["date_filed"] = pd.to_datetime(guidance_df["date_filed"], errors="coerce")
    guidance_df["file_year"] = guidance_df["date_filed"].dt.year
    guidance_df["file_quarter"] = guidance_df["date_filed"].dt.quarter
    guidance_df["file_yearq"] = (
        guidance_df["file_year"].astype(str) + "Q" + guidance_df["file_quarter"].astype(str)
    )
    guidance_df["is_per_share"] = (
        guidance_df["metric"].isin(["EPS", "FFO"])
        | ((guidance_df["val1"].abs() < 200) & (guidance_df["val2"].abs() < 200))
    )

    # Dedup
    before = len(guidance_df)
    guidance_df = guidance_df.drop_duplicates(
        subset=["cik_padded", "date_filed", "metric", "val1", "val2"],
        keep="first",
    )
    log.info(f"  Removed {before - len(guidance_df)} duplicates")

    # Column selection
    cols = [
        "cik_padded", "ticker", "company_name", "company_name_sec",
        "accession_number", "date_filed", "file_year", "file_quarter", "file_yearq",
        "form_type", "metric", "pdicity", "fiscal_year",
        "val1", "val2", "midpoint", "range_width", "is_range",
        "action", "is_per_share", "sentence", "exhibit_url",
    ]
    cols = [c for c in cols if c in guidance_df.columns]
    panel = guidance_df[cols].sort_values(
        ["ticker", "date_filed", "metric"]
    ).reset_index(drop=True)
    panel = panel.rename(columns={"cik_padded": "cik"})

    log.info(f"\n  Final panel: {len(panel)} rows")
    log.info(f"  Unique firms: {panel['ticker'].nunique()}")
    log.info(f"  Metrics: {panel['metric'].value_counts().to_dict()}")
    log.info(f"  Actions: {panel['action'].value_counts().to_dict()}")
    log.info(f"  Periodicities: {panel['pdicity'].value_counts().to_dict()}")

    # Save
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    clean_path = DATA_CLEAN / "guidance_panel.parquet"
    panel.to_parquet(clean_path, index=False)
    log.info(f"  Saved clean panel -> {clean_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_csv = OUTPUT_DIR / "guidance_panel_sample.csv"
    panel.to_csv(sample_csv, index=False)
    log.info(f"  Saved CSV sample -> {sample_csv}")

    # Print a preview
    log.info("\n" + "=" * 60)
    log.info("SAMPLE OUTPUT (first 20 rows):")
    log.info("=" * 60)
    preview_cols = ["ticker", "date_filed", "metric", "val1", "val2", "midpoint",
                    "pdicity", "fiscal_year", "action", "is_range"]
    preview_cols = [c for c in preview_cols if c in panel.columns]
    print(panel[preview_cols].head(20).to_string(index=False))

    log.info(f"\nPipeline test complete. {len(panel)} guidance records from {panel['ticker'].nunique()} firms.")


if __name__ == "__main__":
    main()
