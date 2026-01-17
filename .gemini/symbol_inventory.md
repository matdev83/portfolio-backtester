# Symbol Inventory for MDMP Integration

## Summary

| Category | Count | Already in MDMP | Missing |
|----------|-------|-----------------|---------|
| Universe Files (deduplicated) | 73 | 0 | 73 |
| Hardcoded ETFs/Benchmarks | 6 | 4 | 2 |
| Sector ETFs | 11 | 11 | 0 |
| **Total Unique to Add** | **~65** | - | - |

---

## Already in symbols.json

### ETFs (Already Registered)

| Symbol | Canonical ID | Status |
|--------|--------------|--------|
| SPY | AMEX:SPY | ✅ Present |
| QQQ | NASDAQ:QQQ | ✅ Present |
| IWM | AMEX:IWM | ✅ Present |
| DIA | AMEX:DIA | ✅ Present |
| RSP | AMEX:RSP | ✅ Present |
| XLB | AMEX:XLB | ✅ Present |
| XLE | AMEX:XLE | ✅ Present |
| XLF | AMEX:XLF | ✅ Present |
| XLI | AMEX:XLI | ✅ Present |
| XLP | AMEX:XLP | ✅ Present |
| XLU | AMEX:XLU | ✅ Present |
| XLV | AMEX:XLV | ✅ Present |
| XLK | AMEX:XLK | ✅ Present |
| XLY | AMEX:XLY | ✅ Present |
| XLRE | AMEX:XLRE | ✅ Present |

### Indices (Already Registered)

| Symbol | Canonical ID | Status |
|--------|--------------|--------|
| ^GSPC/SPX | SP:SPX | ✅ Present |
| ^VIX/VIX | CBOE:VIX | ✅ Present |

---

## ETFs/Benchmarks - MISSING (Need to Add)

| Symbol | Proposed ID | Type | Priority |
|--------|-------------|------|----------|
| GLD | AMEX:GLD | Gold ETF | stooq, yfinance, tradingview |
| TLT | NASDAQ:TLT | Treasury Bond ETF | stooq, yfinance, tradingview |
| VTI | AMEX:VTI | Total Stock Market ETF | stooq, yfinance, tradingview |
| UVXY | AMEX:UVXY | VIX Leveraged ETF | yfinance, tradingview |

---

## Individual Stocks - MISSING (From Universe Files)

### S&P 500 Top 50 (All Need to Be Added)

| Symbol | Exchange | Description |
|--------|----------|-------------|
| AAPL | NASDAQ | Apple Inc. |
| MSFT | NASDAQ | Microsoft Corporation |
| GOOGL | NASDAQ | Alphabet Inc. Class A |
| AMZN | NASDAQ | Amazon.com Inc. |
| NVDA | NASDAQ | NVIDIA Corporation |
| META | NASDAQ | Meta Platforms Inc. |
| TSLA | NASDAQ | Tesla Inc. |
| BRK.B | NYSE | Berkshire Hathaway Class B |
| UNH | NYSE | UnitedHealth Group |
| JNJ | NYSE | Johnson & Johnson |
| JPM | NYSE | JPMorgan Chase & Co. |
| V | NYSE | Visa Inc. |
| PG | NYSE | Procter & Gamble |
| XOM | NYSE | Exxon Mobil Corporation |
| HD | NYSE | The Home Depot |
| CVX | NYSE | Chevron Corporation |
| MA | NYSE | Mastercard Incorporated |
| ABBV | NYSE | AbbVie Inc. |
| PFE | NYSE | Pfizer Inc. |
| AVGO | NASDAQ | Broadcom Inc. |
| COST | NASDAQ | Costco Wholesale |
| DIS | NYSE | The Walt Disney Company |
| ADBE | NASDAQ | Adobe Inc. |
| CRM | NYSE | Salesforce Inc. |
| MRK | NYSE | Merck & Co. |
| TMO | NYSE | Thermo Fisher Scientific |
| ABT | NYSE | Abbott Laboratories |
| NFLX | NASDAQ | Netflix Inc. |
| ACN | NYSE | Accenture plc |
| LIN | NYSE | Linde plc |
| VZ | NYSE | Verizon Communications |
| CSCO | NASDAQ | Cisco Systems |
| NKE | NYSE | Nike Inc. |
| DHR | NYSE | Danaher Corporation |
| TXN | NASDAQ | Texas Instruments |
| WMT | NYSE | Walmart Inc. |
| PM | NYSE | Philip Morris International |
| NEE | NYSE | NextEra Energy |
| RTX | NYSE | RTX Corporation |
| QCOM | NASDAQ | Qualcomm Inc. |
| HON | NASDAQ | Honeywell International |
| UPS | NYSE | United Parcel Service |
| LOW | NYSE | Lowe's Companies |
| AMGN | NASDAQ | Amgen Inc. |
| SPGI | NYSE | S&P Global Inc. |
| GS | NYSE | Goldman Sachs Group |
| INTU | NASDAQ | Intuit Inc. |
| CAT | NYSE | Caterpillar Inc. |
| COP | NYSE | ConocoPhillips |
| AXP | NYSE | American Express |

### Additional from Dow Jones (Not in Top 50)

| Symbol | Exchange | Description |
|--------|----------|-------------|
| BA | NYSE | Boeing Company |
| DOW | NYSE | Dow Inc. |
| IBM | NYSE | IBM Corporation |
| INTC | NASDAQ | Intel Corporation |
| KO | NYSE | Coca-Cola Company |
| MCD | NYSE | McDonald's Corporation |
| MMM | NYSE | 3M Company |
| TRV | NYSE | Travelers Companies |
| WBA | NASDAQ | Walgreens Boots Alliance |

### Additional from NASDAQ Top 20 (Not in Above)

| Symbol | Exchange | Description |
|--------|----------|-------------|
| GOOG | NASDAQ | Alphabet Inc. Class C |
| PEP | NASDAQ | PepsiCo Inc. |
| CMCSA | NASDAQ | Comcast Corporation |

---

## Complete Unique Symbol List (65 Stocks to Add)

```
AAPL, ABBV, ABT, ACN, ADBE, AMGN, AMZN, AXP, AVGO, BA,
BRK.B, CAT, CMCSA, COP, COST, CRM, CSCO, CVX, DHR, DIS,
DOW, GOOG, GOOGL, GS, HD, HON, IBM, INTC, INTU, JNJ,
JPM, KO, LIN, LOW, MA, MCD, META, MMM, MRK, MSFT,
NEE, NFLX, NKE, NVDA, PEP, PFE, PG, PM, QCOM, RTX,
SPGI, TMO, TRV, TSLA, TXN, UNH, UPS, V, VZ, WBA,
WMT
```

Plus 4 ETFs: `GLD, TLT, VTI, UVXY`

**Total: 65 unique symbols to add**

---

## Exchange Classification for Provider Mapping

### NASDAQ Stocks (use `{ticker}` for yfinance, `{ticker.lower()}.us` for stooq)

```
AAPL, ADBE, AMGN, AMZN, AVGO, CMCSA, COST, CSCO, GOOG, GOOGL,
HON, INTC, INTU, META, MSFT, NFLX, NVDA, PEP, QCOM, TSLA, TXN, WBA
```

### NYSE Stocks (use `{ticker}` for yfinance, `{ticker.lower()}.us` for stooq)

```
ABBV, ABT, ACN, AXP, BA, BRK.B, CAT, COP, CRM, CVX,
DHR, DIS, DOW, GS, HD, IBM, JNJ, JPM, KO, LIN,
LOW, MA, MCD, MMM, MRK, NEE, NKE, PFE, PG, PM,
RTX, SPGI, TMO, TRV, UNH, UPS, V, VZ, WMT
```

---

## Symbol Template for symbols.json

```json
"NASDAQ:AAPL": {
  "preferred_provider": "stooq",
  "calendar_id": "NYSE",
  "description": "Apple Inc. - Technology hardware company",
  "providers": {
    "stooq": "aapl.us",
    "yfinance": "AAPL",
    "tradingview": "NASDAQ:AAPL"
  },
  "aliases": ["AAPL"]
}
```
