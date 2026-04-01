"""
Brazil Payroll Tax Data Collection
===================================

Collects and structures Brazilian payroll tax rates across multiple margins:
- Desoneração da Folha (CPRB vs. 20% employer INSS) by sector and time period
- SAT/RAT base rates by CNAE code (1%, 2%, 3%)
- Sistema S / Terceiros contribution rates by sector type
- Simples Nacional embedded payroll tax rates by revenue bracket

Primary use: building a panel of effective payroll tax rates by CNAE code and
time period for studying the effect of payroll taxes on AI automation intensity.
"""
