# Venue verification notes (checked 2026-11-26)

## Lebanon access

- **OKX**: Lebanon is NOT on the restricted list in OKX's Risk & Compliance
  Disclosure / Terms of Service (restricted: Afghanistan, Canada, Cuba, Hong
  Kong, India, Iran, Japan, Malaysia, Nepal, North Korea, Syria, Uzbekistan,
  Crimea/Donetsk/Luhansk, US+territories). Lebanon is also not on the
  product-limited list. Source: okx.com/en-us/help/terms-of-service,
  okx.com Risk & Compliance Disclosure (last updated 2026-07-08).
- **Bybit**: Lebanon is NOT on the Excluded Jurisdictions list in clause 11.3
  of the Bybit Platform Terms (excluded: US, Chinese Mainland, Hong Kong,
  Singapore, Canada, France, UK/Dubai for the UAE entity, North Korea, Cuba,
  Iran, Uzbekistan, Crimea + non-govt Donetsk/Luhansk/Kherson/Zaporizhzhia,
  Sudan, Syria, Australia per Sept-2026 terms). Source: bybit.com Platform
  Terms & Conditions (effective 2026-10-16).
- **Caveat (cannot be verified away)**: Banque du Liban Article 10 (revised
  electronic banking rules, incorporating Jan-2026 revision) prohibits
  Lebanese banks/FIs from issuing, dealing in or facilitating virtual assets
  unless authorised by BDL. Practical consequence: funding an exchange
  account via a Lebanese bank card/transfer is likely BLOCKED at the bank
  side. Account opening is not guaranteed — exchanges reserve individual
  screening/sanctions discretion. No VPN/bypass recommended or assumed.
- **Verdict**: neither exchange names Lebanon as excluded, but (a) account
  approval is discretionary, (b) the fiat on-ramp is the real obstacle.
  A Lebanese user would need to already hold USDT/crypto acquired lawfully
  and transfer it on-chain to the exchange.

## Contract specs / fees used in the model

- Perpetuals: USDT-margined linear swaps, 8h funding (Binance/OKX standard).
  Model: isolated margin, IM = notional/L, MMR = 0.40% (OKX BTC tier-1;
  conservative for alts). Fees 0.10% taker + 0.05% slippage per side.
- Quarterly futures (Strategy B): Binance USD-M delivery contracts
  ({BASE}USDT_YYMMDD, last Friday of Mar/Jun/Sep/Dec). Modelled IMR = 20%
  (conservative vs published tier-1), MMR_F = 0.50%, futures fee 0.05% +
  0.05% slippage per side. Actual Binance delivery-futures tier tables were
  NOT retrieved — IMR/MMR are assumptions, sensitivity-tested in
  reports/sensitivity_B.csv.
- Minimum order sizes: not binding at $1,000 scale for BTC/ETH; for small-cap
  alts (TRX, DOT) OKX spot minimums (~$1-5 notional) could matter at small
  allocations — flagged, not modelled.
