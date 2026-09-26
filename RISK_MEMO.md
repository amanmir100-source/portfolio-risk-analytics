# Risk memo: 8-ETF model portfolio

**To:** Portfolio manager · **From:** Aman Mir · **Date:** 26 Sep 2026
**Data:** daily adjusted closes, 27 Sep 2021 – 24 Sep 2026 (crisis replays use history back to 2007). Source numbers: [`reports/live/`](reports/live/).

## Bottom line

The portfolio's risk is mostly US large-cap equity risk. SPY and QQQ are 40% of the money but 54% of the risk, and they move almost as one (daily correlation 0.95). **Recommendation: review the size of the QQQ position** before adding risk anywhere else.

## Current risk

| Measure | Level |
|---|---|
| Volatility (annual) | 13.0%, vs 17.5% if the holdings didn't diversify each other |
| 1-day VaR, 95% / 99% | 1.28% / 2.08% (about $1,280 / $2,080 per $100k) |
| Average loss on the worst 5% of days (CVaR) | 1.82% |
| 1-year VaR, 95% (Monte Carlo) | $12,700 per $100k; 28% chance of losing money over a year |

The VaR numbers can be trusted: out of sample over 1,002 days, the 99% historical VaR was breached 10 times against 10 expected. The normal-distribution version was breached 13 times, so treat parametric VaR as slightly optimistic in the tail.

## What drives it

| | Share of capital | Share of volatility | Share of tail loss |
|---|---|---|---|
| SPY + QQQ | 40% | 54% | 54% |
| All equities (SPY, QQQ, IWM, EFA) | 55% | 72% | 72% |
| Bonds and gold (TLT, LQD, GLD) | 35% | 17% | 17% |

QQQ is the most over-weight in risk terms: 15% of capital, 23% of risk.

## How it held up in 2022

The portfolio fell 26.3% from its December 2021 peak to 14 October 2022 and took 565 trading days to recover (27 March 2024). Replaying past crises on today's weights, 2022 (−25.8%) was nearly as bad as 2008 (−31.9%) even though stocks fell half as far. In 2008 long Treasuries rose 25% and cushioned the fall; in 2022, with rates rising, they fell 29% alongside stocks. In these replays the bond sleeve protected in the recession-type crashes (2008, and COVID when TLT rose 14%) but not in the rate shock.

## Recommendation

**Review the QQQ position size.** It adds 23% of the risk for 15% of the capital, and with a 0.95 correlation to SPY it adds little diversification. Next step: re-run the risk-contribution and stress analysis on two or three candidate weightings and compare volatility, tail loss and the 2022 replay before changing anything.

*Caveats:* five years of history, one of which (2022) dominates the tail; results for a model portfolio built for a portfolio project; not investment advice.
