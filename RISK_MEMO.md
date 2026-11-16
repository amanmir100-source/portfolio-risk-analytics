# Risk memo: 8-ETF model portfolio

**To:** Portfolio manager · **From:** Aman Mir · **Date:** 16 Nov 2026
**Data:** daily adjusted closes, 15 Nov 2021 – 13 Nov 2026 (crisis replays use history back to 2007). Source numbers: [`reports/live/`](reports/live/).

## Bottom line

The portfolio's risk is mostly US large-cap equity risk. SPY and QQQ are 40% of the money but 54% of the risk, and they move almost as one (daily correlation 0.95). Over this window the portfolio returned 6.7% a year against 3.86% on cash, a Sharpe ratio of 0.26. **Recommendation: trim the QQQ position by 5 points**; both alternatives tested lower the risk.

## Current risk

| Measure | Level |
|---|---|
| Volatility (annual) | 13.1%, vs 17.7% if the holdings didn't diversify each other |
| 1-day VaR, 95% / 99% | 1.29% / 2.13% (about $1,290 / $2,130 per $100k) |
| Average loss on the worst 5% of days (CVaR) | 1.84% |
| 1-year VaR, 95% (Monte Carlo) | $13,900 per $100k; 31% chance of losing money over a year |

The VaR numbers held up out of sample: over 1,004 days, the 99% historical VaR was breached 11 times against 10 expected. The normal-distribution version was breached 14 times, so treat parametric VaR as slightly optimistic in the tail. Breaches do cluster (three of the eleven fell on 3, 4 and 10 April 2025); a test for clustering does not reject the model, but it is the closest call (p = 0.11), so expect losses beyond VaR to arrive in runs when volatility jumps.

## What drives it

| | Share of capital | Share of volatility | Share of tail loss |
|---|---|---|---|
| SPY + QQQ | 40% | 54% | 55% |
| All equities (SPY, QQQ, IWM, EFA) | 55% | 72% | 72% |
| Bonds and gold (TLT, LQD, GLD) | 35% | 17% | 17% |

QQQ is the most over-weight in risk terms: 15% of capital, 23% of volatility and 24% of tail loss.

## How it held up in 2022

The portfolio fell 26.3% from its December 2021 peak to 14 October 2022 and took 565 trading days to recover (27 March 2024). Replaying past crises on today's weights, 2022 (−25.8%) was nearly as bad as 2008 (−31.9%) even though stocks fell half as far. In 2008 long Treasuries rose 25% and cushioned the fall; in 2022, with rates rising, they fell 29% alongside stocks. In these replays the bond sleeve protected in the recession-type crashes (2008, and COVID when TLT rose 14%) but not in the rate shock.

## Recommendation

**Trim QQQ from 15% to 10%.** It adds 23% of the risk for 15% of the capital, and with a 0.95 correlation to SPY it adds little diversification. Two candidates were tested on the same data ([`reports/live/portfolio_comparison.csv`](reports/live/portfolio_comparison.csv)):

| | Return | Volatility | CVaR (95%) | Sharpe | 2022 replay |
|---|---|---|---|---|---|
| Current | 6.7% | 13.1% | 1.84% | 0.26 | −25.8% |
| 5 pts QQQ → credit (LQD) | 6.0% | 12.4% | 1.73% | 0.22 | −25.1% |
| 5 pts QQQ → gold (GLD) | 6.9% | 12.5% | 1.75% | 0.29 | −24.4% |

Either move cuts volatility and tail loss by about 5%. Moving to credit gives up 0.7 points of return. Moving to gold did not cost return in this window, but gold returned 15% a year over it, so count on the risk reduction and not on the extra return. Neither candidate changes the main exposure: equities would still drive about two-thirds of the volatility.

*Caveats:* five years of history, one of which (2022) dominates the tail; results for a model portfolio built for a portfolio project; not investment advice.
