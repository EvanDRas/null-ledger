# How Everything Works — The Complete Explainer

*Every component of this project, explained so you can explain it. Each section
ends with a one-liner you can say out loud. Numbers here are the real ones from
the final audited run.*

---

## 0. The one-paragraph version

This is a systematic stock-picking strategy: once a month it ranks every stock
in the S&P 500 using a machine-learning model trained on price patterns, buys
the top ~30 with bigger positions in higher-conviction names, and repeats —
no human judgment anywhere. The headline is the honesty arc: on naive free
data it "returned" 25.7%/yr; each data-integrity fix lowered that — 20%,
then 15% — and completing the delisted-security record (coverage 89% of
member-months) eliminated the edge entirely. **Final result: 11.0%/yr vs the
S&P's 10.8% over 234 out-of-sample months, alpha −0.4%/yr (t = −0.15) —
statistically indistinguishable from the index, at 1.14× the market risk.**
Since 2026-06-30 it runs fully automated, paper-trading a $100k account with
a locked, append-only monthly record that now tests this null out of sample.
The real product was never the return — it's the research process: nine
hypotheses under pre-registered rules, seven rejected, and a measured edge
that the researcher's own data work destroyed and published.

---

## 1. The data layer

### 1.1 Prices (`cache/prices_v2.parquet`)
Daily open/high/low/close/volume for **777 securities, 2003-present** (of
the 994-ticker membership union; the remaining 92 with membership exposure
are unobtainable even in paid retail data),
dividend- and split-adjusted (so returns include dividends automatically).
Sources: Yahoo Finance, rebuilt and validated (below). `update_prices.py`
appends fresh bars for still-trading names each month and keeps a backup.

### 1.2 Point-in-time membership (`cache/sp500_history.csv`)
Daily snapshots of *who was actually in the S&P 500 on every date* since 1996.
The model may only pick from stocks that were members **on the decision date**.
Without this, a backtest quietly trades today's winners in the past — the
single most common way backtests lie. Rename aliases (ANTM→ELV, PCLN→BKNG,
UTX→RTX...) map old membership symbols to the tickers that carry their price
history.

### 1.3 The data rebuild (the forensics story)
The naive universe was missing 240 of the 777 test-window members (31%,
~a quarter of member-months) — disproportionately the *losers* (dead and demoted companies),
which flatters results. The fix:
- built the full **994-ticker membership union**, recovered **365
  identifiers** (184 via Yahoo + 165 via Tiingo for the long-dead names + 16
  rename aliases) → coverage 74% → 89% of member-months;
- **membership-window truncation**: price rows outside
  [first membership − 420d, last membership + 60d] are discarded — this
  mechanically kills *ticker-reuse contamination* (e.g. Yahoo's "BBBY" series
  kept quoting years after Bed Bath & Beyond died — it's a different company);
- **corruption gate**: any series with ≥3 daily moves >80%, or any single
  +150% day, is dropped as garbage data (caught 11, e.g. "TIE" with prices
  from $1.40 to $33,700). Every exclusion is logged.

Each honesty fix **lowered** the backtest: 25.7% → 20% → 15% → **11.0%**
annualized. The final step — buying the delisted data Yahoo couldn't provide
— eliminated the edge entirely. The strategy never changed; the measurement
got honest, and honesty's last word was "index."

> **Say it like this:** "Before modeling anything I rebuilt the universe
> point-in-time, recovered the dead companies, and wrote mechanical gates for
> corrupted data. Every fix made my numbers worse, which is how I knew the
> fixes were real."

---

## 2. The features (13 signals per stock, per day)

All computed from price/volume only — one family: *momentum and trend quality*.

| Feature | What it measures |
|---|---|
| `mom_1m`, `mom_3m`, `mom_6m` | Price change over 1/3/6 months (trend strength) |
| `mom_12_2` | 12-month return *skipping the last month* — the academic-standard momentum measure (recent month tends to reverse) |
| `vol_20d`, `vol_60d` | Annualized volatility over 20/60 days (how violent the ride is) |
| `downside_vol` | Volatility of *down* days only (bad-day risk) |
| `ma50_dist`, `ma200_dist` | % distance above/below the 50- and 200-day moving averages (trend position) |
| `high_52w_dist`, `low_52w_dist` | Distance from the 52-week high/low (breakout/washout position) |
| `vol_trend` | 20d vs 60d average volume (is interest rising?) |
| `rel_str_3m` | 3-month return minus the S&P's (market-relative strength) |

The *combination* matters: the model learns things like "strong 6-month trend
+ low volatility + near the highs = keeps going," while "strong trend +
violent swings + parabolic = tends to snap."

> **Say it like this:** "Thirteen features, all one economic idea — momentum —
> measured from several angles so the model can tell durable trends from
> overstretched ones."

---

## 3. The model

- **XGBoost** (gradient-boosted decision trees) — regression target: each
  stock's **next-month return minus the S&P's** (so it learns relative
  winners, not market direction).
- **Deliberately small**: depth-3 trees, learning rate 0.05, heavy
  regularization, early stopping (averages only ~19 trees per model). With a
  weak signal, model capacity is a liability.
- **Walk-forward training** — the anti-cheating protocol: to predict month
  *T*, the model trains only on a **rolling 60-month window ending before
  *T***, retrains every 3 months, and is evaluated on data it has never seen.
  Every reported number is out-of-sample in this sense.
- **Seed averaging**: 3 models with different random seeds, predictions
  averaged — removes single-seed luck from live picks.
- **Ensemble**: final score = 50% model ranking + 50% plain 6-month momentum
  ranking (percentile blend). The simple signal anchors the clever one.
- Measured contribution — **reversed by complete data**: on the earlier 79%
  universe the ensemble appeared to add ~4pp/yr over pure momentum. With
  coverage at 89%, ensemble ≈ pure momentum ≈ the index (IC 0.0039), and the
  ensemble slightly *trails* the momentum baseline. The ML's apparent edge
  was a survivorship artifact: it had learned patterns of a universe whose
  losers had been deleted.

> **Say it like this:** "A small, heavily-regularized XGBoost trained
> walk-forward, blended 50/50 with plain momentum. On incomplete data it
> appeared to add ~4 points a year; when I completed the delisted-security
> record, that contribution — and the alpha itself — vanished. The model had
> been learning the shape of survivorship bias."

---

## 4. Portfolio construction

Every month-end, all ~500 members are ranked by score. Then:

1. **Top ~30 selected** — enough names to diversify away single-stock luck,
   few enough to differ meaningfully from the index.
2. **Conviction (score) weighting** — higher-ranked names get more money
   (validated: +1.4pp/yr vs equal weight, Sharpe up, on three samples).
3. **10% position cap** — score weighting alone would put ~13% in one name;
   the cap costs zero return and removes that risk.
4. **30% sector cap** — max ~9 names per sector, so a hot sector can't own
   the book.
5. **Turnover controls** — a *buffer* (a held stock stays unless it falls out
   of the top ~45), a *3-month minimum hold*, and a *max ~12 changes/month*.
   Result: ~25% monthly turnover (~7-8 swaps), typical holding ~4 months.
   Every trade costs money, so the book rotates gradually instead of churning.
6. **Costs**: 15 bps per side (10 commission + 5 slippage) charged on all
   turnover, including delisting exits.

> **Say it like this:** "Selection is only half the strategy. Conviction
> weighting, position and sector caps, and turnover limits are where several
> of my nine tested hypotheses lived — concentration, for instance, looked
> like +2pp/yr until I regressed it on the benchmark and found it was pure
> added beta."

---

## 5. The backtest & the numbers that matter

**Mechanics**: hold the portfolio a month; each stock's return = adjusted
month-end close to month-end close (dividends in); subtract costs; compare to
SPY total return. Data begin in 2003; features need ~12 months of history and training
requires a 36-month minimum (growing to a rolling 60-month window), so the
evaluated out-of-sample period is **2007-2026 = 233 tracked months**. (Say
"23 years of data, 233 out-of-sample months" — never blur the two.)

| Number | Value | What it means |
|---|---|---|
| Annualized return | **11.0%** vs 10.8% | Growth rate vs the S&P 500 |
| Excess return | **+0.1pp/yr** compound (arith +1.2pp, t = 0.45) | Statistically nothing |
| **Beta** | **1.14** | The portfolio moves 1.14× the market — it takes meaningfully more risk than the index to earn index-like returns |
| **CAPM alpha** | **−0.4%/yr** (t = −0.15) | Skill after stripping beta — the honest headline: none demonstrated |
| Sharpe ratio | 0.51 vs 0.70 | Return per unit of risk — clearly worse than the index. (Return/vol, no risk-free subtraction, same convention both sides.) |
| Hit rate | 51.7% of months | A coin flip — exactly what zero alpha looks like |
| Max drawdown | −61% vs −51% | Falls *harder* than the index in crashes |
| Mean IC | 0.0039 | Spearman correlation between ranks and next-month returns — indistinguishable from noise |

**How to read this**: on the earlier 79%-coverage data these same rows read
15.0%, alpha +3.1% (t = 1.19), Sharpe 0.71, hit rate 55.4%. Completing the
delisted record to 89% coverage moved every one of them to the index-like
values above. The alpha was never statistically significant — and the complete
data resolved the ambiguity in the direction of *no edge*. Honest forward
expectation: **approximately the index**, at higher risk.

> **Say it like this:** "On incomplete data I measured 3.1 points of alpha at
> t 1.19. I bought the missing delisted data specifically to test that number,
> and it went to zero. The strategy currently matches the index at higher
> risk — and I published exactly that."

---

## 6. The validation discipline (the core credential)

Every improvement idea faced a **pre-registered acceptance rule** — the
pass/fail criteria were committed *before* seeing results, killing the
temptation to move goalposts. The accounting, stated precisely: **nine
hypotheses (the three fundamental factors count individually) — two adopted,
seven rejected** — plus two items that are not hypotheses: a robustness check
(execution timing) and a methodology choice (longer sample).

**July 2026 complete-data retest — the verdict above the verdicts:** when
delisted coverage reached 89%, every variant (ensemble or pure momentum,
score- or equal-weighted) converged to index-like returns with alpha ≈ 0. The
two adopted items' measured gains did not survive; they, like the alpha, were
artifacts of the incomplete universe. The rules held: the retest was allowed
to overturn even the accepted answers.

| Hypothesis | Verdict | Why |
|---|---|---|
| Score weighting | **Adopted** | +1.4pp/yr, Sharpe up, beta flat, confirmed on 3 samples |
| 10% position cap | **Adopted** | Free risk reduction |
| Concentration (top 10-20) | Rejected | The extra return was beta in disguise (1.16→1.31, pre-adoption config); Sharpe fell |
| Inverse-volatility weights | Rejected | Halved the excess return |
| Volatility-target overlay | Rejected | Negative alpha on the tested sample |
| Fundamental factors ×3 (SEC EDGAR, point-in-time by filing date) | Rejected (all 3) | Profitability: right sign, t≈0.55 (too weak). Accruals: sign flipped between halves. Asset growth: wrong sign. Classic published factors have decayed in large caps |
| Residual momentum + consistency features | Rejected | Alpha 3.1%→0.0% — with weak signals, extra correlated features feed the model noise |
| Execution timing (next-open vs close) | Robustness check: ≈ 0 | Full-period return identical — implementable as modeled |
| Longer sample (2003 vs 2010 start) | Methodology choice | Doubled the sample (t-stats ×~1.4); includes 2008 |

Multiple-testing logic: try 1,000 patterns on 233 months and ~50 fake ones
will "pass" by chance. The defense is few hypotheses, strong priors
(published literature), pre-registered gates, and split-sample confirmation.

> **Say it like this:** "Seven of my nine ideas died, several after I'd spent
> days building them. A process that mostly says no is the only reason to
> believe it when it says yes."

---

## 7. The live tracking system (out-of-sample, since July 2026)

The problem it solves: *any* backtest, however careful, was built by someone
who saw the data. The only untouchable evidence is picks locked **before**
returns happen.

- **`paper_track.py` — locked snapshots + ledger.** On each monthly run it (a)
  freezes the new portfolio (tickers + weights) as
  `paper/portfolio_YYYY-MM.csv` — written once, never modified, refuses to
  overwrite; (b) grades last month's locked snapshot at real month-end prices,
  net of turnover costs, vs SPY, appending one row to `paper/ledger.csv` —
  the official record. The ledger also records two context benchmarks each
  month: **RSP** (equal-weight S&P 500 — controls for this portfolio's
  equal-weight-ish tilt) and **SPMO** (S&P 500 momentum-factor ETF — the
  "could I have just bought the momentum ETF?" test). The official excess is
  always measured against SPY.
- **`paper_broker.py` — the $100k simulated account.** Replays every snapshot
  as real orders: whole shares at month-end closes, 15 bps/side fees, cash
  tracks remainders, cost basis and realized P&L maintained. Outputs a full
  order blotter (`orders.csv`), positions (`positions.csv`), and
  `account.html` — a brokerage-statement view re-marked with live prices
  every market close.
- **`buy_list.py`** — prints the current holdings as dollars & share counts
  for any account size (also auto-written to `paper/latest_buy_list.txt`).
- **`paper_report.py`** — builds `dashboard.html`: equity curve vs SPY and
  monthly win/loss bars from the ledger.
- **Protocol**: no parameter changes for 12 months; judge at 12 ledger rows,
  decide at 24; losing streaks of 3-6 months are expected and are not a
  signal to intervene.

### 7.1 The Signal Observatory (forward-only pattern research)

Alongside the ledger, every monthly run passively logs the predictive power
(cross-sectional IC) of a **frozen battery of eight candidate signals** —
short-term reversal, residual momentum, low-volatility, seasonality,
52-week-high, the lottery/MAX effect, volume shocks, and illiquidity — against
the month that just completed (`paper/observatory.csv`, rules in
`OBSERVATORY.md`). The battery was registered (and amended pre-observation) in July 2026 and
cannot be changed; no signal month before July 2026 is ever logged, so every observation is genuinely
out-of-sample. A candidate earns a real hypothesis test only after 24+ logged
months with the literature's sign at t >= 1.5. This is how the system hunts
NEW patterns without the overfitting trap: the specification provably predates
every data point used to judge it. It reads the market; it never steers the
portfolio.

> **Say it like this:** "I can't mine my historical data any further without
> fooling myself, so I registered eight candidate signals in advance and let
> forward months accumulate clean evidence. Discovery, but pre-registered."

> **Say it like this:** "Each month's portfolio is frozen in a timestamped
> file before returns occur, then graded at real prices with costs. In two
> years I'll have evidence no backtest can fake — and the discipline rule is
> that I don't touch it in between."

---

## 8. The automation

- **`run_monthly.bat`** — the full monthly cycle, scheduled as Windows task
  `EquityStrategyMonthly`: **1st of each month, 3:15 PM Central** (4:15 PM ET —
  15 min after the close). Order: update prices → run model & rebalance → lock snapshot &
  score ledger → buy list → dashboard → account. If the PC is off, it runs
  at next power-on ("catch-up").
- **`daily_mark.bat`** — task `EquityStrategyDailyMark`, weekdays 3:20 PM Central:
  re-marks account.html at the day's close. Read-only.
- **Failure tripwire**: any failed step creates `paper/ATTENTION_NEEDED.txt`
  naming the failure; logs in `paper/run_log.txt`. No file = healthy.
- Timing deliberately doesn't matter: measurement is month-end-close to
  month-end-close from recorded history, and signals use only completed
  months — a late run computes identical numbers.
- Zero external/AI dependencies: plain Python + Task Scheduler;
  `README_OPERATIONS.md` documents every failure mode and fix.

> **Say it like this:** "It's a monthly-cadence system, so it's built to
> survive neglect: calendar-anchored measurement, catch-up scheduling, and a
> tripwire file instead of a human watching logs."

---

## 9. The presentation layer

- **`SCSES.html`** — hub linking everything (desktop shortcut: "Stock
  Strategy").
- **`research_overview.html`** — self-contained shareable tear sheet:
  summary stats, embedded audited charts, methodology, the hypothesis table,
  honest limitations. The thing you email.
- **`paper/dashboard.html`** — live performance vs SPY.
- **`paper/account.html`** — the brokerage view: positions, P&L, every order.
- **`presentations/Strategy_Research_Deck.pptx`** (+PDF) — the 10-slide
  interview deck; speaker notes on every slide. Generated by `gen_deck.js`
  with charts fed from the actual results CSVs.
- Design: one validated palette (blue #2A78D6 accent), stat tiles, tabular
  numbers, S&P always a muted reference line.

---

## 10. Interview Q&A — the questions you'll actually get

**"Why momentum?"** Strongest and most replicated cross-sectional anomaly in
the literature (Jegadeesh-Titman 1993 onward); behaviorally grounded
(underreaction, herding); and it survives *my own* out-of-sample gates,
unlike the value/quality factors I tested, which have decayed in large caps.

**"So the strategy doesn't work?"** As an investment, on complete data, it
has no demonstrated edge — index-like returns at 1.14× the risk — and I say
that plainly. What *worked* is the research system around it: the pipeline
measured an apparent +3.1%/yr alpha (t = 1.19), I treated my own data as the
prime suspect, spent $30 completing the delisted record, and watched the
alpha go to zero. Then I published that. Most backtests anyone will ever show
you are this project from before that purchase.

**"Give me a confidence interval on your alpha."** Point estimate −0.4%/yr
with t = −0.15, so the standard error is ~2.7pp and the 95% interval is
roughly **[−5.7%, +4.9%] per year** — centered on nothing. Under a flat prior
that's ~44% probability the true alpha is even positive (a Bayesian statement,
not a property of the interval). I quote the interval, not the point — and
this interval says "indistinguishable from the index." 

**"How would you bet on this?"** Small, and unlevered. Kelly-style sizing says
bet proportional to edge over variance — but my *edge estimate itself* has
huge variance, and betting full Kelly on a parameter you don't know is how
you go broke being right. So: paper record first, then modest real size, and
no leverage until the live evidence tightens the interval. Leverage is only
safe on edges you've *proven*, because one bad year at high leverage ends the
experiment before the statistics can vindicate it.

**"What would change your mind?"** Pre-committed before the live record
started: at 24 ledger months, if the live excess is negative or the alpha
estimate is drifting toward zero, I conclude luck or decay and stop —
regardless of how attached I am. Conversely, live results inside the
backtest's confidence band raise my posterior and would justify size. The
answer I'm *not* allowed to give is "re-examine and tune it" — that's how you
launder a dead strategy back to life on paper.

**"Why did adding features make it worse?"** Bias-variance. The signal is
faint, the new features were correlated with the old ones, so they added
model capacity without adding information — the trees fit noise dimensions
in-sample that didn't generalize. With weak signals, small models win.

**"Why monthly rebalancing?"** Momentum's horizon is months, not days; daily
trading triples costs with no extra signal; and monthly cadence makes the
system robust to operational reality (a late run changes nothing). Also
honesty: at higher frequencies my free data and cost model would no longer
be trustworthy.

**"So why not just buy SPY?"** That is literally the project's conclusion.
Risk-adjusted the strategy is clearly behind (Sharpe 0.51 vs 0.70), and its
compound return gap of +0.1pp/yr is noise. The deliverable was never "fire
your index fund" — it's a correctly-measured null result, and machinery that
proved strong enough to find it.

**"What breaks it?"** A momentum crash (2009-style sharp reversal — beta 1.14
and the factor both hurt at once); crowding in the factor; my remaining data
gap (11% of member-months, unobtainable in retail data — and the last
gap-closing moved returns by −4pp, so gaps are never assumed benign); and any
future me who starts tweaking parameters after a losing quarter — which the
12-month no-touch rule exists to prevent.

**"What would you do with real resources?"** Point-in-time fundamentals
(CRSP/Compustat) to retest the factor gates properly; the mid-cap universe
where my cross-market test shows a bigger edge; more markets rather than more
parameters — breadth is the honest way to scale a weak signal (IR ≈ IC ×
√breadth).

**"What did you actually learn?"** Excess return isn't alpha until you
regress out beta. Single stocks are 51/49 coins; systems are built from
breadth and repetition. More model is often worse. Published factors decay.
And the most valuable line on any results table is the one that says
*rejected*.

---

## Glossary (fast definitions)

- **Alpha** — return unexplained by market exposure; the "skill" component.
- **Beta** — sensitivity to the market (1.14 = moves 14% more than the S&P).
- **Sharpe ratio** — return per unit of volatility.
- **t-statistic** — how many standard errors an estimate is from zero; ≥2 ≈
  "statistically significant."
- **IC (information coefficient)** — rank correlation between predicted and
  realized returns; 0.01 is a *good* month for a real signal.
- **Walk-forward** — always training on the past, predicting the never-seen
  next period; the honest alternative to in-sample fitting.
- **Survivorship bias** — testing only on companies that survived; inflates
  results because the dead (mostly losers) are invisible.
- **Point-in-time** — using only information that existed on the decision
  date (membership, prices, filings).
- **Turnover** — fraction of the portfolio replaced per rebalance (~25%/mo here).
- **Drawdown** — peak-to-trough loss (worst here: −56%).
- **Pre-registration** — committing the pass/fail rule before seeing results,
  so you can't move goalposts.
- **Basis point (bp)** — 0.01%. Costs here: 15 bps per trade side.
