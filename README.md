# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/itaro9110/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/itaro9110/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one — 5 quantiles per trophy, from a per-trophy logit-space log-time trend model.

## Run it

```sh
python3 main.py
```

Reads `/input/<id>.in` (schema makefun/gta6-trophies-case/v1) and writes `/output/<id>.out`: 55 trophies, 5 quantiles each (q05, q25, q50, q75, q95), 0–100, non-decreasing. Python standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ptcn8f06&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## Method

Per trophy, in logit space (rates clamped to 0.01–99.99 before the logit):

1. Collapse history to distinct states (consecutive reads with identical players + rates are one state — the page only changes at Exophase refreshes).
2. Fit weighted least squares: logit(rate) = a + b · ln(1 + t), t = hours since 2026-11-19T00:00Z, recency-weighted (half-life 4 h).
3. Shrink the slope toward 0 by fit quality (few distinct states / insignificant t-stat → shrink hard; <3 distinct states → persistence median).
4. Median at target = inv_logit(a + b_shrunk · ln(1 + t_target)), clamped to at most ±3.0 logit from the last rate.
5. Spread sigma = sqrt(sigma_rw² + sigma_fit²): random-walk term 0.5·sqrt(horizon_days) plus residual std × prediction-interval factor; flat history gets a wide spread (≥1.5 logit) to cover a refresh jump.
6. Quantiles from the median logit ± z·sigma, z = (−1.645, −0.674, 0, 0.674, 1.645), rounded to 4 decimals half-up. Near-100% trophies get a small spread.

## Validation

- Practice (p00/p01): format + determinism (two runs byte-identical). Loss vs a reconstructed practice answer: **mine 0.7929** vs persistence baseline 1.4009 and curve baseline 1.4009; 90% coverage 0.98 vs the baselines' 0.00.
- Synthetic final-like cases (11 scenarios: slow/fast growth, fast/slow refresh, extra seeds, low/high caps, noisy refresh, regime change): **mean pinball loss 0.2174, beats the curve baseline (0.3463) in 11/11 scenarios and persistence (0.5789) in 11/11.**
- Edge cases (platinum at 0.00% + null rates, single-read history, 288 reads with rising states, short rates lists) all produce valid output.

## License

MIT — see `LICENSE`.
