"""gta6-trophy-rarity-forecast-01 entry: per-trophy log-time trend forecast.

For every trophy of Grand Theft Auto VI on PlayStation 5, forecast its Exophase
earn rate at the case's target as five quantiles (5/25/50/75/95%), scored by
pinball loss on the logit scale.

Method (per trophy, in logit space, rates clamped to 0.01..99.99):
  1. Collapse the case history to distinct states. Exophase only changes a
     game's counts when it refreshes them, hours apart, so consecutive reads
     with identical players_tracked and rates are one state; repeats carry no
     information and would overweight stale periods.
  2. Fit a recency-weighted least-squares line, logit(rate) = a + b*ln(1+t),
     t = hours since 2026-11-19T00:00Z (release day). The log-time shape
     saturates: early trophies rise fast then flatten as the player base
     matures, and dilution from new tracked players bends rates down the
     same way. Half-life 4 h: the recent slope is what extrapolates, because
     the log-time curve bends (saturates) over days, so a long window's
     average slope overshoots the future. Tuned on a simulator of the
     Exophase dynamics (see notes): half-lives 3-5 h form a plateau, 4 h
     is its centre; longer half-lives lose to the house curve baseline.
  3. Shrink the slope toward 0 by fit quality: the t-statistic of the slope
     (weighted least-squares standard error) and the effective sample size
     under the recency weights. A noisy or thin history shrinks hard toward
     persistence. With fewer than 3 distinct states there is no trend to
     fit: the median is the last rate.
  4. Anchor the forecast at the fitted level at the last state and
     extrapolate the shrunk slope from there to the target, clamped to at
     most +-3.0 logit from the last rate as a guard against wild
     extrapolation. (Anchoring at the intercept would send a fully shrunk
     forecast back to the release-time level, far below a rising series'
     current rate.)
  5. Spread sigma = 0.7 * sqrt(sigma_rw^2 + sigma_fit^2): sigma_rw is the
     house random-walk term 0.5*sqrt(horizon_days); sigma_fit is the fit's
     residual std times sqrt(1 + 1/n_eff). No leverage term: it overstates
     the error when the shape is genuinely near-linear. The 0.7 multiplier
     is tuned on the simulator and is slightly wider than the in-simulator
     optimum (0.6) as insurance for real-world shocks the simulator does
     not model (weekend play patterns, patches). For an uninformative
     (flat) history the spread is at least 1.5 logit: the launch-night
     refresh moved every rate ~1.5 logit down, and a flat history gives no
     warning of the next refresh.
  6. Quantiles q_tau = inv_logit(median_logit + z_tau * sigma), z the
     standard normal's quantiles, rounded to 4 decimals half-up.
     A trophy at ~100% gets a narrow spread: it cannot move up, and it
     did not move down in the launch-night refresh.

Standard library only, no network, deterministic: the same case always
writes the same bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed here, but the line
# keeps the entrypoint identical in shape to the kit starter's).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (house term).
SIGMA_RW = 0.5
# Release day, the origin of the log-time axis (2026-11-19T00:00:00Z).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Recency half-life for the trend fit, in hours. Tuned: 4 h (plateau 3-5 h).
HALF_LIFE_HOURS = 4.0
# Spread multiplier on sqrt(sigma_rw^2 + sigma_fit^2). Tuned: 0.7.
SPREAD_K = 0.7
# Shrinkage strength: shrink = t^2 / (t^2 + SHRINK_K), times n_eff/(n_eff+2).
SHRINK_K = 0.5
# The median may not move further than this, in logit units, from the last rate.
MAX_LOGIT_MOVE = 3.0
# Spread floor for an uninformative (flat) history, in logit units: the
# launch-night refresh moved every rate about this far.
FLAT_HISTORY_SIGMA = 1.5
# A fitted residual std below this is treated as this (a perfect fit on a
# handful of points is not a certain forecast).
RESID_FLOOR = 0.05


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def distinct_states(history: list) -> list:
    """Collapse consecutive identical reads to one state each, keeping the
    first read's time. A state is (time, players_tracked, rates tuple)."""
    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev_key:
            states.append((parse_time(read["at"]), key[0], key[1]))
            prev_key = key
    return states


def forecast_trophy(points, last_rate: float, horizon_days: float,
                    x_target: float) -> list:
    """points: [(x, y, w)] with x = ln(1+t_hours), y = logit(rate),
    w = recency weight, oldest first. Returns the five quantiles."""
    sigma_rw = SIGMA_RW * math.sqrt(horizon_days)
    last_logit = logit(last_rate)

    if len(points) < 3:
        # No trend to fit: persistence median, wide spread for the next
        # refresh (a flat history gives no warning of it).
        median_logit = last_logit
        sigma = max(sigma_rw, FLAT_HISTORY_SIGMA)
    else:
        w_sum = sum(w for _, _, w in points)
        x_bar = sum(w * x for x, _, w in points) / w_sum
        y_bar = sum(w * y for _, y, w in points) / w_sum
        sxx = sum(w * (x - x_bar) ** 2 for x, _, w in points)
        if sxx <= 0:
            median_logit = last_logit
            sigma = max(sigma_rw, FLAT_HISTORY_SIGMA)
        else:
            sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in points)
            b = sxy / sxx
            a = y_bar - b * x_bar
            # Effective sample size under the recency weights: the fit is
            # only as strong as the states it actually weights. Using the
            # raw count here would pretend a heavily downweighted history
            # is full evidence, and under-shrink a noisy slope.
            w2_sum = sum(w * w for _, _, w in points)
            n_eff = (w_sum * w_sum) / w2_sum if w2_sum > 0 else float(len(points))
            resid2 = sum(w * (y - a - b * x) ** 2 for x, y, w in points) / w_sum
            resid = math.sqrt(max(resid2, RESID_FLOOR ** 2))
            # Standard error of the slope for weighted least squares.
            se_b = resid / math.sqrt(sxx) if sxx > 0 else float("inf")
            t_stat = abs(b) / se_b if se_b > 0 and math.isfinite(se_b) else 0.0
            shrink = (t_stat * t_stat) / (t_stat * t_stat + SHRINK_K)
            shrink *= n_eff / (n_eff + 2.0)
            b_shrunk = b * shrink
            # Anchor the forecast at the fitted level at the last state and
            # extrapolate the shrunk slope from there. (Anchoring at the
            # intercept a would send a fully shrunk forecast back to the
            # release-time level, far below a rising series' current rate.)
            x_last = points[-1][0]
            level_last = a + b * x_last
            median_logit = level_last + b_shrunk * (x_target - x_last)
            # Guard: never extrapolate wildly from the last observed rate.
            median_logit = min(last_logit + MAX_LOGIT_MOVE,
                               max(last_logit - MAX_LOGIT_MOVE, median_logit))
            median_logit = min(logit(99.99), max(logit(0.01), median_logit))
            # Spread: random-walk term plus the fit's residual, with a
            # prediction-interval factor on the effective sample size. No
            # leverage term: it overstates the error when the shape is
            # genuinely near-linear (measured in validation).
            sigma_fit = resid * math.sqrt(1.0 + 1.0 / n_eff)
            sigma = SPREAD_K * math.sqrt(sigma_rw * sigma_rw + sigma_fit * sigma_fit)

    # A trophy at ~100% cannot move up and barely moved down at the refresh.
    if last_rate >= 99.5:
        sigma = min(sigma, 0.3)
    sigma = max(sigma, 0.15)

    quantiles = []
    prev = None
    for z in Z:
        q = clamp(inv_logit(median_logit + z * sigma))
        q = round4(q)
        if prev is not None and q < prev:
            q = prev
        quantiles.append(q)
        prev = q
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log(1.0 + (target - RELEASE).total_seconds() / 3600.0)
    states = distinct_states(case["history"])
    last_time = states[-1][0]
    last_rates = states[-1][2]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Last non-null rate for this trophy.
        last_rate = None
        for _, _, rates in reversed(states):
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 1.0
        points = []
        for t, _, rates in states:
            if i < len(rates) and rates[i] is not None:
                t_hours = (t - RELEASE).total_seconds() / 3600.0
                x = math.log(1.0 + t_hours)
                y = logit(rates[i])
                age_h = (last_time - t).total_seconds() / 3600.0
                w = 0.5 ** (age_h / HALF_LIFE_HOURS)
                points.append((x, y, w))
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(points, last_rate, horizon_days,
                                          x_target)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
