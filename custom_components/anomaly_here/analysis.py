import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# List any sensor names here that should NOT be treated as behavioural

EXCLUDE_SENSORS = set()

LEARNING_WINDOW_DAYS = 14
SWEEP_SECONDS = list(range(10, 601, 10))
MIN_GAPS_FOR_ESTIMATE = 20
K_SENSITIVITY_MULTIPLIERS = [0.5, 1.0, 2.0]

ON_VALUES = {"ON", "OPEN"}
OFF_VALUES = {"OFF", "CLOSE"}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------


def truncate_events(events, days):
    """Restricts to the first N days from the dataset's own start."""
    cutoff = events["time"].min() + pd.Timedelta(days=days)
    return events[events["time"] < cutoff]


def discover_entities(events, exclude=None):

    exclude = exclude or set()
    return sorted(s for s in events["sensor"].unique() if s not in exclude)


def estimate_hold_times(events):

    holds = {}
    for sensor, group in events.groupby("sensor"):
        group = group.sort_values("time")
        on_times = group[group["value"] == "ON"]["time"].tolist()
        off_times = group[group["value"] == "OFF"]["time"].tolist()
        durations, oi = [], 0
        for on in on_times:
            while oi < len(off_times) and off_times[oi] <= on:
                oi += 1
            if oi < len(off_times):
                durations.append((off_times[oi] - on).total_seconds())
        if durations:
            holds[sensor] = float(np.percentile(durations, 1))
    return holds


def build_activations(events, holds, sensor=None):
    """Hold-corrected (on, off) intervals. sensor=None -> all sensors pooled (house scope)."""
    subset = events if sensor is None else events[events["sensor"] == sensor]
    activations = []
    for s, group in subset.groupby("sensor"):
        group = group.sort_values("time")
        on_times = group[group["value"] == "ON"]["time"].tolist()
        off_times = group[group["value"] == "OFF"]["time"].tolist()
        oi = 0
        for on in on_times:
            while oi < len(off_times) and off_times[oi] <= on:
                oi += 1
            off = off_times[oi] if oi < len(off_times) else on
            hold = holds.get(s, 0.0)
            true_off = max(on, off - pd.Timedelta(seconds=hold))
            activations.append((on, true_off))
    return activations


def merge_episodes(activations, gap_seconds):
    """Merges activations into episodes: gap <= gap_seconds -> same episode."""
    items = sorted(activations)
    gap = pd.Timedelta(seconds=gap_seconds)
    episodes, cur_start, cur_end = [], None, None
    for start, end in items:
        if cur_start is not None and start - cur_end <= gap:
            cur_end = max(cur_end, end) if cur_end is not None else end
        else:
            if cur_start is not None:
                episodes.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        episodes.append((cur_start, cur_end))
    return pd.DataFrame(episodes, columns=["start", "end"])


def raw_gaps_seconds(activations):
    """Inter-activation gaps in seconds, before merging."""
    items = sorted(activations)
    return np.array(
        [
            (items[i + 1][0] - items[i][1]).total_seconds()
            for i in range(len(items) - 1)
            if (items[i + 1][0] - items[i][1]).total_seconds() > 0
        ]
    )


# ---------------------------------------------------------------------------
# THRESHOLD ESTIMATION: mixture-fit, knee, stability
# ---------------------------------------------------------------------------
def mixture_fit_crossing(gaps_seconds):

    if len(gaps_seconds) < MIN_GAPS_FOR_ESTIMATE:
        return None
    log_gaps = np.log10(gaps_seconds).reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(log_gaps)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(log_gaps)
    if gmm2.bic(log_gaps) >= gmm1.bic(log_gaps):
        return None
    means = np.asarray(gmm2.means_).flatten()
    lo_idx, hi_idx = np.argsort(means)[0], np.argsort(means)[1]
    scan = np.linspace(means[lo_idx], means[hi_idx], 500).reshape(-1, 1)
    posterior = gmm2.predict_proba(scan)
    for i in range(len(scan) - 1):
        if posterior[i, lo_idx] >= 0.5 and posterior[i + 1, lo_idx] < 0.5:
            return float(10 ** scan[i, 0])
    return None


def episodes_per_day_sweep(activations, sweep_seconds):
    if not activations:
        return np.array([]), np.array([])
    n_days = max(len({t[0].date() for t in activations}), 1)
    counts = [len(merge_episodes(activations, g)) / n_days for g in sweep_seconds]
    return np.array(sweep_seconds, dtype=float), np.array(counts)


def find_knee(x, y):
    """Kneedle-style elbow: max perpendicular distance from the line through the endpoints."""
    if len(x) < 3:
        return None
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
    x1, y1, x2, y2 = x_n[0], y_n[0], x_n[-1], y_n[-1]
    denom = np.hypot(x2 - x1, y2 - y1)
    if denom < 1e-9:
        return None
    dist = np.abs((y2 - y1) * x_n - (x2 - x1) * y_n + x2 * y1 - y2 * x1) / denom
    return float(x[int(np.argmax(dist))])


def stability_plateau(x, y, tolerance_frac=0.05):

    if len(x) < 3:
        return None
    best_len, best_range, i = 0, None, 0
    while i < len(y):
        j, window = i, [y[i]]
        while j + 1 < len(y):
            trial = window + [y[j + 1]]
            med = np.median(trial)
            if med == 0 or max(abs(v - med) / med for v in trial) > tolerance_frac:
                break
            window.append(y[j + 1])
            j += 1
        if (j - i) > best_len:
            best_len, best_range = j - i, (i, j)
        i = j + 1 if j > i else i + 1
    if best_range is None:
        return None
    lo, hi = best_range
    return float((x[lo] + x[hi]) / 2)


def estimate_scope_threshold(activations, sweep_seconds=SWEEP_SECONDS):

    gaps = raw_gaps_seconds(activations)
    n_gaps = len(gaps)
    if n_gaps < MIN_GAPS_FOR_ESTIMATE:
        return {
            "n_gaps": n_gaps,
            "mixture_crossing": None,
            "knee": None,
            "stability_mid": None,
            "raw_estimate": None,
        }
    mixture_crossing = mixture_fit_crossing(gaps)
    x, y = episodes_per_day_sweep(activations, sweep_seconds)
    knee = find_knee(x, y)
    stability_mid = stability_plateau(x, y)
    raw_estimate = (
        knee
        if knee is not None
        else (stability_mid if stability_mid is not None else mixture_crossing)
    )
    return {
        "n_gaps": n_gaps,
        "mixture_crossing": mixture_crossing,
        "knee": knee,
        "stability_mid": stability_mid,
        "raw_estimate": raw_estimate,
    }


# ---------------------------------------------------------------------------
# HIERARCHICAL POOLING
# ---------------------------------------------------------------------------
def pool_toward_house(n, raw_estimate, house_G, k):
    """w = n/(n+k). Returns (final_G, weight)."""
    if raw_estimate is None:
        return house_G, 0.0
    w = n / (n + k)
    return w * raw_estimate + (1 - w) * house_G, w


def choose_k(entity_results):
    """K = the gap-count of a modestly-used entity (lower-third by volume)
    at the current learning window — entities busier than 'typical modest'
    are trusted close to fully; sparser ones lean on the house baseline."""
    counts = sorted(r["n_gaps"] for r in entity_results.values())
    return counts[len(counts) // 3]


# ---------------------------------------------------------------------------
# MAIN: single entry point, clean output
# ---------------------------------------------------------------------------
def run_merge(
    events_full,
    entities=None,
    learning_window_days=LEARNING_WINDOW_DAYS,
    exclude_sensors=EXCLUDE_SENSORS,
):

    if entities is None:
        entities = discover_entities(events_full, exclude=exclude_sensors)
        print(f"Discovered {len(entities)} entities: {entities}")

    events = truncate_events(events_full, learning_window_days)
    holds = estimate_hold_times(events)

    house_result = estimate_scope_threshold(build_activations(events, holds))
    house_G = house_result["raw_estimate"]
    if house_G is None:
        raise SystemExit(
            f"House-scope estimate failed at {learning_window_days} days — "
            f"not enough data yet. Try a longer window."
        )

    entity_results = {
        e: estimate_scope_threshold(build_activations(events, holds, sensor=e))
        for e in entities
    }
    k = choose_k(entity_results)

    rows = [
        {
            "entity": "HOUSE",
            "n_gaps": house_result["n_gaps"],
            "raw_estimate_s": round(house_G, 1),
            "pooling_weight": None,
            "final_G_s": round(house_G, 1),
            "sensitivity_range_s": None,
            "rationale": "house-scope baseline, no pooling applied",
        }
    ]
    for entity, res in entity_results.items():
        n, raw = res["n_gaps"], res["raw_estimate"]
        final_G, w = pool_toward_house(n, raw, house_G, k)

        sensitivity_Gs = [
            pool_toward_house(n, raw, house_G, k * m)[0]
            for m in K_SENSITIVITY_MULTIPLIERS
        ]
        sensitivity_range = max(sensitivity_Gs) - min(sensitivity_Gs)

        if raw is None:
            rationale = f"only {n} gaps (<{MIN_GAPS_FOR_ESTIMATE}) -> fully pooled to house value"
        else:
            rationale = f"n={n}, w={w:.2f} -> {'own estimate trusted' if w > 0.7 else 'blended with house' if w > 0.3 else 'mostly house value'}"

        rows.append(
            {
                "entity": entity,
                "n_gaps": n,
                "raw_estimate_s": round(raw, 1) if raw is not None else None,
                "pooling_weight": round(w, 3),
                "final_G_s": round(final_G, 1),
                "sensitivity_range_s": round(sensitivity_range, 1),
                "rationale": rationale,
            }
        )

    result_df = pd.DataFrame(rows)

    # result_df.to_csv("merge_thresholds.csv", index=False)
    # result_df.to_json("merge_thresholds.json", orient="records", indent=2)

    # print(
    #    f"Learning window: {learning_window_days} days | K = {k} "
    #    f"(sensitivity checked at {K_SENSITIVITY_MULTIPLIERS})"
    # )
    # print(result_df.to_string(index=False))
    # print("\nWritten: merge_thresholds.csv, merge_thresholds.json")
    return result_df
