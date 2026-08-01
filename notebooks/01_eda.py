"""
Phase 1 - Exploratory data analysis and defect sweep.

Run:  python notebooks/01_eda.py

Profiles data/train_test.csv and data/validation.csv before any modelling
decision is made. Findings are summarised in docs/findings.md; this script is
the reproducible evidence behind them. Read-only - it writes nothing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EARTH_RADIUS_MILES = 3958.8


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def load():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    # Derived columns used throughout. rate-per-mile is the natural unit for
    # freight: it strips out the dominant distance effect.
    train["rpm"] = train["posted_rate"] / train["distance"]
    for df in (train, valid):
        df["lane"] = df["pickup"] + " -> " + df["delivery"]
        df["haversine"] = haversine(df.pickup_lat, df.pickup_lon,
                                    df.delivery_lat, df.delivery_lon)
        df["circuity"] = df["distance"] / df["haversine"]
    return train, valid, december


# --------------------------------------------------------------------------
# 1. Shape, completeness, integrity
# --------------------------------------------------------------------------
def profile_completeness(train, valid):
    section("1. SHAPE / COMPLETENESS / INTEGRITY")
    print("train:", train.shape, " validation:", valid.shape)
    print("\nnulls (train):\n", train.isna().sum().loc[lambda s: s > 0].to_string() or "  none")
    print("\nnulls (validation):\n", valid.isna().sum().loc[lambda s: s > 0].to_string() or "  none")
    print("\nduplicate load_id  train:", train.load_id.duplicated().sum(),
          " validation:", valid.load_id.duplicated().sum())
    feat = [c for c in train.columns if c not in ("load_id", "posted_rate", "rpm", "lane",
                                                  "haversine", "circuity")]
    print("duplicate feature rows (train):", train.duplicated(feat).sum())
    for df, name, prefix in [(train, "train", "TR"), (valid, "validation", "TE")]:
        ok = df.load_id.str.match(rf"^{prefix}-\d{{6}}$").sum()
        print(f"{name}: load_id format ok {ok}/{len(df)} | date-sorted: {df.date.is_monotonic_increasing}")

    print("\nstring hygiene (whitespace / casing collisions):")
    for df, name in [(train, "train"), (valid, "validation")]:
        for col in ["pickup", "delivery", "equipment"]:
            s = df[col].astype(str)
            ws = (s != s.str.strip()).sum()
            print(f"  {name:10s} {col:10s} stray-whitespace {ws:3d} | "
                  f"raw {s.nunique():3d} vs casefolded {s.str.lower().nunique():3d}")


# --------------------------------------------------------------------------
# 2. Target structure
# --------------------------------------------------------------------------
def profile_target(train):
    section("2. TARGET: posted_rate")
    print(train.posted_rate.describe(percentiles=[.01, .05, .5, .95, .99]).to_string())
    print("\nskew raw %.3f -> skew log %.3f  (log transform is the right scale)"
          % (train.posted_rate.skew(), np.log(train.posted_rate).skew()))
    print("non-positive:", (train.posted_rate <= 0).sum(), "| nulls:", train.posted_rate.isna().sum())

    print("\nrate-per-mile by distance band (economies of scale):")
    bands = pd.cut(train.distance, [0, 250, 500, 1000, 1500, 2000, 3000, 10_000])
    print(train.groupby(bands, observed=True)
          .agg(n=("rpm", "size"), rpm_mean=("rpm", "mean"), rate_mean=("posted_rate", "mean"))
          .round(3).to_string())

    print("\nequipment:")
    print(train.groupby("equipment")
          .agg(n=("rpm", "size"), rate_mean=("posted_rate", "mean"), rpm_mean=("rpm", "mean"))
          .round(3).to_string())


# --------------------------------------------------------------------------
# 3. Label contamination
# --------------------------------------------------------------------------
def profile_label_noise(train):
    section("3. LABEL CONTAMINATION (rate-per-mile tail)")
    print("A genuine heavy tail decays smoothly. This one plateaus, which means")
    print("the extreme rows are a separate injected population, not real loads.\n")
    for threshold in [3, 3.5, 4, 5, 6, 8, 10]:
        n = (train.rpm > threshold).sum()
        print(f"  rpm > {threshold:<4}: {n:5d} rows ({n / len(train):.3%})")
    print(f"\n  rpm < 1.0 : {(train.rpm < 1).sum():5d} rows "
          f"({(train.rpm < 1).mean():.3%})   <- implausibly cheap, mirror image")

    print("\nworst offenders (short hauls priced like cross-country runs):")
    cols = ["pickup", "delivery", "distance", "weight", "equipment", "date", "posted_rate", "rpm"]
    print(train.nlargest(5, "rpm")[cols].round(2).to_string())

    print("\ncontamination rate by month (uniform => injected noise, not a market event):")
    flag = (train.rpm > 4) | (train.rpm < 1)
    print(train.assign(bad=flag).groupby(train.date.dt.to_period("M"))["bad"]
          .agg(["mean", "sum"]).round(4).to_string())


# --------------------------------------------------------------------------
# 4. Feature defects
# --------------------------------------------------------------------------
def profile_feature_defects(train, valid):
    section("4. FEATURE DEFECTS")

    print("-- weight --")
    for df, name in [(train, "train"), (valid, "validation")]:
        w = df.weight
        print(f"  {name:10s} null {w.isna().sum():4d} | negative {(w < 0).sum():4d} | "
              f"zero {(w == 0).sum():3d} | at cap 47500 {(w == 47500).sum():5d} "
              f"({(w == 47500).mean():.2%})")
    neg = train.loc[train.weight < 0, "weight"].abs()
    pos = train.loc[train.weight > 0, "weight"]
    print(f"  abs(negative) median {neg.median():,.0f} vs positive median {pos.median():,.0f}"
          "  -> sign flips, recoverable with abs()")

    print("\n-- distance --")
    for df, name in [(train, "train"), (valid, "validation")]:
        print(f"  {name:10s} min {df.distance.min()} | exactly 70.0: {(df.distance == 70.0).sum():3d}"
              f" | max {df.distance.max()}")
    print("  -> hard floor at 70 miles: short hauls are censored, not measured")

    print("\n-- market_index / quote_signal --")
    for df, name in [(train, "train"), (valid, "validation")]:
        for col in ["market_index", "quote_signal"]:
            s = df[col]
            print(f"  {name:10s} {col:13s} null {s.isna().sum():4d} | "
                  f"range [{s.min():.3f}, {s.max():.3f}]")
    print("\n  market_index nulls by month (uniform => missing at random):")
    print("  " + train.assign(m=train.market_index.isna())
          .groupby(train.date.dt.to_period("M"))["m"].sum().to_string().replace("\n", "\n  "))


# --------------------------------------------------------------------------
# 5. Geography
# --------------------------------------------------------------------------
def profile_geography(train, valid):
    section("5. GEOGRAPHY")
    print("corr(haversine, distance) = %.4f" % train.haversine.corr(train.distance))
    print("circuity (distance / great-circle):")
    print(train.circuity.describe(percentiles=[.01, .5, .99]).round(3).to_string())
    print("\n  median ~1.18 matches the real-world road-circuity factor, so the")
    print("  coordinates are internally consistent even though they do not match")
    print("  true US geography.\n")
    for df, name in [(train, "train"), (valid, "validation")]:
        print(f"  {name:10s} circuity > 2.0: {(df.circuity > 2.0).sum():3d} rows "
              f"(max {df.circuity.max():.1f}) -- city pairs placed too close together")

    print("\ncity coordinate consistency:")
    multi = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique()
    print("  train cities with >1 coordinate:", (multi > 1).any(axis=1).sum(), "of", len(multi))
    tc = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    vc = valid.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    both = tc.join(vc, how="inner", lsuffix="_t", rsuffix="_v")
    mismatch = ((~np.isclose(both.pickup_lat_t, both.pickup_lat_v))
                | (~np.isclose(both.pickup_lon_t, both.pickup_lon_v))).sum()
    print(f"  shared cities {len(both)} | train/validation coordinate mismatches {mismatch}")

    print("\nunseen geography in validation:")
    unseen_cities = (set(valid.pickup) | set(valid.delivery)) - (set(train.pickup) | set(train.delivery))
    unseen_lanes = set(valid.lane) - set(train.lane)
    print("  cities:", len(unseen_cities), sorted(unseen_cities))
    print(f"  lanes : {len(unseen_lanes)} of {valid.lane.nunique()} "
          f"({len(unseen_lanes) / valid.lane.nunique():.1%})")
    print(f"  train rows per lane: median {train.lane.value_counts().median():.0f}, "
          f"p10 {train.lane.value_counts().quantile(.1):.0f}")
    print("  -> lane-level target encoding misses ~1 row in 6; coordinates must carry the fallback")


# --------------------------------------------------------------------------
# 6. Time structure - the axis the whole assignment turns on
# --------------------------------------------------------------------------
def profile_time(train, valid):
    section("6. TIME STRUCTURE")
    print("train     ", train.date.min().date(), "->", train.date.max().date(),
          f"({train.date.nunique()} days)")
    print("validation", valid.date.min().date(), "->", valid.date.max().date(),
          f"({valid.date.nunique()} days)")
    full = pd.date_range(train.date.min(), train.date.max(), freq="D")
    print("missing calendar days in train:", len(set(full) - set(train.date.unique())))

    print("\nis market_index a per-day series or per-load noise?")
    for col in ["market_index", "quote_signal"]:
        g = train.groupby("date")[col]
        within, across = g.std().mean(), g.mean().std()
        verdict = "daily series + small noise" if within < across else "mostly per-load noise"
        print(f"  {col:13s} within-day std {within:.4f} | across-day std {across:.4f} "
              f"| ratio {within / across:5.2f}  -> {verdict}")

    print("\nmonthly means:")
    m = train.groupby(train.date.dt.to_period("M")).agg(
        rpm=("rpm", "mean"), market_index=("market_index", "mean"),
        quote_signal=("quote_signal", "mean"))
    m["rpm_vs_jan"] = (m.rpm / m.rpm.iloc[0] - 1).map("{:+.1%}".format)
    print(m.round(3).to_string())
    print("\nvalidation months:")
    print(valid.groupby(valid.date.dt.to_period("M"))
          .agg(n=("load_id", "size"), market_index=("market_index", "mean"),
               quote_signal=("quote_signal", "mean")).round(3).to_string())

    print("\nday-of-week effect on rpm (negligible):")
    print(train.groupby(train.date.dt.dayofweek)["rpm"].mean().round(3).to_string())


# --------------------------------------------------------------------------
# 7. Does market_index actually matter?
# --------------------------------------------------------------------------
def profile_market_index_effect(train):
    section("7. market_index: MARGINAL vs CONDITIONAL EFFECT")
    print("marginal correlation with posted_rate: %.3f  (looks like a decoy)"
          % train.market_index.corr(train.posted_rate))
    print("marginal correlation with rpm        : %.3f"
          % train.market_index.corr(train.rpm))
    print("\nbut distance confounds it. Within distance bands, on rpm:\n")
    bands = pd.cut(train.distance, [0, 500, 1000, 2000, 10_000])
    for band, grp in train.groupby(bands, observed=True):
        grp = grp.dropna(subset=["market_index"])
        quintile = pd.qcut(grp.market_index, 5, labels=["q1", "q2", "q3", "q4", "q5"])
        means = grp.groupby(quintile, observed=True)["rpm"].mean()
        print(f"  {str(band):16s} n={len(grp):6d}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in means.items())
              + f"   spread {means.max() - means.min():.3f}")
    print("\n  -> monotonic in every band. The feature is real; the raw correlation was masked.")


# --------------------------------------------------------------------------
# 8. Train/validation drift
# --------------------------------------------------------------------------
def profile_drift(train, valid):
    section("8. TRAIN vs VALIDATION DRIFT")
    for col in ["distance", "weight", "market_index", "quote_signal"]:
        a, b = train[col].dropna(), valid[col].dropna()
        if col == "weight":
            a, b = a[a > 0], b[b > 0]
        print(f"  {col:14s} train mean {a.mean():10.3f} | valid mean {b.mean():10.3f} "
              f"| shift {(b.mean() - a.mean()) / a.mean():+7.2%}")
    print("\n  market_index range: train [%.3f, %.3f] vs validation [%.3f, %.3f]"
          % (train.market_index.min(), train.market_index.max(),
             valid.market_index.min(), valid.market_index.max()))
    print("  -> validation sits entirely in a softer market than the training average.")
    print("     Sep-Oct train months share that regime, which is what makes a")
    print("     late-window temporal holdout the honest choice.")

    print("\nequipment mix:")
    print(pd.concat([train.equipment.value_counts(normalize=True).rename("train"),
                     valid.equipment.value_counts(normalize=True).rename("valid")],
                    axis=1).round(4).to_string())


# --------------------------------------------------------------------------
# 9. December chart anchor
# --------------------------------------------------------------------------
def profile_december_lane(train, december):
    section("9. DECEMBER CHART LANE: Lexington -> Fort Wayne")
    lane = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne")]
    print("matching rows in train:", len(lane))
    if len(lane):
        print(lane[["distance", "weight", "posted_rate", "rpm"]]
              .describe(percentiles=[.5]).round(2).to_string())
        print(f"\n  anchor: a 360-mile Dry Van on this lane historically prices near "
              f"${lane.posted_rate.mean():,.0f}.")
        print("  December predictions landing far outside ~$800-1000 indicate a bug.")
    print("\ndecember_chart_inputs: constant distance/weight:",
          december.distance.nunique() == 1, december.weight.nunique() == 1,
          "| predicted_rate empty:", december.predicted_rate.isna().all())
    print("note: file carries only 7 columns - no lat/lon, no market_index, no quote_signal.")


def main():
    train, valid, december = load()
    profile_completeness(train, valid)
    profile_target(train)
    profile_label_noise(train)
    profile_feature_defects(train, valid)
    profile_geography(train, valid)
    profile_time(train, valid)
    profile_market_index_effect(train)
    profile_drift(train, valid)
    profile_december_lane(train, december)


if __name__ == "__main__":
    main()
