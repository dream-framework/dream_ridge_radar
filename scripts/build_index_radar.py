#!/usr/bin/env python3
"""Build the RidgePulse / DREAM Index Ridge Radar JSON bundle.

Live-only policy:
- Downloads public daily index history through yfinance.
- Computes ridge, operational dust cloud, rolling S2 retention fits, event scores,
  and historical crash/bull-run scorecards.
- Writes data/derived/market_ridge_radar.json for the static GitHub Pages app.
- Does not generate synthetic/demo fallback data. If live data cannot be fetched
  or computed for enough indices, the build fails.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "indices.yml"
OUT_DIR = ROOT / "data" / "derived"
OUT_PATH = OUT_DIR / "market_ridge_radar.json"

WINDOWS_S2 = np.array([3, 5, 8, 13, 21, 34, 55, 89, 144], dtype=float)
EPS = 1e-12

CRASH_WINDOWS = [
    {"key": "crash_1987", "label": "1987 crash", "start": "1987-08-25", "end": "1987-12-31", "kind": "crash"},
    {"key": "dotcom", "label": "Dot-com unwind", "start": "2000-03-24", "end": "2002-10-09", "kind": "crash"},
    {"key": "gfc", "label": "Global Financial Crisis", "start": "2007-10-09", "end": "2009-03-09", "kind": "crash"},
    {"key": "covid", "label": "COVID liquidity shock", "start": "2020-02-19", "end": "2020-03-23", "kind": "crash"},
    {"key": "rate_2022", "label": "2022 rates / inflation reset", "start": "2022-01-03", "end": "2022-10-12", "kind": "crash"},
]

BULL_WINDOWS = [
    {"key": "post_gfc_bull", "label": "Post-GFC bull leg", "start": "2009-03-09", "end": "2010-04-23", "kind": "bull"},
    {"key": "covid_recovery", "label": "COVID recovery / liquidity bull", "start": "2020-03-23", "end": "2021-12-31", "kind": "bull"},
    {"key": "ai_liquidity_bull", "label": "AI / liquidity bull leg", "start": "2023-10-27", "end": "2025-02-19", "kind": "bull"},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_float(value: Any, digits: Optional[int] = None) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return round(x, digits) if digits is not None else x


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_records(df: pd.DataFrame, cols: List[str], digits: int = 4) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if df.empty:
        return records
    for _, row in df.iterrows():
        item: Dict[str, Any] = {}
        for col in cols:
            if col == "date":
                item[col] = pd.Timestamp(row[col]).strftime("%Y-%m-%d")
            else:
                item[col] = clean_float(row[col], digits)
        records.append(item)
    return records


def robust_z(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    med = series.rolling(window, min_periods=min_periods).median()
    mad = (series - med).abs().rolling(window, min_periods=min_periods).median()
    z = (series - med) / (1.4826 * mad.replace(0, np.nan))
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clip_score(x: Any, center: float = 50.0, scale: float = 12.5) -> np.ndarray:
    return np.clip(center + scale * np.asarray(x, dtype=float), 0, 100)


def fetch_yfinance(symbol: str, period: str = "max") -> Tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"yfinance unavailable: {exc}") from exc

    data = yf.download(symbol, period=period, auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise RuntimeError("empty yfinance result")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    close_col = "Close" if "Close" in data.columns else data.columns[-1]
    out = data[[close_col]].rename(columns={close_col: "close"}).dropna().copy()
    out = out[out["close"] > 0]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise RuntimeError("no valid positive close values")
    return out, "yfinance"


def future_min_return(close: pd.Series, horizon: int) -> pd.Series:
    vals = close.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals) - horizon - 1):
        out[i] = np.nanmin(vals[i + 1 : i + horizon + 1]) / vals[i] - 1.0
    return pd.Series(out, index=close.index)


def future_max_return(close: pd.Series, horizon: int) -> pd.Series:
    vals = close.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals) - horizon - 1):
        out[i] = np.nanmax(vals[i + 1 : i + horizon + 1]) / vals[i] - 1.0
    return pd.Series(out, index=close.index)


def fit_s2_retention(ret_window: pd.Series) -> Dict[str, Any]:
    x = ret_window.dropna().to_numpy(dtype=float)
    n = len(x)
    raw_std = float(np.nanstd(x))
    if n < 220 or raw_std <= EPS:
        return {"ok": False, "reason": "small_or_flat"}

    ws = WINDOWS_S2[WINDOWS_S2 < max(10, n / 3)]
    r_vals: List[float] = []
    used_ws: List[float] = []
    s = pd.Series(x)

    for w_float in ws:
        w = int(w_float)
        sm = s.rolling(w, min_periods=max(2, int(w * 0.7))).mean().dropna().to_numpy(dtype=float)
        if len(sm) < 10:
            continue
        val = float(np.nanstd(sm) / raw_std)
        if math.isfinite(val) and 0 < val < 1:
            r_vals.append(min(max(val, 1e-5), 0.99999))
            used_ws.append(float(w))

    if len(r_vals) < 5:
        return {"ok": False, "reason": "insufficient_retention_points"}

    w_arr = np.asarray(used_ws, dtype=float)
    r_arr = np.asarray(r_vals, dtype=float)
    y = np.log(-np.log(r_arr))
    xlog = np.log(w_arr)
    if not np.isfinite(y).all():
        return {"ok": False, "reason": "nonfinite_fit_points"}

    beta, intercept = np.polyfit(xlog, y, 1)
    beta = float(beta)
    intercept = float(intercept)
    if not math.isfinite(beta) or abs(beta) <= EPS:
        return {"ok": False, "reason": "bad_beta"}

    lam = float(np.exp(-intercept / beta))
    pred = intercept + beta * xlog
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > EPS else 0.0

    intercept_d1 = float(np.mean(y - xlog))
    pred_d1 = intercept_d1 + xlog
    sse_d1 = float(np.sum((y - pred_d1) ** 2))
    m = len(y)
    bic_s2 = m * math.log(max(sse / m, EPS)) + 2 * math.log(m)
    bic_d1 = m * math.log(max(sse_d1 / m, EPS)) + 1 * math.log(m)
    boundary = beta < 0.15 or beta > 5.0 or lam < min(w_arr) / 3 or lam > max(w_arr) * 20

    return {
        "ok": True,
        "lambda_q": lam,
        "beta": beta,
        "r2": r2,
        "delta_bic_vs_d1": bic_d1 - bic_s2,
        "n_points": m,
        "boundary": boundary,
        "w_min": float(np.min(w_arr)),
        "w_max": float(np.max(w_arr)),
    }


def rolling_s2(ret: pd.Series, window: int, stride: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if len(ret.dropna()) < window:
        return pd.DataFrame()

    for pos in range(window, len(ret), stride):
        fit = fit_s2_retention(ret.iloc[pos - window : pos])
        row = {"date": ret.index[pos], "ok": bool(fit.get("ok", False))}
        row.update(fit)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    fits = pd.DataFrame(rows).set_index("date").sort_index()
    for col in ["lambda_q", "beta", "r2", "delta_bic_vs_d1"]:
        fits[col] = pd.to_numeric(fits.get(col), errors="coerce")

    good_lambda = np.log(fits["lambda_q"].where(fits["lambda_q"] > 0))
    fits["lambda_flicker_raw"] = good_lambda.rolling(20, min_periods=5).std()
    fits["boundary_rate"] = fits.get("boundary", False).astype(float).rolling(20, min_periods=5).mean()
    fits["fit_failure_rate"] = (~fits["ok"].astype(bool)).astype(float).rolling(20, min_periods=5).mean()
    fits["lambda_flicker_score"] = 100 * (
        0.55 * np.clip(fits["lambda_flicker_raw"].fillna(0) / 0.60, 0, 1)
        + 0.25 * fits["boundary_rate"].fillna(0)
        + 0.20 * fits["fit_failure_rate"].fillna(0)
    )
    fits["lambda_flicker_score"] = fits["lambda_flicker_score"].clip(0, 100)
    return fits


def compute_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = df.copy().sort_index()
    s = s[s["close"] > 0]
    if len(s) < int(cfg["settings"].get("min_rows", 650)):
        raise RuntimeError(f"not enough rows after cleaning: {len(s)}")

    s["log_close"] = np.log(s["close"])
    s["ret"] = s["log_close"].diff()

    spans = cfg["settings"].get("ridge_spans", [21, 63, 126, 252])
    ridge_cols: List[str] = []
    for span in spans:
        span_i = int(span)
        col = f"ridge_{span_i}"
        s[col] = s["log_close"].ewm(span=span_i, adjust=False, min_periods=max(5, span_i // 3)).mean()
        ridge_cols.append(col)

    s["ridge"] = s[ridge_cols].median(axis=1)
    s["ridge_price"] = np.exp(s["ridge"])
    s["dust"] = s["log_close"] - s["ridge"]

    dust_w = int(cfg["settings"].get("dust_window", 63))
    base_w = int(cfg["settings"].get("baseline_window", 252))
    dust_med = s["dust"].rolling(dust_w, min_periods=max(15, dust_w // 3)).median()
    dust_mad = (s["dust"] - dust_med).abs().rolling(dust_w, min_periods=max(15, dust_w // 3)).median()
    s["dust_sigma"] = (1.4826 * dust_mad).replace(0, np.nan).ffill()
    s["dust_z"] = robust_z(s["dust_sigma"], base_w, min_periods=80).clip(-4, 6)
    s["dust_accel"] = np.log(s["dust_sigma"].replace(0, np.nan)).diff(21)
    s["dust_accel_z"] = robust_z(s["dust_accel"], base_w, min_periods=80).clip(-4, 6)
    s["excursion"] = (s["dust"] / s["dust_sigma"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-8, 8)
    s["outside_envelope"] = (s["excursion"].abs() > 1.75).astype(float)
    s["pullback_persist"] = s["outside_envelope"].rolling(21, min_periods=5).mean()
    s["ridge_slope_21"] = s["ridge"].diff(21)
    s["ridge_slope_63"] = s["ridge"].diff(63)
    s["ridge_curvature"] = s["ridge_slope_21"] - (s["ridge_slope_63"] / 3.0)
    s["vol_20"] = s["ret"].rolling(20, min_periods=8).std() * math.sqrt(252)
    s["vol_z"] = robust_z(s["vol_20"], base_w, min_periods=80).clip(-4, 6)
    s["drawdown_252"] = s["close"] / s["close"].rolling(252, min_periods=80).max() - 1.0
    s["envelope_hi"] = np.exp(s["ridge"] + 1.75 * s["dust_sigma"].fillna(0))
    s["envelope_lo"] = np.exp(s["ridge"] - 1.75 * s["dust_sigma"].fillna(0))

    s2 = rolling_s2(
        s["ret"],
        int(cfg["settings"].get("rolling_s2_window", 512)),
        int(cfg["settings"].get("rolling_s2_stride", 5)),
    )
    for col in ["lambda_q", "beta", "r2", "delta_bic_vs_d1", "lambda_flicker_score"]:
        if not s2.empty and col in s2:
            s[col] = s2[col].reindex(s.index).ffill()
        else:
            s[col] = np.nan
    s["lambda_flicker_score"] = s["lambda_flicker_score"].fillna(0).clip(0, 100)

    dust_pressure = pd.Series(clip_score(s["dust_z"], 45, 13), index=s.index)
    dust_accel_score = pd.Series(clip_score(s["dust_accel_z"], 45, 13), index=s.index)
    pullback_score = 100 * (
        0.65 * np.clip((s["excursion"].abs() - 1.0) / 2.0, 0, 1)
        + 0.35 * s["pullback_persist"].fillna(0)
    )
    flatten_score = pd.Series(clip_score(-s["ridge_curvature"].fillna(0) * 850, 40, 1), index=s.index).clip(0, 100)
    lambda_score = s["lambda_flicker_score"].fillna(0)
    s["risk_score"] = (
        0.27 * dust_pressure
        + 0.18 * dust_accel_score
        + 0.24 * pullback_score
        + 0.19 * lambda_score
        + 0.12 * flatten_score
    ).clip(0, 100)

    up_slope = pd.Series(clip_score(s["ridge_slope_63"].fillna(0) * 550, 45, 1), index=s.index).clip(0, 100)
    above_ridge = pd.Series(clip_score(s["excursion"].clip(-3, 3), 48, 11), index=s.index).clip(0, 100)
    lambda_stable = (100 - s["lambda_flicker_score"].fillna(50)).clip(0, 100)
    dust_ok = (100 - (s["dust_z"].clip(lower=0) * 16)).clip(0, 100)
    s["bull_raw_score"] = (0.42 * up_slope + 0.24 * above_ridge + 0.20 * lambda_stable + 0.14 * dust_ok).clip(0, 100)

    s["ret_21"] = s["log_close"].diff(21)
    s["ret_63"] = s["log_close"].diff(63)
    risk_gate = ((s["risk_score"].fillna(0) - 45.0) / 35.0).clip(0, 1)
    neg_21_gate = ((-s["ret_21"].fillna(0) - 0.02) / 0.08).clip(0, 1)
    neg_63_gate = ((-s["ret_63"].fillna(0) - 0.04) / 0.14).clip(0, 1)
    drawdown_gate = ((-s["drawdown_252"].fillna(0) - 0.05) / 0.15).clip(0, 1)
    below_ridge_gate = ((-s["excursion"].fillna(0) - 0.25) / 1.75).clip(0, 1)
    lambda_gate = ((s["lambda_flicker_score"].fillna(0) - 45.0) / 35.0).clip(0, 1)
    s["crash_gate"] = pd.concat(
        [risk_gate, neg_21_gate, neg_63_gate, drawdown_gate, below_ridge_gate, lambda_gate],
        axis=1,
    ).max(axis=1).fillna(0).clip(0, 1)
    s["bull_score"] = (s["bull_raw_score"] * np.power(1.0 - s["crash_gate"], 1.5)).clip(0, 100)

    s["state"] = np.select([s["risk_score"] >= 75, s["risk_score"] >= 55], ["RED", "YELLOW"], default="GREEN")
    return s, s2


def auc_score(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    mask = np.isfinite(y_true) & np.isfinite(score)
    y = y_true[mask].astype(int)
    sc = score[mask].astype(float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(sc).rank(method="average").to_numpy()
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metric_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(score)
    y = y_true[mask].astype(int)
    pred = (score[mask] >= threshold).astype(int)
    if len(y) == 0:
        return {"n": 0, "events": 0, "precision": None, "recall": None, "false_alert_rate": None, "hit_rate": None}

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return {
        "n": int(len(y)),
        "events": int(np.sum(y == 1)),
        "precision": clean_float(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": clean_float(tp / (tp + fn), 4) if (tp + fn) else None,
        "false_alert_rate": clean_float(fp / (fp + tn), 4) if (fp + tn) else None,
        "hit_rate": clean_float((tp + tn) / len(y), 4),
    }


def backtest_targets(feat: pd.DataFrame, cfg: Dict[str, Any], key: str, name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    close = feat["close"]

    for target in cfg["settings"].get("crash_targets", []):
        horizon = int(target["horizon_days"])
        threshold = float(target["threshold"])
        future = future_min_return(close, horizon)
        y = (future <= threshold).astype(float).where(future.notna(), np.nan).to_numpy()
        s2_score = feat["risk_score"].to_numpy(dtype=float)
        base_score = pd.Series(clip_score(feat["vol_z"].fillna(0), 45, 13), index=feat.index).to_numpy(dtype=float)
        rows.append({
            "index": key,
            "name": name,
            "target": target["key"],
            "label": target["label"],
            "model": "S2 ridge+dust+lambda",
            "auc": clean_float(auc_score(y, s2_score), 4),
            **metric_at_threshold(y, s2_score, 70),
        })
        rows.append({
            "index": key,
            "name": name,
            "target": target["key"],
            "label": target["label"],
            "model": "baseline vol-only",
            "auc": clean_float(auc_score(y, base_score), 4),
            **metric_at_threshold(y, base_score, 70),
        })

    for target in cfg["settings"].get("bull_targets", []):
        horizon = int(target["horizon_days"])
        threshold = float(target["threshold"])
        future = future_max_return(close, horizon)
        y = (future >= threshold).astype(float).where(future.notna(), np.nan).to_numpy()
        s2_score = feat["bull_score"].to_numpy(dtype=float)
        base_score = pd.Series(clip_score(feat["ridge_slope_63"].fillna(0) * 550, 45, 1), index=feat.index).clip(0, 100).to_numpy(dtype=float)
        rows.append({
            "index": key,
            "name": name,
            "target": target["key"],
            "label": target["label"],
            "model": "S2 coherent bull-thrust",
            "auc": clean_float(auc_score(y, s2_score), 4),
            **metric_at_threshold(y, s2_score, 65),
        })
        rows.append({
            "index": key,
            "name": name,
            "target": target["key"],
            "label": target["label"],
            "model": "baseline trend-only",
            "auc": clean_float(auc_score(y, base_score), 4),
            **metric_at_threshold(y, base_score, 65),
        })

    return rows


def case_studies(feat: pd.DataFrame, key: str, name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for win in CRASH_WINDOWS + BULL_WINDOWS:
        start = pd.Timestamp(win["start"])
        end = pd.Timestamp(win["end"])
        if feat.index.max() < start or feat.index.min() > end:
            continue

        pre = feat.loc[(feat.index >= start - pd.Timedelta(days=120)) & (feat.index < start)]
        during = feat.loc[(feat.index >= start) & (feat.index <= end)]
        if pre.empty or during.empty:
            continue

        start_price = float(during["close"].iloc[0])
        if win["kind"] == "crash":
            move = float(during["close"].min()) / start_price - 1.0
            score_col = "risk_score"
            trigger_level = 55
        else:
            move = float(during["close"].max()) / start_price - 1.0
            score_col = "bull_score"
            trigger_level = 60

        trigger = pre[pre[score_col] >= trigger_level]
        first_trigger = trigger.index[0] if not trigger.empty else None
        rows.append({
            "index": key,
            "name": name,
            "window": win["label"],
            "kind": win["kind"],
            "start": win["start"],
            "end": win["end"],
            "realized_move": clean_float(move, 4),
            "max_pre_score": clean_float(float(pre[score_col].max()), 2),
            "max_during_score": clean_float(float(during[score_col].max()), 2),
            "first_warning": first_trigger.strftime("%Y-%m-%d") if first_trigger is not None else None,
            "lead_days": int((start - first_trigger).days) if first_trigger is not None else None,
        })
    return rows


def make_narrative(cur: Dict[str, Any]) -> str:
    state = cur.get("state") or "NA"
    risk = cur.get("risk_score") or 0
    dust = cur.get("dust_z") or 0
    flicker = cur.get("lambda_flicker_score") or 0
    excursion = cur.get("excursion") or 0
    beta = cur.get("beta")
    lam = cur.get("lambda_q")
    gate = cur.get("crash_gate")

    if state == "RED":
        tone = "Red: ridge coherence is stressed and the correction-risk stack is active."
    elif state == "YELLOW":
        tone = "Yellow: the retained ridge is present, but dust pressure or lambda stability is no longer quiet."
    else:
        tone = "Green: the retained ridge is absorbing noise; no structural break flag is active."

    beta_txt = f"beta {beta:.2f}" if isinstance(beta, (int, float)) and math.isfinite(beta) else "beta unavailable"
    lam_txt = f"lambda_q {lam:.1f} sessions" if isinstance(lam, (int, float)) and math.isfinite(lam) else "lambda_q unavailable"
    gate_txt = f"crash gate {gate:.2f}" if isinstance(gate, (int, float)) and math.isfinite(gate) else "crash gate unavailable"
    return (
        f"{tone} Composite risk {risk:.1f}/100. Dust z {dust:.2f}, envelope excursion {excursion:.2f}, "
        f"lambda flicker {flicker:.1f}/100; {lam_txt}, {beta_txt}; {gate_txt}. "
        "Research-only structural health read."
    )


def build_index_payload(info: Dict[str, Any], raw: pd.DataFrame, source: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    feat, s2 = compute_features(raw, cfg)
    key = info["key"]
    name = info["name"]
    latest = feat.dropna(subset=["close"]).iloc[-1]

    cur = {
        "date": latest.name.strftime("%Y-%m-%d"),
        "close": clean_float(latest["close"], 4),
        "ridge_price": clean_float(latest["ridge_price"], 4),
        "risk_score": clean_float(latest["risk_score"], 2),
        "bull_score": clean_float(latest["bull_score"], 2),
        "bull_raw_score": clean_float(latest["bull_raw_score"], 2),
        "crash_gate": clean_float(latest["crash_gate"], 4),
        "state": str(latest["state"]),
        "dust_z": clean_float(latest["dust_z"], 2),
        "dust_accel_z": clean_float(latest["dust_accel_z"], 2),
        "excursion": clean_float(latest["excursion"], 2),
        "lambda_q": clean_float(latest["lambda_q"], 2),
        "beta": clean_float(latest["beta"], 3),
        "s2_r2": clean_float(latest["r2"], 3),
        "delta_bic_vs_d1": clean_float(latest["delta_bic_vs_d1"], 2),
        "lambda_flicker_score": clean_float(latest["lambda_flicker_score"], 2),
        "vol_20": clean_float(latest["vol_20"], 4),
        "drawdown_252": clean_float(latest["drawdown_252"], 4),
    }

    feats_for_json = feat.reset_index().rename(columns={"index": "date"})
    keep_cols = [
        "date", "close", "ridge_price", "risk_score", "bull_score", "bull_raw_score", "crash_gate",
        "dust_z", "dust_accel_z", "excursion", "lambda_q", "beta", "lambda_flicker_score",
        "drawdown_252", "vol_20", "envelope_hi", "envelope_lo",
    ]

    recent = feats_for_json.tail(1300)
    monthly = feats_for_json.iloc[::21].copy()
    event_mask = pd.Series(False, index=feat.index)
    for win in CRASH_WINDOWS + BULL_WINDOWS:
        start = pd.Timestamp(win["start"]) - pd.Timedelta(days=80)
        end = pd.Timestamp(win["end"]) + pd.Timedelta(days=30)
        event_mask |= (feat.index >= start) & (feat.index <= end)
    event_rows = feats_for_json.loc[event_mask.to_numpy()].iloc[::5].copy()
    long_df = pd.concat([monthly, event_rows]).drop_duplicates(subset=["date"]).sort_values("date")

    latest_s2 = pd.DataFrame()
    if not s2.empty:
        latest_s2 = s2.reset_index().rename(columns={"index": "date"}).tail(260)
    s2_tail_cols = ["date", "lambda_q", "beta", "r2", "delta_bic_vs_d1", "lambda_flicker_score"]

    return {
        "key": key,
        "symbol": info["symbol"],
        "name": name,
        "region": info.get("region"),
        "role": info.get("role"),
        "source": source,
        "rows": int(len(feat)),
        "start": feat.index.min().strftime("%Y-%m-%d"),
        "end": feat.index.max().strftime("%Y-%m-%d"),
        "current": cur,
        "narrative": make_narrative(cur),
        "series_recent": as_records(recent, keep_cols, 4),
        "series_long": as_records(long_df, keep_cols, 4),
        "s2_tail": as_records(latest_s2, s2_tail_cols, 4) if not latest_s2.empty else [],
        "backtest": backtest_targets(feat, cfg, key, name),
        "case_studies": case_studies(feat, key, name),
        "data_quality": {
            "download_source": source,
            "min_rows_met": bool(len(feat) >= int(cfg["settings"].get("min_rows", 650))),
            "s2_fits": int(len(s2)) if not s2.empty else 0,
            "s2_fit_ok_share": clean_float(float(s2["ok"].mean()), 4) if not s2.empty and "ok" in s2 else None,
        },
    }


def aggregate_summary(indices: List[Dict[str, Any]]) -> Dict[str, Any]:
    states = [idx["current"].get("state") for idx in indices]
    risk_scores = [idx["current"].get("risk_score") for idx in indices if idx["current"].get("risk_score") is not None]
    bull_scores = [idx["current"].get("bull_score") for idx in indices if idx["current"].get("bull_score") is not None]
    dust = [idx["current"].get("dust_z") for idx in indices if idx["current"].get("dust_z") is not None]
    flicker = [idx["current"].get("lambda_flicker_score") for idx in indices if idx["current"].get("lambda_flicker_score") is not None]

    red = states.count("RED")
    yellow = states.count("YELLOW")
    if red >= 3 or (red >= 1 and yellow >= 4):
        global_state = "RED"
    elif red >= 1 or yellow >= 3:
        global_state = "YELLOW"
    else:
        global_state = "GREEN"

    leaders = sorted(indices, key=lambda x: x["current"].get("risk_score") or -1, reverse=True)[:5]
    bulls = sorted(indices, key=lambda x: x["current"].get("bull_score") or -1, reverse=True)[:5]
    return {
        "global_state": global_state,
        "index_count": len(indices),
        "red_count": red,
        "yellow_count": yellow,
        "green_count": states.count("GREEN"),
        "median_risk": clean_float(np.nanmedian(risk_scores), 2) if risk_scores else None,
        "median_bull": clean_float(np.nanmedian(bull_scores), 2) if bull_scores else None,
        "median_dust_z": clean_float(np.nanmedian(dust), 2) if dust else None,
        "median_lambda_flicker": clean_float(np.nanmedian(flicker), 2) if flicker else None,
        "top_risk": [{"key": x["key"], "name": x["name"], "risk_score": x["current"].get("risk_score"), "state": x["current"].get("state")} for x in leaders],
        "top_bull": [{"key": x["key"], "name": x["name"], "bull_score": x["current"].get("bull_score")} for x in bulls],
    }


def flatten(groups: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group in groups:
        out.extend(group)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-indices", type=int, default=0, help="debug limit")
    args = parser.parse_args()

    cfg = load_config()
    universe = cfg.get("universe", [])
    if args.max_indices:
        universe = universe[: args.max_indices]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    indices: List[Dict[str, Any]] = []
    health: List[Dict[str, Any]] = []
    failure_messages: List[str] = []

    for info in universe:
        key = info["key"]
        try:
            raw, source = fetch_yfinance(info["symbol"], cfg["settings"].get("lookback_period", "max"))
            if len(raw) < int(cfg["settings"].get("min_rows", 650)):
                raise RuntimeError(f"too few live rows from yfinance: {len(raw)}")
            payload = build_index_payload(info, raw, source, cfg)
        except Exception as exc:
            msg = f"{key}: live fetch/compute failed: {str(exc)[:220]}"
            failure_messages.append(msg)
            health.append({
                "index": key,
                "source": "live_only",
                "status": "failed",
                "message": str(exc)[:220],
            })
            continue

        health.append({
            "index": key,
            "source": source,
            "status": "ok",
            "message": f"{len(raw)} live rows",
        })
        indices.append(payload)

    if not indices:
        raise SystemExit(
            "LIVE-ONLY BUILD FAILED: no live index datasets were built. "
            + " | ".join(failure_messages[:10])
        )

    min_live_indices = int(cfg["settings"].get("min_live_indices", 8))
    if len(indices) < min_live_indices:
        raise SystemExit(
            f"LIVE-ONLY BUILD FAILED: only {len(indices)} live indices built; "
            f"minimum required is {min_live_indices}. "
            + " | ".join(failure_messages[:10])
        )

    backtests = flatten([x["backtest"] for x in indices])
    cases = flatten([x["case_studies"] for x in indices])
    summary = aggregate_summary(indices)
    mode = "public_index_history_live_only"

    payload = {
        "schema_version": "ridge-radar-v1",
        "metadata": {
            "generated_at": now_iso(),
            "mode": mode,
            "data_source_note": "Front end reads this static bundle only. Scheduled GitHub Action builds from live public index history only. Synthetic/demo fallback is disabled.",
            "research_policy": "Research only. No Alpaca, no live orders, no h1 trading. Signals are structural-health labels and historical event backtests.",
            "dream_mapping": {
                "ridge": "retained low-frequency structure",
                "dust_cloud": "operational residual cloud after ridge extraction; not resurrected S1",
                "lambda_q": "rolling S2 coherence scale fitted from multiscale return retention",
                "beta": "effective stretched-exponential retention exponent",
            },
            "caveats": [
                "This is not investment advice and not a price forecast.",
                "Backtests are non-leaky feature/label tests but remain exploratory until verified with independent data vendors.",
                "Live-only build: if public market history cannot be fetched or computed, this script fails instead of publishing demo data.",
            ],
            "live_failures": len(failure_messages),
            "minimum_live_indices": min_live_indices,
        },
        "summary": summary,
        "targets": {
            "crash": cfg["settings"].get("crash_targets", []),
            "bull": cfg["settings"].get("bull_targets", []),
        },
        "indices": indices,
        "backtests": backtests,
        "case_studies": cases,
        "health": health,
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    print(
        f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB), "
        f"mode={mode}, indices={len(indices)}, failures={len(failure_messages)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
