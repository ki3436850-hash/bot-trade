"""
TradeMind — Trading-Only Telegram Bot (No Hardcoded APIs)
==========================================================
គ្មាន API Key ណាមួយក្នុងកូដ — ទាំងអស់មកពី Environment Variables។
បើ Variable ណាមួយទទេ → Feature នោះបង្ហាញ "Not configured" (មិន Fabricate data)។

ត្រូវការ Environment Variables:
  BOT_TOKEN                  — Telegram Bot Token (ទាល់តែមាន)
  ADMIN_ID / ADMIN_IDS       — Admin User ID(s)

  --- Market Data (យ៉ាងតិច ១ ត្រូវមាន) ---
  TWELVE_DATA_API_KEY        — Forex/Metals (សំខាន់បំផុត)
  ALPHA_VANTAGE_API_KEY      — Backup Forex
  FINNHUB_API_KEY            — Backup Forex + Crypto
  COINGECKO_API_KEY          — Crypto (ឥតគិតថ្លៃ មិនចាំបាច់ Key ក៏បាន)

  --- News / Macro ---
  NEWSAPI_KEY                — News headlines
  FRED_API_KEY               — Fed Funds Rate

  --- AI (ស្រេចចិត្ត — បើគ្មាន AI Synthesis/Vision មិនដំណើរការ) ---
  GEMINI_API_KEY
  DEEPSEEK_API_KEY
  OPENROUTER_API_KEY
  GROQ_API_KEY
  CEREBRAS_API_KEY
  CLAUDE_API_KEY

  --- Chart / Vision ---
  CHART_IMG_API_KEY          — TradingView chart snapshots

  --- EA Bridge ---
  EA_BRIDGE_TOKEN            — MT5 EA authentication
  EA_BRIDGE_HMAC_SECRET      — (ស្រេចចិត្ត) HMAC signing
"""

import json
import uuid
import logging
import os
import sys
import random
import re
import threading
import base64
import asyncio
import time
import secrets
import hmac
import hashlib
import math
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import requests
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters,
)
from telegram.request import HTTPXRequest

# ============================================================
# EMBEDDED TRADING MODULES (direction_core / prediction_tracker / strategy_ensemble)
# ============================================================
import types as _types
import sys as _sys

# ---- direction_core ----
_DC_SRC = r'''"""direction_core.py - shared Direction Validator."""
from datetime import datetime, timedelta

DEFAULTS = {
    "min_alignment": 0.60, "ltf_strong": 60, "htf_strong": 65,
    "min_room_atr": 1.0, "min_score": 55, "hv_min_alignment": 0.75,
    "hv_min_score": 65, "stale_bars": 3, "regime_gate": True,
    "min_strategy_score": 60,
}
_NO_TREND_REGIMES = ("RANGING", "LOW_VOLATILITY", "UNKNOWN")

def _cfg(risk, key):
    return (risk or {}).get("dv_" + key, DEFAULTS[key])

def drop_forming_candle(df, tf_seconds, now=None, utc_offset_hours=0.0):
    meta = {"forming_dropped": False, "tf_seconds": tf_seconds,
            "age_since_close_s": None, "last_bar": None}
    if df is None or not tf_seconds or "datetime" not in getattr(df, "columns", []) or len(df) < 2:
        return df, meta
    now = now or datetime.utcnow()
    try:
        last = df["datetime"].iloc[-1]
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_convert("UTC").tz_localize(None)
        last = last.to_pydatetime() - timedelta(hours=utc_offset_hours)
    except Exception:
        return df, meta
    close_ts = last + timedelta(seconds=tf_seconds)
    if close_ts > now:
        df = df.iloc[:-1].reset_index(drop=True)
        meta["forming_dropped"] = True
        try:
            prev = df["datetime"].iloc[-1]
            if getattr(prev, "tzinfo", None) is not None:
                prev = prev.tz_convert("UTC").tz_localize(None)
            prev = prev.to_pydatetime() - timedelta(hours=utc_offset_hours)
            last, close_ts = prev, prev + timedelta(seconds=tf_seconds)
        except Exception:
            return df, meta
    meta["last_bar"] = last.isoformat()
    meta["age_since_close_s"] = max(0, int((now - close_ts).total_seconds()))
    return df, meta

def stale_timeframes(valid, risk=None):
    limit = _cfg(risk, "stale_bars")
    out = []
    for r in valid:
        m = r.get("bar_meta") or {}
        age, tfs_ = m.get("age_since_close_s"), m.get("tf_seconds")
        if age is None or not tfs_ or tfs_ >= 86400:
            continue
        if age > limit * tfs_:
            out.append((r["timeframe"], age))
    return out

def entry_confirmation(ltf_results, htf_bias, strong=None):
    strong = DEFAULTS["ltf_strong"] if strong is None else strong
    if not ltf_results:
        return {"status": "NONE", "reasons": ["No entry-timeframe data"]}
    want = "BULL" if htf_bias == "BULLISH" else ("BEAR" if htf_bias == "BEARISH" else None)
    if want is None:
        return {"status": "NEUTRAL", "reasons": ["No HTF bias to confirm against"]}
    agree = [r for r in ltf_results if r["bias"] == want]
    oppose = [r for r in ltf_results if r["bias"] not in (want, "NEUTRAL") and r["strength"] >= strong]
    if oppose:
        return {"status": "CONFLICT",
                "reasons": [f"{r['timeframe']}: {r['bias']} against HTF ({r['strength']}/100)" for r in oppose]}
    if agree:
        return {"status": "CONFIRMED",
                "reasons": [f"{r['timeframe']}: {r['bias']} confirms ({r['strength']}/100)" for r in agree]}
    return {"status": "PULLBACK", "reasons": ["Entry timeframe neutral - no confirmation yet"]}

def mtf_alignment(valid, tfs, want):
    order = {tf: i for i, tf in enumerate(tfs)}
    rows = sorted(valid, key=lambda r: order.get(r["timeframe"], 99))
    if want is None:
        bulls = sum(1 for r in rows if r["bias"] == "BULL")
        bears = sum(1 for r in rows if r["bias"] == "BEAR")
        want = "BULL" if bulls >= bears else "BEAR"
    agree = sum(1 for r in rows if r["bias"] == want)
    return {"agree": agree, "total": len(rows), "label": f"{agree}/{len(rows)}", "side": want,
            "rows": [{"tf": r["timeframe"], "bias": r["bias"], "strength": r["strength"]} for r in rows]}

def _is_fallback_level(level, price):
    if level is None or not price:
        return True
    return abs(level - price * 1.02) < price * 1e-9 or abs(level - price * 0.98) < price * 1e-9

def validate_direction(*, valid, tfs, htf_tfs, ltf_tfs, htf, entry_conf,
                       confluence_score, risk=None, strategy_summary=None):
    wait, block, checks = [], [], {}
    want = {"BULLISH": "BULL", "BEARISH": "BEAR"}.get(htf.get("bias"))
    proposed = {"BULL": "BUY", "BEAR": "SELL"}.get(want)
    mtf = mtf_alignment(valid, tfs, want)
    stale = stale_timeframes(valid, risk)
    if stale:
        tf_, age = stale[0]
        block.append(f"Stale data ({tf_}: last closed bar {age // 60} min old)")
        checks["fresh_data"] = "fail"
    else:
        checks["fresh_data"] = "pass"
    if want is None:
        wait.append("HTF bias NEUTRAL")
        checks["htf_direction"] = "fail"
        mtf["status"] = "MIXED"
        return {"wait_reasons": wait, "block_reasons": block, "mtf": mtf, "checks": checks, "proposed": None}
    checks["htf_direction"] = "pass"
    st = entry_conf["status"]
    if st == "CONFLICT":
        wait.append("HTF/LTF conflict")
        wait.extend(entry_conf["reasons"])
    elif st == "PULLBACK":
        wait.append(f"HTF {htf['bias']} but no LTF confirmation yet")
    elif st == "NONE":
        wait.append("Insufficient LTF data")
    checks["entry_confirmation"] = "pass" if st == "CONFIRMED" else "fail"
    htf_strong = _cfg(risk, "htf_strong")
    opp_htf = [r for r in valid if r["timeframe"] in htf_tfs
               and r["bias"] not in (want, "NEUTRAL") and r["strength"] >= htf_strong]
    if opp_htf:
        wait.append("HTF disagreement")
        checks["htf_consistency"] = "fail"
    else:
        checks["htf_consistency"] = "pass"
    exec_r = next((r for r in reversed(valid) if r["timeframe"] in ltf_tfs), None) or valid[0]
    regime_exec = (exec_r.get("components") or {}).get("regime") or {}
    hv = regime_exec.get("regime") == "HIGH_VOLATILITY"
    min_align = _cfg(risk, "hv_min_alignment") if hv else _cfg(risk, "min_alignment")
    ratio = mtf["agree"] / mtf["total"] if mtf["total"] else 0
    if ratio + 1e-9 < min_align:
        wait.append(f"MTF alignment too low ({mtf['label']})")
        checks["mtf_alignment"] = "fail"
    else:
        checks["mtf_alignment"] = "pass"
    if opp_htf or st == "CONFLICT":
        mtf["status"] = "CONFLICT"
    elif checks["mtf_alignment"] == "pass":
        mtf["status"] = "ALIGNED"
    else:
        mtf["status"] = "MIXED"
    primary = next((r for r in valid if r["timeframe"] == tfs[0]), valid[0])
    price, atr = primary.get("price"), primary.get("atr")
    if not atr or atr != atr or atr <= 0 or not price:
        checks["sr_room"] = "skipped"
    else:
        need = _cfg(risk, "min_room_atr")
        near = []
        for r in [primary] + [x for x in valid if x["timeframe"] in htf_tfs and x is not primary]:
            lvl = r.get("resistance") if want == "BULL" else r.get("support")
            if _is_fallback_level(lvl, r.get("price")):
                continue
            dist = ((lvl - price) if want == "BULL" else (price - lvl)) / atr
            if 0 < dist < need:
                near.append((r["timeframe"], lvl, dist))
        if near:
            name = "resistance" if want == "BULL" else "support"
            tf_, lvl, dist = min(near, key=lambda x: x[2])
            wait.append(f"Major {name} nearby")
            checks["sr_room"] = "fail"
        else:
            checks["sr_room"] = "pass"
    rg0 = regime_exec.get("regime")
    if strategy_summary is not None:
        if rg0 in ("UNSTABLE", "UNKNOWN"):
            wait.append(f"Regime {rg0} - no strategy may trade")
            checks["strategy_support"] = "fail"
        else:
            need = _cfg(risk, "min_strategy_score")
            sup, opp = strategy_summary.get("support", 0), strategy_summary.get("oppose", 0)
            if strategy_summary.get("active_valid", 0) == 0 or sup < need:
                wait.append("No regime-appropriate strategy confirms")
                checks["strategy_support"] = "fail"
            elif opp and opp >= sup:
                wait.append("Active opposing strategy outweighs")
                checks["strategy_support"] = "fail"
            else:
                checks["strategy_support"] = "pass"
        if rg0 in ("TRENDING", "BREAKOUT") and regime_exec.get("bias") not in (want, "NEUTRAL", None):
            wait.append("Regime direction opposes")
            checks["regime"] = "fail"
        else:
            checks["regime"] = "pass"
    elif _cfg(risk, "regime_gate"):
        rg = regime_exec.get("regime")
        if rg in _NO_TREND_REGIMES:
            wait.append(f"Regime {rg} - trend setup not valid")
            checks["regime"] = "fail"
        elif rg in ("TRENDING", "BREAKOUT") and regime_exec.get("bias") not in (want, "NEUTRAL", None):
            wait.append("Regime opposes")
            checks["regime"] = "fail"
        else:
            checks["regime"] = "pass"
    else:
        checks["regime"] = "skipped"
    floor = _cfg(risk, "hv_min_score") if hv else _cfg(risk, "min_score")
    if confluence_score is not None and confluence_score < floor:
        wait.append(f"Signal score {confluence_score} < {floor}")
        checks["score_floor"] = "fail"
    else:
        checks["score_floor"] = "pass"
    return {"wait_reasons": wait, "block_reasons": block, "mtf": mtf, "checks": checks, "proposed": proposed}
'''
DC = _types.ModuleType("direction_core")
exec(compile(_DC_SRC, "<direction_core>", "exec"), DC.__dict__)
_sys.modules["direction_core"] = DC

# ---- prediction_tracker ----
_PT_SRC = r'''"""prediction_tracker.py"""
import calendar, math, uuid
from datetime import datetime, timedelta
import direction_core as DC

LOGIC_VERSION = "dv1"
NEUTRAL_ATR_K = 0.10
MIN_SAMPLE = 30
ROLLING_N = 100
EXPIRE_BARS = 10
GAP_BARS = 3
MAX_LIVE = 1000

def _ts(dt):
    return calendar.timegm(dt.utctimetuple()) + dt.microsecond / 1e6

BULL, BEAR, NEUT = "BULLISH", "BEARISH", "NEUTRAL"
_SIDE = {"BUY": BULL, "SELL": BEAR}

def build_prediction(result, symbol, timeframe, preset=None, now=None, k=NEUTRAL_ATR_K):
    if not result or not result.get("ok"):
        return None
    final = result.get("direction")
    proposed = (result.get("validator") or {}).get("proposed")
    side = _SIDE.get(final) or _SIDE.get(proposed)
    if side is None:
        return None
    per_tf = result.get("per_tf") or []
    prim = next((r for r in per_tf if r.get("timeframe") == timeframe), None)
    if not prim:
        return None
    meta = prim.get("bar_meta") or {}
    ref_bar, ref_close, atr = meta.get("ref_bar_raw"), meta.get("ref_close"), prim.get("atr")
    if not ref_bar or ref_close is None or not atr or atr != atr or atr <= 0:
        return None
    tf_seconds = meta.get("tf_seconds")
    if not tf_seconds:
        return None
    mtf = result.get("mtf") or {}
    regime = ((prim.get("components") or {}).get("regime") or {}).get("regime")
    return {
        "id": uuid.uuid4().hex[:8],
        "at": _ts(now or datetime.utcnow()),
        "symbol": symbol, "tf": timeframe, "preset": preset, "tf_seconds": tf_seconds,
        "ref_bar": str(ref_bar), "ref_close": float(ref_close), "atr": float(atr),
        "thr": round(float(k) * float(atr), 10), "k": k,
        "pred": side, "final": final, "taken": final in ("BUY", "SELL"),
        "score": result.get("confidence"),
        "mtf": mtf.get("label"), "mtf_status": mtf.get("status"),
        "regime": regime, "news": (result.get("news") or {}).get("level"),
        "strat": list((result.get("strategy_summary") or {}).get("supporters") or []),
        "prev_dir": meta.get("prev_dir"), "ver": LOGIC_VERSION,
        "status": "PENDING", "actual": None, "result": None,
        "target_bar": None, "target_close": None, "resolved_at": None,
    }

def prediction_key(p):
    return (p.get("symbol"), p.get("tf"), p.get("ref_bar"))

def add_prediction(preds, pred):
    if pred is None:
        return False
    k = prediction_key(pred)
    if any(prediction_key(p) == k for p in preds):
        return False
    preds.insert(0, pred)
    return True

def classify(pred_dir, actual):
    if actual == NEUT:
        return "INCONCLUSIVE"
    return "CORRECT" if pred_dir == actual else "WRONG"

def resolve_prediction(p, df, now=None, utc_offset_hours=0.0):
    if p.get("status") != "PENDING":
        return None
    now = now or datetime.utcnow()
    tfs = p.get("tf_seconds") or 0
    if df is None or len(df) == 0 or not tfs:
        return _maybe_expire(p, now, "no data")
    closed, _ = DC.drop_forming_candle(df, tfs, now=now, utc_offset_hours=utc_offset_hours)
    if closed is None or len(closed) == 0:
        return _maybe_expire(p, now, "no closed bars")
    import pandas as pd
    ref_ts = pd.Timestamp(p["ref_bar"])
    if closed["datetime"].iloc[0] > ref_ts:
        return _expired(now, "reference bar older than available data")
    later = closed[closed["datetime"] > ref_ts]
    if later.empty:
        return _maybe_expire(p, now, "target candle not closed yet")
    row = later.iloc[0]
    gap = (row["datetime"] - ref_ts).total_seconds()
    if tfs < 86400 and gap > GAP_BARS * tfs:
        return _expired(now, f"gap of {int(gap // tfs)} bars before next candle")
    move = float(row["close"]) - float(p["ref_close"])
    thr = float(p["thr"])
    actual = BULL if move > thr else (BEAR if move < -thr else NEUT)
    return {"status": "RESOLVED", "actual": actual, "result": classify(p["pred"], actual),
            "target_bar": row["datetime"].isoformat(), "target_close": float(row["close"]),
            "move_atr": round(move / float(p["atr"]), 3), "resolved_at": _ts(now)}

def _expired(now, why):
    return {"status": "EXPIRED", "actual": None, "result": "EXPIRED", "why": why, "resolved_at": _ts(now)}

def _maybe_expire(p, now, why):
    at, tfs = p.get("at"), p.get("tf_seconds") or 0
    if at and tfs and _ts(now) - at > EXPIRE_BARS * tfs:
        return _expired(now, why + f" after {EXPIRE_BARS} bars")
    return None

def fold_old(preds, archive, keep=MAX_LIVE):
    while len(preds) > keep:
        old = preds.pop()
        if old.get("status") == "RESOLVED" and old.get("taken"):
            for key in _buckets(old):
                row = archive.setdefault(key, {"n": 0, "c": 0, "w": 0, "x": 0})
                row["n"] += 1
                row["c" if old["result"] == "CORRECT" else "w" if old["result"] == "WRONG" else "x"] += 1

def _buckets(p):
    out = ["all", "side:" + p["pred"], "sym:" + str(p.get("symbol")), "tf:" + str(p.get("tf"))]
    if p.get("regime"):
        out.append("reg:" + str(p["regime"]))
    return out

def wilson(c, n, z=1.96):
    if n <= 0:
        return None
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)

def _tally(items):
    c = sum(1 for p in items if p["result"] == "CORRECT")
    w = sum(1 for p in items if p["result"] == "WRONG")
    x = sum(1 for p in items if p["result"] == "INCONCLUSIVE")
    return c, w, x

def _block(c, w, x, min_sample):
    dec, n = c + w, c + w + x
    out = {"n": n, "decided": dec, "correct": c, "wrong": w, "inconclusive": x,
           "neutral_rate": (x / n) if n else None, "accuracy": None, "ci": None,
           "reliable": dec >= min_sample}
    if dec:
        out["accuracy"] = c / dec
        out["ci"] = wilson(c, dec)
    return out

def compute_stats(preds, archive=None, symbol=None, tf=None, min_sample=MIN_SAMPLE, rolling_n=ROLLING_N):
    live = [p for p in preds if (symbol is None or p.get("symbol") == symbol) and (tf is None or p.get("tf") == tf)]
    done = [p for p in live if p.get("status") == "RESOLVED"]
    taken = [p for p in done if p.get("taken")]
    vetoed = [p for p in done if not p.get("taken")]
    st = {"pending": sum(1 for p in live if p.get("status") == "PENDING"),
          "expired": sum(1 for p in live if p.get("status") == "EXPIRED"),
          "filters": {"symbol": symbol, "tf": tf}}
    arc = archive if (archive and symbol is None and tf is None) else {}
    ac = arc.get("all", {"n": 0, "c": 0, "w": 0, "x": 0})
    c, w, x = _tally(taken)
    st["overall"] = _block(c + ac["c"], w + ac["w"], x + ac["x"], min_sample)
    st["archived_n"] = ac["n"]
    for side, key in ((BULL, "buy"), (BEAR, "sell")):
        sub = [p for p in taken if p["pred"] == side]
        a = arc.get("side:" + side, {"c": 0, "w": 0, "x": 0})
        sc, sw, sx = _tally(sub)
        st[key] = _block(sc + a["c"], sw + a["w"], sx + a["x"], min_sample)
    recent = taken[:rolling_n]
    st["rolling"] = _block(*_tally(recent), min_sample) if recent else _block(0, 0, 0, min_sample)
    st["rolling"]["window"] = len(recent)
    def group(field):
        out = {}
        for p in taken:
            if p.get(field):
                out.setdefault(p[field], []).append(p)
        return {k: _block(*_tally(v), min_sample) for k, v in out.items()}
    st["by_symbol"], st["by_tf"], st["by_regime"] = group("symbol"), group("tf"), group("regime")
    by_strat = {}
    for p in taken:
        for name in p.get("strat") or []:
            by_strat.setdefault(name, []).append(p)
    st["by_strategy"] = {k: _block(*_tally(v), min_sample) for k, v in by_strat.items()}
    st["by_score_band"] = {}
    for lo, hi, name in ((0, 60, "<60"), (60, 70, "60-69"), (70, 80, "70-79"), (80, 101, "80+")):
        band = [p for p in taken if p.get("score") is not None and lo <= p["score"] < hi]
        if band:
            st["by_score_band"][name] = _block(*_tally(band), min_sample)
    st["vetoed"] = _block(*_tally(vetoed), min_sample)
    dec = [p for p in taken if p["result"] in ("CORRECT", "WRONG")]
    base = {"n": len(dec)}
    if dec:
        bull_actual = sum(1 for p in dec if p["actual"] == BULL)
        bear_actual = len(dec) - bull_actual
        base["majority"] = max(bull_actual, bear_actual) / len(dec)
        pd_ = [p for p in dec if p.get("prev_dir") in (BULL, BEAR)]
        if pd_:
            base["prev_dir"] = sum(1 for p in pd_ if p["prev_dir"] == p["actual"]) / len(pd_)
            base["prev_dir_n"] = len(pd_)
        best = max(v for k, v in base.items() if k in ("majority", "prev_dir"))
        st["overall"]["edge_vs_best_baseline"] = (st["overall"]["accuracy"] - best) if st["overall"]["accuracy"] is not None else None
    st["baseline"] = base
    return st

def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"

def accuracy_line(block):
    if not block or not block.get("reliable"):
        return None
    return _pct(block["accuracy"]), block["decided"]

def format_stats(st):
    f = st["filters"]
    title = "VALIDATED ACCURACY" + (f" - {f['symbol']}" if f.get("symbol") else "") + (f" {f['tf']}" if f.get("tf") else "")
    o = st["overall"]
    lines = [f"\U0001F3AF {title}", "(next-candle direction, completed predictions only)", ""]
    if not o["decided"]:
        lines.append("No completed predictions yet.")
        lines.append(f"Pending: {st['pending']}  Expired: {st['expired']}")
        return "\n".join(lines)
    if o["reliable"]:
        lo, hi = o["ci"]
        lines.append(f"Overall: {_pct(o['accuracy'])}  (95% range {_pct(lo)} - {_pct(hi)})")
    else:
        lines.append(f"Overall: Insufficient validation data ({o['decided']} decided, need {MIN_SAMPLE})")
        lines.append(f"Raw so far: {_pct(o['accuracy'])} - not reliable yet")
    lines.append(f"Sample: {o['decided']} decided  |  Neutral moves: {_pct(o['neutral_rate'])}")
    for label, key in (("BUY", "buy"), ("SELL", "sell")):
        b = st[key]
        if b["decided"]:
            lines.append(f"{label}: {_pct(b['accuracy'])} ({b['decided']})" + ("" if b["reliable"] else "  - small sample"))
    r = st["rolling"]
    if r["decided"]:
        lines.append(f"Last {r['window']}: {_pct(r['accuracy'])} ({r['decided']} decided)")
    base = st["baseline"]
    if base.get("n"):
        lines += ["", "Baselines (same outcomes):"]
        if "majority" in base:
            lines.append(f"  Always-majority-direction: {_pct(base['majority'])}")
        if "prev_dir" in base:
            lines.append(f"  Repeat previous candle: {_pct(base['prev_dir'])} ({base['prev_dir_n']})")
        e = o.get("edge_vs_best_baseline")
        if e is not None and o["reliable"]:
            lines.append(f"  Edge vs best baseline: {e * 100:+.1f} pts")
    v = st["vetoed"]
    if v["decided"]:
        lines += ["", f"Validator-vetoed calls: {_pct(v['accuracy'])} ({v['decided']})"]
    for name, grp in (("By symbol", st["by_symbol"]), ("By timeframe", st["by_tf"]),
                      ("By regime", st["by_regime"]), ("By strategy", st.get("by_strategy", {})),
                      ("By signal score", st["by_score_band"])):
        rows = [(k, b) for k, b in grp.items() if b["decided"]]
        if rows:
            lines += ["", name + ":"]
            for k, b in sorted(rows, key=lambda kv: -kv[1]["decided"])[:6]:
                lines.append(f"  {k}: {_pct(b['accuracy'])} ({b['decided']})")
    lines += ["", f"Pending: {st['pending']}  Expired: {st['expired']}"]
    return "\n".join(lines)
'''

PT = _types.ModuleType("prediction_tracker")
exec(compile(_PT_SRC, "<prediction_tracker>", "exec"), PT.__dict__)
_sys.modules["prediction_tracker"] = PT

# ---- strategy_ensemble ----
_SE_SRC = r'''"""strategy_ensemble.py - 8 regime-gated strategies."""
import math
import pandas as pd

BUY, SELL, NONE = "BUY", "SELL", "NONE"
REGIME_ACTIVE = {
    "TRENDING":        {"trend_following", "pullback", "momentum", "market_structure"},
    "BREAKOUT":        {"breakout", "momentum", "trend_following", "market_structure"},
    "RANGING":         {"mean_reversion", "support_resistance", "market_structure"},
    "LOW_VOLATILITY":  {"mean_reversion", "support_resistance"},
    "HIGH_VOLATILITY": {"breakout", "trend_following", "pullback", "market_structure"},
    "UNSTABLE":        set(),
    "UNKNOWN":         set(),
}
ALWAYS_ACTIVE = {"news_event"}
STRATEGY_NAMES = ["trend_following", "breakout", "pullback", "mean_reversion",
                  "momentum", "support_resistance", "market_structure", "news_event"]

def _ev(name, direction, score, reason, valid, ctx):
    return {"strategy": name, "direction": direction, "score": int(max(0, min(100, round(score)))),
            "reason": reason, "valid": bool(valid), "active": True, "tf": (ctx or {}).get("tf")}

def _none(name, reason, ctx):
    return _ev(name, NONE, 0, reason, False, ctx)

def _ema(df, span):
    return df["close"].ewm(span=span, adjust=False).mean()

def _rsi_last(df, period=14):
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    v = (100 - 100 / (1 + g / l)).iloc[-1]
    return None if pd.isna(v) else float(v)

def _macd_hist(df):
    macd = _ema(df, 12) - _ema(df, 26)
    return macd - macd.ewm(span=9, adjust=False).mean()

def swing_points(df, window=3):
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    highs, lows = [], []
    for i in range(window, len(hi) - window):
        if hi[i] == hi[i - window:i + window + 1].max():
            highs.append((i, float(hi[i])))
        if lo[i] == lo[i - window:i + window + 1].min():
            lows.append((i, float(lo[i])))
    def collapse(points, keep_high):
        out = []
        for p in points:
            if out and p[0] - out[-1][0] <= window:
                better = p[1] > out[-1][1] if keep_high else p[1] < out[-1][1]
                if better:
                    out[-1] = p
            else:
                out.append(p)
        return out
    return collapse(highs, True), collapse(lows, False)

def _bar(df, k=-1):
    r = df.iloc[k]
    return float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])

def _atr_ok(ctx):
    a = (ctx or {}).get("atr")
    return a if (a is not None and a == a and a > 0) else None

def trend_following(df, ctx):
    n = "trend_following"
    atr = _atr_ok(ctx)
    if atr is None or len(df) < 60:
        return _none(n, "Not enough data/ATR", ctx)
    e20, e50 = _ema(df, 20), _ema(df, 50)
    close = float(df["close"].iloc[-1])
    slope = float(e20.iloc[-1] - e20.iloc[-6])
    adx = ctx.get("adx")
    up = e20.iloc[-1] > e50.iloc[-1] and close > e50.iloc[-1] and slope > 0
    dn = e20.iloc[-1] < e50.iloc[-1] and close < e50.iloc[-1] and slope < 0
    if not (up or dn):
        return _none(n, "EMA20/EMA50 not stacked with slope", ctx)
    if adx is not None and adx < 20:
        return _none(n, f"ADX {adx:.0f} < 20 - trend too weak", ctx)
    side = BUY if up else SELL
    ext = abs(close - float(e20.iloc[-1])) / atr
    st = ctx.get("structure_bias")
    agrees = (st == "BULL" and up) or (st == "BEAR" and dn)
    score = 55 + (min(25, max(0, (adx or 20) - 20)) if adx else 5) + (12 if agrees else 0)
    if ext > 2.5:
        return _ev(n, side, score - 25, f"Trend {side} but extended", False, ctx)
    return _ev(n, side, score, f"EMA20/50 stacked {side}", True, ctx)

def breakout(df, ctx, lookback=20):
    n = "breakout"
    atr = _atr_ok(ctx)
    if atr is None or len(df) < lookback + 3:
        return _none(n, "Not enough data/ATR", ctx)
    prior = df.iloc[-(lookback + 1):-1]
    res, sup = float(prior["high"].max()), float(prior["low"].min())
    o, h, l, c = _bar(df)
    rng = max(h - l, 1e-12)
    body = abs(c - o) / rng
    st = ctx.get("structure_bias")
    if h > res:
        if c <= res:
            return _ev(n, NONE, 0, f"Wick above {res:g} but closed under", False, ctx)
        confirmed = (c - res) >= 0.1 * atr and body >= 0.5 and (c - l) / rng >= 0.7
        late = (c - res) > 1.5 * atr
        if not confirmed:
            return _ev(n, BUY, 40, "Lacks momentum", False, ctx)
        if late:
            return _ev(n, BUY, 45, "Extended - late", False, ctx)
        return _ev(n, BUY, 70 + (10 if st == "BULL" else 0), f"Closed above {lookback}-bar high", True, ctx)
    if l < sup:
        if c >= sup:
            return _ev(n, NONE, 0, f"Wick below {sup:g} but closed above", False, ctx)
        confirmed = (sup - c) >= 0.1 * atr and body >= 0.5 and (h - c) / rng >= 0.7
        late = (sup - c) > 1.5 * atr
        if not confirmed:
            return _ev(n, SELL, 40, "Lacks momentum", False, ctx)
        if late:
            return _ev(n, SELL, 45, "Extended - late", False, ctx)
        return _ev(n, SELL, 70 + (10 if st == "BEAR" else 0), f"Closed below {lookback}-bar low", True, ctx)
    return _none(n, "No break of the recent range", ctx)

def pullback(df, ctx):
    n = "pullback"
    atr = _atr_ok(ctx)
    if atr is None or len(df) < 60:
        return _none(n, "Not enough data/ATR", ctx)
    e20, e50 = _ema(df, 20), _ema(df, 50)
    up = e20.iloc[-1] > e50.iloc[-1] and e50.iloc[-1] > e50.iloc[-6]
    dn = e20.iloc[-1] < e50.iloc[-1] and e50.iloc[-1] < e50.iloc[-6]
    if not (up or dn):
        return _none(n, "No established trend", ctx)
    st = ctx.get("structure_bias")
    rsi = _rsi_last(df)
    o, h, l, c = _bar(df)
    po, ph, pl, pc = _bar(df, -2)
    recent = df.iloc[-6:-1]
    if up:
        if st == "BEAR":
            return _none(n, "Uptrend EMAs but bearish structure", ctx)
        touched = float(recent["low"].min()) <= float(e20.iloc[-1]) + 0.25 * atr
        dipped = float(recent["close"].min()) < float(recent["close"].max()) - 0.4 * atr
        confirm = c > o and c > float(e20.iloc[-1]) and (c > ph or (c > po and pc < po))
        if not (touched and dipped):
            return _none(n, "No pullback yet", ctx)
        if not confirm:
            return _ev(n, BUY, 40, "No bullish confirmation", False, ctx)
        if rsi is not None and rsi > 75:
            return _ev(n, BUY, 40, f"RSI {rsi:.0f} stretched", False, ctx)
        return _ev(n, BUY, 68 + (8 if st == "BULL" else 0), "Uptrend pullback + confirm", True, ctx)
    if st == "BULL":
        return _none(n, "Downtrend EMAs but bullish structure", ctx)
    touched = float(recent["high"].max()) >= float(e20.iloc[-1]) - 0.25 * atr
    rallied = float(recent["close"].max()) > float(recent["close"].min()) + 0.4 * atr
    confirm = c < o and c < float(e20.iloc[-1]) and (c < pl or (c < po and pc > po))
    if not (touched and rallied):
        return _none(n, "No rally yet", ctx)
    if not confirm:
        return _ev(n, SELL, 40, "No bearish confirmation", False, ctx)
    if rsi is not None and rsi < 25:
        return _ev(n, SELL, 40, f"RSI {rsi:.0f} stretched", False, ctx)
    return _ev(n, SELL, 68 + (8 if st == "BEAR" else 0), "Downtrend rally + confirm", True, ctx)

def mean_reversion(df, ctx, lookback=30):
    n = "mean_reversion"
    atr = _atr_ok(ctx)
    adx = ctx.get("adx")
    if atr is None or len(df) < lookback + 2:
        return _none(n, "Not enough data/ATR", ctx)
    if adx is not None and adx >= 25:
        return _none(n, f"ADX {adx:.0f} >= 25", ctx)
    win = df.iloc[-lookback - 1:-1]
    lo, hi = float(win["low"].min()), float(win["high"].max())
    rng = hi - lo
    if rng < 1.5 * atr:
        return _none(n, "Range too narrow", ctx)
    o, h, l, c = _bar(df)
    candle_rng = max(h - l, 1e-12)
    rsi = _rsi_last(df)
    pos = (c - lo) / rng
    if pos <= 0.2 and c > o and (min(o, c) - l) / candle_rng >= 0.3 and (rsi is None or rsi < 45):
        return _ev(n, BUY, 66, f"Range low rejection", True, ctx)
    if pos >= 0.8 and c < o and (h - max(o, c)) / candle_rng >= 0.3 and (rsi is None or rsi > 55):
        return _ev(n, SELL, 66, f"Range high rejection", True, ctx)
    return _none(n, "Mid-range", ctx)

def momentum(df, ctx):
    n = "momentum"
    if len(df) < 40:
        return _none(n, "Not enough data", ctx)
    h = _macd_hist(df)
    rsi = _rsi_last(df)
    e20 = float(_ema(df, 20).iloc[-1])
    c = float(df["close"].iloc[-1])
    rising = h.iloc[-1] > h.iloc[-2] > h.iloc[-3]
    falling = h.iloc[-1] < h.iloc[-2] < h.iloc[-3]
    if h.iloc[-1] > 0 and rising and c > e20 and rsi is not None and 50 <= rsi <= 72:
        return _ev(n, BUY, 62 + min(15, (rsi - 50)), f"MACD rising >0, RSI {rsi:.0f}", True, ctx)
    if h.iloc[-1] < 0 and falling and c < e20 and rsi is not None and 28 <= rsi <= 50:
        return _ev(n, SELL, 62 + min(15, 50 - rsi), f"MACD falling <0, RSI {rsi:.0f}", True, ctx)
    return _none(n, "Momentum not aligned", ctx)

def support_resistance(df, ctx):
    n = "support_resistance"
    atr = _atr_ok(ctx)
    sup, res = ctx.get("support"), ctx.get("resistance")
    price = float(df["close"].iloc[-1]) if len(df) else None
    if atr is None or not price:
        return _none(n, "Not enough data/ATR", ctx)
    def real(level):
        return level is not None and abs(level - price * 1.02) > price * 1e-9 and abs(level - price * 0.98) > price * 1e-9
    o, h, l, c = _bar(df)
    rng = max(h - l, 1e-12)
    if real(sup) and abs(price - sup) <= 0.6 * atr:
        if c > o and (min(o, c) - l) / rng >= 0.35:
            return _ev(n, BUY, 66, "Bullish rejection at support", True, ctx)
        return _ev(n, BUY, 35, "Near support, no rejection", False, ctx)
    if real(res) and abs(res - price) <= 0.6 * atr:
        if c < o and (h - max(o, c)) / rng >= 0.35:
            return _ev(n, SELL, 66, "Bearish rejection at resistance", True, ctx)
        return _ev(n, SELL, 35, "Near resistance, no rejection", False, ctx)
    return _none(n, "Not at real S/R", ctx)

def market_structure(df, ctx):
    n = "market_structure"
    if len(df) < 20:
        return _none(n, "Not enough data", ctx)
    highs, lows = swing_points(df, 3)
    if len(highs) < 2 or len(lows) < 2:
        return _none(n, "No clear swing structure", ctx)
    c = float(df["close"].iloc[-1])
    hh, hl = highs[-1][1] > highs[-2][1], lows[-1][1] > lows[-2][1]
    lh, ll = highs[-1][1] < highs[-2][1], lows[-1][1] < lows[-2][1]
    if hh and hl:
        if c < lows[-1][1]:
            return _ev(n, SELL, 55, "Broke last higher low", False, ctx)
        bos = c > highs[-1][1]
        return _ev(n, BUY, 78 if bos else 66, "HH + HL" + (", BOS" if bos else ""), True, ctx)
    if lh and ll:
        if c > highs[-1][1]:
            return _ev(n, BUY, 55, "Broke last lower high", False, ctx)
        bos = c < lows[-1][1]
        return _ev(n, SELL, 78 if bos else 66, "LH + LL" + (", BOS" if bos else ""), True, ctx)
    return _none(n, "Mixed swings", ctx)

def news_event(news, recent_bars, atr, tf_seconds, now_utc=None, bar_utc_offset_hours=0.0):
    n = "news_event"
    ctx = {"tf": None}
    if not news or news.get("level") in (None, "LOW"):
        return _none(n, "No relevant event", ctx)
    mins = news.get("minutes_to_event")
    detail = news.get("detail") or "event"
    if mins is None or mins >= 0:
        return _ev(n, NONE, 0, f"Event ahead: {detail}", False, ctx)
    if not recent_bars or atr is None or not (atr > 0) or not tf_seconds:
        return _ev(n, NONE, 0, f"Released: {detail}; reaction unavailable", False, ctx)
    from datetime import datetime, timedelta
    now = now_utc or datetime.utcnow()
    event_t = now + timedelta(minutes=mins)
    pre, post = None, None
    for iso, close in recent_bars:
        try:
            t = pd.Timestamp(iso).to_pydatetime() - timedelta(hours=bar_utc_offset_hours)
        except Exception:
            continue
        if t + timedelta(seconds=tf_seconds) <= event_t:
            pre = close
        else:
            post = close
    if pre is None or post is None:
        return _ev(n, NONE, 0, f"Released: {detail}; no post-release candle", False, ctx)
    mv = (post - pre) / atr
    if abs(mv) < 1.0:
        return _ev(n, NONE, 0, f"Released: {detail}; reaction {mv:+.1f} ATR", False, ctx)
    side = BUY if mv > 0 else SELL
    return _ev(n, side, 55 + min(20, abs(mv) * 8), f"Released: {detail}; reacted {mv:+.1f} ATR", True, ctx)

_PRICE_STRATS = {"trend_following": trend_following, "breakout": breakout, "pullback": pullback,
                 "mean_reversion": mean_reversion, "momentum": momentum,
                 "support_resistance": support_resistance, "market_structure": market_structure}

def evaluate_all(df, ctx):
    ctx = ctx or {}
    regime = ((ctx.get("regime") or {}).get("regime")) or "UNKNOWN"
    active = REGIME_ACTIVE.get(regime, set())
    out = []
    for name in STRATEGY_NAMES:
        fn = _PRICE_STRATS.get(name)
        if fn is None:
            continue
        try:
            e = fn(df, ctx)
        except Exception as ex:
            e = _none(name, f"error: {type(ex).__name__}", ctx)
        e["active"] = name in active
        out.append(e)
    return out

def regime_support(evidence, want_side, news_ev=None, min_score=60):
    items = list(evidence or []) + ([news_ev] if news_ev else [])
    live = [e for e in items if e.get("active") and e.get("valid") and e.get("direction") in (BUY, SELL)]
    sup = [e for e in live if e["direction"] == want_side]
    opp = [e for e in live if e["direction"] != want_side]
    best = lambda xs: max(xs, key=lambda e: e["score"]) if xs else None
    bs, bo = best(sup), best(opp)
    return {"support": bs["score"] if bs else 0, "oppose": bo["score"] if bo else 0,
            "supporters": [e["strategy"] for e in sup], "opposers": [e["strategy"] for e in opp],
            "best_support": bs, "best_oppose": bo, "active_valid": len(live), "min_score": min_score}
'''

SE = _types.ModuleType("strategy_ensemble")
exec(compile(_SE_SRC, "<strategy_ensemble>", "exec"), SE.__dict__)
_sys.modules["strategy_ensemble"] = SE

# ============================================================
# LOGGING & CONFIG (NO HARDCODED API KEYS)
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def _env(name: str, default: str = "") -> str:
    """អាន Environment Variable — បើទទេ ឬ 'YOUR_KEY' → ត្រឡប់ '' (Not configured)។"""
    v = os.environ.get(name, default)
    if not v or v.strip().upper() in ("YOUR_KEY", "YOUR_KEY_HERE", "CHANGE_ME", "NONE", "NULL"):
        return ""
    return v.strip()

def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default

# --- Required ---
BOT_TOKEN = _env("BOT_TOKEN")
ADMIN_ID = _env_int("ADMIN_ID", 0)
ADMIN_IDS = set()
for _part in os.environ.get("ADMIN_IDS", "").split(","):
    _part = _part.strip()
    if _part.isdigit():
        ADMIN_IDS.add(int(_part))
if ADMIN_ID:
    ADMIN_IDS.add(ADMIN_ID)

def is_admin(user_id) -> bool:
    try:
        return int(user_id) in ADMIN_IDS
    except (TypeError, ValueError):
        return False

# --- Market Data Providers (all optional; code checks before use) ---
TWELVE_DATA_API_KEY = _env("TWELVE_DATA_API_KEY")
ALPHA_VANTAGE_API_KEY = _env("ALPHA_VANTAGE_API_KEY")
FINNHUB_API_KEY = _env("FINNHUB_API_KEY")
COINGECKO_API_KEY = _env("COINGECKO_API_KEY")  # optional — public endpoint works without key
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# --- News / Macro ---
NEWSAPI_KEY = _env("NEWSAPI_KEY")
FRED_API_KEY = _env("FRED_API_KEY")

# --- AI Providers (all optional) ---
GEMINI_API_KEY = _env("GEMINI_API_KEY")
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
GROQ_API_KEY = _env("GROQ_API_KEY")
CEREBRAS_API_KEY = _env("CEREBRAS_API_KEY")
CLAUDE_API_KEY = _env("CLAUDE_API_KEY")

# --- Chart / Vision ---
CHART_IMG_API_KEY = _env("CHART_IMG_API_KEY")

# --- EA Bridge ---
EA_BRIDGE_TOKEN = _env("EA_BRIDGE_TOKEN")
EA_BRIDGE_HMAC_SECRET = _env("EA_BRIDGE_HMAC_SECRET")
EA_BRIDGE_TIMESTAMP_TOLERANCE_SECONDS = _env_int("EA_BRIDGE_TIMESTAMP_TOLERANCE_SECONDS", 120)
EA_HEARTBEAT_TIMEOUT_SECONDS = 90
EA_LOCK = threading.Lock()
EA_NONCE_LOCK = threading.Lock()
EA_SEEN_NONCES = {}
EA_AUTH_FAIL_LOCK = threading.Lock()
EA_AUTH_FAILURES = {}
EA_AUTH_MAX_FAILURES = 8
EA_AUTH_FAILURE_WINDOW = 60
EA_AUTH_LOCKOUT_SECONDS = 300

# --- Data File ---
DATA_FILE = _env("DATA_FILE") or "trademind_data.json"

# --- Provider availability map (used by /status command) ---
def provider_status_summary():
    """បញ្ជី Provider ដែលបាន Setup vs មិនបាន Setup។"""
    checks = [
        ("BOT_TOKEN",            bool(BOT_TOKEN),            "Required"),
        ("EA_BRIDGE_TOKEN",      bool(EA_BRIDGE_TOKEN),      "MT5 EA bridge"),
        ("EA_BRIDGE_HMAC_SECRET",bool(EA_BRIDGE_HMAC_SECRET),"EA request signing"),
        ("TWELVE_DATA_API_KEY",  bool(TWELVE_DATA_API_KEY),  "Forex/Metals data"),
        ("ALPHA_VANTAGE_API_KEY",bool(ALPHA_VANTAGE_API_KEY),"Forex backup"),
        ("FINNHUB_API_KEY",      bool(FINNHUB_API_KEY),      "Forex+Crypto backup"),
        ("COINGECKO_API_KEY",    bool(COINGECKO_API_KEY),    "Crypto (optional)"),
        ("NEWSAPI_KEY",          bool(NEWSAPI_KEY),          "News headlines"),
        ("FRED_API_KEY",         bool(FRED_API_KEY),         "Fed Funds Rate"),
        ("GEMINI_API_KEY",       bool(GEMINI_API_KEY),       "AI: Gemini (text+vision)"),
        ("DEEPSEEK_API_KEY",     bool(DEEPSEEK_API_KEY),     "AI: DeepSeek"),
        ("OPENROUTER_API_KEY",   bool(OPENROUTER_API_KEY),   "AI: OpenRouter"),
        ("GROQ_API_KEY",         bool(GROQ_API_KEY),         "AI: Groq (text+vision)"),
        ("CEREBRAS_API_KEY",     bool(CEREBRAS_API_KEY),     "AI: Cerebras"),
        ("CLAUDE_API_KEY",       bool(CLAUDE_API_KEY),       "AI: Claude (commentary)"),
        ("CHART_IMG_API_KEY",    bool(CHART_IMG_API_KEY),    "Chart snapshots"),
    ]
    lines = ["🔑 *Provider Status*", ""]
    for name, ok, purpose in checks:
        icon = "✅" if ok else "⬜"
        lines.append(f"{icon} `{name}` — {purpose}")
    return "\n".join(lines)

# ============================================================
# USER DATA
# ============================================================

user_data = {}
_dirty_uids = set()

def load_data():
    global user_data
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            logger.info(f"Loaded {len(user_data)} user records from {DATA_FILE}")
        except Exception as e:
            logger.error(f"Load failed: {e}")
            user_data = {}

def save_data():
    global _dirty_uids
    snapshot = set(_dirty_uids)
    _dirty_uids = set()
    if not snapshot:
        return
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Save failed: {e}")

def ensure_ea_fields(ud):
    ud.setdefault("ea_default_preset", "intraday")
    ud.setdefault("ea_last_signal", None)
    ud.setdefault("ea_last_signal_id", None)
    ud.setdefault("ea_signal_journal", [])
    ud.setdefault("ea_trade_journal", [])
    ud.setdefault("ea_predictions", [])
    ud.setdefault("ea_pred_archive", {})
    ud.setdefault("ea_pending_execution", None)
    ud.setdefault("ea_pending_execution_ts", None)
    ud.setdefault("ea_pending_result", None)
    ud.setdefault("ea_pending_close", None)
    ud.setdefault("ea_pending_close_ts", None)
    ud.setdefault("ea_pending_closeall_ts", None)
    ud.setdefault("ea_pending_modify", None)
    ud.setdefault("ea_pending_modify_ts", None)
    ud.setdefault("ea_pending_partial", None)
    ud.setdefault("ea_pending_partial_ts", None)
    ud.setdefault("ea_state", None)
    ud.setdefault("ea_modify_target", None)
    ud.setdefault("ea_risk", {
        "risk_per_trade_pct": 1.0, "max_open_trades": 3, "max_trades_per_symbol": 1,
        "min_confidence": 70, "max_daily_loss_pct": 5.0,
        "sl_atr_mult": 1.5, "sl_mode": "atr", "fixed_sl_pct": 0.5, "tp_rr": 2.0,
        "partial_tp_enabled": False, "partial_tp_trigger_rr": 1.0, "partial_tp_close_pct": 50,
        "live_trading_enabled": False,
        "breakeven_enabled": True, "breakeven_trigger_rr": 1.0,
        "trailing_enabled": False, "trailing_mult": 1.5,
        "news_filter_enabled": True, "news_pre_minutes": 30, "news_post_minutes": 15,
        "max_drawdown_pct": 15.0, "max_consecutive_losses": 3,
        "max_spread_pips": None, "max_total_exposure_lots": None,
        "allowed_sessions": [], "cooldown_after_loss_minutes": 10,
        "abnormal_market_filter_enabled": True,
        "regime_filter_enabled": True, "blocked_regimes": ["UNSTABLE"],
        "chart_vision_enabled": False,
    })
    ud.setdefault("ea_perf_filters", {"period": "7D", "symbol": "ALL",
                                       "timeframe": "ALL", "direction": "ALL", "preset": "ALL"})
    ud.setdefault("ea_notify_types", ["TRADE_OPENED", "TRADE_CLOSED", "ERROR",
                                        "EA_STARTED", "EA_PAUSED", "EMERGENCY_STOP_ACK"])
    ud.setdefault("ea_peak_equity", None)
    ud.setdefault("ea_equity_curve", [])

def get_user_data(user_id):
    key = str(user_id)
    _dirty_uids.add(key)
    if key not in user_data:
        user_data[key] = {}
    ensure_ea_fields(user_data[key])
    return user_data[key]

def clear_conversation_modes(ud, keep=None):
    keep = keep or set()
    for k, v in (("ea_state", None), ("ea_modify_target", None)):
        if k not in keep:
            ud[k] = v

# ============================================================
# TRADING CONSTANTS
# ============================================================

FOREX_SYMBOLS = {
    "EURUSD": "🇪🇺 EUR/USD", "GBPUSD": "🇬🇧 GBP/USD", "USDJPY": "🇯🇵 USD/JPY",
    "AUDUSD": "🇦🇺 AUD/USD", "XAUUSD": "🥇 XAU/USD (Gold)", "XAGUSD": "🥈 XAG/USD (Silver)",
    "AUDJPY": "🇦🇺🇯🇵 AUD/JPY", "USDCAD": "🇺🇸🇨🇦 USD/CAD", "USDCHF": "🇺🇸🇨🇭 USD/CHF",
    "NZDUSD": "🇳🇿 NZD/USD", "EURJPY": "🇪🇺🇯🇵 EUR/JPY", "GBPJPY": "🇬🇧🇯🇵 GBP/JPY",
    "EURGBP": "🇪🇺🇬🇧 EUR/GBP",
}
CRYPTO_SYMBOLS = {
    "BTCUSD": "₿ Bitcoin", "ETHUSD": "Ξ Ethereum", "SOLUSD": "◎ Solana",
    "XRPUSD": "✕ Ripple", "BNBUSD": "🔶 BNB", "DOGEUSD": "🐕 Dogecoin",
}
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
TIMEFRAME_TO_INTERVAL = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1day", "W1": "1week", "MN1": "1month",
}
EA_RESAMPLE_MAP = {
    "M2": ("M1", "2min"), "M3": ("M1", "3min"), "M4": ("M1", "4min"),
    "M6": ("M1", "6min"), "M10": ("M5", "10min"), "M12": ("M1", "12min"),
    "M20": ("M5", "20min"), "H2": ("H1", "2h"), "H3": ("H1", "3h"),
    "H6": ("H1", "6h"), "H8": ("H1", "8h"), "H12": ("H1", "12h"),
}
TIMEFRAME_SECONDS = {
    "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300, "M6": 360,
    "M10": 600, "M12": 720, "M15": 900, "M20": 1200, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600,
    "H8": 28800, "H12": 43200, "D1": 86400, "W1": 604800, "MN1": 2592000,
}
EA_MTF_PRESETS = {
    "scalping": {"label": "⚡ Scalping", "desc": "M1→M5→M15→H1", "timeframes": ["M1", "M5", "M15", "H1"]},
    "intraday": {"label": "📆 Intraday", "desc": "M5→M15→H1→H4", "timeframes": ["M5", "M15", "H1", "H4"]},
    "swing":    {"label": "🌊 Swing",    "desc": "H1→H4→D1→W1", "timeframes": ["H1", "H4", "D1", "W1"]},
    "full":     {"label": "🧠 Full",     "desc": "M1→M5→M15→M30→H1→H4→D1→W1→MN1",
                 "timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]},
}
BACKTEST_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D1", "W1"]
BACKTEST_MIN_BARS = 60
WALKFORWARD_FOLDSSPLIT = (("Development", 0.50), ("Validation", 0.25), ("Out-of-sample test", 0.25))
WALKFORWARD_BARS = "1000"
EA_PERF_PERIODS = ["24H", "7D", "30D", "90D", "ALL"]
EA_PERF_PERIOD_SECONDS = {"24H": 86400, "7D": 7*86400, "30D": 30*86400, "90D": 90*86400, "ALL": None}
EA_PERF_LOW_SAMPLE = 5
EA_CONFLUENCE_WEIGHTS = {
    "htf_bias": 0.20, "structure": 0.20, "entry_confirmation": 0.15,
    "momentum": 0.10, "price_action": 0.10, "sr_proximity": 0.10,
    "mtf": 0.05, "news": 0.05, "volatility": 0.05,
}
EA_CONFLUENCE_WEIGHTS_WITH_VISION = {
    "htf_bias": 0.18, "structure": 0.18, "entry_confirmation": 0.135,
    "momentum": 0.09, "price_action": 0.09, "sr_proximity": 0.09,
    "mtf": 0.045, "news": 0.045, "volatility": 0.045, "chart_vision": 0.10,
}
EA_TP_TIER_MULTIPLIERS = (1.0, 1.6, 2.4)
EA_SIGNAL_EXPIRY_BARS = 300
EA_JOURNAL_MAX = 500
EA_EQUITY_CURVE_MAX = 2000
EA_EQUITY_CURVE_MIN_INTERVAL = 900
EA_ACTION_CONFIRM_TTL_SECONDS = 90
EA_OHLC_CACHE = {}
EA_OHLC_CACHE_TTL = 60
EA_BAR_UTC_OFFSET_HOURS = float(_env("EA_BAR_UTC_OFFSET_HOURS", "0") or 0)

# ============================================================
# HTTP ASYNC WRAPPERS
# ============================================================
async def async_get(*a, **k): return await asyncio.to_thread(requests.get, *a, **k)
async def async_post(*a, **k): return await asyncio.to_thread(requests.post, *a, **k)
async def async_put(*a, **k): return await asyncio.to_thread(requests.put, *a, **k)

# ============================================================
# DATA FETCHERS (each checks its own key first)
# ============================================================

CRYPTO_MAP = {
    "BTCUSD": "bitcoin", "ETHUSD": "ethereum", "SOLUSD": "solana",
    "XRPUSD": "ripple", "BNBUSD": "binancecoin", "DOGEUSD": "dogecoin",
}
FINNHUB_RESOLUTION_MAP = {"M1": "1", "M5": "5", "M15": "15", "M30": "30",
                           "H1": "60", "H4": "60", "D1": "D", "W1": "W"}
ALPHA_VANTAGE_INTERVAL_MAP = {"M1": "1min", "M5": "5min", "M15": "15min",
                                "M30": "30min", "H1": "60min"}

def to_pair_symbol(symbol):
    s = symbol.upper()
    return f"{s[:3]}/{s[3:]}" if len(s) == 6 and s.isalpha() else s

async def fetch_coingecko_data(coin_id, days="1"):
    """CoinGecko — ដំណើរការទាំង Key និងគ្មាន Key (public tier)។"""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=hourly"
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
    try:
        res = await async_get(url, headers=headers, timeout=10)
        data = res.json()
        if "prices" in data:
            df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            return df[["datetime", "open", "high", "low", "close"]]
        logger.warning(f"CoinGecko no prices for {coin_id}")
    except Exception as e:
        logger.warning(f"CoinGecko error: {e}")
    return None

async def fetch_twelve_data(symbol, interval="1h", outputsize="500"):
    if not TWELVE_DATA_API_KEY:
        return None
    pair = to_pair_symbol(symbol)
    url = (f"https://api.twelvedata.com/time_series?symbol={pair}&interval={interval}"
           f"&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}")
    try:
        res = await async_get(url, timeout=10)
        data = res.json()
        if "values" in data:
            df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            return df
    except Exception as e:
        logger.warning(f"Twelve Data error for {pair}: {e}")
    return None

async def fetch_alpha_vantage_data(symbol, timeframe="H1"):
    if not ALPHA_VANTAGE_API_KEY:
        return None
    av_interval = ALPHA_VANTAGE_INTERVAL_MAP.get(timeframe)
    if not av_interval:
        return None
    s = symbol.upper()
    if len(s) == 6 and s.isalpha():
        url = ("https://www.alphavantage.co/query?function=FX_INTRADAY"
               f"&from_symbol={s[:3]}&to_symbol={s[3:]}"
               f"&interval={av_interval}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}")
    else:
        url = ("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
               f"&symbol={s}&interval={av_interval}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}")
    try:
        res = await async_get(url, timeout=10)
        data = res.json()
        series_key = next((k for k in data if "Time Series" in k), None)
        if not series_key:
            return None
        rows = [{"datetime": ts, "open": float(v["1. open"]), "high": float(v["2. high"]),
                 "low": float(v["3. low"]), "close": float(v["4. close"])}
                for ts, v in data[series_key].items()]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Alpha Vantage error: {e}")
        return None

async def fetch_finnhub_data(symbol, timeframe="H1", count=500):
    if not FINNHUB_API_KEY:
        return None
    resolution = FINNHUB_RESOLUTION_MAP.get(timeframe, "60")
    s = symbol.upper()
    try:
        if s in CRYPTO_MAP:
            fh_symbol = f"BINANCE:{s[:-3]}USDT"
            endpoint = "crypto/candle"
        elif len(s) == 6 and s.isalpha():
            fh_symbol = f"OANDA:{s[:3]}_{s[3:]}"
            endpoint = "forex/candle"
        else:
            return None
        secs = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600,
                "D": 86400, "W": 604800}.get(resolution, 3600)
        now_ts = int(datetime.utcnow().timestamp())
        url = (f"https://finnhub.io/api/v1/{endpoint}?symbol={fh_symbol}"
               f"&resolution={resolution}&from={now_ts - secs * count}&to={now_ts}&token={FINNHUB_API_KEY}")
        res = await async_get(url, timeout=10)
        data = res.json()
        if data.get("s") != "ok":
            return None
        return pd.DataFrame({
            "datetime": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"], "high": data["h"], "low": data["l"], "close": data["c"],
        })
    except Exception as e:
        logger.warning(f"Finnhub error: {e}")
        return None

async def fetch_market_data(symbol, interval="1h", outputsize="500", timeframe="H1"):
    s = symbol.upper()
    tried = []
    if s in CRYPTO_MAP:
        df = await fetch_coingecko_data(CRYPTO_MAP[s]); tried.append("CoinGecko")
        if df is not None and len(df) >= 30:
            return df, tried
        df = await fetch_twelve_data(symbol, interval, outputsize); tried.append("Twelve Data")
        if df is not None and len(df) >= 30:
            return df, tried
        df = await fetch_finnhub_data(symbol, timeframe); tried.append("Finnhub")
        return df, tried
    df = await fetch_twelve_data(symbol, interval, outputsize); tried.append("Twelve Data")
    if df is not None and len(df) >= 30:
        return df, tried
    df = await fetch_alpha_vantage_data(symbol, timeframe); tried.append("Alpha Vantage")
    if df is not None and len(df) >= 30:
        return df, tried
    df = await fetch_finnhub_data(symbol, timeframe); tried.append("Finnhub")
    return df, tried

async def fetch_fred_trend(series_id="DFF", limit=5):
    if not FRED_API_KEY:
        return None
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
           f"&sort_order=desc&limit={limit}")
    try:
        res = await async_get(url, timeout=10)
        obs = [o for o in res.json().get("observations", []) if o.get("value") not in (".", None)]
        if not obs:
            return None
        latest_val = float(obs[0]["value"])
        result = {"latest_value": latest_val, "latest_date": obs[0]["date"],
                  "trend": None, "window_label": None}
        if len(obs) < 2:
            return result
        diff = latest_val - float(obs[1]["value"])
        try:
            latest_dt = datetime.strptime(obs[0]["date"], "%Y-%m-%d")
            hrs = max((datetime.utcnow() - latest_dt).total_seconds() / 3600, 0)
        except Exception:
            hrs = 24
        result["window_label"] = "4 ម៉ោង" if hrs <= 6 else ("12 ម៉ោង" if hrs <= 18 else "24 ម៉ោង")
        result["trend"] = (f"⬆️ កើនឡើង {diff:.2f}%" if diff > 0.001
                           else f"⬇️ ថយចុះ {abs(diff):.2f}%" if diff < -0.001
                           else "➖ រក្សាស្ថេរភាព")
        return result
    except Exception as e:
        logger.warning(f"FRED trend error: {e}")
        return None

CHART_IMG_INTERVAL_MAP = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
                           "H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W"}

def to_chart_img_symbol(symbol):
    s = symbol.upper()
    return f"BINANCE:{s[:-3]}USDT" if s in CRYPTO_MAP else f"OANDA:{s}"

async def fetch_chart_image(symbol, timeframe):
    if not CHART_IMG_API_KEY:
        return None
    interval = CHART_IMG_INTERVAL_MAP.get(timeframe, "30m")
    try:
        res = await async_post(
            "https://api.chart-img.com/v2/tradingview/advanced-chart",
            headers={"x-api-key": CHART_IMG_API_KEY, "content-type": "application/json"},
            json={"symbol": to_chart_img_symbol(symbol), "interval": interval,
                  "width": 800, "height": 500, "theme": "dark",
                  "studies": [{"name": "Volume", "forceOverlay": True}]},
            timeout=20,
        )
        if res.status_code == 200 and res.headers.get("content-type", "").startswith("image"):
            return res.content
    except Exception as e:
        logger.warning(f"Chart-IMG error: {e}")
    return None

async def get_claude_commentary(prompt):
    if not CLAUDE_API_KEY:
        return None
    try:
        res = await async_post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-3-5-haiku-latest", "max_tokens": 300,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=20,
        )
        data = res.json()
        return "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "text") or None
    except Exception as e:
        logger.warning(f"Claude API error: {e}")
        return None

# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]

def calculate_macd(df):
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1], (macd_line - signal_line).iloc[-1]

def calculate_ema(df, period):
    return df["close"].ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_sma(df, period):
    return df["close"].rolling(window=period).mean().iloc[-1]

def calculate_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
                    axis=1).max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]

def calculate_bollinger(df, period=20, num_std=2):
    if len(df) < period:
        return None, None, None
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    u, m, l = (sma + num_std * std).iloc[-1], sma.iloc[-1], (sma - num_std * std).iloc[-1]
    return (None, None, None) if (pd.isna(u) or pd.isna(m) or pd.isna(l)) else (float(u), float(m), float(l))

def calculate_stochastic(df, k_period=14, d_period=3):
    if len(df) < k_period + d_period:
        return None, None
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    k = ((df["close"] - low_min) / rng * 100).iloc[-1]
    d = ((df["close"] - low_min) / rng * 100).rolling(window=d_period).mean().iloc[-1]
    return (None, None) if (pd.isna(k) or pd.isna(d)) else (float(k), float(d))

def calculate_adx(df, period=14):
    if len(df) < period * 2:
        return None, None, None
    high, low, close = df["high"], df["low"], df["close"]
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)
    up, dn = high - ph, pl - low
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up > dn) & (up > 0)] = up[(up > dn) & (up > 0)]
    minus_dm[(dn > up) & (dn > 0)] = dn[(dn > up) & (dn > 0)]
    tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    mdi = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    a, p, m = adx.iloc[-1], pdi.iloc[-1], mdi.iloc[-1]
    return (None, None, None) if (pd.isna(a) or pd.isna(p) or pd.isna(m)) else (float(a), float(p), float(m))

def _swing_extremes(values, window, find_min):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2 * window + 1:
        return arr[:0]
    wins = np.lib.stride_tricks.sliding_window_view(arr, 2 * window + 1)
    ext = wins.min(axis=1) if find_min else wins.max(axis=1)
    centre = arr[window:n - window]
    return centre[centre == ext]

def find_support_resistance(df, window=5):
    sups = [float(v) for v in _swing_extremes(df["low"].to_numpy(), window, True)]
    ress = [float(v) for v in _swing_extremes(df["high"].to_numpy(), window, False)]
    price = df["close"].iloc[-1]
    support = max([s for s in sups if s < price], default=price * 0.98)
    resistance = min([r for r in ress if r > price], default=price * 1.02)
    return support, resistance

# ============================================================
# AI ROUTER (all providers optional)
# ============================================================

_provider_stats = {}
PROVIDER_COOLDOWN_SECONDS = 60
PROVIDER_COOLDOWN_THRESHOLD = 3

def _provider_stat(n): return _provider_stats.setdefault(n, {"success": 0, "fail": 0, "last_error": None, "last_fail_ts": None})

def _provider_on_cooldown(n):
    st = _provider_stat(n)
    if st["fail"] < PROVIDER_COOLDOWN_THRESHOLD or st["last_fail_ts"] is None:
        return False
    return (time.time() - st["last_fail_ts"]) < PROVIDER_COOLDOWN_SECONDS

def _record_success(n):
    st = _provider_stat(n); st["success"] += 1; st["fail"] = 0

def _record_failure(n, err):
    st = _provider_stat(n); st["fail"] += 1; st["last_error"] = err; st["last_fail_ts"] = time.time()

def provider_health_summary():
    if not _provider_stats:
        return "គ្មានទិន្នន័យ Provider នៅឡើយទេ។"
    out = []
    for n, st in _provider_stats.items():
        status = "🟡 Cooldown" if _provider_on_cooldown(n) else "🟢 OK"
        out.append(f"{status} *{n}* — success: {st['success']}, fail: {st['fail']}")
        if st["last_error"]:
            out.append(f"   {st['last_error'][:120]}")
    return "\n".join(out)

# ---- Individual provider callers (each checks its own key) ----

async def call_deepseek(messages, max_tokens=1200):
    if not DEEPSEEK_API_KEY: return None, "DeepSeek not configured"
    try:
        r = await async_post("https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": messages, "max_tokens": max_tokens}, timeout=30)
        if r.status_code != 200: return None, f"DeepSeek HTTP {r.status_code}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e: return None, str(e)

async def call_gemini_text(messages, max_tokens=1200):
    if not GEMINI_API_KEY: return None, "Gemini not configured"
    try:
        contents, sysinst = [], None
        for m in messages:
            if m["role"] == "system": sysinst = m["content"]; continue
            contents.append({"role": "user" if m["role"] == "user" else "model",
                              "parts": [{"text": m["content"]}]})
        payload = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
        if sysinst: payload["system_instruction"] = {"parts": [{"text": sysinst}]}
        r = await async_post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            params={"key": GEMINI_API_KEY}, json=payload, timeout=30)
        if r.status_code != 200: return None, f"Gemini HTTP {r.status_code}"
        return r.json()["candidates"][0]["content"]["parts"][0]["text"], None
    except Exception as e: return None, str(e)

async def call_openrouter(messages, max_tokens=1200, model="openrouter/auto"):
    if not OPENROUTER_API_KEY: return None, "OpenRouter not configured"
    try:
        r = await async_post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens}, timeout=30)
        if r.status_code != 200: return None, f"OpenRouter HTTP {r.status_code}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e: return None, str(e)

async def call_cerebras(messages, max_tokens=1200):
    if not CEREBRAS_API_KEY: return None, "Cerebras not configured"
    try:
        r = await async_post("https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-oss-120b", "messages": messages, "max_completion_tokens": max_tokens}, timeout=30)
        if r.status_code != 200: return None, f"Cerebras HTTP {r.status_code}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e: return None, str(e)

async def call_groq_text(messages, max_tokens=1200):
    if not GROQ_API_KEY: return None, "Groq not configured"
    try:
        r = await async_post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-120b", "messages": messages, "max_completion_tokens": max_tokens}, timeout=30)
        if r.status_code != 200: return None, f"Groq HTTP {r.status_code}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e: return None, str(e)

PROVIDER_CALL_FUNCS = {
    "deepseek": call_deepseek, "gemini": call_gemini_text,
    "openrouter": call_openrouter, "cerebras": call_cerebras, "groq": call_groq_text,
}

_RETRYABLE = ("429", "500", "502", "503", "504", "timeout", "Timed out")

async def _call_with_retry(provider_name, call_fn, retries=2):
    delay, last_err = 1.0, None
    for attempt in range(retries + 1):
        try:
            text, err = await call_fn()
        except Exception as e:
            text, err = None, str(e)
        if text:
            _record_success(provider_name); return text, None
        last_err = err
        if attempt < retries and err and any(m in str(err) for m in _RETRYABLE):
            await asyncio.sleep(delay); delay *= 2; continue
        break
    _record_failure(provider_name, last_err)
    return None, last_err

AI_PROVIDER_CHAINS = {"prediction": ["gemini", "openrouter", "deepseek"]}
AI_MODE_MAX_TOKENS = {"prediction": 2200}

async def ai_router_generate(mode, messages, max_tokens=None):
    if max_tokens is None:
        max_tokens = AI_MODE_MAX_TOKENS.get(mode, 1200)
    errors = []
    for provider_name in AI_PROVIDER_CHAINS.get(mode, []):
        func = PROVIDER_CALL_FUNCS.get(provider_name)
        if not func: continue
        if _provider_on_cooldown(provider_name):
            errors.append(f"{provider_name}: cooldown"); continue
        text, err = await _call_with_retry(provider_name,
                                            lambda f=func: f(messages, max_tokens=max_tokens))
        if text: return text, provider_name, None
        errors.append(f"{provider_name}: {err}")
    return None, None, " | ".join(errors) if errors else "គ្មាន AI Provider ណាមួយ Setup ទេ"

# Vision routers
async def call_gemini_vision(image_bytes, mime_type, prompt, max_tokens=1500):
    if not GEMINI_API_KEY: return None, "Gemini not configured"
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}}]}],
            "generationConfig": {"maxOutputTokens": max_tokens}}
        r = await async_post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            params={"key": GEMINI_API_KEY}, json=payload, timeout=45)
        if r.status_code != 200: return None, f"Gemini Vision HTTP {r.status_code}"
        return r.json()["candidates"][0]["content"]["parts"][0]["text"], None
    except Exception as e: return None, str(e)

async def call_openrouter_vision(image_bytes, mime_type, prompt, max_tokens=1500):
    if not OPENROUTER_API_KEY: return None, "OpenRouter not configured"
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"model": "openrouter/auto",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}]}],
            "max_tokens": max_tokens}
        r = await async_post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=45)
        if r.status_code != 200: return None, f"OpenRouter Vision HTTP {r.status_code}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e: return None, str(e)

VISION_CHAIN = [("gemini", call_gemini_vision), ("openrouter", call_openrouter_vision)]

async def ai_router_generate_vision(image_bytes, mime_type, prompt, max_tokens=1500):
    errors = []
    for name, func in VISION_CHAIN:
        if _provider_on_cooldown(name):
            errors.append(f"{name}: cooldown"); continue
        text, err = await _call_with_retry(name,
                                            lambda f=func: f(image_bytes, mime_type, prompt, max_tokens))
        if text: return text, name, None
        errors.append(f"{name}: {err}")
    return None, None, " | ".join(errors) if errors else "គ្មាន Vision Provider ណាមួយ Setup ទេ"

# ============================================================
# NEWS / CALENDAR
# ============================================================

EA_NEWS_CACHE = {"data": None, "fetched_at": 0}
EA_NEWS_CACHE_TTL = 900

def ea_symbol_currencies(symbol):
    s = (symbol or "").upper()
    return [s[:3], s[3:]] if (len(s) == 6 and s.isalpha()) else []

async def ea_fetch_calendar():
    now = time.time()
    if EA_NEWS_CACHE["data"] is not None and (now - EA_NEWS_CACHE["fetched_at"]) < EA_NEWS_CACHE_TTL:
        return EA_NEWS_CACHE["data"]
    try:
        res = await async_get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        data = res.json()
        EA_NEWS_CACHE["data"] = data
        EA_NEWS_CACHE["fetched_at"] = now
    except Exception as e:
        logger.warning(f"Calendar fetch error: {e}")
        data = EA_NEWS_CACHE["data"] or []
    return data

async def ea_news_risk(symbol, pre_minutes=30, post_minutes=15):
    currencies = ea_symbol_currencies(symbol)
    if not currencies:
        return {"level": "LOW", "blocking": False, "detail": None}
    events = await ea_fetch_calendar()
    now = datetime.now(timezone.utc)
    worst = None
    for ev in events:
        if ev.get("country") not in currencies: continue
        impact = ev.get("impact")
        if impact not in ("High", "Medium"): continue
        try:
            ev_time = datetime.fromisoformat(ev["date"])
        except Exception:
            continue
        if ev_time.tzinfo is None: continue
        delta_min = (ev_time - now).total_seconds() / 60
        if not (-post_minutes <= delta_min <= pre_minutes): continue
        when = f"in {round(delta_min)}min" if delta_min >= 0 else f"{round(-delta_min)}min ago"
        actual = (ev.get("actual") or "").strip() if delta_min < 0 else ""
        cand = {"level": "HIGH" if impact == "High" else "MEDIUM",
                "blocking": impact == "High",
                "detail": f"{ev.get('title')} ({ev.get('country')}) {when}",
                "minutes_to_event": round(delta_min, 1),
                "event": {"title": ev.get("title"), "currency": ev.get("country"), "impact": impact,
                          "forecast": (ev.get("forecast") or "").strip() or None,
                          "previous": (ev.get("previous") or "").strip() or None,
                          "actual": actual or None}}
        if cand["blocking"]: return cand
        worst = worst or cand
    return worst or {"level": "LOW", "blocking": False, "detail": None}

NEWS_BULLISH_KW = ["rally", "surge", "gain", "rises", "record high", "bullish", "jumps",
                   "soars", "recovery", "growth", "boost", "climbs", "rebound"]
NEWS_BEARISH_KW = ["plunge", "crash", "drops", "falls", "decline", "bearish", "slump",
                   "selloff", "recession", "fears", "tumbles", "rout"]

def compute_news_signal(articles):
    blob = " ".join(f"{a.get('title', '')} {a.get('description') or ''}" for a in articles).lower()
    bull = sum(blob.count(k) for k in NEWS_BULLISH_KW)
    bear = sum(blob.count(k) for k in NEWS_BEARISH_KW)
    if bull > bear: return "BUY (positive bias)", "🟢📈", bull, bear
    if bear > bull: return "SELL (negative bias)", "🔴📉", bear, bull
    return "HOLD (unclear)", "🟡➖", bull, bear

NEWS_QUERY_MAP = {
    "EURUSD": "EUR/USD euro dollar forex", "GBPUSD": "GBP/USD pound dollar forex",
    "USDJPY": "USD/JPY dollar yen forex", "AUDUSD": "AUD/USD australian dollar",
    "XAUUSD": "gold price XAU", "AUDJPY": "AUD/JPY australian dollar yen",
    "USDCAD": "USD/CAD dollar canadian", "BTCUSD": "bitcoin BTC price",
    "ETHUSD": "ethereum ETH price", "SOLUSD": "solana SOL price",
}

# ============================================================
# EA SIGNAL ENGINE (indicators → bias → confluence → direction)
# ============================================================

def ea_resample_ohlc(df, rule):
    d = df.set_index("datetime").sort_index()
    return d.resample(rule).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"}).dropna().reset_index()

async def ea_get_ohlc(symbol, tf, outputsize="500"):
    if tf in TIMEFRAME_TO_INTERVAL:
        return await fetch_market_data(symbol, interval=TIMEFRAME_TO_INTERVAL[tf],
                                        outputsize=outputsize, timeframe=tf)
    if tf in EA_RESAMPLE_MAP:
        base_tf, rule = EA_RESAMPLE_MAP[tf]
        base_df, tried = await fetch_market_data(
            symbol, interval=TIMEFRAME_TO_INTERVAL[base_tf], outputsize=outputsize, timeframe=base_tf)
        if base_df is None: return None, tried
        return ea_resample_ohlc(base_df, rule), tried
    return None, []

async def ea_get_ohlc_cached(symbol, tf, outputsize="500"):
    key = (symbol, tf, outputsize)
    now = time.time()
    hit = EA_OHLC_CACHE.get(key)
    if hit and (now - hit[0]) < EA_OHLC_CACHE_TTL:
        return hit[1], hit[2]
    df, tried = await ea_get_ohlc(symbol, tf, outputsize=outputsize)
    if len(EA_OHLC_CACHE) > 300: EA_OHLC_CACHE.clear()
    EA_OHLC_CACHE[key] = (now, df, tried)
    return df, tried

def ea_prepare_df(df, tf):
    df2, meta = DC.drop_forming_candle(df, TIMEFRAME_SECONDS.get(tf),
                                        utc_offset_hours=EA_BAR_UTC_OFFSET_HOURS)
    try:
        if df2 is not None and len(df2) >= 2:
            meta["ref_bar_raw"] = str(df2["datetime"].iloc[-1])
            meta["ref_close"] = float(df2["close"].iloc[-1])
            d = float(df2["close"].iloc[-1]) - float(df2["close"].iloc[-2])
            meta["prev_dir"] = "BULLISH" if d > 0 else ("BEARISH" if d < 0 else "NEUTRAL")
    except Exception:
        pass
    return df2, meta

def _ea_bias(bull, bear):
    if bull > bear: return "BULL"
    if bear > bull: return "BEAR"
    return "NEUTRAL"

def ea_score_trend(df):
    e20, e50 = calculate_ema(df, 20), calculate_ema(df, 50)
    price = df["close"].iloc[-1]
    bull, bear, reasons = 0, 0, []
    if e20 > e50: bull += 1; reasons.append("EMA20>EMA50")
    else: bear += 1; reasons.append("EMA20<EMA50")
    if len(df) >= 100:
        e200 = calculate_ema(df, 200)
        if price > e200: bull += 1; reasons.append("Price>EMA200")
        else: bear += 1; reasons.append("Price<EMA200")
    adx, pdi, mdi = calculate_adx(df)
    if adx is not None:
        if pdi > mdi: bull += 1; reasons.append(f"+DI>-DI (ADX {adx:.1f})")
        else: bear += 1; reasons.append(f"-DI>+DI (ADX {adx:.1f})")
    total = bull + bear
    score = round(max(bull, bear) / total * 100) if total else 50
    if adx is not None:
        if adx < 20: score = round(score * 0.6 + 50 * 0.4); reasons.append(f"Weak ADX {adx:.1f}")
        elif adx >= 25: reasons.append(f"Strong ADX {adx:.1f}")
    return {"bias": _ea_bias(bull, bear), "score": score, "reasons": reasons, "adx": adx}

def ea_score_momentum(df, trend=None):
    strong = None
    if trend and trend.get("adx") is not None and trend["adx"] >= 25 and trend.get("bias") in ("BULL", "BEAR"):
        strong = trend["bias"]
    rsi = calculate_rsi(df)
    macd_line, signal_line, _ = calculate_macd(df)
    k, d = calculate_stochastic(df)
    bull, bear, reasons = 0, 0, []
    if rsi <= 35:
        if strong == "BEAR": reasons.append(f"RSI {rsi:.1f} oversold (ignored)")
        else: bull += 1; reasons.append(f"RSI oversold {rsi:.1f}")
    elif rsi >= 65:
        if strong == "BULL": reasons.append(f"RSI {rsi:.1f} overbought (ignored)")
        else: bear += 1; reasons.append(f"RSI overbought {rsi:.1f}")
    else: reasons.append(f"RSI neutral {rsi:.1f}")
    if macd_line > signal_line: bull += 1; reasons.append("MACD bullish")
    else: bear += 1; reasons.append("MACD bearish")
    if k is not None:
        if k <= 20:
            if strong == "BEAR": reasons.append(f"Stoch %K {k:.1f} oversold (ignored)")
            else: bull += 1; reasons.append(f"Stoch oversold {k:.1f}")
        elif k >= 80:
            if strong == "BULL": reasons.append(f"Stoch %K {k:.1f} overbought (ignored)")
            else: bear += 1; reasons.append(f"Stoch overbought {k:.1f}")
        else: reasons.append(f"Stoch neutral {k:.1f}")
    total = bull + bear
    return {"bias": _ea_bias(bull, bear) if total else "NEUTRAL",
            "score": round(max(bull, bear) / total * 100) if total else 50,
            "reasons": reasons, "rsi": rsi, "stoch_k": k, "stoch_d": d}

def ea_score_structure(df, window=3):
    n = len(df)
    if n < window * 2 + 4:
        return {"bias": "NEUTRAL", "score": 50, "reasons": ["Not enough bars"]}
    highs = [float(v) for v in _swing_extremes(df["high"].to_numpy(), window, False)]
    lows = [float(v) for v in _swing_extremes(df["low"].to_numpy(), window, True)]
    if len(highs) < 2 or len(lows) < 2:
        return {"bias": "NEUTRAL", "score": 50, "reasons": ["No clear swings"]}
    hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
    lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
    if hh and hl: return {"bias": "BULL", "score": 80, "reasons": ["HH/HL"]}
    if lh and ll: return {"bias": "BEAR", "score": 80, "reasons": ["LH/LL"]}
    return {"bias": "NEUTRAL", "score": 45, "reasons": ["Mixed structure"]}

def ea_score_price_action(df):
    o, h, l, c = (df["open"].iloc[-1], df["high"].iloc[-1],
                  df["low"].iloc[-1], df["close"].iloc[-1])
    po, pc = df["open"].iloc[-2], df["close"].iloc[-2]
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    uw = h - max(c, o)
    lw = min(c, o) - l
    if c > o and o <= pc and c >= po and pc < po: return {"bias": "BULL", "score": 75, "reasons": ["Bullish engulfing"]}
    if c < o and o >= pc and c <= po and pc > po: return {"bias": "BEAR", "score": 75, "reasons": ["Bearish engulfing"]}
    if lw > body * 2 and c > o: return {"bias": "BULL", "score": 70, "reasons": ["Bullish pin bar"]}
    if uw > body * 2 and c < o: return {"bias": "BEAR", "score": 70, "reasons": ["Bearish pin bar"]}
    if body / rng > 0.7: return {"bias": "BULL" if c > o else "BEAR", "score": 60, "reasons": ["Momentum candle"]}
    return {"bias": "NEUTRAL", "score": 50, "reasons": ["Indecision"]}

def ea_score_volatility(df, period=14):
    atr_now = calculate_atr(df, period)
    high, low, close = df["high"], df["low"], df["close"]
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period).mean()
    lb = atr_series.iloc[-40:] if len(atr_series) >= 40 else atr_series
    avg = lb.mean()
    if not avg or avg != avg or avg == 0:
        return {"label": "UNKNOWN", "ratio": 1.0, "score": 60}
    r = atr_now / avg
    if r >= 2.5: return {"label": "EXTREME", "ratio": round(r, 2), "score": 20}
    if r >= 1.6: return {"label": "HIGH", "ratio": round(r, 2), "score": 45}
    if r <= 0.5: return {"label": "LOW", "ratio": round(r, 2), "score": 55}
    return {"label": "NORMAL", "ratio": round(r, 2), "score": 75}

def ea_detect_abnormal_price(df, spike_ratio=5.0):
    if len(df) < 15:
        return {"abnormal": False, "detail": None}
    close = df["close"]
    rc = close.diff().abs().iloc[-15:-2]
    avg = rc.mean()
    if not avg or avg != avg or avg == 0:
        return {"abnormal": False, "detail": None}
    last = abs(close.iloc[-1] - close.iloc[-2])
    if last / avg >= spike_ratio:
        return {"abnormal": True, "detail": f"Sudden move ({last/avg:.1f}x avg)"}
    gap = abs(df["open"].iloc[-1] - close.iloc[-2])
    if gap / avg >= spike_ratio:
        return {"abnormal": True, "detail": f"Gap ({gap/avg:.1f}x avg)"}
    return {"abnormal": False, "detail": None}

def ea_score_bbands_width(df, period=20):
    close = df["close"]
    if len(close) < period + 5:
        return {"width": None, "ratio": 1.0}
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    width = (4 * std) / sma
    wn = width.iloc[-1]
    lb = width.iloc[-60:-1] if len(width) >= 61 else width.iloc[:-1]
    wa = lb.mean() if len(lb) else wn
    if wn != wn or not wa or wa != wa or wa == 0:
        return {"width": None, "ratio": 1.0}
    return {"width": round(float(wn), 6), "ratio": round(float(wn / wa), 2)}

def ea_detect_market_regime(df, trend=None, structure=None, volatility=None, anomaly=None):
    if len(df) < 25:
        return {"regime": "UNKNOWN", "label": "❔ Unknown", "bias": "NEUTRAL", "detail": "Not enough bars"}
    trend = trend or ea_score_trend(df)
    structure = structure or ea_score_structure(df)
    volatility = volatility or ea_score_volatility(df)
    anomaly = anomaly if anomaly is not None else ea_detect_abnormal_price(df)
    bbw = ea_score_bbands_width(df)
    if anomaly.get("abnormal") or volatility["label"] == "EXTREME":
        return {"regime": "UNSTABLE", "label": "⚠️ Unstable", "bias": "NEUTRAL",
                "detail": anomaly.get("detail") or f"ATR {volatility['ratio']}x avg"}
    if (bbw["ratio"] >= 1.5 and trend["score"] >= 60
            and trend["bias"] == structure["bias"] and structure["bias"] != "NEUTRAL"):
        return {"regime": "BREAKOUT", "label": f"🚀 Breakout ({trend['bias']})", "bias": trend["bias"],
                "detail": f"Bands {bbw['ratio']}x avg"}
    if volatility["label"] == "HIGH":
        return {"regime": "HIGH_VOLATILITY", "label": "🌪️ High Vol", "bias": "NEUTRAL",
                "detail": f"ATR {volatility['ratio']}x avg"}
    if trend["bias"] == structure["bias"] and structure["bias"] != "NEUTRAL" and trend["score"] >= 60:
        return {"regime": "TRENDING", "label": f"📈 Trending ({trend['bias']})", "bias": trend["bias"],
                "detail": f"Trend {trend['score']}/100"}
    if volatility["label"].startswith("LOW"):
        return {"regime": "LOW_VOLATILITY", "label": "😴 Low Vol", "bias": "NEUTRAL",
                "detail": f"ATR {volatility['ratio']}x avg"}
    return {"regime": "RANGING", "label": "↔️ Ranging", "bias": "NEUTRAL",
            "detail": "No trend/structure agreement"}

def ea_analyze_timeframe_bias(df, tf):
    price = df["close"].iloc[-1]
    trend = ea_score_trend(df)
    momentum = ea_score_momentum(df, trend)
    structure = ea_score_structure(df)
    price_action = ea_score_price_action(df)
    volatility = ea_score_volatility(df)
    anomaly = ea_detect_abnormal_price(df)
    regime = ea_detect_market_regime(df, trend, structure, volatility, anomaly)
    directional = [trend, momentum, structure, price_action]
    bull = sum(1 for c in directional if c["bias"] == "BULL")
    bear = sum(1 for c in directional if c["bias"] == "BEAR")
    bias = _ea_bias(bull, bear)
    dw = {"trend": 0.35, "momentum": 0.25, "structure": 0.25, "price_action": 0.15}
    comps = {"trend": trend, "momentum": momentum, "structure": structure, "price_action": price_action}
    aw = sum(dw[k] for k, c in comps.items() if c["bias"] == bias) or 1e-9
    ds = sum(dw[k] * c["score"] for k, c in comps.items() if c["bias"] == bias) / aw
    strength = round(ds * 0.8 + volatility["score"] * 0.2)
    reasons = [r for c in (trend, momentum, structure, price_action) for r in c["reasons"]]
    support, resistance = find_support_resistance(df)
    atr_val = calculate_atr(df)
    return {
        "timeframe": tf, "bias": bias, "strength": strength, "reasons": reasons,
        "price": price, "rsi": momentum.get("rsi"), "adx": trend.get("adx"),
        "stoch_k": momentum.get("stoch_k"),
        "support": support, "resistance": resistance, "atr": atr_val,
        "components": {"trend": trend, "momentum": momentum, "structure": structure,
                        "price_action": price_action, "volatility": volatility,
                        "anomaly": anomaly, "regime": regime},
    }

def ea_score_df(df, tf):
    res = ea_analyze_timeframe_bias(df, tf)
    try:
        comps = res.get("components") or {}
        sctx = {"tf": tf, "atr": res.get("atr"), "regime": comps.get("regime"),
                "adx": res.get("adx"), "structure_bias": (comps.get("structure") or {}).get("bias"),
                "support": res.get("support"), "resistance": res.get("resistance")}
        res["strategies"] = SE.evaluate_all(df, sctx)
        res["recent_bars"] = [(str(t), float(c)) for t, c in
                              zip(df["datetime"].iloc[-8:], df["close"].iloc[-8:])]
    except Exception as e:
        logger.warning(f"strategy ensemble failed on {tf}: {e}")
        res["strategies"] = None
    return res

async def ea_fetch_and_score(symbol, tf):
    if tf not in TIMEFRAME_TO_INTERVAL and tf not in EA_RESAMPLE_MAP:
        return {"timeframe": tf, "error": True, "providers_tried": []}
    df, tried = await ea_get_ohlc(symbol, tf)
    if df is not None:
        df, meta = ea_prepare_df(df, tf)
    if df is None or len(df) < 30:
        return {"timeframe": tf, "error": True, "providers_tried": tried}
    res = ea_score_df(df, tf)
    res["bar_meta"] = meta
    return res

def ea_split_htf_ltf(tfs):
    n = len(tfs)
    if n <= 1:
        return tfs, tfs
    s = max(1, n // 2)
    return tfs[s:], tfs[:s]

def ea_compute_htf_bias(htf_results):
    if not htf_results:
        return {"bias": "NEUTRAL", "bull_weight": 0, "bear_weight": 0,
                "reasons": ["No HTF data"]}
    bw = sum(r["strength"] for r in htf_results if r["bias"] == "BULL")
    ew = sum(r["strength"] for r in htf_results if r["bias"] == "BEAR")
    if bw == 0 and ew == 0: bias = "NEUTRAL"
    elif bw > ew * 1.15: bias = "BULLISH"
    elif ew > bw * 1.15: bias = "BEARISH"
    else: bias = "NEUTRAL"
    return {"bias": bias, "bull_weight": bw, "bear_weight": ew,
            "reasons": [f"{r['timeframe']}: {r['bias']} ({r['strength']}/100)" for r in htf_results]}

def ea_compute_entry_confirmation(ltf_results, htf_bias):
    return DC.entry_confirmation(ltf_results, htf_bias)

def ea_score_htf_bias(htf):
    t = htf.get("bull_weight", 0) + htf.get("bear_weight", 0)
    return 50 if t <= 0 else round(max(htf["bull_weight"], htf["bear_weight"]) / t * 100)

def ea_score_entry_confirmation(ec):
    return {"CONFIRMED": 85, "PULLBACK": 55, "CONFLICT": 15,
            "NEUTRAL": 50, "NONE": 45}.get(ec["status"], 50)

def ea_score_sr_proximity(bias, price, sup, res, atr):
    if not atr or atr != atr or atr <= 0 or sup is None or res is None or bias not in ("BULL", "BEAR"):
        return 60, None
    dist = (res - price) if bias == "BULL" else (price - sup)
    if dist is None or dist != dist or dist <= 0:
        return 60, None
    ratio = dist / atr
    if ratio < 0.3:
        opp = "resistance" if bias == "BULL" else "support"
        return 20, f"Entry sits against {opp}"
    if ratio < 0.6:
        opp = "resistance" if bias == "BULL" else "support"
        return 45, f"Close to opposing {opp}"
    return 80, None

async def ea_generate_signal(symbol, preset_key, risk=None, snap=None, timeframes_override=None):
    tfs = timeframes_override if timeframes_override else EA_MTF_PRESETS[preset_key]["timeframes"]
    risk = risk or {}
    tf_tasks = [ea_fetch_and_score(symbol, tf) for tf in tfs]
    vision_task = (asyncio.ensure_future(ea_chart_vision_bias(symbol, tfs[0]))
                   if risk.get("chart_vision_enabled", False) else None)
    results = await asyncio.gather(*tf_tasks)
    vision = await vision_task if vision_task else None
    if not [r for r in results if not r.get("error")]:
        return {"ok": False, "failed": [r["timeframe"] for r in results if r.get("error")]}
    news = {"level": "LOW", "blocking": False, "detail": None}
    if risk.get("news_filter_enabled", True):
        news = await ea_news_risk(symbol, risk.get("news_pre_minutes", 30),
                                   risk.get("news_post_minutes", 15))
    return ea_decide_signal(symbol, preset_key, tfs, results, news, vision, risk, snap)

def ea_decide_signal(symbol, preset_key, tfs, results, news, vision, risk, snap=None):
    valid = [r for r in results if not r.get("error")]
    failed = [r["timeframe"] for r in results if r.get("error")]
    if not valid:
        return {"ok": False, "failed": failed}
    bulls = sum(1 for r in valid if r["bias"] == "BULL")
    bears = sum(1 for r in valid if r["bias"] == "BEAR")
    avg_strength = sum(r["strength"] for r in valid) / len(valid)
    htf_tfs, ltf_tfs = ea_split_htf_ltf(tfs)
    htf_results = [r for r in valid if r["timeframe"] in htf_tfs] or valid
    ltf_results = [r for r in valid if r["timeframe"] in ltf_tfs] or valid
    htf = ea_compute_htf_bias(htf_results)
    ec = ea_compute_entry_confirmation(ltf_results, htf["bias"])
    if htf["bias"] == "BULLISH": dominant = "BULL"
    elif htf["bias"] == "BEARISH": dominant = "BEAR"
    else: dominant = "BULL" if bulls >= bears else "BEAR"

    def _agg(k):
        sc = []
        for r in valid:
            c = r["components"][k]
            if c["bias"] == dominant: sc.append(c["score"])
            elif c["bias"] == "NEUTRAL": sc.append(c["score"])
            else: sc.append(c["score"] * 0.5)
        return round(sum(sc) / len(sc)) if sc else 50

    mtf_ag = round(max(bulls, bears) / len(valid) * 100)
    avg_vol = round(sum(r["components"]["volatility"]["score"] for r in valid) / len(valid))
    news_s = {"LOW": 100, "MEDIUM": 60, "HIGH": 0}.get(news["level"], 100)
    primary = next((r for r in valid if r["timeframe"] == tfs[0]), valid[0])
    sr_s, sr_note = ea_score_sr_proximity(dominant, primary.get("price"),
                                            primary.get("support"), primary.get("resistance"),
                                            primary.get("atr"))

    components = {"trend": _agg("trend"), "momentum": _agg("momentum"),
                  "structure": _agg("structure"), "price_action": _agg("price_action"),
                  "mtf": mtf_ag, "news": news_s, "volatility": avg_vol,
                  "htf_bias": ea_score_htf_bias(htf),
                  "entry_confirmation": ea_score_entry_confirmation(ec),
                  "sr_proximity": sr_s}

    vision_conflict = None
    if vision:
        if vision["bias"] == dominant:
            components["chart_vision"] = 60 + round(vision["confidence"] * 0.4)
        elif vision["bias"] == "NEUTRAL":
            components["chart_vision"] = 50
        else:
            components["chart_vision"] = max(0, 40 - round(vision["confidence"] * 0.4))
            if vision["confidence"] >= 55:
                vision_conflict = f"Vision {vision['bias']} vs indicators {dominant}"
        weights = EA_CONFLUENCE_WEIGHTS_WITH_VISION
    else:
        weights = EA_CONFLUENCE_WEIGHTS

    cs = round(sum(components[k] * w for k, w in weights.items()))
    cs = max(0, min(100, cs))

    abnormal = []
    if risk.get("abnormal_market_filter_enabled", True):
        for r in valid:
            a = r["components"].get("anomaly") or {}
            if a.get("abnormal"):
                abnormal.append(f"Abnormal ({r['timeframe']}): {a['detail']}")
                break
        if snap:
            sp = (snap.get("spreads") or {}).get(symbol)
            sp_cap = risk.get("max_spread_pips")
            if sp is not None and sp_cap and sp > sp_cap * 3:
                abnormal.append(f"Extreme spread {sp}")

    ntr = []
    if failed: ntr.append(f"Data unavailable: {', '.join(failed)}")
    if avg_strength < 50: ntr.append("Weak Momentum")
    if news["blocking"]: ntr.append(f"News HIGH: {news['detail']}")
    if avg_vol <= 20: ntr.append("Extreme Volatility")
    if vision_conflict: ntr.append(vision_conflict)
    if sr_note and sr_s <= 20: ntr.append(f"S/R: {sr_note}")
    ntr.extend(abnormal)

    strat_all, strat_summary = None, None
    prim = next((r for r in valid if r["timeframe"] == tfs[0]), None)
    if prim and prim.get("strategies") is not None:
        news_ev = SE.news_event(news, prim.get("recent_bars"), prim.get("atr"),
                                 TIMEFRAME_SECONDS.get(prim["timeframe"]),
                                 bar_utc_offset_hours=EA_BAR_UTC_OFFSET_HOURS)
        news_ev["tf"] = prim["timeframe"]
        strat_all = list(prim["strategies"]) + [news_ev]
        want = {"BULLISH": "BUY", "BEARISH": "SELL"}.get(htf["bias"])
        if want:
            strat_summary = SE.regime_support(prim["strategies"], want, news_ev,
                                                min_score=(risk or {}).get("dv_min_strategy_score", 60))

    verdict = DC.validate_direction(valid=valid, tfs=tfs, htf_tfs=htf_tfs, ltf_tfs=ltf_tfs,
                                     htf=htf, entry_conf=ec, confluence_score=cs,
                                     risk=risk, strategy_summary=strat_summary)
    wait_reasons = verdict["wait_reasons"]
    ntr.extend(verdict["block_reasons"])

    regime = valid[0]["components"]["regime"]
    if risk.get("regime_filter_enabled", True) and regime["regime"] in (risk.get("blocked_regimes") or ["UNSTABLE"]):
        ntr.append(f"Regime: {regime['label']}")

    if ntr:
        direction = "NO TRADE"
    elif wait_reasons:
        direction = "WAIT"
        ntr = wait_reasons
    else:
        direction = "BUY" if dominant == "BULL" else "SELL"

    return {"ok": True, "symbol": symbol, "preset": preset_key, "direction": direction,
            "confidence": cs, "components": components, "per_tf": valid,
            "failed": failed, "bulls": bulls, "bears": bears, "news": news,
            "chart_vision": vision, "no_trade_reasons": ntr, "regime": regime,
            "htf_bias": htf, "entry_confirmation": ec,
            "mtf": verdict["mtf"],
            "validator": {"checks": verdict["checks"], "proposed": verdict["proposed"]},
            "strategies": strat_all, "strategy_summary": strat_summary}

# ============================================================
# FORMATTERS
# ============================================================

def ea_format_signal_text(result):
    symbol = result["symbol"]
    preset = EA_MTF_PRESETS[result["preset"]]
    d = result["direction"]
    c = result["confidence"]
    icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡", "NO TRADE": "🚫"}.get(d, "⚪")
    lines = [f"{icon} *{d}* — {symbol}",
             f"MTF Preset: {preset['label']} ({preset['desc']})", "",
             f"🧠 *Confidence: {c}/100* (analytical confluence — NOT a win rate)", ""]
    htf = result.get("htf_bias")
    ec = result.get("entry_confirmation")
    if htf:
        lines.append(f"📐 HTF Bias: *{htf['bias']}*")
        if ec: lines.append(f"📍 Entry Confirmation: *{ec['status']}*")
        lines.append("")
    comp = result.get("components")
    if comp:
        lines.append("🧩 *Confluence breakdown:*")
        labels = {"htf_bias": "HTF Bias", "structure": "Structure",
                  "entry_confirmation": "Entry Confirmation", "momentum": "Momentum",
                  "price_action": "Price Action", "sr_proximity": "S/R + Liquidity",
                  "mtf": "MTF Agreement", "news": "News", "volatility": "Volatility",
                  "trend": "Trend (info)", "chart_vision": "Chart Vision"}
        for k, lb in labels.items():
            if k in comp: lines.append(f"   {lb}: {comp[k]}")
        lines.append(f"   *Final: {c}/100*"); lines.append("")
    v = result.get("chart_vision")
    if v:
        vi = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(v["bias"], "⚪")
        lines.append(f"📸 Chart Vision: {vi} {v['bias']} ({v['confidence']}%)"); lines.append("")
    lines.append("📊 *Per-timeframe:*")
    for r in result["per_tf"]:
        i = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(r["bias"], "⚪")
        lines.append(f"   {i} {r['timeframe']}: {r['bias']} ({r['strength']}/100) — {', '.join(r['reasons'])}")
    if result.get("failed"):
        lines.append(f"   ⚠️ No data: {', '.join(result['failed'])}")
    news = result.get("news") or {"level": "LOW", "detail": None}
    ni = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(news["level"], "⚪")
    nl = f"📰 News Risk: {ni} {news['level']}"
    if news.get("detail"): nl += f" — {news['detail']}"
    lines.append(""); lines.append(nl)
    regime = result.get("regime")
    if regime: lines.append(f"🌐 Market Regime: {regime['label']}")
    strat = result.get("strategies")
    if strat:
        lines.append(""); lines.append("🧠 *Strategy evidence*")
        clean = lambda t: str(t).replace("_", " ").replace("*", "").replace("`", "")
        for e in sorted(strat, key=lambda x: (not x["active"], not x["valid"], -x["score"])):
            if not e["active"]:
                lines.append(f"⚪ {clean(e['strategy'])}: off in this regime")
            elif e["valid"] and e["direction"] in ("BUY", "SELL"):
                lines.append(f"✅ {clean(e['strategy'])}: {e['direction']} {e['score']}/100 - {clean(e['reason'])}")
            else:
                lines.append(f"➖ {clean(e['strategy'])}: {clean(e['reason'])}")
    lines.append("")
    if d in ("BUY", "SELL"):
        lines.append(f"🧠 *WHY {d}?*")
        for r in result["per_tf"]:
            ok = (r["bias"] == "BULL" and d == "BUY") or (r["bias"] == "BEAR" and d == "SELL")
            lines.append(f"{'✓' if ok else '⚠️'} {r['timeframe']}: {', '.join(r['reasons'])}")
    else:
        lines.append(f"🚫 *WHY {d}?*")
        lines.append(f"• Bullish: {result['bulls']} | Bearish: {result['bears']}")
        for r in result.get("no_trade_reasons") or []:
            lines.append(f"• {r}")
    lines.append("")
    lines.append("⚠️ Confidence Score is NOT a win probability. Not financial advice.")
    return "\n".join(lines)

def ea_format_signal_card(symbol, tf_label, result, plan, risk_ok=None, risk_reasons=None, acc=None):
    d = result["direction"]
    sc = result.get("confidence")
    mtf = result.get("mtf") or {}
    news = result.get("news") or {"level": "LOW", "detail": None}
    ni = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(news.get("level"), "⚪")
    regime = result.get("regime")
    acc_lines = ([f"Validated Accuracy: {acc['pct']}", f"Sample: {acc['n']} ({acc['scope']})"]
                 if acc else ["Validated Accuracy: Insufficient data"])
    mtf_block = ["📊 MTF"]
    for row in (mtf.get("rows") or []):
        dot = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(row["bias"], "⚪")
        word = {"BULL": "BUY", "BEAR": "SELL", "NEUTRAL": "NEUTRAL"}.get(row["bias"], row["bias"])
        mtf_block.append(f"`{row['tf']:<4} {dot} {word}`")
    if mtf.get("label"):
        mtf_block.append(f"Alignment: {mtf['label']}" + (f" - {mtf['status']}" if mtf.get("status") else ""))
    ctx = [f"Regime: {regime['label']}" if regime else "Regime: N/A",
           f"News Risk: {ni} {news.get('level', 'N/A')}"]

    if d in ("BUY", "SELL") and plan:
        icon = "🟢" if d == "BUY" else "🔴"
        lines = ["🎯 *TRADE SIGNAL*", "", f"*{symbol}* · {tf_label}", "",
                 f"{icon} *{d}*", "", f"Signal Score: {sc}/100 (heuristic, not a probability)"]
        lines += acc_lines
        lines += ["", *mtf_block, "", *ctx, "",
                  f"ENTRY\n{plan['reference_price']}", "",
                  f"STOP LOSS\n{plan['sl']}", "",
                  f"TAKE PROFIT\nTP1 {plan['tp1']}\nTP2 {plan['tp2']}\nTP3 {plan['tp3']}",
                  f"R:R  1:{plan['rr1']:.1f} / 1:{plan['rr2']:.1f} / 1:{plan['rr3']:.1f}", "",
                  f"⚠️ *INVALIDATION*\nClose {'below' if d == 'BUY' else 'above'} {plan['sl']}"]
        if risk_ok is False:
            lines += ["", "⛔ *TRADE BLOCKED*"] + [f"• {r}" for r in (risk_reasons or [])]
        return "\n".join(lines)

    if d in ("BUY", "SELL"):
        return "\n".join(["🎯 *TRADE SIGNAL*", "", f"*{symbol}* · {tf_label}", "",
                          f"🟡 *{d} - levels unavailable*", "",
                          f"Signal Score: {sc}/100", *acc_lines, "", *mtf_block, "",
                          "Entry / SL / TP could not be computed."])

    head = "🛑 *NO TRADE*" if d == "NO TRADE" else "⚠️ *WAIT*"
    reasons = result.get("no_trade_reasons") or []
    if not reasons:
        reasons = [f"No majority direction - {result.get('bulls', 0)} bull vs {result.get('bears', 0)} bear"]
    lines = [head, "", f"*{symbol}* · {tf_label}", "", *mtf_block, "", *ctx, "",
             f"Decision: 🟡 *{d}*", "", "Reason:"]
    lines += [f"• {r}" for r in reasons[:5]]
    if len(reasons) > 5: lines.append(f"• +{len(reasons)-5} more")
    lines += ["", f"Signal Score: {sc}/100", *acc_lines, "",
              "Waiting is a valid result — no levels shown."]
    return "\n".join(lines)

# ============================================================
# RISK ENGINE
# ============================================================

def ea_session_for_ts(ts):
    if not ts: return "UNKNOWN"
    h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    if h < 7: return "Asian"
    if h < 15: return "London"
    return "New York"

def ea_confidence_bucket(c):
    if c is None: return "UNKNOWN"
    if c >= 90: return "90-100"
    if c >= 80: return "80-89"
    if c >= 70: return "70-79"
    return "<70"

def ea_daily_pl(snap, balance):
    if not balance: return 0.0
    today = datetime.now().date()
    total = 0.0
    for e in snap.get("events", []):
        if e.get("type") != "TRADE_CLOSED": continue
        try:
            if datetime.fromtimestamp(e["at"]).date() != today: continue
            total += float(e.get("pl", 0) or 0)
        except Exception: continue
    return total

def ea_update_peak_equity(ud, equity):
    try: equity = float(equity)
    except: return
    peak = ud.get("ea_peak_equity")
    if peak is None or equity > peak:
        ud["ea_peak_equity"] = equity

def ea_current_drawdown_pct(ud, equity):
    peak = ud.get("ea_peak_equity")
    if not peak or not equity: return 0.0
    return max(0.0, (peak - equity) / peak * 100)

def ea_consecutive_losses(ud):
    journal = ud.get("ea_trade_journal", [])
    closed = sorted([t for t in journal if t.get("status") == "CLOSED" and t.get("pl") is not None],
                     key=lambda t: t.get("closed_at") or 0, reverse=True)
    n = 0
    for t in closed:
        if t["pl"] < 0: n += 1
        else: break
    return n

def ea_cooldown_remaining_minutes(ud, cd_min):
    if not cd_min: return 0
    journal = ud.get("ea_trade_journal", [])
    losses = [t for t in journal if t.get("status") == "CLOSED" and t.get("pl") is not None and t["pl"] < 0]
    if not losses: return 0
    last = max(t.get("closed_at") or 0 for t in losses)
    if not last: return 0
    return max(0, round(cd_min - (time.time() - last) / 60, 1))

def ea_total_exposure_lots(positions):
    return sum(abs(float(p.get("lot", 0) or 0)) for p in (positions or []))

def ea_risk_check(ud, snap, signal_result):
    reasons = []
    risk = ud.get("ea_risk", {})
    if not snap["connected"]: return False, ["EA Offline"]
    if snap["run_state"] != "RUNNING":
        return False, [f"EA is {snap['run_state']} — press ▶️ Start first"]
    if signal_result["direction"] not in ("BUY", "SELL"):
        return False, [f"Signal is {signal_result['direction']}"]
    if signal_result["confidence"] < risk.get("min_confidence", 70):
        reasons.append(f"Confidence {signal_result['confidence']} < min {risk.get('min_confidence', 70)}")
    mode = (snap.get("mode") or "").upper()
    if mode == "LIVE" and not risk.get("live_trading_enabled"):
        reasons.append("Live trading disabled in ⚙️ Settings")
    positions = snap.get("positions") or []
    if len(positions) >= risk.get("max_open_trades", 3):
        reasons.append(f"Max open trades reached ({risk.get('max_open_trades', 3)})")
    same = [p for p in positions if p.get("symbol") == signal_result["symbol"]]
    if len(same) >= risk.get("max_trades_per_symbol", 1):
        reasons.append(f"Max exposure on {signal_result['symbol']}")
    max_exp = risk.get("max_total_exposure_lots")
    if max_exp:
        tot = ea_total_exposure_lots(positions)
        if tot >= max_exp: reasons.append(f"Max total exposure ({tot:.2f}/{max_exp:.2f})")
    max_sp = risk.get("max_spread_pips")
    if max_sp:
        sp = (snap.get("spreads") or {}).get(signal_result["symbol"])
        if sp is not None and sp > max_sp:
            reasons.append(f"High spread {signal_result['symbol']} ({sp})")
    acct = snap.get("account") or {}
    balance = acct.get("balance")
    equity = acct.get("equity", balance)
    if balance:
        dl = ea_daily_pl(snap, balance)
        ml = balance * (risk.get("max_daily_loss_pct", 5.0) / 100)
        if dl <= -ml: reasons.append(f"Daily loss limit ({dl:.2f} / -{ml:.2f})")
    max_dd = risk.get("max_drawdown_pct")
    if max_dd and equity:
        dd = ea_current_drawdown_pct(ud, equity)
        if dd >= max_dd: reasons.append(f"Max DD reached ({dd:.1f}% / {max_dd:.1f}%)")
    max_l = risk.get("max_consecutive_losses")
    if max_l:
        streak = ea_consecutive_losses(ud)
        if streak >= max_l: reasons.append(f"Max consecutive losses ({streak}/{max_l})")
    cd = ea_cooldown_remaining_minutes(ud, risk.get("cooldown_after_loss_minutes", 10))
    if cd > 0: reasons.append(f"Cooldown active ({cd}min)")
    allowed = risk.get("allowed_sessions") or []
    if allowed:
        sess = ea_session_for_ts(time.time())
        if sess not in allowed: reasons.append(f"Session disabled ({sess})")
    return (len(reasons) == 0), reasons

def ea_confidence_bucket(c):
    if c is None: return "UNKNOWN"
    if c >= 90: return "90-100"
    if c >= 80: return "80-89"
    if c >= 70: return "70-79"
    return "<70"

# ============================================================
# EXECUTION PLAN & JOURNAL
# ============================================================

async def ea_build_execution_plan(symbol, preset_key, direction, risk, snap=None, timeframe_override=None):
    fastest = timeframe_override or EA_MTF_PRESETS[preset_key]["timeframes"][0]
    interval = TIMEFRAME_TO_INTERVAL.get(fastest, "1h")
    df, _ = await fetch_market_data(symbol, interval=interval, timeframe=fastest)
    if df is None or len(df) < 20: return None
    price = df["close"].iloc[-1]
    atr = calculate_atr(df)
    if not atr or atr <= 0: return None
    sd, sl_mode, sl_note = ea_compute_sl_distance(df, direction, price, atr, risk)
    base_rr = risk.get("tp_rr", 2.0)
    td = sd * base_rr
    if direction == "BUY": sl, tp = price - sd, price + td
    else: sl, tp = price + sd, price - td
    tier_prices = []
    for m in (1.0, 1.6, 2.4):
        d = sd * base_rr * m
        tier_prices.append(price + d if direction == "BUY" else price - d)
    lot_info = None
    if snap is not None:
        bal = (snap.get("account") or {}).get("balance")
        specs = (snap.get("symbol_specs") or {}).get(symbol)
        lot_info = ea_calculate_lot_size(bal, risk.get("risk_per_trade_pct", 1.0), sd, specs)
    return {"symbol": symbol, "direction": direction,
            "reference_price": round(float(price), 5),
            "sl": round(float(sl), 5), "tp": round(float(tp), 5),
            "risk_pct": risk.get("risk_per_trade_pct", 1.0), "rr": base_rr,
            "sl_mode": sl_mode, "sl_note": sl_note,
            "lot_size": lot_info["lot"] if lot_info else None,
            "lot_note": lot_info["note"] if lot_info else None,
            "tp1": round(float(tier_prices[0]), 5),
            "tp2": round(float(tier_prices[1]), 5),
            "tp3": round(float(tier_prices[2]), 5),
            "rr1": round(base_rr * 1.0, 2),
            "rr2": round(base_rr * 1.6, 2),
            "rr3": round(base_rr * 2.4, 2)}

def ea_format_execution_confirm(plan, snap):
    mode = "🟡 DEMO" if (snap.get("mode") or "").upper() == "DEMO" else \
            ("🔴 LIVE — REAL MONEY" if (snap.get("mode") or "").upper() == "LIVE" else "⚪ UNKNOWN")
    lines = [f"🎯 *EXECUTE CONFIRMATION*", "", f"{plan['direction']} {plan['symbol']} — {mode}", "",
             f"Reference price: {plan['reference_price']}",
             f"Stop Loss: {plan['sl']}  ({plan.get('sl_mode', 'atr')})"]
    if plan.get("sl_note"): lines.append(f"⚠️ {plan['sl_note']}")
    lines += [f"Take Profit: {plan['tp']}  (R:R 1:{plan['rr']})",
              f"Risk: {plan['risk_pct']}% of balance"]
    if plan.get("lot_size") is not None:
        lines.append(f"Suggested Lot: {plan['lot_size']} (EA re-validates)")
        if plan.get("lot_note"): lines.append(f"⚠️ {plan['lot_note']}")
    else:
        lines.append("Lot Size: EA computes from Risk %")
    lines += ["", "EA places market order at whatever price it sees.",
              "", "⚠️ This sends a real order. Confirm?"]
    return "\n".join(lines)

def ea_record_signal(ud, symbol, preset_key, result, plan=None, timeframe=None):
    tradeable = result["direction"] in ("BUY", "SELL") and plan is not None
    if tradeable and not timeframe:
        timeframe = EA_MTF_PRESETS[preset_key]["timeframes"][0]
    entry = {"id": str(uuid.uuid4())[:8], "at": time.time(), "symbol": symbol,
             "preset": preset_key, "direction": result["direction"],
             "confidence": result["confidence"], "components": result.get("components"),
             "news_level": (result.get("news") or {}).get("level"),
             "bulls": result.get("bulls"), "bears": result.get("bears"),
             "timeframe": timeframe if tradeable else None,
             "reference_price": plan.get("reference_price") if plan else None,
             "sl": plan.get("sl") if plan else None,
             "tp": plan.get("tp") if plan else None,
             "rr_target": plan.get("rr") if plan else None,
             "status": "PENDING" if tradeable else "N/A",
             "outcome": None, "exit_price": None, "r_multiple": None, "evaluated_at": None}
    journal = ud.setdefault("ea_signal_journal", [])
    ref_bar = None
    try:
        prim = next((r for r in (result.get("per_tf") or [])
                     if r.get("timeframe") == (timeframe or EA_MTF_PRESETS[preset_key]["timeframes"][0])), None)
        ref_bar = ((prim or {}).get("bar_meta") or {}).get("ref_bar_raw")
    except Exception: pass
    if ref_bar:
        entry["ref_bar"] = ref_bar
        for old in journal[:20]:
            if (old.get("ref_bar") == ref_bar and old.get("symbol") == symbol
                    and old.get("preset") == preset_key
                    and old.get("direction") == result["direction"]
                    and (old.get("status") == "N/A") == (not tradeable)):
                try: ea_record_prediction(ud, symbol, preset_key, result, timeframe)
                except Exception: pass
                return old["id"]
    journal.insert(0, entry)
    del journal[EA_JOURNAL_MAX:]
    try: ea_record_prediction(ud, symbol, preset_key, result, timeframe)
    except Exception as e: logger.warning(f"prediction record failed: {e}")
    return entry["id"]

def ea_record_prediction(ud, symbol, preset_key, result, timeframe=None):
    tf = timeframe or EA_MTF_PRESETS[preset_key]["timeframes"][0]
    pred = PT.build_prediction(result, symbol, tf, preset=preset_key)
    preds = ud.setdefault("ea_predictions", [])
    if PT.add_prediction(preds, pred):
        PT.fold_old(preds, ud.setdefault("ea_pred_archive", {}))
        return pred["id"]
    return None

async def ea_evaluate_pending_predictions(ud):
    preds = ud.get("ea_predictions") or []
    pending = [p for p in preds if p.get("status") == "PENDING"]
    if not pending: return False
    changed = False
    groups = {}
    for p in pending:
        groups.setdefault((p["symbol"], p["tf"]), []).append(p)
    for (symbol, tf), items in list(groups.items())[:12]:
        try:
            df, _ = await ea_get_ohlc_cached(symbol, tf, outputsize="500")
        except Exception as e:
            logger.warning(f"pred eval fetch failed: {e}"); continue
        for p in items:
            try:
                upd = PT.resolve_prediction(p, df, utc_offset_hours=EA_BAR_UTC_OFFSET_HOURS)
            except Exception as e:
                logger.warning(f"pred eval error: {e}"); upd = None
            if upd: p.update(upd); changed = True
    return changed

def ea_accuracy_for_card(ud, symbol, tf):
    if not ud: return None
    try:
        preds, arch = ud.get("ea_predictions") or [], ud.get("ea_pred_archive")
        for sym, t in ((symbol, tf), (None, None)):
            st = PT.compute_stats(preds, arch if sym is None else None, symbol=sym, tf=t)
            line = PT.accuracy_line(st["overall"])
            if line:
                return {"pct": line[0], "n": line[1],
                        "scope": "this symbol and TF" if sym else "all signals"}
    except Exception as e:
        logger.warning(f"accuracy for card failed: {e}")
    return None

def ea_validated_accuracy(ud, symbol=None, tf=None):
    st = PT.compute_stats(ud.get("ea_predictions") or [], ud.get("ea_pred_archive"),
                           symbol=symbol, tf=tf)
    return PT.accuracy_line(st["overall"])

# ============================================================
# CHART VISION
# ============================================================

async def ea_chart_vision_bias(symbol, timeframe):
    if not (GEMINI_API_KEY or OPENROUTER_API_KEY):
        return None
    try:
        chart_bytes = await asyncio.wait_for(fetch_chart_image(symbol, timeframe), timeout=12)
    except Exception:
        return None
    if not chart_bytes: return None
    prompt = (f"Look at this {symbol} ({timeframe}) candlestick chart. Base your read purely on "
              "what is visible. Reply EXACTLY:\nBIAS: BULL or BEAR or NEUTRAL\nCONFIDENCE: <0-100>")
    text, _prov, _err = await ai_router_generate_vision(chart_bytes, "image/jpeg", prompt)
    if not text: return None
    bm = re.search(r"BIAS:\s*(BULL|BEAR|NEUTRAL)", text, re.IGNORECASE)
    if not bm: return None
    cm = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
    conf = max(0, min(100, int(cm.group(1)))) if cm else 60
    return {"bias": bm.group(1).upper(), "confidence": conf}

# ============================================================
# MENUS
# ============================================================

def ea_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎯 Signals", callback_data="ea_signals"),
         InlineKeyboardButton("🧠 Analysis", callback_data="ea_analysis")],
        [InlineKeyboardButton("📡 Live Chart", callback_data="ea_livechart"),
         InlineKeyboardButton("📊 Live Dashboard", callback_data="ea_dashboard")],
        [InlineKeyboardButton("📡 EA Status", callback_data="ea_status")],
        [InlineKeyboardButton("💼 Account", callback_data="ea_account"),
         InlineKeyboardButton("📈 Positions", callback_data="ea_positions")],
        [InlineKeyboardButton("📜 Trade History", callback_data="ea_history"),
         InlineKeyboardButton("🧪 Backtesting", callback_data="ea_backtest_menu")],
        [InlineKeyboardButton("📈 Strategy Performance", callback_data="ea_performance")],
        [InlineKeyboardButton("▶️ Start EA", callback_data="ea_start"),
         InlineKeyboardButton("⏸️ Pause EA", callback_data="ea_pause")],
        [InlineKeyboardButton("🛑 Emergency Stop", callback_data="ea_estop_confirm"),
         InlineKeyboardButton("🔔 Notifications", callback_data="ea_notifications")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="ea_settings")],
    ])

def ea_symbol_keyboard():
    kb = [[InlineKeyboardButton(label, callback_data=f"ea_symbol_{s}")]
          for s, label in FOREX_SYMBOLS.items()]
    kb += [[InlineKeyboardButton(label, callback_data=f"ea_symbol_{s}")]
           for s, label in CRYPTO_SYMBOLS.items()]
    kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_menu")])
    return InlineKeyboardMarkup(kb)

def ea_preset_keyboard(symbol):
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"{p['label']} ({p['desc']})", callback_data=f"ea_preset_{symbol}_{k}")]
         for k, p in EA_MTF_PRESETS.items()] +
        [[InlineKeyboardButton("🔙 Back", callback_data="ea_signals")]])

def ea_analysis_symbol_keyboard():
    kb = [[InlineKeyboardButton(label, callback_data=f"ea_analysis_symbol_{s}")]
          for s, label in FOREX_SYMBOLS.items()]
    kb += [[InlineKeyboardButton(label, callback_data=f"ea_analysis_symbol_{s}")]
           for s, label in CRYPTO_SYMBOLS.items()]
    kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_menu")])
    return InlineKeyboardMarkup(kb)

def ea_analysis_preset_keyboard(symbol):
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"{p['label']} ({p['desc']})", callback_data=f"ea_analysis_preset_{symbol}_{k}")]
         for k, p in EA_MTF_PRESETS.items()] +
        [[InlineKeyboardButton("🔙 Back", callback_data="ea_analysis")]])

def ea_backtest_mode_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧪 Standard Backtest", callback_data="ea_bt_symbol_menu")],
        [InlineKeyboardButton("🔬 Walk-Forward", callback_data="ea_wf_symbol_menu")],
        [InlineKeyboardButton("🆚 A/B Test", callback_data="ea_ab_symbol_menu")],
        [InlineKeyboardButton("🔙 Back", callback_data="ea_menu")],
    ])

def ea_backtest_symbol_keyboard(prefix):
    kb = [[InlineKeyboardButton(label, callback_data=f"{prefix}{s}")] for s, label in FOREX_SYMBOLS.items()]
    kb += [[InlineKeyboardButton(label, callback_data=f"{prefix}{s}")] for s, label in CRYPTO_SYMBOLS.items()]
    kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_backtest_menu")])
    return InlineKeyboardMarkup(kb)

def ea_backtest_tf_keyboard(symbol, prefix):
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(tf, callback_data=f"{prefix}{symbol}_{tf}")] for tf in BACKTEST_TIMEFRAMES] +
        [[InlineKeyboardButton("🔙 Back", callback_data="ea_backtest_menu")]])

def ea_livechart_symbol_keyboard():
    kb = [[InlineKeyboardButton(label, callback_data=f"ea_livechart_symbol_{s}")]
          for s, label in FOREX_SYMBOLS.items()]
    kb += [[InlineKeyboardButton(label, callback_data=f"ea_livechart_symbol_{s}")]
           for s, label in CRYPTO_SYMBOLS.items()]
    kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_menu")])
    return InlineKeyboardMarkup(kb)

def ea_livechart_tf_keyboard(symbol):
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"⏱ {tf}", callback_data=f"ea_livechart_tf_{symbol}_{tf}")]
         for tf in TIMEFRAMES] +
        [[InlineKeyboardButton("🔙 Back", callback_data="ea_livechart")]])

def ea_livechart_result_keyboard(symbol, tf):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh", callback_data=f"ea_livechart_tf_{symbol}_{tf}")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="ea_settings")],
        [InlineKeyboardButton("🔙 Back", callback_data="ea_livechart")],
    ])

def ea_signal_card_keyboard(symbol, preset_key, primary_tf, direction=None, back=None):
    refresh_cb = f"ea_analysis_preset_{symbol}_{preset_key}" if back else f"ea_preset_{symbol}_{preset_key}"
    rows = [[InlineKeyboardButton("🔄 Refresh", callback_data=refresh_cb),
             InlineKeyboardButton("📊 Details", callback_data=f"ea_sigdetail_{symbol}_{preset_key}"),
             InlineKeyboardButton("📈 Chart", callback_data=f"ea_livechart_tf_{symbol}_{primary_tf}")]]
    if direction in ("BUY", "SELL") and not back:
        rows.append([InlineKeyboardButton("🚀 Send to EA (Execute)",
                                          callback_data=f"ea_execute_{symbol}_{preset_key}_{direction}")])
    if back:
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data=back)])
    return InlineKeyboardMarkup(rows)

def ea_settings_keyboard(ud):
    risk = ud.get("ea_risk", {})
    default = ud.get("ea_default_preset", "intraday")
    kb = []
    for k, p in EA_MTF_PRESETS.items():
        mark = "✅ " if k == default else ""
        kb.append([InlineKeyboardButton(f"{mark}{p['label']} default", callback_data=f"ea_setdefault_{k}")])
    kb.append([InlineKeyboardButton(f"💰 Risk/trade: {risk.get('risk_per_trade_pct', 1.0)}%",
                                     callback_data="ea_cycle_risk_pct")])
    kb.append([InlineKeyboardButton(f"🧠 Min confidence: {risk.get('min_confidence', 70)}",
                                     callback_data="ea_cycle_min_conf")])
    kb.append([InlineKeyboardButton(f"📊 Max open trades: {risk.get('max_open_trades', 3)}",
                                     callback_data="ea_cycle_max_trades")])
    live = risk.get("live_trading_enabled", False)
    kb.append([InlineKeyboardButton(
        "🔴 LIVE trading: ON (tap to disable)" if live else "🟢 LIVE trading: OFF (tap to enable)",
        callback_data="ea_disable_live" if live else "ea_toggle_live")])
    be = risk.get("breakeven_enabled", True)
    kb.append([InlineKeyboardButton(f"{'✅' if be else '⬜'} Break-even", callback_data="ea_toggle_be"),
               InlineKeyboardButton(f"Trigger: {risk.get('breakeven_trigger_rr', 1.0)}R",
                                    callback_data="ea_cycle_be_trigger")])
    tr = risk.get("trailing_enabled", False)
    kb.append([InlineKeyboardButton(f"{'✅' if tr else '⬜'} Trailing", callback_data="ea_toggle_trailing"),
               InlineKeyboardButton(f"Dist: {risk.get('trailing_mult', 1.5)}x",
                                    callback_data="ea_cycle_trailing_mult")])
    sl_modes = {"atr": "🎯 SL: ATR", "structure": "🎯 SL: Structure", "fixed": "🎯 SL: Fixed %"}
    kb.append([InlineKeyboardButton(sl_modes.get(risk.get("sl_mode", "atr"), "🎯 SL: ATR"),
                                     callback_data="ea_cycle_sl_mode"),
               InlineKeyboardButton(f"Fixed: {risk.get('fixed_sl_pct', 0.5)}%",
                                    callback_data="ea_cycle_fixed_sl_pct")])
    ptp = risk.get("partial_tp_enabled", False)
    kb.append([InlineKeyboardButton(f"{'✅' if ptp else '⬜'} Partial TP", callback_data="ea_toggle_partial_tp"),
               InlineKeyboardButton(f"@ {risk.get('partial_tp_trigger_rr', 1.0)}R",
                                    callback_data="ea_cycle_partial_tp_rr")])
    nf = risk.get("news_filter_enabled", True)
    kb.append([InlineKeyboardButton(f"{'✅' if nf else '⬜'} News filter", callback_data="ea_toggle_news"),
               InlineKeyboardButton(f"±{risk.get('news_pre_minutes', 30)}min",
                                    callback_data="ea_cycle_news_window")])
    ab = risk.get("abnormal_market_filter_enabled", True)
    kb.append([InlineKeyboardButton(f"{'✅' if ab else '⬜'} Abnormal Market",
                                     callback_data="ea_toggle_abnormal")])
    rg = risk.get("regime_filter_enabled", True)
    blocked = risk.get("blocked_regimes") or ["UNSTABLE"]
    kb.append([InlineKeyboardButton(f"{'✅' if rg else '⬜'} Regime filter", callback_data="ea_toggle_regime"),
               InlineKeyboardButton(f"Blocks: {'/'.join(blocked)}",
                                    callback_data="ea_cycle_blocked_regimes")])
    cv = risk.get("chart_vision_enabled", False)
    kb.append([InlineKeyboardButton(f"{'✅' if cv else '⬜'} Chart Vision", callback_data="ea_toggle_chart_vision")])
    kb.append([InlineKeyboardButton(f"📉 Max DD: {risk.get('max_drawdown_pct', 15.0)}%",
                                     callback_data="ea_cycle_max_dd"),
               InlineKeyboardButton(f"❌ Max losses: {risk.get('max_consecutive_losses', 3)}",
                                    callback_data="ea_cycle_max_losses")])
    spread_v = risk.get("max_spread_pips")
    exp_v = risk.get("max_total_exposure_lots")
    kb.append([InlineKeyboardButton(f"📏 Max spread: {spread_v or 'Off'} pips",
                                     callback_data="ea_cycle_max_spread"),
               InlineKeyboardButton(f"📦 Max exposure: {exp_v or 'Off'}",
                                    callback_data="ea_cycle_max_exposure")])
    sess = risk.get("allowed_sessions") or []
    kb.append([InlineKeyboardButton(f"🕐 Sessions: {', '.join(sess) if sess else 'All'}",
                                     callback_data="ea_cycle_sessions")])
    cd = risk.get("cooldown_after_loss_minutes", 10)
    kb.append([InlineKeyboardButton(f"⏳ Cooldown: {cd if cd else 'Off'}{'min' if cd else ''}",
                                     callback_data="ea_cycle_cooldown")])
    kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_menu")])
    return InlineKeyboardMarkup(kb)

# ============================================================
# COMMANDS
# ============================================================

async def start_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Bot នេះសម្រាប់តែ Admin ប៉ុណ្ណោះ។")
        return
    ud = get_user_data(update.effective_user.id)
    clear_conversation_modes(ud)
    save_data()
    await update.message.reply_text(
        "📊 *TradeMind — Trading Control Center*\n\n"
        "Analyze • Signal • EA Control • Backtest • Performance",
        parse_mode="Markdown", reply_markup=ea_main_keyboard())

async def ea_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only.")
        return
    await update.message.reply_text(
        "🤖 *EA CONTROL CENTER*\n\nSelect a menu:", parse_mode="Markdown",
        reply_markup=ea_main_keyboard())

async def ea_signal_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    await update.message.reply_text("🎯 *SIGNALS*\n\nChoose a symbol:", parse_mode="Markdown",
                                     reply_markup=ea_symbol_keyboard())

async def ea_analyze_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    await update.message.reply_text("🧠 *ANALYSIS*\n\nChoose a symbol:", parse_mode="Markdown",
                                     reply_markup=ea_analysis_symbol_keyboard())

async def ea_status_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    snap = ea_snapshot()
    if snap["connected"]:
        mode = "🟡 DEMO" if (snap.get("mode") or "").upper() == "DEMO" else \
                ("🔴 LIVE" if (snap.get("mode") or "").upper() == "LIVE" else "⚪ UNKNOWN")
        age = int(time.time() - snap["last_heartbeat"])
        text = (f"📡 *EA STATUS*\n\n🟢 ONLINE — {age}s ago\nMode: {mode}\n"
                f"Run State: *{snap['run_state']}*\nBroker: {snap.get('broker') or 'N/A'}")
    else:
        text = f"📡 *EA STATUS*\n\n🔴 *EA OFFLINE*\n\nNo heartbeat from MT5 EA yet."
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=ea_main_keyboard())

async def ea_settings_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    ud = get_user_data(update.effective_user.id)
    await update.message.reply_text("⚙️ *EA SETTINGS*", parse_mode="Markdown",
                                     reply_markup=ea_settings_keyboard(ud))

async def ea_backtest_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    await update.message.reply_text("🧪 *BACKTESTING*", parse_mode="Markdown",
                                     reply_markup=ea_backtest_mode_keyboard())

async def ea_performance_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    ud = get_user_data(update.effective_user.id)
    if await ea_evaluate_pending_signals(ud):
        save_data()
    filters = ud.get("ea_perf_filters") or {"period": "7D", "symbol": "ALL",
                                              "timeframe": "ALL", "direction": "ALL", "preset": "ALL"}
    signals = ea_perf_filtered_signals(ud, filters)
    stats = ea_compute_strategy_performance(signals)
    text = ea_format_strategy_performance(stats, filters)
    await update.message.reply_text(text, parse_mode="Markdown",
                                     reply_markup=ea_strategy_performance_keyboard(filters))

async def accuracy_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    ud = get_user_data(update.effective_user.id)
    args = [a.upper() for a in (context.args or [])]
    symbol = next((a for a in args if a in FOREX_SYMBOLS or a in CRYPTO_SYMBOLS), None)
    tf = next((a for a in args if a in TIMEFRAME_SECONDS), None)
    msg = await update.message.reply_text("⏳ Scoring predictions...")
    try:
        if await ea_evaluate_pending_predictions(ud):
            save_data()
        st = PT.compute_stats(ud.get("ea_predictions") or [], ud.get("ea_pred_archive"),
                               symbol=symbol, tf=tf)
        text = PT.format_stats(st)
    except Exception as e:
        logger.exception(f"/accuracy failed: {e}")
        text = "❌ Could not compute accuracy."
    try: await msg.edit_text(text[:4000])
    except Exception: await update.message.reply_text(text[:4000])

async def aistatus_cmd(update, context):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    await update.message.reply_text(f"📊 *Provider Health*\n\n{provider_health_summary()}",
                                     parse_mode="Markdown")

async def providers_cmd(update, context):
    """បង្ហាញ Provider ណាដែល Setup រួច vs មិនទាន់ — មិនបង្ហាញតម្លៃ Key ទេ។"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🔒 Admin only."); return
    await update.message.reply_text(provider_status_summary(), parse_mode="Markdown")

async def help_cmd(update, context):
    await update.message.reply_text(
        "📊 *TradeMind*\n\n"
        "/start — Control Center\n"
        "/ea — EA Control Center\n"
        "/signal — Signals flow\n"
        "/analyze — Analysis flow\n"
        "/ea_status — EA heartbeat status\n"
        "/ea_settings — Risk settings\n"
        "/ea_backtest — Backtest menu\n"
        "/ea_performance — Strategy Performance\n"
        "/accuracy [SYMBOL] [TF] — Validated accuracy\n"
        "/providers — Show configured providers\n"
        "/aistatus — AI provider health\n"
        "/cancel — Exit current flow",
        parse_mode="Markdown")

async def cancel_cmd(update, context):
    ud = get_user_data(update.effective_user.id)
    clear_conversation_modes(ud)
    save_data()
    await update.message.reply_text("✅ Cleared all active flows.")

# ============================================================
# LIVE CHART
# ============================================================

def ea_technical_readout(symbol, df, tf):
    bias = ea_analyze_timeframe_bias(df, tf)
    price = df["close"].iloc[-1]
    ema20, ema50 = calculate_ema(df, 20), calculate_ema(df, 50)
    ml, sl, _ = calculate_macd(df)
    atr = calculate_atr(df)
    bu, bm, bl = calculate_bollinger(df)
    sk, sd = calculate_stochastic(df)
    sup, res = find_support_resistance(df)
    return {"symbol": symbol, "tf": tf, "price": price, "bias": bias,
            "ema20": ema20, "ema50": ema50, "macd_bullish": ml > sl,
            "atr": atr, "bb_upper": bu, "bb_mid": bm, "bb_lower": bl,
            "stoch_k": sk, "stoch_d": sd, "support": sup, "resistance": res}

async def ea_build_live_chart(symbol, tf, risk=None):
    risk = risk or {}
    df, tried = await ea_get_ohlc(symbol, tf, outputsize="200")
    if df is None or len(df) < 30:
        return None, tried
    tech = ea_technical_readout(symbol, df, tf)
    try:
        chart_bytes = await fetch_chart_image(symbol, tf)
    except Exception:
        chart_bytes = None
    vision = None
    if risk.get("chart_vision_enabled", False):
        try: vision = await ea_chart_vision_bias(symbol, tf)
        except Exception: vision = None
    return {**tech, "chart_bytes": chart_bytes, "vision": vision,
            "data_source": tried[-1] if tried else "N/A",
            "updated_at": datetime.now().strftime("%H:%M:%S")}, tried

def ea_format_live_chart(d):
    comps = d["bias"]["components"]
    regime = comps["regime"]
    rsi = d["bias"].get("rsi")
    lines = [f"📡 *LIVE CHART — {d['symbol']} ({d['tf']})*", f"💰 Price: {d['price']:.5f}", "",
             "📊 *MARKET DATA*",
             f"Status: 🟢 Live   |   Updated: {d['updated_at']}",
             f"Data Source: {d['data_source']}", "",
             "📈 *TECHNICAL STATUS*",
             f"Trend: {comps['trend']['bias']} ({comps['trend']['score']})",
             f"RSI: {rsi:.2f}" if rsi is not None else "RSI: N/A",
             f"MACD: {'Bullish' if d['macd_bullish'] else 'Bearish'}",
             f"EMA20: {d['ema20']:.5f}   |   EMA50: {d['ema50']:.5f}",
             f"ATR: {d['atr']:.5f}" if d["atr"] is not None and not pd.isna(d["atr"]) else "ATR: N/A"]
    if d["bb_upper"] is not None:
        lines.append(f"BB: {d['bb_lower']:.5f} / {d['bb_mid']:.5f} / {d['bb_upper']:.5f}")
    if d["stoch_k"] is not None:
        lines.append(f"Stoch: %K {d['stoch_k']:.1f} / %D {d['stoch_d']:.1f}")
    lines += [f"Support: {d['support']:.5f}   |   Resistance: {d['resistance']:.5f}",
              "", "🧠 *MARKET CONDITION*",
              f"{regime['label']} — {regime['detail']}",
              "", "🤖 *AI CHART ANALYSIS*"]
    if d.get("vision"):
        lines.append(f"Visual: {d['vision']['bias']} ({d['vision']['confidence']}%)")
    else:
        lines.append("_Unavailable — enable Chart Vision in ⚙️ Settings, or no Vision provider configured._")
    lines += ["", "⚠️ Not financial advice."]
    return "\n".join(lines)

# ============================================================
# STRATEGY PERFORMANCE
# ============================================================

def ea_perf_filtered_signals(ud, filters):
    journal = ud.get("ea_signal_journal", [])
    ps = EA_PERF_PERIOD_SECONDS.get(filters.get("period", "ALL"))
    now = time.time()
    out = []
    for s in journal:
        if ps is not None and (now - s.get("at", 0)) > ps: continue
        if filters.get("symbol", "ALL") != "ALL" and s.get("symbol") != filters["symbol"]: continue
        if filters.get("timeframe", "ALL") != "ALL" and s.get("timeframe") != filters["timeframe"]: continue
        if filters.get("direction", "ALL") != "ALL" and s.get("direction") != filters["direction"]: continue
        if filters.get("preset", "ALL") != "ALL" and s.get("preset") != filters["preset"]: continue
        out.append(s)
    return out

def ea_compute_strategy_performance(signals):
    total = len(signals)
    buys = sum(1 for s in signals if s.get("direction") == "BUY")
    sells = sum(1 for s in signals if s.get("direction") == "SELL")
    waits = sum(1 for s in signals if s.get("direction") in ("WAIT", "NO TRADE"))
    resolved = [s for s in signals if s.get("status") == "RESOLVED"]
    pending = [s for s in signals if s.get("status") == "PENDING"]
    expired = [s for s in signals if s.get("status") == "EXPIRED"]
    trades = [{"direction": s["direction"], "r": s["r_multiple"]}
              for s in resolved if s.get("r_multiple") is not None]
    if trades:
        wins = [t for t in trades if t["r"] > 0]
        losses = [t for t in trades if t["r"] <= 0]
        gw = sum(t["r"] for t in wins)
        gl = abs(sum(t["r"] for t in losses))
        pf = (gw / gl) if gl else (float("inf") if gw else 0.0)
        eq, cum, peak, mdd = [], 0.0, 0.0, 0.0
        for t in trades:
            cum += t["r"]; eq.append(cum)
            peak = max(peak, cum); mdd = max(mdd, peak - cum)
    else:
        wins, losses = [], []
        gw, gl, pf, mdd, eq = 0, 0, None, 0, []
    conf = [s["confidence"] for s in signals if s.get("confidence") is not None]
    rr = [s["rr_target"] for s in signals if s.get("rr_target") is not None]
    return {
        "total_signals": total, "buy_signals": buys, "sell_signals": sells, "wait_signals": waits,
        "evaluated": len(trades), "pending": len(pending), "expired": len(expired),
        "wins": len(wins), "losses": len(losses), "breakeven": 0,
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else None,
        "loss_rate": round(len(losses) / len(trades) * 100, 2) if trades else None,
        "avg_confidence": round(sum(conf) / len(conf), 1) if conf else None,
        "total_pl_r": round(sum(t["r"] for t in trades), 2) if trades else None,
        "avg_win_r": round(sum(t["r"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss_r": round(sum(t["r"] for t in losses) / len(losses), 2) if losses else None,
        "profit_factor": round(pf, 2) if isinstance(pf, (int, float)) and pf != float("inf") else pf,
        "max_drawdown_r": round(mdd, 2) if trades else None,
        "avg_rr": round(sum(rr) / len(rr), 2) if rr else None,
        "sample_size": len(trades),
    }

def ea_group_strategy_performance(signals, key_fn):
    groups = {}
    for s in signals:
        k = key_fn(s)
        if k is None: continue
        groups.setdefault(k, []).append(s)
    return {k: ea_compute_strategy_performance(v) for k, v in groups.items()}

def _ea_pf_label(pf):
    if pf is None: return "—"
    if pf == float("inf"): return "∞"
    return str(pf)

def ea_format_strategy_performance(stats, filters):
    header = ("📈 *STRATEGY PERFORMANCE*\n"
              "_Signal-based, historical-simulation — no MT5 required._\n\n"
              f"Period: {filters.get('period', 'ALL')}   Symbol: {filters.get('symbol', 'ALL')}   "
              f"TF: {filters.get('timeframe', 'ALL')}")
    lines = [header, "━━━━━━━━━━━━━━"]
    if stats["total_signals"] == 0:
        lines.append("\n📭 No signal history yet for this filter.")
        return "\n".join(lines)
    lines += ["", "📊 *OVERVIEW*", "",
              f"Signals: {stats['total_signals']}  (BUY {stats['buy_signals']} · SELL {stats['sell_signals']} · WAIT/NT {stats['wait_signals']})",
              f"Evaluated: {stats['evaluated']}   Pending: {stats['pending']}   Expired: {stats['expired']}"]
    if not stats["evaluated"]:
        lines += ["", "No resolved signals yet."]
        return "\n".join(lines)
    lines += ["", f"🟢 Wins: {stats['wins']}   🔴 Losses: {stats['losses']}", "",
              f"Win Rate: {stats['win_rate']}%   Loss Rate: {stats['loss_rate']}%",
              "", "━━━━━━━━━━━━━━", "", "💰 *SIMULATION (R-multiples)*", ""]
    if stats['total_pl_r'] is not None: lines.append(f"Total P/L: {stats['total_pl_r']:+.2f}R")
    lines.append(f"Profit Factor: {_ea_pf_label(stats['profit_factor'])}")
    if stats['avg_win_r'] is not None: lines.append(f"Avg Win: {stats['avg_win_r']:+.2f}R")
    if stats['avg_loss_r'] is not None: lines.append(f"Avg Loss: {stats['avg_loss_r']:+.2f}R")
    if stats['max_drawdown_r'] is not None: lines.append(f"Max DD: {stats['max_drawdown_r']:.2f}R")
    lines += ["", "━━━━━━━━━━━━━━", "", "🎯 *SIGNAL QUALITY*", ""]
    if stats['avg_confidence'] is not None: lines.append(f"Avg Confidence: {stats['avg_confidence']}%")
    if stats['avg_rr'] is not None: lines.append(f"Avg R:R (target): {stats['avg_rr']}")
    if stats["sample_size"] < EA_PERF_LOW_SAMPLE:
        lines += ["", f"⚠️ LOW sample ({stats['sample_size']} evaluated) — treat as preliminary."]
    lines += ["", "_Simulated performance — not a real trading account._"]
    return "\n".join(lines)

def ea_strategy_performance_keyboard(filters):
    period = filters.get("period", "ALL")
    nxt = EA_PERF_PERIODS[(EA_PERF_PERIODS.index(period) + 1) % len(EA_PERF_PERIODS)] if period in EA_PERF_PERIODS else "7D"
    d = filters.get("direction", "ALL")
    nc = {"ALL": "BUY", "BUY": "SELL", "SELL": "ALL"}
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🕐 Period: {period}", callback_data=f"ea_sp_period_{nxt}")],
        [InlineKeyboardButton(f"💱 Symbol: {filters.get('symbol', 'ALL')}",
                               callback_data="ea_sp_symbol_menu"),
         InlineKeyboardButton(f"⏱ TF: {filters.get('timeframe', 'ALL')}",
                               callback_data="ea_sp_tf_menu")],
        [InlineKeyboardButton(f"↕️ Direction: {d}", callback_data=f"ea_sp_dir_{nc.get(d, 'ALL')}"),
         InlineKeyboardButton("🔁 Reset", callback_data="ea_sp_reset")],
        [InlineKeyboardButton("📊 By Symbol", callback_data="ea_sp_bysymbol"),
         InlineKeyboardButton("⏱ By TF", callback_data="ea_sp_bytf")],
        [InlineKeyboardButton("↕️ By Direction", callback_data="ea_sp_bydir"),
         InlineKeyboardButton("🧩 By Preset", callback_data="ea_sp_bypreset")],
        [InlineKeyboardButton("📜 Signal History", callback_data="ea_sp_history")],
        [InlineKeyboardButton("🔙 Back", callback_data="ea_menu")],
    ])

async def ea_evaluate_signal_outcome(entry):
    symbol, tf = entry.get("symbol"), entry.get("timeframe")
    direction = entry.get("direction")
    entry_price, sl, tp = (entry.get("reference_price"), entry.get("sl"), entry.get("tp"))
    at = entry.get("at")
    if entry.get("status") != "PENDING" or direction not in ("BUY", "SELL"):
        return None
    if not (symbol and tf and entry_price and sl and tp and at):
        return None
    try:
        df, _ = await ea_get_ohlc_cached(symbol, tf, outputsize="500")
    except Exception:
        return None
    if df is None or df.empty: return None
    entry_dt = pd.Timestamp(datetime.utcfromtimestamp(at))
    future = df[df["datetime"] > entry_dt]
    if future.empty: return None
    for _, row in future.iterrows():
        hit_sl = (row["low"] <= sl) if direction == "BUY" else (row["high"] >= sl)
        hit_tp = (row["high"] >= tp) if direction == "BUY" else (row["low"] <= tp)
        if hit_sl:
            return {"status": "RESOLVED", "outcome": "LOSS",
                    "outcome_at": row["datetime"].isoformat(),
                    "exit_price": float(sl), "r_multiple": -1.0, "evaluated_at": time.time()}
        if hit_tp:
            rr = entry.get("rr_target") or 2.0
            return {"status": "RESOLVED", "outcome": "WIN",
                    "outcome_at": row["datetime"].isoformat(),
                    "exit_price": float(tp), "r_multiple": round(float(rr), 2),
                    "evaluated_at": time.time()}
    return None

async def ea_evaluate_pending_signals(ud, limit=200):
    journal = ud.get("ea_signal_journal", [])
    checked = 0; changed = False
    for e in journal:
        if e.get("status") != "PENDING": continue
        if checked >= limit: break
        checked += 1
        try:
            upd = await ea_evaluate_signal_outcome(e)
        except Exception as ex:
            logger.warning(f"eval error: {ex}"); upd = None
        if upd: e.update(upd); changed = True
    try:
        if await ea_evaluate_pending_predictions(ud): changed = True
    except Exception as ex:
        logger.warning(f"pred eval error: {ex}")
    return changed

# ============================================================
# BACKTEST (simplified)
# ============================================================

def _ea_backtest_fetch_sync(df, timeframe, risk, i_start, i_end):
    """No-look-ahead replay; simplified - SL first if both touched."""
    trades = []
    i = max(i_start, BACKTEST_MIN_BARS)
    i_end = min(i_end, len(df) - 1)
    min_conf = (risk or {}).get("min_confidence", 70)
    while i < i_end:
        window = df.iloc[:i + 1]
        try:
            res = ea_score_df(window, timeframe)
        except Exception:
            i += 1; continue
        if res.get("bias") == "NEUTRAL":
            i += 1; continue
        direction = "BUY" if res["bias"] == "BULL" else "SELL"
        score = res["strength"]
        if score < min_conf:
            i += 1; continue
        atr = calculate_atr(window)
        if not atr or atr <= 0 or pd.isna(atr):
            i += 1; continue
        entry = window["close"].iloc[-1]
        sl_d, _, _ = ea_compute_sl_distance(window, direction, entry, atr, risk)
        tp_rr = (risk or {}).get("tp_rr", 2.0)
        tp_d = sl_d * tp_rr
        sl = entry - sl_d if direction == "BUY" else entry + sl_d
        tp = entry + tp_d if direction == "BUY" else entry - tp_d
        exit_r, exit_j = None, None
        for j in range(i + 1, len(df)):
            hi_j, lo_j = df["high"].iloc[j], df["low"].iloc[j]
            hit_sl = (lo_j <= sl) if direction == "BUY" else (hi_j >= sl)
            hit_tp = (hi_j >= tp) if direction == "BUY" else (lo_j <= tp)
            if hit_sl: exit_r, exit_j = -1.0, j; break
            if hit_tp: exit_r, exit_j = tp_rr, j; break
        if exit_r is None: break
        trades.append({"direction": direction, "entry_i": i, "exit_i": exit_j, "r": exit_r})
        i = exit_j + 1
    return trades

async def ea_run_backtest(symbol, timeframe, risk):
    df, tried = await ea_get_ohlc(symbol, timeframe, outputsize="500")
    if df is None: return {"ok": False, "reason": "no_data", "tried": tried}
    df, _ = ea_prepare_df(df, timeframe)
    if df is None or len(df) < 80:
        return {"ok": False, "reason": "not_enough_bars"}
    n = len(df)
    loop = asyncio.get_event_loop()
    trades = await loop.run_in_executor(None, _ea_backtest_fetch_sync, df, timeframe, risk,
                                          BACKTEST_MIN_BARS, n - 1)
    return {"ok": True, "symbol": symbol, "timeframe": timeframe, "bars_used": n, "trades": trades}

def ea_format_backtest(result):
    if not result.get("ok"):
        return f"📊 *BACKTEST*\n\nNot enough data ({result.get('reason')})."
    trades = result["trades"]
    lines = [f"📊 *BACKTEST* — {result['symbol']} {result['timeframe']} ({result['bars_used']} bars)", ""]
    if not trades:
        lines.append("No qualifying trades.")
    else:
        wins = [t for t in trades if t["r"] > 0]
        losses = [t for t in trades if t["r"] <= 0]
        gw = sum(t["r"] for t in wins)
        gl = abs(sum(t["r"] for t in losses))
        pf = (gw / gl) if gl else (float("inf") if gw else 0)
        avg = sum(t["r"] for t in trades) / len(trades)
        pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
        lines += [f"Trades: {len(trades)}", f"Win rate: {len(wins)/len(trades)*100:.1f}%",
                  f"Profit factor: {pf_str}", f"Avg R: {avg:+.2f}R"]
        if len(trades) < 30: lines.append("Small sample — indicative only.")
    lines += ["", "⚠️ No news replay, spread, slippage or commission modeled. Not a forecast."]
    return "\n".join(lines)

# ============================================================
# BUTTON HANDLER
# ============================================================

async def safe_alert(q, text, show_alert=True):
    try:
        await q.answer(text, show_alert=show_alert); return
    except Exception: pass
    try: await q.message.reply_text(text)
    except Exception: pass

async def ea_show_card(q, text, kb):
    try:
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=kb); return
    except Exception as e:
        if "not modified" in str(e).lower(): return
    plain = text.replace("*", "").replace("`", "")
    try:
        await q.edit_message_text(plain, reply_markup=kb)
    except Exception as e:
        if "not modified" not in str(e).lower():
            await q.message.reply_text(plain, reply_markup=kb)

async def button_handler(update, context):
    q = update.callback_query
    data = q.data
    uid = q.from_user.id
    try: await q.answer()
    except Exception: pass
    ud = get_user_data(uid)
    if not is_admin(uid):
        await safe_alert(q, "🔒 Admin only.")
        return
    logger.info(f"Callback: {data}")
    try:
        await _dispatch_button(update, context, q, data, uid, ud)
    except (ValueError, IndexError, KeyError, TypeError) as e:
        logger.warning(f"Bad callback {data!r}: {e}")
        await safe_alert(q, "⚠️ Invalid or expired button.", show_alert=True)

async def _dispatch_button(update, context, q, data, uid, ud):
    if data == "ea_menu":
        await q.edit_message_text("🤖 *EA CONTROL CENTER*", parse_mode="Markdown",
                                   reply_markup=ea_main_keyboard())
        return
    if data == "ea_signals":
        await q.edit_message_text("🎯 *SIGNALS*\n\nChoose symbol:", parse_mode="Markdown",
                                   reply_markup=ea_symbol_keyboard()); return
    if data.startswith("ea_symbol_"):
        symbol = data[len("ea_symbol_"):]
        await q.edit_message_text(f"✅ *{symbol}*\n\nChoose preset:", parse_mode="Markdown",
                                   reply_markup=ea_preset_keyboard(symbol)); return
    if data.startswith("ea_preset_"):
        parts = data.split("_")
        symbol, preset_key = parts[2], parts[3]
        await q.edit_message_text(f"⏳ Analyzing *{symbol}* ({EA_MTF_PRESETS[preset_key]['label']})...",
                                   parse_mode="Markdown")
        snap = ea_snapshot()
        result = await ea_generate_signal(symbol, preset_key, ud.get("ea_risk", {}), snap=snap)
        if not result.get("ok"):
            await q.message.reply_text(
                f"❌ Could not fetch data for {symbol}\n"
                f"(No data: {', '.join(result.get('failed', []))})\n\n"
                f"💡 ត្រូវការ API Key — សាកល្បង /providers ដើម្បីមើលអ្វីដែល Setup រួច។")
            return
        ud["ea_last_signal"] = {"symbol": symbol, "preset": preset_key,
                                 "direction": result["direction"], "confidence": result["confidence"]}
        fastest = EA_MTF_PRESETS[preset_key]["timeframes"][0]
        plan = None
        if result["direction"] in ("BUY", "SELL"):
            plan = await ea_build_execution_plan(symbol, preset_key, result["direction"],
                                                   ud.get("ea_risk", {}), snap=snap)
        ud["ea_last_signal_id"] = ea_record_signal(ud, symbol, preset_key, result,
                                                     plan=plan, timeframe=fastest)
        save_data()
        if result["direction"] in ("BUY", "SELL"):
            risk_ok, risk_reasons = ea_risk_check(ud, snap, result)
        else:
            risk_ok, risk_reasons = None, []
        card = ea_format_signal_card(symbol, EA_MTF_PRESETS[preset_key]["label"],
                                       result, plan, risk_ok, risk_reasons,
                                       ea_accuracy_for_card(ud, symbol, fastest))
        await ea_show_card(q, card, ea_signal_card_keyboard(symbol, preset_key, fastest, result["direction"]))
        return
    if data == "ea_analysis":
        await q.edit_message_text("🧠 *ANALYSIS*\n\nChoose symbol:", parse_mode="Markdown",
                                   reply_markup=ea_analysis_symbol_keyboard()); return
    if data.startswith("ea_analysis_symbol_"):
        symbol = data[len("ea_analysis_symbol_"):]
        await q.edit_message_text(f"✅ *{symbol}*\n\nChoose preset:", parse_mode="Markdown",
                                   reply_markup=ea_analysis_preset_keyboard(symbol)); return
    if data.startswith("ea_analysis_preset_"):
        parts = data.split("_")
        symbol, preset_key = parts[3], parts[4]
        await q.edit_message_text(f"⏳ Analyzing *{symbol}*...", parse_mode="Markdown")
        snap = ea_snapshot()
        result = await ea_generate_signal(symbol, preset_key, ud.get("ea_risk", {}), snap=snap)
        if not result.get("ok"):
            await q.message.reply_text(
                f"❌ Could not fetch data. Try /providers to check API keys.")
            return
        fastest = EA_MTF_PRESETS[preset_key]["timeframes"][0]
        plan = None
        if result["direction"] in ("BUY", "SELL"):
            plan = await ea_build_execution_plan(symbol, preset_key, result["direction"],
                                                   ud.get("ea_risk", {}), snap=snap)
        card = ea_format_signal_card(symbol, EA_MTF_PRESETS[preset_key]["label"],
                                       result, plan, None, None,
                                       ea_accuracy_for_card(ud, symbol, fastest))
        await ea_show_card(q, card,
                            ea_signal_card_keyboard(symbol, preset_key, fastest,
                                                     result["direction"], back="ea_analysis"))
        return
    if data.startswith("ea_sigdetail_"):
        parts = data.split("_")
        symbol, preset_key = parts[2], parts[3]
        snap = ea_snapshot()
        result = await ea_generate_signal(symbol, preset_key, ud.get("ea_risk", {}), snap=snap)
        if not result.get("ok"):
            await safe_alert(q, "Could not load details."); return
        await q.message.reply_text(ea_format_signal_text(result), parse_mode="Markdown")
        return
    if data.startswith("ea_execute_"):
        parts = data.split("_")
        symbol, preset_key, direction = parts[2], parts[3], parts[4]
        snap = ea_snapshot()
        result = await ea_generate_signal(symbol, preset_key, ud.get("ea_risk", {}), snap=snap)
        if not result.get("ok"):
            await safe_alert(q, "❌ Could not re-fetch data."); return
        ok, reasons = ea_risk_check(ud, snap, result)
        if not ok:
            await q.message.reply_text("🚫 *NO TRADE — Risk blocked:*\n\n" +
                                        "\n".join(f"• {r}" for r in reasons),
                                        parse_mode="Markdown")
            return
        plan = await ea_build_execution_plan(symbol, preset_key, direction,
                                              ud.get("ea_risk", {}), snap=snap)
        if not plan:
            await safe_alert(q, "❌ Could not compute SL/TP."); return
        fastest = EA_MTF_PRESETS[preset_key]["timeframes"][0]
        signal_id = ea_record_signal(ud, symbol, preset_key, result, plan=plan, timeframe=fastest)
        ud["ea_pending_execution"] = plan
        ud["ea_pending_execution_ts"] = time.time()
        ud["ea_pending_result"] = {"preset": preset_key, "confidence": result["confidence"],
                                     "components": result.get("components"),
                                     "news": result.get("news"), "signal_id": signal_id}
        save_data()
        await q.message.reply_text(
            ea_format_execution_confirm(plan, snap), parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Confirm & Send", callback_data="ea_xgo")],
                [InlineKeyboardButton("🔙 Cancel", callback_data="ea_menu")]]))
        return
    if data == "ea_xgo":
        plan = ud.get("ea_pending_execution")
        ts = ud.get("ea_pending_execution_ts")
        pending = ud.get("ea_pending_result") or {}
        if not plan:
            await safe_alert(q, "Already sent, or expired."); return
        ud["ea_pending_execution"] = None
        ud["ea_pending_execution_ts"] = None
        ud["ea_pending_result"] = None
        save_data()
        if ts is None or (time.time() - ts) > EA_ACTION_CONFIRM_TTL_SECONDS:
            await q.edit_message_text("⌛ Confirmation too old.", reply_markup=ea_main_keyboard())
            return
        snap = ea_snapshot()
        if not snap["connected"] or snap["run_state"] != "RUNNING":
            await safe_alert(q, "🔴 EA not online/running."); return
        cmd = ea_enqueue_command("EXECUTE", plan)
        ea_add_event({"type": "EXECUTE_REQUESTED",
                       "detail": f"{plan['direction']} {plan['symbol']} (cmd {cmd['id']})"})
        save_data()
        await q.edit_message_text(
            f"📤 *Sent to EA* — {plan['direction']} {plan['symbol']} (cmd `{cmd['id']}`)",
            parse_mode="Markdown", reply_markup=ea_main_keyboard())
        return
    if data == "ea_livechart":
        await q.edit_message_text("📡 *LIVE CHART*\n\nChoose symbol:", parse_mode="Markdown",
                                   reply_markup=ea_livechart_symbol_keyboard()); return
    if data.startswith("ea_livechart_symbol_"):
        symbol = data[len("ea_livechart_symbol_"):]
        await q.edit_message_text(f"✅ *{symbol}*\n\nChoose timeframe:", parse_mode="Markdown",
                                   reply_markup=ea_livechart_tf_keyboard(symbol)); return
    if data.startswith("ea_livechart_tf_"):
        parts = data.split("_")
        symbol, tf = parts[3], parts[4]
        await q.edit_message_text(f"⏳ Loading {symbol} {tf}...", parse_mode="Markdown")
        dash, tried = await ea_build_live_chart(symbol, tf, ud.get("ea_risk", {}))
        if dash is None:
            await q.message.reply_text(
                f"❌ Could not fetch data for {symbol} {tf}.\n"
                f"Tried: {', '.join(tried) if tried else 'N/A'}\n"
                f"💡 Check /providers for API configuration.",
                reply_markup=ea_livechart_result_keyboard(symbol, tf))
            return
        if dash.get("chart_bytes"):
            try: await q.message.reply_photo(photo=dash["chart_bytes"], caption=f"📡 {symbol} ({tf})")
            except Exception: pass
        await q.message.reply_text(ea_format_live_chart(dash), parse_mode="Markdown",
                                     reply_markup=ea_livechart_result_keyboard(symbol, tf))
        return
    if data == "ea_status":
        snap = ea_snapshot()
        if snap["connected"]:
            mode = "🟡 DEMO" if (snap.get("mode") or "").upper() == "DEMO" else \
                    ("🔴 LIVE" if (snap.get("mode") or "").upper() == "LIVE" else "⚪ UNKNOWN")
            age = int(time.time() - snap["last_heartbeat"])
            text = (f"📡 *EA STATUS*\n\n🟢 ONLINE — {age}s\nMode: {mode}\n"
                    f"Run: *{snap['run_state']}*\nBroker: {snap.get('broker') or 'N/A'}")
        else:
            text = "📡 *EA STATUS*\n\n🔴 *EA OFFLINE* — no heartbeat."
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=ea_main_keyboard()); return
    if data == "ea_dashboard":
        snap = ea_snapshot()
        if not snap["connected"]:
            text = "📊 *LIVE DASHBOARD*\n\n🔴 EA OFFLINE"
        else:
            acct = snap.get("account") or {}
            text = (f"📊 *LIVE DASHBOARD*\n\n"
                    f"Mode: {'DEMO' if (snap.get('mode') or '').upper() == 'DEMO' else 'LIVE'}\n"
                    f"Run: {snap['run_state']}\n"
                    f"Balance: {acct.get('balance', 'N/A')}\n"
                    f"Equity: {acct.get('equity', 'N/A')}\n"
                    f"Open positions: {len(snap.get('positions', []))}")
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=ea_main_keyboard()); return
    if data == "ea_account":
        snap = ea_snapshot()
        if not snap["connected"]:
            await q.edit_message_text("💼 *ACCOUNT*\n\n🔴 EA OFFLINE",
                                       parse_mode="Markdown", reply_markup=ea_main_keyboard()); return
        acct = snap.get("account") or {}
        lines = ["💼 *ACCOUNT*", ""]
        for k in ("balance", "equity", "margin", "free_margin", "currency", "leverage"):
            if k in acct: lines.append(f"{k.replace('_', ' ').title()}: {acct[k]}")
        await q.edit_message_text("\n".join(lines), parse_mode="Markdown",
                                   reply_markup=ea_main_keyboard()); return
    if data == "ea_positions":
        snap = ea_snapshot()
        positions = snap.get("positions") or []
        if not positions:
            text = "📈 *POSITIONS*\n\nNo open positions."
        else:
            lines = ["📈 *OPEN POSITIONS*", ""]
            for i, p in enumerate(positions):
                lines.append(f"{i+1}. {p.get('symbol')} {p.get('direction')} {p.get('lot')} — "
                              f"Entry {p.get('entry')} → {p.get('current')} | P/L {p.get('pl')}")
            text = "\n".join(lines)
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=ea_main_keyboard()); return
    if data == "ea_history":
        snap = ea_snapshot()
        events = [e for e in (snap.get("events") or [])
                  if e.get("type") in ("TRADE_OPENED", "TRADE_CLOSED")][:20]
        if not events:
            text = "📜 *HISTORY*\n\nNo trade events yet."
        else:
            lines = ["📜 *HISTORY*", ""]
            for e in events:
                when = datetime.fromtimestamp(e["at"]).strftime("%Y-%m-%d %H:%M")
                lines.append(f"• [{when}] {e['type']}: {e.get('detail', '')}")
            text = "\n".join(lines)
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=ea_main_keyboard()); return
    if data == "ea_start":
        snap = ea_snapshot()
        if not snap["connected"]:
            await safe_alert(q, "🔴 No EA connected."); return
        ea_enqueue_command("START")
        await safe_alert(q, "▶️ Start requested."); return
    if data == "ea_pause":
        snap = ea_snapshot()
        if not snap["connected"]:
            await safe_alert(q, "🔴 No EA connected."); return
        ea_enqueue_command("PAUSE")
        await safe_alert(q, "⏸️ Pause requested."); return
    if data == "ea_estop_confirm":
        await q.edit_message_text(
            "🛑 *EMERGENCY STOP*\n\nStops new entries. Does NOT close existing.\n\nConfirm?",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Yes", callback_data="ea_estop_go")],
                [InlineKeyboardButton("🔙 Cancel", callback_data="ea_menu")]]))
        return
    if data == "ea_estop_go":
        snap = ea_snapshot()
        if not snap["connected"]:
            await safe_alert(q, "🔴 EA offline."); return
        ea_enqueue_command("EMERGENCY_STOP")
        ea_add_event({"type": "EMERGENCY_STOP_REQUESTED"})
        await q.edit_message_text("🛑 Emergency Stop sent.", parse_mode="Markdown",
                                   reply_markup=ea_main_keyboard()); return
    if data == "ea_settings":
        await q.edit_message_text("⚙️ *EA SETTINGS*", parse_mode="Markdown",
                                   reply_markup=ea_settings_keyboard(ud)); return
    if data.startswith("ea_setdefault_"):
        key = data[len("ea_setdefault_"):]
        if key in EA_MTF_PRESETS: ud["ea_default_preset"] = key; save_data()
        await q.edit_message_text("⚙️ *EA SETTINGS*", parse_mode="Markdown",
                                   reply_markup=ea_settings_keyboard(ud)); return
    # Cycle settings
    cycles = {
        "ea_cycle_risk_pct": ("risk_per_trade_pct", [0.5, 1.0, 1.5, 2.0, 3.0]),
        "ea_cycle_min_conf": ("min_confidence", [50, 60, 70, 80, 90]),
        "ea_cycle_max_trades": ("max_open_trades", [1, 2, 3, 5, 10]),
        "ea_cycle_be_trigger": ("breakeven_trigger_rr", [0.5, 1.0, 1.5, 2.0]),
        "ea_cycle_trailing_mult": ("trailing_mult", [1.0, 1.5, 2.0, 2.5]),
        "ea_cycle_sl_mode": ("sl_mode", ["atr", "structure", "fixed"]),
        "ea_cycle_fixed_sl_pct": ("fixed_sl_pct", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
        "ea_cycle_partial_tp_rr": ("partial_tp_trigger_rr", [0.5, 1.0, 1.5, 2.0]),
        "ea_cycle_partial_tp_pct": ("partial_tp_close_pct", [25, 33, 50, 66, 75]),
        "ea_cycle_news_window": ("news_pre_minutes", [15, 30, 45, 60]),
        "ea_cycle_max_dd": ("max_drawdown_pct", [10.0, 15.0, 20.0, 25.0, 30.0]),
        "ea_cycle_max_losses": ("max_consecutive_losses", [2, 3, 4, 5]),
        "ea_cycle_max_spread": ("max_spread_pips", [None, 5, 10, 15, 20]),
        "ea_cycle_max_exposure": ("max_total_exposure_lots", [None, 1.0, 2.0, 5.0, 10.0]),
        "ea_cycle_sessions": ("allowed_sessions", [[], ["Asian"], ["London"], ["New York"],
                                                     ["Asian", "London"], ["London", "New York"]]),
        "ea_cycle_cooldown": ("cooldown_after_loss_minutes", [0, 5, 10, 15, 30, 60]),
        "ea_cycle_blocked_regimes": ("blocked_regimes",
                                       [["UNSTABLE"], ["UNSTABLE", "HIGH_VOLATILITY"],
                                        ["UNSTABLE", "HIGH_VOLATILITY", "RANGING"], []]),
    }
    if data in cycles:
        key, options = cycles[data]
        cur = ud["ea_risk"].get(key)
        if cur in options:
            nxt = options[(options.index(cur) + 1) % len(options)]
        else:
            nxt = options[0]
        ud["ea_risk"][key] = nxt
        save_data()
        await q.edit_message_text("⚙️ *EA SETTINGS*", parse_mode="Markdown",
                                   reply_markup=ea_settings_keyboard(ud)); return
    toggles = {
        "ea_toggle_be": ("breakeven_enabled", True),
        "ea_toggle_trailing": ("trailing_enabled", False),
        "ea_toggle_partial_tp": ("partial_tp_enabled", False),
        "ea_toggle_news": ("news_filter_enabled", True),
        "ea_toggle_abnormal": ("abnormal_market_filter_enabled", True),
        "ea_toggle_regime": ("regime_filter_enabled", True),
        "ea_toggle_chart_vision": ("chart_vision_enabled", False),
    }
    if data in toggles:
        key, default = toggles[data]
        ud["ea_risk"][key] = not ud["ea_risk"].get(key, default)
        save_data()
        await q.edit_message_text("⚙️ *EA SETTINGS*", parse_mode="Markdown",
                                   reply_markup=ea_settings_keyboard(ud)); return
    if data == "ea_toggle_live":
        await q.edit_message_text(
            "⚠️ *ENABLE LIVE TRADING?*\n\nReal money will be used.\n\nConfirm?",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Yes, enable LIVE", callback_data="ea_confirm_live_on")],
                [InlineKeyboardButton("🔙 Cancel", callback_data="ea_settings")]]))
        return
    if data == "ea_confirm_live_on":
        ud["ea_risk"]["live_trading_enabled"] = True; save_data()
        ea_add_event({"type": "LIVE_TRADING_ENABLED"})
        await q.edit_message_text("🔴 LIVE enabled.", parse_mode="Markdown",
                                   reply_markup=ea_settings_keyboard(ud)); return
    if data == "ea_disable_live":
        ud["ea_risk"]["live_trading_enabled"] = False; save_data()
        ea_add_event({"type": "LIVE_TRADING_DISABLED"})
        await q.edit_message_text("🟢 LIVE disabled.", parse_mode="Markdown",
                                   reply_markup=ea_settings_keyboard(ud)); return
    if data == "ea_notifications":
        active = ud.get("ea_notify_types", [])
        cats = [("TRADE_OPENED", "📈 Trade Opened"), ("TRADE_CLOSED", "📉 Trade Closed"),
                ("ERROR", "🚨 Errors"), ("SL_MODIFIED", "🎯 SL Modified"),
                ("EA_STARTED", "▶️ Started"), ("EA_PAUSED", "⏸️ Paused"),
                ("EMERGENCY_STOP_ACK", "🛑 E-Stop Ack")]
        kb = [[InlineKeyboardButton(f"{'✅' if k in active else '⬜'} {lb}",
                                     callback_data=f"ea_notify_toggle_{k}")] for k, lb in cats]
        kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_menu")])
        await q.edit_message_text("🔔 *NOTIFICATIONS*", parse_mode="Markdown",
                                   reply_markup=InlineKeyboardMarkup(kb)); return
    if data.startswith("ea_notify_toggle_"):
        key = data[len("ea_notify_toggle_"):]
        types = ud.get("ea_notify_types", [])
        if key in types: types.remove(key)
        else: types.append(key)
        ud["ea_notify_types"] = types; save_data()
        active = types
        cats = [("TRADE_OPENED", "📈 Trade Opened"), ("TRADE_CLOSED", "📉 Trade Closed"),
                ("ERROR", "🚨 Errors"), ("SL_MODIFIED", "🎯 SL Modified"),
                ("EA_STARTED", "▶️ Started"), ("EA_PAUSED", "⏸️ Paused"),
                ("EMERGENCY_STOP_ACK", "🛑 E-Stop Ack")]
        kb = [[InlineKeyboardButton(f"{'✅' if k in active else '⬜'} {lb}",
                                     callback_data=f"ea_notify_toggle_{k}")] for k, lb in cats]
        kb.append([InlineKeyboardButton("🔙 Back", callback_data="ea_menu")])
        await q.edit_message_text("🔔 *NOTIFICATIONS*", parse_mode="Markdown",
                                   reply_markup=InlineKeyboardMarkup(kb)); return
    # Backtest
    if data == "ea_backtest_menu":
        await q.edit_message_text("🧪 *BACKTESTING*", parse_mode="Markdown",
                                   reply_markup=ea_backtest_mode_keyboard()); return
    if data == "ea_bt_symbol_menu":
        await q.edit_message_text("🧪 *STANDARD BACKTEST*\n\nChoose symbol:", parse_mode="Markdown",
                                   reply_markup=ea_backtest_symbol_keyboard("ea_bt_symbol_")); return
    if data.startswith("ea_bt_symbol_"):
        symbol = data[len("ea_bt_symbol_"):]
        await q.edit_message_text(f"✅ *{symbol}*\n\nChoose timeframe:", parse_mode="Markdown",
                                   reply_markup=ea_backtest_tf_keyboard(symbol, "ea_bt_tf_")); return
    if data.startswith("ea_bt_tf_"):
        parts = data.split("_")
        symbol, tf = parts[3], parts[4]
        await q.edit_message_text(f"⏳ Running backtest — {symbol} {tf}...", parse_mode="Markdown")
        result = await ea_run_backtest(symbol, tf, ud.get("ea_risk", {}))
        if not result.get("ok"):
            await q.message.reply_text(
                f"❌ Could not run backtest for {symbol} {tf}.\n"
                f"Reason: {result.get('reason', 'unknown')}\n"
                f"💡 Check /providers for API keys.",
                reply_markup=ea_main_keyboard())
            return
        await q.message.reply_text(ea_format_backtest(result), parse_mode="Markdown",
                                     reply_markup=ea_main_keyboard()); return
    if data == "ea_wf_symbol_menu":
        await q.edit_message_text("🔬 *WALK-FORWARD* (coming soon)", parse_mode="Markdown",
                                   reply_markup=ea_backtest_mode_keyboard()); return
    if data == "ea_ab_symbol_menu":
        await q.edit_message_text("🆚 *A/B TEST* (coming soon)", parse_mode="Markdown",
                                   reply_markup=ea_backtest_mode_keyboard()); return
    # Performance
    if data == "ea_performance":
        if await ea_evaluate_pending_signals(ud): save_data()
        filters = ud.get("ea_perf_filters") or {"period": "7D", "symbol": "ALL",
                                                  "timeframe": "ALL", "direction": "ALL", "preset": "ALL"}
        signals = ea_perf_filtered_signals(ud, filters)
        stats = ea_compute_strategy_performance(signals)
        await q.edit_message_text(ea_format_strategy_performance(stats, filters),
                                   parse_mode="Markdown",
                                   reply_markup=ea_strategy_performance_keyboard(filters)); return
    if data.startswith("ea_sp_period_"):
        p = data[len("ea_sp_period_"):]
        if p in EA_PERF_PERIODS:
            ud.setdefault("ea_perf_filters", {})["period"] = p; save_data()
        filters = ud["ea_perf_filters"]
        signals = ea_perf_filtered_signals(ud, filters)
        stats = ea_compute_strategy_performance(signals)
        await q.edit_message_text(ea_format_strategy_performance(stats, filters),
                                   parse_mode="Markdown",
                                   reply_markup=ea_strategy_performance_keyboard(filters)); return
    if data.startswith("ea_sp_dir_"):
        d = data[len("ea_sp_dir_"):]
        if d in ("ALL", "BUY", "SELL"):
            ud.setdefault("ea_perf_filters", {})["direction"] = d; save_data()
        filters = ud["ea_perf_filters"]
        signals = ea_perf_filtered_signals(ud, filters)
        stats = ea_compute_strategy_performance(signals)
        await q.edit_message_text(ea_format_strategy_performance(stats, filters),
                                   parse_mode="Markdown",
                                   reply_markup=ea_strategy_performance_keyboard(filters)); return
    if data == "ea_sp_reset":
        ud["ea_perf_filters"] = {"period": "7D", "symbol": "ALL", "timeframe": "ALL",
                                   "direction": "ALL", "preset": "ALL"}
        save_data()
        filters = ud["ea_perf_filters"]
        signals = ea_perf_filtered_signals(ud, filters)
        stats = ea_compute_strategy_performance(signals)
        await q.edit_message_text(ea_format_strategy_performance(stats, filters),
                                   parse_mode="Markdown",
                                   reply_markup=ea_strategy_performance_keyboard(filters)); return
    if data in ("ea_sp_bysymbol", "ea_sp_bytf", "ea_sp_bydir", "ea_sp_bypreset"):
        if await ea_evaluate_pending_signals(ud): save_data()
        filters = ud.get("ea_perf_filters") or {}
        signals = ea_perf_filtered_signals(ud, filters)
        keys = {"ea_sp_bysymbol": (lambda s: s.get("symbol"), "Symbol"),
                "ea_sp_bytf": (lambda s: s.get("timeframe") or "—", "Timeframe"),
                "ea_sp_bydir": (lambda s: s.get("direction"), "Direction"),
                "ea_sp_bypreset": (lambda s: EA_MTF_PRESETS.get(s.get("preset"), {}).get("label", s.get("preset")), "Preset")}
        kf, lb = keys[data]
        groups = ea_group_strategy_performance(signals, kf)
        lines = [f"📊 *By {lb}*", ""]
        rows = sorted([(k, s) for k, s in groups.items() if s["total_signals"]],
                       key=lambda kv: -(kv[1]["sample_size"] or 0))
        for k, s in rows[:10]:
            if not s["evaluated"]:
                lines.append(f"*{k}*: {s['total_signals']} signals (no eval yet)")
            else:
                lines.append(f"*{k}*: {s['total_signals']} signals, {s['win_rate']}% win, "
                              f"P/L {s['total_pl_r']:+.2f}R, sample {s['sample_size']}")
        if not rows: lines.append("📭 No signal history.")
        await q.edit_message_text("\n".join(lines), parse_mode="Markdown",
                                   reply_markup=ea_strategy_performance_keyboard(filters)); return
    if data == "ea_sp_history":
        if await ea_evaluate_pending_signals(ud): save_data()
        filters = ud.get("ea_perf_filters") or {}
        signals = ea_perf_filtered_signals(ud, filters)
        lines = ["📜 *SIGNAL HISTORY*", ""]
        for s in signals[:15]:
            ts = datetime.fromtimestamp(s["at"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            ic = {"WIN": "🟢", "LOSS": "🔴", "UNRESOLVED": "⚪"}.get(s.get("outcome"), "⏳")
            rl = (f"{ic} {s['outcome']} ({s.get('r_multiple', 0):+.2f}R)"
                  if s.get("status") == "RESOLVED" else
                  "⏳ PENDING" if s.get("status") == "PENDING" else
                  "— (WAIT/NT)" if s.get("status") == "N/A" else "⚪ EXPIRED")
            lines.append(f"{ts} — {s['symbol']} {s.get('timeframe') or ''}")
            lines.append(f"{s['direction']} · Conf {s['confidence']} · {rl}")
            lines.append("")
        if not signals: lines.append("📭 No signal history.")
        await q.edit_message_text("\n".join(lines), parse_mode="Markdown",
                                   reply_markup=ea_strategy_performance_keyboard(filters)); return
    await safe_alert(q, "Unknown action.")

# ============================================================
# MAIN
# ============================================================

def main():
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN មិនទាន់កំណត់ក្នុង Environment Variables ទេ។")
        sys.exit(1)
    if not ADMIN_IDS:
        logger.warning("⚠️ ADMIN_ID / ADMIN_IDS មិនទាន់កំណត់ — គ្មានអ្នកណាអាចប្រើ Bot បានទេ។")
    load_data()
    threading.Thread(target=start_health_server, daemon=True).start()
    threading.Thread(target=ea_heartbeat_watcher, daemon=True).start()

    request = HTTPXRequest(connect_timeout=30.0, read_timeout=60.0, write_timeout=60.0,
                            pool_timeout=30.0)
    app = Application.builder().token(BOT_TOKEN).request(request).build()

    commands = {
        "start": start_cmd,
        "ea": ea_cmd,
        "signal": ea_signal_cmd,
        "analyze": ea_analyze_cmd,
        "ea_status": ea_status_cmd,
        "ea_settings": ea_settings_cmd,
        "ea_backtest": ea_backtest_cmd,
        "ea_performance": ea_performance_cmd,
        "accuracy": accuracy_cmd,
        "aistatus": aistatus_cmd,
        "providers": providers_cmd,
        "help": help_cmd,
        "cancel": cancel_cmd,
    }
    for cmd, handler in commands.items():
        app.add_handler(CommandHandler(cmd, handler))
    app.add_handler(CallbackQueryHandler(button_handler))
    logger.info("Starting TradeMind bot...")
    logger.info(f"Admin IDs: {ADMIN_IDS or 'NONE SET'}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
