#!/usr/bin/env python3
"""每日收盘信号（D2V策略）：增量更新6个标的数据 -> 计算信号 -> 输出JSON/日志/记录。

D2V规则（2026-08-06定稿）：
  进场 = close>MA3 且 MA3>MA30 且 个股RSI<80 且 +DI>-DI
         且 医药ETF>MA30 且 医药ETF RSI<70 且 创业板>MA20
         且 换手率∈[1%,10%]
  仓位 = min(50%/20日年化波动, 100%)（百分比仓位，非全仓）
  成交 = 次日开盘，大跳空≥4%等回撤（脚本只报信号与目标仓位，成交由实盘执行）

时间因果：全部指标只用截至当日收盘的数据；持仓信号 shift(1) 次日生效。
数据源：东财优先，失败回退新浪；某标的当天取不到则沿用旧数据并标记门控关闭(保守)。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import akshare as ak

# 绕过macOS系统代理（系统代理常不可用）
_session = requests.Session()
_session.trust_env = False
requests.get = _session.get

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

INSTRUMENTS = [
    {"name": "贝瑞基因", "symbol": "sz000710", "file": "sz000710_daily_qfq.csv", "kind": "stock"},
    {"name": "上证指数", "symbol": "sh000001", "file": "sh000001_daily.csv", "kind": "index"},
    {"name": "深证成指", "symbol": "sz399001", "file": "sz399001_daily.csv", "kind": "index"},
    {"name": "创业板指", "symbol": "sz399006", "file": "sz399006_daily.csv", "kind": "index"},
    {"name": "医药ETF", "symbol": "sh512010", "file": "sh512010_daily.csv", "kind": "etf"},
    {"name": "医疗ETF", "symbol": "sh512170", "file": "sh512170_daily.csv", "kind": "etf"},
]

RENAME = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
          "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"}
KEEP = ["date", "open", "high", "low", "close", "volume", "amount", "turnover", "outstanding_share"]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, file=sys.stderr)
    try:
        with open(os.path.join(DATA_DIR, "daily_signal.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _try(fn, label, retries=3, backoff=2.0):
    last = None
    for i in range(retries):
        try:
            df = fn()
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            last = e
        if i < retries - 1:
            time.sleep(backoff * (i + 1))
    log(f"[warn] {label} 连续{retries}次失败: {last}")
    return None


def fetch_em(instr, start, end):
    code = instr["symbol"][-6:]
    if instr["kind"] == "index":
        return ak.index_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end)
    return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")


def fetch_sina(instr, start, end):
    sym = instr["symbol"]
    if instr["kind"] == "index":
        return ak.stock_zh_index_daily(symbol=sym)
    if instr["kind"] == "etf":
        return ak.fund_etf_hist_sina(symbol=sym)
    return ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="qfq")


def norm(df):
    df = df.rename(columns=RENAME)
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date", keep="last")


def update_instrument(instr):
    path = os.path.join(DATA_DIR, instr["file"])
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["date"])
        start = (old["date"].max() - timedelta(days=15)).strftime("%Y%m%d")
    else:
        old = None
        start = "20150101"
    end = datetime.now().strftime("%Y%m%d")
    df = _try(lambda: fetch_em(instr, start, end), f"{instr['name']} 东财")
    src = "东财"
    if df is None:
        df = _try(lambda: fetch_sina(instr, start, end), f"{instr['name']} 新浪")
        src = "新浪"
    if df is None:
        log(f"[warn] {instr['name']} 本次未更新，沿用旧数据")
        return old is not None
    df = norm(df)
    if old is not None:
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates("date", keep="last")
    else:
        merged = df
    # 保持既有列顺序（个股含 outstanding_share 等）
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[ok] {instr['name']} 更新自{src}源，共{len(merged)}行，最新 {merged['date'].max().date()}")
    return True


def rsi14(close):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def dmi(dfx):
    high, low, close = dfx["high"], dfx["low"], dfx["close"]
    up = high.diff(); dn = -low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=dfx.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=dfx.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    return pdi, mdi


def load(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def compute_signal():
    stock = load(os.path.join(DATA_DIR, "sz000710_daily_qfq.csv"))
    med_a = load(os.path.join(DATA_DIR, "sh512010_daily.csv"))
    med_b = load(os.path.join(DATA_DIR, "sh512170_daily.csv"))
    cyb = load(os.path.join(DATA_DIR, "sz399006_daily.csv"))
    sh = load(os.path.join(DATA_DIR, "sh000001_daily.csv"))

    close = stock["close"]
    ma3 = close.rolling(3).mean(); ma30 = close.rolling(30).mean()
    rsi = rsi14(close)
    pdi, mdi = dmi(stock)
    to = stock["turnover"].ffill()
    vwap = stock["amount"] / stock["volume"].replace(0, np.nan)
    vr = stock["volume"] / stock["volume"].rolling(20).mean()
    rv20 = close.pct_change().rolling(20).std(ddof=0) * (252 ** 0.5)

    ma_a30 = med_a["close"].rolling(30).mean()
    ma_a20 = med_a["close"].rolling(20).mean()
    rsi_a = rsi14(med_a["close"])
    ma_b20 = med_b["close"].rolling(20).mean()
    ma_cyb20 = cyb["close"].rolling(20).mean()
    ma_sh20 = sh["close"].rolling(20).mean()

    last = len(stock) - 1
    gates = {
        "个股_close>MA3": bool(close.iloc[last] > ma3.iloc[last]),
        "个股_MA3>MA30": bool(ma3.iloc[last] > ma30.iloc[last]),
        "个股_RSI<80": bool(rsi.iloc[last] < 80),
        "个股_+DI>-DI": bool(pdi.iloc[last] > mdi.iloc[last]),
        "医药ETF>MA30": bool(med_a["close"].iloc[-1] > ma_a30.iloc[-1]),
        "医药ETF_RSI<70": bool(rsi_a.iloc[-1] < 70),
        "创业板>MA20": bool(cyb["close"].iloc[-1] > ma_cyb20.iloc[-1]),
        "换手率1%-10%": bool(0.01 <= to.iloc[last] <= 0.10),
    }
    sig_today = all(gates.values())
    scale_today = min(0.5 / rv20.iloc[last], 1.0) if np.isfinite(rv20.iloc[last]) else 0.0
    target_tomorrow = sig_today * scale_today

    # 昨日信号与今日应持仓（近似：按昨日信号×昨日scale）
    sig_y = (bool(close.iloc[last - 1] > ma3.iloc[last - 1]) and
             bool(ma3.iloc[last - 1] > ma30.iloc[last - 1]) and
             bool(rsi.iloc[last - 1] < 80) and bool(pdi.iloc[last - 1] > mdi.iloc[last - 1]) and
             bool(med_a["close"].iloc[-2] > ma_a30.iloc[-2]) and bool(rsi_a.iloc[-2] < 70) and
             bool(cyb["close"].iloc[-2] > ma_cyb20.iloc[-2]) and
             bool(0.01 <= to.iloc[last - 1] <= 0.10))
    scale_y = min(0.5 / rv20.iloc[last - 1], 1.0) if np.isfinite(rv20.iloc[last - 1]) else 0.0
    pos_today = sig_y * scale_y

    if sig_today:
        action = "买入/加仓至目标仓位" if pos_today < target_tomorrow - 1e-9 else "持有"
    elif pos_today > 1e-6:
        action = "卖出/清仓(明日开盘执行，大低开≥4%等回撤)"
    else:
        action = "空仓等待"

    out = {
        "as_of": str(stock["date"].iloc[last].date()),
        "strategy": "D2V",
        "last_close": round(float(close.iloc[last]), 3),
        "gates": {k: bool(v) for k, v in gates.items()},
        "signal_today": bool(sig_today),
        "target_position_tomorrow_pct": round(float(target_tomorrow) * 100, 2),
        "position_today_pct": round(float(pos_today) * 100, 2),
        "action": action,
        "indicators": {
            "ma3": round(float(ma3.iloc[last]), 3), "ma30": round(float(ma30.iloc[last]), 3),
            "rsi": round(float(rsi.iloc[last]), 1), "pdi": round(float(pdi.iloc[last]), 1),
            "mdi": round(float(mdi.iloc[last]), 1),
            "turnover_pct": round(float(to.iloc[last]) * 100, 2),
            "volume_ratio": round(float(vr.iloc[last]), 2),
            "close_vs_vwap_pct": round((float(close.iloc[last]) / float(vwap.iloc[last]) - 1) * 100, 2),
            "vol20_annual": round(float(rv20.iloc[last]), 3),
        },
        "sector": {
            "医药ETF_close": float(med_a["close"].iloc[-1]),
            "医药ETF_MA30": round(float(ma_a30.iloc[-1]), 3),
            "医药ETF_RSI": round(float(rsi_a.iloc[-1]), 1),
            "医疗ETF_close": float(med_b["close"].iloc[-1]),
            "医疗ETF_MA20": round(float(ma_b20.iloc[-1]), 3),
            "创业板close": float(cyb["close"].iloc[-1]),
            "创业板MA20": round(float(ma_cyb20.iloc[-1]), 3),
            "上证close": float(sh["close"].iloc[-1]),
            "上证MA20": round(float(ma_sh20.iloc[-1]), 3),
        },
        "data_latest": {
            "贝瑞基因": str(stock["date"].iloc[last].date()),
            "医药ETF": str(med_a["date"].iloc[-1].date()),
            "医疗ETF": str(med_b["date"].iloc[-1].date()),
            "创业板": str(cyb["date"].iloc[-1].date()),
            "上证": str(sh["date"].iloc[-1].date()),
        },
        "note": "创业板/ETF等数据源缺最新日时门控按关闭处理(保守)；仓位为百分比目标，实盘按次日开盘成交、大跳空≥4%等回撤。历史表现不代表未来。",
    }
    return out


def append_log(out):
    path = os.path.join(DATA_DIR, "signal_log.csv")
    row = {
        "date": out["as_of"], "close": out["last_close"], "signal": int(out["signal_today"]),
        "target_pos_pct": out["target_position_tomorrow_pct"],
        "pos_today_pct": out["position_today_pct"], "action": out["action"],
    }
    cols = ["date", "close", "signal", "target_pos_pct", "pos_today_pct", "action"]
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["date"] != row["date"]]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([row])[cols].to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="D2V每日收盘信号：增量更新数据+计算信号")
    ap.add_argument("--no-update", action="store_true", help="跳过数据更新，仅重算信号")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    if not args.no_update:
        ok = [update_instrument(i) for i in INSTRUMENTS]
        log(f"数据更新完成: {sum(ok)}/{len(ok)} 成功")
    try:
        out = compute_signal()
    except Exception as e:
        log(f"[error] 信号计算失败: {e}")
        raise
    append_log(out)
    path = os.path.join(DATA_DIR, "last_signal.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
