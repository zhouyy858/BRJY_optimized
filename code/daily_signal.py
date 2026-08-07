#!/usr/bin/env python3
"""每日收盘信号（D3V策略）：增量更新6个标的数据 -> 计算信号 -> 输出JSON/日志/记录。

D3V规则（2026-08-06重训定稿，样本内2015-2018选参+2019后样本外验证）：
  进场 = close>MA3 且 MA3>MA20 且 个股RSI<80 且 +DI>-DI
         且 医药ETF>MA20 且 创业板>MA30
         且 换手率∈[0.8%,10%]
  仓位 = min(40%/20日年化波动, 100%)（百分比仓位，非全仓）
  成交 = 次日开盘，大跳空≥3%等回撤（脚本只报信号与目标仓位，成交由实盘执行）

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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

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
    if instr["kind"] == "etf":
        # ETF 必须走东财前复权接口；stock_zh_a_hist 对 ETF 兼容性差，作为兜底
        try:
            return ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start,
                                       end_date=end, adjust="qfq")
        except Exception:
            return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                                      end_date=end, adjust="qfq")
    return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")


def fetch_sina(instr, start, end):
    sym = instr["symbol"]
    if instr["kind"] == "index":
        return ak.stock_zh_index_daily(symbol=sym)
    if instr["kind"] == "etf":
        # 新浪 fund_etf_hist_sina 返回不复权价，与东财前复权口径不兼容；
        # 静默换源会重写ETF历史、翻转板块门控并改变回测结果，故拒绝回退(沿用旧qfq数据)
        log(f"[warn] {instr['name']} 新浪ETF数据为不复权口径，拒绝换源回退(仅用东财qfq)")
        return None
    return ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="qfq")


def norm(df):
    df = df.rename(columns=RENAME)
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date", keep="last")


def update_instrument(instr):
    path = os.path.join(DATA_DIR, instr["file"])
    full_start = "19900101"
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
        # 前复权锚点一致性检查：重叠区间收盘价偏差>1%说明期间发生除权除息或源口径变化，
        # 增量拼接会污染历史价格，需全量重下校准（锚点变化会整体平移历史价）。
        ov = df.merge(old[["date", "close"]], on="date", suffixes=("_new", "_old"))
        drift = float((ov["close_new"] / ov["close_old"] - 1).abs().max()) if len(ov) else 0.0
        if drift > 0.01:
            log(f"[warn] {instr['name']} 前复权重叠价偏差{drift:.2%}，触发全量重下")
            full = _try(lambda: fetch_em(instr, full_start, end), f"{instr['name']} 东财全量")
            src = "东财"
            if full is None:
                full = _try(lambda: fetch_sina(instr, full_start, end), f"{instr['name']} 新浪全量")
                src = "新浪"
            if full is not None:
                df = norm(full)
                old = None
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
    # 全部门控按个股交易日对齐：某标的缺最新日时取值为NaN，比较结果为False=门控关闭(保守)
    stock_dates = stock["date"]
    med_a_close = med_a.set_index("date")["close"].reindex(stock_dates)
    med_b_close = med_b.set_index("date")["close"].reindex(stock_dates)
    cyb_close = cyb.set_index("date")["close"].reindex(stock_dates)
    sh_close = sh.set_index("date")["close"].reindex(stock_dates)

    ma3 = close.rolling(3).mean(); ma20 = close.rolling(20).mean()
    rsi = rsi14(close)
    pdi, mdi = dmi(stock)
    to = stock["turnover"].ffill()
    vwap = stock["amount"] / stock["volume"].replace(0, np.nan)
    vr = stock["volume"] / stock["volume"].rolling(20).mean()
    rv20 = close.pct_change().rolling(20).std(ddof=0) * (252 ** 0.5)

    ma_a20 = med_a_close.rolling(20).mean()
    ma_cyb30 = cyb_close.rolling(30).mean()

    last = len(stock) - 1
    gates = {
        "个股_close>MA3": bool(close.iloc[last] > ma3.iloc[last]),
        "个股_MA3>MA20": bool(ma3.iloc[last] > ma20.iloc[last]),
        "个股_RSI<80": bool(rsi.iloc[last] < 80),
        "个股_+DI>-DI": bool(pdi.iloc[last] > mdi.iloc[last]),
        "医药ETF>MA20": bool(med_a_close.iloc[last] > ma_a20.iloc[last]),
        "创业板>MA30": bool(cyb_close.iloc[last] > ma_cyb30.iloc[last]),
        "换手率0.8%-10%": bool(0.008 <= to.iloc[last] <= 0.10),
    }
    sig_today = all(gates.values())
    scale_today = min(0.4 / rv20.iloc[last], 1.0) if np.isfinite(rv20.iloc[last]) else 0.0
    target_tomorrow = sig_today * scale_today

    # 昨日信号与今日应持仓（近似：按昨日信号×昨日scale）
    sig_y = (bool(close.iloc[last - 1] > ma3.iloc[last - 1]) and
             bool(ma3.iloc[last - 1] > ma20.iloc[last - 1]) and
             bool(rsi.iloc[last - 1] < 80) and bool(pdi.iloc[last - 1] > mdi.iloc[last - 1]) and
             bool(med_a_close.iloc[last - 1] > ma_a20.iloc[last - 1]) and
             bool(cyb_close.iloc[last - 1] > ma_cyb30.iloc[last - 1]) and
             bool(0.008 <= to.iloc[last - 1] <= 0.10))
    scale_y = min(0.4 / rv20.iloc[last - 1], 1.0) if np.isfinite(rv20.iloc[last - 1]) else 0.0
    pos_today = sig_y * scale_y

    if sig_today:
        action = "买入/加仓至目标仓位" if pos_today < target_tomorrow - 1e-9 else "持有"
    elif pos_today > 1e-6:
        action = "卖出/清仓(明日开盘执行，大低开≥3%等回撤)"
    else:
        action = "空仓等待"

    out = {
        "as_of": str(stock["date"].iloc[last].date()),
        "strategy": "D3V",
        "last_close": round(float(close.iloc[last]), 3),
        "gates": {k: bool(v) for k, v in gates.items()},
        "signal_today": bool(sig_today),
        "target_position_tomorrow_pct": round(float(target_tomorrow) * 100, 2),
        "position_today_pct": round(float(pos_today) * 100, 2),
        "action": action,
        "indicators": {
            "ma3": round(float(ma3.iloc[last]), 3), "ma20": round(float(ma20.iloc[last]), 3),
            "rsi": round(float(rsi.iloc[last]), 1), "pdi": round(float(pdi.iloc[last]), 1),
            "mdi": round(float(mdi.iloc[last]), 1),
            "turnover_pct": round(float(to.iloc[last]) * 100, 2),
            "volume_ratio": round(float(vr.iloc[last]), 2),
            "close_vs_vwap_pct": round((float(close.iloc[last]) / float(vwap.iloc[last]) - 1) * 100, 2),
            "vol20_annual": round(float(rv20.iloc[last]), 3),
        },
        "sector": {
            "医药ETF_close": float(med_a["close"].iloc[-1]),
            "医药ETF_MA20": round(float(med_a["close"].rolling(20).mean().iloc[-1]), 3),
            "医药ETF_RSI": round(float(rsi14(med_a["close"]).iloc[-1]), 1),
            "医疗ETF_close": float(med_b["close"].iloc[-1]),
            "医疗ETF_MA20": round(float(med_b["close"].rolling(20).mean().iloc[-1]), 3),
            "创业板close": float(cyb["close"].iloc[-1]),
            "创业板MA30": round(float(cyb["close"].rolling(30).mean().iloc[-1]), 3),
            "上证close": float(sh["close"].iloc[-1]),
            "上证MA20": round(float(sh["close"].rolling(20).mean().iloc[-1]), 3),
        },
        "data_latest": {
            "贝瑞基因": str(stock["date"].iloc[last].date()),
            "医药ETF": str(med_a["date"].iloc[-1].date()),
            "医疗ETF": str(med_b["date"].iloc[-1].date()),
            "创业板": str(cyb["date"].iloc[-1].date()),
            "上证": str(sh["date"].iloc[-1].date()),
        },
        "data_stale": {
            "医药ETF": med_a["date"].iloc[-1].date() < stock["date"].iloc[last].date(),
            "医疗ETF": med_b["date"].iloc[-1].date() < stock["date"].iloc[last].date(),
            "创业板": cyb["date"].iloc[-1].date() < stock["date"].iloc[last].date(),
            "上证": sh["date"].iloc[-1].date() < stock["date"].iloc[last].date(),
        },
        "note": "门控全部按个股交易日对齐，数据源缺最新日时该门控强制关闭(保守)；仓位为百分比目标，实盘按次日开盘成交、大跳空≥3%等回撤。历史表现不代表未来。",
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
    ap = argparse.ArgumentParser(description="D3V每日收盘信号：增量更新数据+计算信号")
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
