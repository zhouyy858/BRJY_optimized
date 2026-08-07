#!/usr/bin/env python3
"""盘中实时决策回测试点：T日14:45用盘中数据计算D3V信号并即时成交（买卖同规则）。

数据：baostock 5分钟前复权(adjustflag=2)，2020-01-02起有分钟线；日线同源baostock。
口径（无未来函数）：
  - 个股指标(MA3/MA20/RSI/DI)用"截至T-1的日线 + T日14:45盘中价/盘中高低"计算；
  - 板块门控(医药ETF>MA20)、创业板(>MA30)、换手率、波动率目标全部用T-1已知值；
  - 成交价=T日14:45分钟线收盘价，买卖同规则；成本0.1%/边；
  - 持仓部分按日线收盘标记，交易部分按14:45成交价标记。
对比：官方retrace(次日开盘)引擎在同一窗口(同源baostock日线+仓库CSV门控)的结果。
"""
import argparse
import importlib.util
import json
import os
import sys

import baostock as bs
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("optimize", os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize.py"))
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


def _rows(rs):
    out = []
    while rs.error_code == "0" and rs.next():
        out.append(rs.get_row_data())
    return out


def fetch_daily(code, start, end):
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount,turn",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    df = pd.DataFrame(_rows(rs), columns=["date", "open", "high", "low", "close", "volume", "amount", "turn"])
    for c in ("open", "high", "low", "close", "volume", "amount", "turn"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_minute(code, start, end, cache, refetch=False):
    if cache and not refetch and os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["date"])
    else:
        rs = bs.query_history_k_data_plus(
            code, "date,time,open,high,low,close,volume,amount",
            start_date=start, end_date=end, frequency="5", adjustflag="2")
        df = pd.DataFrame(_rows(rs), columns=["date", "time", "open", "high", "low", "close", "volume", "amount"])
        for c in ("open", "high", "low", "close", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["hhmm"] = df["time"].astype(str).str[8:12].astype(int)
        df = df.sort_values(["date", "hhmm"]).reset_index(drop=True)
        if cache:
            df.to_csv(cache, index=False)
    return df


def rsi14(close):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def dmi14(high, low, close):
    up = high.diff()
    dn = -low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=high.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=high.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    return pdi, mdi


def stats_of(net):
    tot = float((1 + net).prod() - 1)
    n = int(net.shape[0])
    ann = (1 + tot) ** (252.0 / n) - 1 if tot > -1 else -1.0
    g = (1 + net).cumprod()
    mdd = float((g / g.cummax() - 1).min())
    return {"ret": round(tot, 4), "ann": round(ann, 4), "maxdd": round(mdd, 4),
            "calmar": round(ann / abs(mdd), 3) if mdd < 0 else None, "days": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2026-08-06")
    ap.add_argument("--decision", default="1445", choices=["1430", "1445"],
                    help="决策时点：1430=14:30 用截至14:30的分钟线成交；1445=14:45")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--cache", default="/tmp/intraday_000710_5min.csv")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    bs.login()
    try:
        d_stock = fetch_daily("sz.000710", args.start, args.end)
        d_cyb = fetch_daily("sz.399006", args.start, args.end)
        d_etf = pd.read_csv("data/sh512010_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        m = fetch_minute("sz.000710", args.start, args.end, args.cache, refetch=args.refetch)
    finally:
        bs.logout()

    # 每日14:45盘中状态
    cutoff = int(args.decision)
    day = m[m["hhmm"] <= cutoff].groupby("date").agg(
        p=("close", "last"), h=("high", "max"), l=("low", "min")).reset_index()
    day = day.sort_values("date").reset_index(drop=True)

    d_stock = d_stock.merge(day, on="date", how="left")
    closes = d_stock["close"]
    pc = closes.shift(1)
    etf_close = d_etf.set_index("date")["close"]
    etf_flag = (etf_close > etf_close.rolling(20).mean()).reindex(d_stock["date"]).shift(1).fillna(False)
    cyb_close = d_cyb.set_index("date")["close"]
    cyb_flag = (cyb_close > cyb_close.rolling(30).mean()).reindex(d_stock["date"]).shift(1).fillna(False)
    turn_t1 = (d_stock["turn"] / 100.0).shift(1).fillna(0)
    vol20 = closes.pct_change().rolling(20).std(ddof=0) * (252 ** 0.5)
    scale = (0.4 / vol20).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)

    actual = 0.0
    net_list = []
    sig_list = []
    for i in range(len(d_stock)):
        c_t = closes.iloc[i]
        p_1445 = d_stock["p"].iloc[i]
        if not np.isfinite(c_t) or not np.isfinite(p_1445) or i == 0:
            net_list.append(0.0)
            sig_list.append(0)
            continue
        # 用截至T-1的日线+T日盘中价构造指标序列
        hh = pd.concat([d_stock["high"].iloc[:i], pd.Series([d_stock["h"].iloc[i]])], ignore_index=True)
        ll = pd.concat([d_stock["low"].iloc[:i], pd.Series([d_stock["l"].iloc[i]])], ignore_index=True)
        cc = pd.concat([closes.iloc[:i], pd.Series([p_1445])], ignore_index=True)
        ma3 = cc.rolling(3).mean().iloc[-1]
        ma20 = cc.rolling(20).mean().iloc[-1]
        rsi = rsi14(cc).iloc[-1]
        pdi, mdi = dmi14(hh, ll, cc)
        sig = bool(
            p_1445 > ma3 and ma3 > ma20 and rsi < 80 and pdi.iloc[-1] > mdi.iloc[-1]
            and bool(etf_flag.iloc[i]) and bool(cyb_flag.iloc[i])
            and 0.008 <= turn_t1.iloc[i] <= 0.10)
        goal = sig * float(scale.iloc[i])
        delta = goal - actual
        kept = min(actual, goal)
        bought = max(delta, 0.0)
        sold = max(-delta, 0.0)
        # 三段记账（与optimize.py open口径一致）：持仓按昨收→收盘，买入按14:45→收盘，卖出按昨收→14:45
        day_ret = kept * (c_t / pc.iloc[i] - 1) + bought * (c_t / p_1445 - 1) + sold * (p_1445 / pc.iloc[i] - 1)
        day_ret -= abs(delta) * args.cost
        actual = goal
        net_list.append(day_ret)
        sig_list.append(int(sig))
    net = pd.Series(net_list, index=d_stock["date"])
    intra = stats_of(net)
    intra["trades"] = int((pd.Series(sig_list).diff().abs().sum()) / 2)

    # 官方retrace基线：同源baostock日线 + 仓库CSV门控（T-1信号次日执行）
    etf_flag_full = (d_etf.set_index("date")["close"] > d_etf.set_index("date")["close"].rolling(20).mean())
    cyb_flag_full = (d_cyb.set_index("date")["close"] > d_cyb.set_index("date")["close"].rolling(30).mean())
    df_bt = d_stock[["date", "open", "high", "low", "close"]].copy()
    df_bt["volume"] = d_stock["volume"].fillna(0)
    df_bt["amount"] = d_stock["amount"].fillna(0)
    df_bt["turnover"] = (d_stock["turn"] / 100.0).fillna(0)
    eq, net2, pos2 = opt.run_backtest(
        df_bt, 3, 20, 80, None, 0.4, 0, args.cost, etf_flag_full, trend_dir="di",
        extra_flag=cyb_flag_full, turnover_range=(0.008, 0.10),
        execution="retrace", retrace_gap=0.03)
    base = stats_of(net2)
    base["trades"] = int(pos2.diff().abs().fillna(0).sum() / 2)

    result = {
        "window": [str(d_stock["date"].iloc[0].date()), str(d_stock["date"].iloc[-1].date())],
        "decision_time": args.decision,
        "intraday_" + args.decision: intra,
        "official_retrace_baseline": base,
        "cost": args.cost,
        "notes": [
            "个股指标用盘中价实时计算；板块/创业板/换手/波动率门控用T-1值",
            "ETF门控取仓库CSV(东财qfq)，创业板与个股日线取baostock qfq",
            "盘中模型持仓部分按日线收盘标记，交易部分按14:45价标记",
            "试点窗口2020+：baostock分钟线自2020-01-02起可用",
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
