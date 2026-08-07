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


def run_rule_triggered(d_stock, d_cyb, d_etf, m, cost=0.001, confirm_bars=1):
    """规则触发模式：每个5分钟bar实时算D3V信号，目标仓位与当前仓位不一致时当根bar成交。
    无时钟参数；T+1约束：当日买入部分当日不可卖，卖不掉的顺延至次日第一根bar。
    个股指标增量计算：MA3/MA20用滚动和，RSI/DI用EWM状态递推(截至T-1日线+当日盘中值)。"""
    closes = d_stock["close"].reset_index(drop=True)
    pc = closes.shift(1)
    etf_flag = (d_etf.set_index("date")["close"] > d_etf.set_index("date")["close"].rolling(20).mean())
    etf_flag = etf_flag.reindex(d_stock["date"]).shift(1).fillna(False).reset_index(drop=True)
    cyb_flag = (d_cyb.set_index("date")["close"] > d_cyb.set_index("date")["close"].rolling(30).mean())
    cyb_flag = cyb_flag.reindex(d_stock["date"]).shift(1).fillna(False).reset_index(drop=True)
    turn_t1 = (d_stock["turn"] / 100.0).shift(1).fillna(0).reset_index(drop=True)
    vol20 = closes.pct_change().rolling(20).std(ddof=0) * (252 ** 0.5)
    scale = (0.4 / vol20).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0).reset_index(drop=True)

    # 日线EWM状态（截至T-1）与MA滚动和
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    ag = gain.ewm(alpha=1 / 14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False).mean()
    high = d_stock["high"].reset_index(drop=True)
    low = d_stock["low"].reset_index(drop=True)
    up = high.diff()
    dn = -low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low, (high - closes.shift(1)).abs(), (low - closes.shift(1)).abs()], axis=1).max(axis=1)
    ep = pd.Series(pdm, index=closes.index).ewm(alpha=1 / 14, adjust=False).mean()
    em = pd.Series(mdm, index=closes.index).ewm(alpha=1 / 14, adjust=False).mean()
    eatr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    s19 = closes.rolling(19).sum().shift(1)  # T-1时点的最近19日收盘和
    a14 = 1 / 14.0

    m = m.sort_values(["date", "hhmm"]).reset_index(drop=True)
    by_date = {d: g for d, g in m.groupby("date")}
    actual = 0.0
    nets = []
    sigs = []
    trade_count = 0
    for t in range(1, len(d_stock)):
        dt = d_stock["date"].iloc[t]
        bars = by_date.get(dt)
        c_t = closes.iloc[t]
        pc_t = pc.iloc[t]
        if bars is None or len(bars) == 0 or not np.isfinite(c_t) or not np.isfinite(pc_t):
            nets.append(0.0)
            sigs.append(0)
            continue
        day_start = actual
        sold_today = 0.0
        day_ret = 0.0
        day_h = -np.inf
        day_l = np.inf
        state_count = 0
        prev_state = None
        for _, b in bars.iterrows():
            p_b = float(b["close"])
            if not np.isfinite(p_b):
                continue
            day_h = max(day_h, float(b["high"]))
            day_l = min(day_l, float(b["low"]))
            # 增量指标
            ma3 = (closes.iloc[t - 2] + closes.iloc[t - 1] + p_b) / 3.0
            ma20 = (float(s19.iloc[t]) + p_b) / 20.0
            g_b = max(p_b - pc_t, 0.0)
            l_b = max(pc_t - p_b, 0.0)
            ag_b = ag.iloc[t - 1] + a14 * (g_b - ag.iloc[t - 1])
            al_b = al.iloc[t - 1] + a14 * (l_b - al.iloc[t - 1])
            rsi_b = 100.0 - 100.0 / (1.0 + ag_b / max(al_b, 1e-12))
            up_b = day_h - high.iloc[t - 1]
            dn_b = low.iloc[t - 1] - day_l
            pdm_b = up_b if (up_b > dn_b and up_b > 0) else 0.0
            mdm_b = dn_b if (dn_b > up_b and dn_b > 0) else 0.0
            tr_b = max(day_h - day_l, abs(day_h - pc_t), abs(day_l - pc_t))
            ep_b = ep.iloc[t - 1] + a14 * (pdm_b - ep.iloc[t - 1])
            em_b = em.iloc[t - 1] + a14 * (mdm_b - em.iloc[t - 1])
            eatr_b = eatr.iloc[t - 1] + a14 * (tr_b - eatr.iloc[t - 1])
            pdi_b = 100.0 * ep_b / max(eatr_b, 1e-12)
            mdi_b = 100.0 * em_b / max(eatr_b, 1e-12)
            sig_b = int(p_b > ma3 and ma3 > ma20 and rsi_b < 80 and pdi_b > mdi_b
                        and bool(etf_flag.iloc[t]) and bool(cyb_flag.iloc[t])
                        and 0.008 <= turn_t1.iloc[t] <= 0.10)
            # 规则确认：信号状态需连续稳定 confirm_bars 根bar才动作（纯规则，无时钟参数）
            state_count = state_count + 1 if sig_b == prev_state else 1
            prev_state = sig_b
            if state_count >= confirm_bars:
                goal = sig_b * float(scale.iloc[t])
                delta = goal - actual
                if delta < -1e-12:
                    max_sell = max(0.0, day_start - sold_today)  # T+1：只能卖当日开盘前持仓
                    delta = max(delta, -max_sell)
                if abs(delta) > 1e-12:
                    trade_count += 1
                    if delta > 0:
                        day_ret += delta * (c_t / p_b - 1)
                        actual += delta
                    else:
                        sold = -delta
                        day_ret += sold * (p_b / pc_t - 1)
                        actual -= sold
                        sold_today += sold
                    day_ret -= abs(delta) * cost
        day_ret += (day_start - sold_today) * (c_t / pc_t - 1)
        nets.append(day_ret)
        sigs.append(int(actual > 1e-12))
    net = pd.Series(nets, index=d_stock["date"].iloc[1:])
    st = stats_of(net)
    st["trades"] = trade_count
    return st


PRICE_RULES = ["prevhl", "orb30", "vwap", "ma5", "pct2", "pct15"]


def run_price_rule(d_stock, d_cyb, d_etf, m, rule, use_gates, cost=0.001):
    """纯盘中价格触发规则：触发条件只看盘中价(不含收盘价决策)；
    触发即当根bar成交，T+1：当日买入不可卖，卖不掉顺延次日首bar。
    规则：
      prevhl = 突破昨高做多/跌破昨低做空(离场)
      orb30  = 开盘30分钟区间突破(高破做多/低破离场，区间内保持)
      vwap   = 价格上穿盘中累计VWAP做多/下穿离场
      ma5    = 上穿盘中5根bar均线做多/下穿离场
      pct2/pct15 = 突破昨收+2%/+1.5%做多、跌破昨收-2%/-1.5%离场(带内保持)
    """
    closes = d_stock["close"].reset_index(drop=True)
    high = d_stock["high"].reset_index(drop=True)
    low = d_stock["low"].reset_index(drop=True)
    pc = closes.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    if use_gates:
        etf_flag = (d_etf.set_index("date")["close"] > d_etf.set_index("date")["close"].rolling(20).mean())
        etf_flag = etf_flag.reindex(d_stock["date"]).shift(1).fillna(False).reset_index(drop=True)
        cyb_flag = (d_cyb.set_index("date")["close"] > d_cyb.set_index("date")["close"].rolling(30).mean())
        cyb_flag = cyb_flag.reindex(d_stock["date"]).shift(1).fillna(False).reset_index(drop=True)
        turn_t1 = (d_stock["turn"] / 100.0).shift(1).fillna(0).reset_index(drop=True)
    else:
        etf_flag = pd.Series(True, index=closes.index)
        cyb_flag = pd.Series(True, index=closes.index)
        turn_t1 = pd.Series(0.05, index=closes.index)
    vol20 = closes.pct_change().rolling(20).std(ddof=0) * (252 ** 0.5)
    scale = (0.4 / vol20).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0).reset_index(drop=True)
    m = m.sort_values(["date", "hhmm"]).reset_index(drop=True)
    by_date = {d: g for d, g in m.groupby("date")}
    actual = 0.0
    nets = []
    trade_count = 0
    for t in range(1, len(d_stock)):
        dt = d_stock["date"].iloc[t]
        bars = by_date.get(dt)
        c_t = closes.iloc[t]
        pc_t = pc.iloc[t]
        if bars is None or len(bars) == 0 or not np.isfinite(c_t) or not np.isfinite(pc_t):
            nets.append(0.0)
            continue
        day_start = actual
        sold_today = 0.0
        day_ret = 0.0
        cum_amt = 0.0
        cum_vol = 0.0
        bar_closes = []
        orb_hi = None
        orb_lo = None
        bar_i = 0
        for _, b in bars.iterrows():
            p_b = float(b["close"])
            if not np.isfinite(p_b):
                continue
            bar_i += 1
            cum_amt += float(b["amount"])
            cum_vol += float(b["volume"])
            bar_closes.append(p_b)
            if rule == "prevhl":
                sig = 1 if p_b > prev_high.iloc[t] else 0
            elif rule == "orb30":
                if bar_i <= 6:
                    if orb_hi is None:
                        orb_hi = orb_lo = p_b
                    orb_hi = max(orb_hi, float(b["high"]))
                    orb_lo = min(orb_lo, float(b["low"]))
                    continue
                if p_b > orb_hi:
                    sig = 1
                elif p_b < orb_lo:
                    sig = 0
                else:
                    continue
            elif rule == "vwap":
                vwap = cum_amt / cum_vol if cum_vol > 0 else p_b
                sig = 1 if p_b > vwap else 0
            elif rule == "ma5":
                ma = float(np.mean(bar_closes[-5:]))
                sig = 1 if p_b > ma else 0
            else:  # pct2 / pct15
                thr = 0.02 if rule == "pct2" else 0.015
                if p_b >= pc_t * (1 + thr):
                    sig = 1
                elif p_b <= pc_t * (1 - thr):
                    sig = 0
                else:
                    continue
            gate_ok = (not use_gates) or (bool(etf_flag.iloc[t]) and bool(cyb_flag.iloc[t])
                                          and 0.008 <= turn_t1.iloc[t] <= 0.10)
            sig = sig if gate_ok else 0
            goal = sig * float(scale.iloc[t])
            delta = goal - actual
            if delta < -1e-12:
                max_sell = max(0.0, day_start - sold_today)  # T+1
                delta = max(delta, -max_sell)
            if abs(delta) > 1e-12:
                trade_count += 1
                if delta > 0:
                    day_ret += delta * (c_t / p_b - 1)
                    actual += delta
                else:
                    sold = -delta
                    day_ret += sold * (p_b / pc_t - 1)
                    actual -= sold
                    sold_today += sold
                day_ret -= abs(delta) * cost
        day_ret += (day_start - sold_today) * (c_t / pc_t - 1)
        nets.append(day_ret)
    net = pd.Series(nets, index=d_stock["date"].iloc[1:])
    st = stats_of(net)
    st["trades"] = trade_count
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2026-08-06")
    ap.add_argument("--decision", default="1445", choices=["1430", "1445"],
                    help="决策时点：1430=14:30 用截至14:30的分钟线成交；1445=14:45")
    ap.add_argument("--trigger", default="fixed", choices=["fixed", "rule", "price"],
                    help="fixed=固定时点决策；rule=规则触发(D3V信号变了就成交)；price=纯价格触发规则")
    ap.add_argument("--price-rule", choices=PRICE_RULES, default="orb30",
                    help="price触发模式的价格规则: prevhl/orb30/vwap/ma5/pct2/pct15")
    ap.add_argument("--no-gates", action="store_true",
                    help="price触发模式不带D3V板块/换手门控(纯价格规则)")
    ap.add_argument("--confirm", type=int, default=1,
                    help="规则触发模式的确认bar数：信号连续N根bar成立才成交(默认1=即时)")
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

    if args.trigger == "price":
        intra = run_price_rule(d_stock, d_cyb, d_etf, m, args.price_rule,
                               use_gates=not args.no_gates, cost=args.cost)
        intra_key = "price_%s_gates_%s" % (args.price_rule, "no" if args.no_gates else "yes")
    elif args.trigger == "rule":
        intra = run_rule_triggered(d_stock, d_cyb, d_etf, m, args.cost, confirm_bars=args.confirm)
        intra_key = "intraday_rule"
    else:
        # 固定时点决策（原有逻辑）
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
        intra_key = "intraday_" + args.decision

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
        "decision_time": "rule(盘中触发)" if args.trigger == "rule" else args.decision,
        intra_key: intra,
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
