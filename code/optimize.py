#!/usr/bin/env python3
"""参数网格优化：MA快慢线+RSI超买过滤，全部点时序、次日生效；输出帕累托前沿与简单样本外校验。"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd


def rsi14(close):
    delta = close.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr14(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False).mean()


def dmi14(df):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, dx.ewm(alpha=1.0 / 14, adjust=False).mean()


def adx14(df):
    return dmi14(df)[2]


def run_backtest(df, fast, slow, rsi_ceiling, stop_mult, vol_target, adx_floor, cost, sector_flag=None, vol_floor=0.0, trend_dir="adx", extra_flag=None, turnover_range=None, close_vwap=False, vol_ratio_min=None, execution="open", retrace_gap=0.03, sector_rsi_flag=None, limit_block=False):
    close = df["close"]
    ret = close.pct_change()
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    sig = ((close > ma_fast) & (ma_fast > ma_slow)).astype(int)
    if rsi_ceiling is not None:
        sig = (sig & (rsi14(close) < rsi_ceiling).fillna(True)).astype(int)
    if trend_dir in ("adx", "both") and adx_floor:
        sig = (sig & (adx14(df) >= adx_floor).fillna(False)).astype(int)
    if trend_dir in ("di", "both"):
        pdi, mdi, _ = dmi14(df)
        sig = (sig & (pdi > mdi).fillna(False)).astype(int)
    if sector_flag is not None:
        # 板块过滤：ETF收盘>其MA，按日期对齐到个股交易序列，缺失(ETF未上市)视为不满足->空仓
        sig = (sig & sector_flag.reindex(df["date"]).fillna(False).to_numpy()).astype(int)
    if sector_rsi_flag is not None:
        # 板块过热过滤：ETF RSI<上限才允许进场(避免追高过热板块)
        sig = (sig & sector_rsi_flag.reindex(df["date"]).fillna(False).to_numpy()).astype(int)
    if extra_flag is not None:
        sig = (sig & extra_flag.reindex(df["date"]).fillna(False).to_numpy()).astype(int)
    if turnover_range is not None and "turnover" in df.columns:
        # 换手率区间过滤：排除缩量冷清日与放量冲高日；缺失值前向填充(只用过去信息)
        to = df["turnover"].ffill()
        sig = (sig & ((to >= turnover_range[0]) & (to <= turnover_range[1])).fillna(False).to_numpy()).astype(int)
    if close_vwap and "amount" in df.columns:
        # 均价过滤：收盘价须站上当日成交均价(VWAP=成交额/成交量)
        vwap = df["amount"] / df["volume"].replace(0, np.nan)
        sig = (sig & (df["close"] > vwap).fillna(False).to_numpy()).astype(int)
    if vol_ratio_min is not None and "volume" in df.columns:
        # 量比过滤：当日成交量/20日均量 >= 阈值
        vr = df["volume"] / df["volume"].rolling(20).mean()
        sig = (sig & (vr >= vol_ratio_min).fillna(False).to_numpy()).astype(int)
    if stop_mult is None:
        pos = sig.shift(1).fillna(0)
    else:
        # 跟踪止损：持仓期间以收盘价峰值为基准，跌破 峰值-倍数*ATR 则次日离场（决策只用当日及以前数据）
        atr = atr14(df).to_numpy()
        closes = close.to_numpy()
        sigs = sig.to_numpy()
        pos_arr = np.zeros(len(df), dtype=int)
        peak = 0.0
        for t in range(1, len(df)):
            if pos_arr[t - 1]:
                peak = max(peak, closes[t - 1])
                if closes[t - 1] < peak - stop_mult * atr[t - 1]:
                    pos_arr[t] = 0
                    peak = 0.0
                else:
                    pos_arr[t] = 1
            elif sigs[t - 1]:
                pos_arr[t] = 1
                peak = closes[t - 1]
        pos = pd.Series(pos_arr, index=df.index)
    if vol_target is not None:
        # 波动率目标：按前20日已实现波动动态缩放仓位（t-1日波动决定t日仓位），高波动自动降仓
        realized_vol = close.pct_change().rolling(20).std(ddof=0) * (252 ** 0.5)
        scale = (vol_target / realized_vol).clip(lower=vol_floor, upper=1.0).shift(1).fillna(1.0)
        pos = (pos * scale).fillna(0)
    turnover = pos.diff().abs().fillna(0)
    if execution == "retrace":
        # 回撤成交：目标仓位t-1收盘决定；高开不追买/低开不杀跌，等价格回到昨收价再成交，未成交顺延
        # limit_block=True：一字涨停/跌停日(开盘即封板且全天未开板)视为无法成交，顺延到下一天（仅retrace支持，open/vwap无顺延逻辑）
        target = pos.to_numpy()
        closes = df["close"].to_numpy(); opens = df["open"].to_numpy()
        highs = df["high"].to_numpy(); lows = df["low"].to_numpy()
        actual = np.zeros(len(df)); day_ret = np.zeros(len(df))
        for t in range(1, len(df)):
            pc = closes[t - 1]
            if not np.isfinite(pc) or pc <= 0:
                actual[t] = actual[t - 1]
                continue
            prev_p = actual[t - 1]; goal = target[t]
            delta = goal - prev_p
            if limit_block:
                # 一字板：全天OHLC同价且封在涨/跌停价（主板上限±10%，ST为±5%未单独识别）
                # 方向约束：一字涨停只拦买入(卖出可成交)，一字跌停只拦卖出(买入可成交)
                one_word = opens[t] == highs[t] == lows[t] == closes[t]
                if one_word and closes[t] >= round(pc * 1.1, 2) - 1e-9 and delta > 1e-12:
                    actual[t] = actual[t - 1]
                    day_ret[t] = actual[t - 1] * (closes[t] / pc - 1)
                    continue
                if one_word and closes[t] <= round(pc * 0.9, 2) + 1e-9 and delta < -1e-12:
                    actual[t] = actual[t - 1]
                    day_ret[t] = actual[t - 1] * (closes[t] / pc - 1)
                    continue
            gap_buy = pc * (1.0 + retrace_gap)
            gap_sell = pc * (1.0 - retrace_gap)
            if delta > 1e-12:
                if opens[t] <= gap_buy:
                    fill = opens[t]  # 跳空未超阈值，开盘照常成交
                elif lows[t] <= pc:
                    fill = pc  # 大高开后盘中回落到昨收，按昨收挂单成交
                else:
                    fill = np.nan  # 大高开且未回落，放弃本日买入，顺延
                if np.isfinite(fill):
                    actual[t] = goal
                    day_ret[t] = prev_p * (closes[t] / pc - 1) + delta * (closes[t] / fill - 1)
                else:
                    actual[t] = prev_p
                    day_ret[t] = prev_p * (closes[t] / pc - 1)
            elif delta < -1e-12:
                if opens[t] >= gap_sell:
                    fill = opens[t]  # 低开未超阈值，开盘照常成交
                elif highs[t] >= pc:
                    fill = pc  # 大低开后盘中反弹回昨收，按昨收挂单成交
                else:
                    fill = np.nan  # 大低开且未反弹，放弃本日卖出，顺延
                if np.isfinite(fill):
                    actual[t] = goal
                    sold = -delta
                    day_ret[t] = (prev_p - sold) * (closes[t] / pc - 1) + sold * (fill / pc - 1)
                else:
                    actual[t] = prev_p
                    day_ret[t] = prev_p * (closes[t] / pc - 1)
            else:
                actual[t] = prev_p
                day_ret[t] = prev_p * (closes[t] / pc - 1)
            day_ret[t] -= abs(actual[t] - prev_p) * cost
        pos = pd.Series(actual, index=df.index)
        net = pd.Series(day_ret, index=df.index)
        return (1 + net).cumprod() - 1, net, pos

    if execution in ("open", "vwap"):
        if limit_block:
            raise ValueError("limit_block 仅支持 retrace 成交口径（open/vwap 无未成交顺延逻辑，启用会得到错误结果）")
        # 严格次日成交：信号t-1收盘决定，t日开盘(vwap时按当日成交均价)执行；加减仓切片分别计价
        prev = pos.shift(1).fillna(0)
        delta = pos - prev
        kept = pd.concat([prev, pos], axis=1).min(axis=1)
        bought = delta.clip(lower=0)
        sold = (-delta).clip(lower=0)
        if execution == "open":
            fill = df["open"].replace(0, np.nan)
        else:
            # 成交均价VWAP=成交额/成交量，缺失时回退收盘价
            fill = (df["amount"] / df["volume"].replace(0, np.nan)).fillna(df["close"])
        ret_fc = df["close"] / fill - 1
        ret_cf = fill / df["close"].shift(1).replace(0, np.nan) - 1
        day_ret = kept * ret + bought * ret_fc + sold * ret_cf
        net = day_ret.fillna(0) - turnover * cost
    else:
        # 收盘口径：信号当日收盘价成交，偏乐观，仅作参照
        net = pos * ret - turnover * cost
    equity = (1 + net).cumprod() - 1
    return equity, net, pos


def stats(df, equity, net):
    n = int(equity.shape[0])
    if n == 0:
        return None
    total = float(equity.iloc[-1])
    ann = (1 + total) ** (252.0 / n) - 1 if total > -1 else -1.0
    growth = 1 + equity
    mdd = float((growth / growth.cummax() - 1).min())
    cur_dd = float(growth.iloc[-1] / growth.cummax().iloc[-1] - 1)
    vol = float(net.std(ddof=0) * (252 ** 0.5)) if n > 1 else 0.0
    calmar = ann / abs(mdd) if mdd < 0 else 0.0
    return {
        "total_return": round(total, 4),
        "annual_return": round(ann, 4),
        "max_drawdown": round(mdd, 4),
        "current_drawdown": round(cur_dd, 4),
        "annual_vol": round(vol, 4),
        "calmar": round(calmar, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="MA+RSI策略参数网格优化")
    ap.add_argument("csv")
    ap.add_argument("--fasts", default="5,10,20,30")
    ap.add_argument("--slows", default="30,60,90,120")
    ap.add_argument("--rsi-ceiling", default="70", help="RSI超买过滤上限，None表示不过滤")
    ap.add_argument("--stops", default="none,3,5", help="ATR跟踪止损倍数，none表示不止损")
    ap.add_argument("--vol-target", default="none,0.25", help="波动率目标(年化)，none表示不缩放仓位")
    ap.add_argument("--adx", default="0,20", help="ADX趋势强度下限，0表示不过滤")
    ap.add_argument("--sector-csv", help="板块/行业ETF日线CSV，传入后启用ETF>MA过滤（如data/sh512010_daily.csv）")
    ap.add_argument("--sector-ma", type=int, default=20, help="板块ETF均线周期")
    ap.add_argument("--vol-floor", type=float, default=0.0, help="波动率目标缩放下限(0表示无下限)")
    ap.add_argument("--trend", default="adx", choices=["adx", "di", "both"],
                    help="趋势过滤方式：adx=仅ADX阈值; di=仅+DI>-DI方向; both=两者都满足")
    ap.add_argument("--extra-csv", help="第二过滤指数/ETF日线CSV(如创业板sz399006_daily.csv)，与板块过滤AND叠加")
    ap.add_argument("--extra-ma", type=int, default=20, help="第二过滤均线周期")
    ap.add_argument("--turnover-range", default="none", help="换手率区间过滤，如0.01,0.12(排除冷清日与放量冲高日)，none表示不过滤")
    ap.add_argument("--close-vwap", action="store_true", help="要求收盘价站上当日成交均价(VWAP)")
    ap.add_argument("--vol-ratio-min", type=float, default=None, help="量比下限(当日量/20日均量)，None表示不过滤")
    ap.add_argument("--execution", default="open", choices=["close", "open", "vwap", "retrace"],
                    help="成交口径：open=次日开盘成交(默认); retrace=回撤成交(跳空超阈值时不追买/不杀跌,等回到昨收价成交,未成交顺延); vwap=次日成交均价; close=信号当日收盘(偏乐观,仅参照)")
    ap.add_argument("--retrace-gap", type=float, default=0.03, help="retrace模式的跳空阈值(默认0.03=3%): 跳空小于阈值按开盘成交,超过阈值才等回撤")
    ap.add_argument("--sector-rsi-max", type=float, default=None, help="板块ETF RSI上限(如70): 板块过热时不进场,None表示不过滤")
    ap.add_argument("--cost", type=float, default=0.001, help="单边交易成本(默认0.1%/边，往返0.2%；调仓按成交额每边计费)")
    ap.add_argument("--limit-block", action="store_true", help="一字涨停/跌停日不可成交并顺延（仅retrace口径支持）")
    ap.add_argument("--split", default="2017-01-01", help="样本外起始日(样本外按连续运行口径：指标全历史预热、起点归一化)")
    ap.add_argument("--out")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    sector_flag = None
    if args.sector_csv:
        etf = pd.read_csv(args.sector_csv, parse_dates=["date"]).sort_values("date")
        etf_close = etf.set_index("date")["close"]
        sector_flag = etf_close > etf_close.rolling(args.sector_ma).mean()
    fasts = [int(x) for x in args.fasts.split(",")]
    slows = [int(x) for x in args.slows.split(",")]
    ceilings = [None] if args.rsi_ceiling.strip().lower() == "none" else [int(x) for x in args.rsi_ceiling.split(",")]
    stops = []
    for x in args.stops.split(","):
        x = x.strip().lower()
        stops.append(None if x in ("none", "") else float(x))
    vol_targets = []
    for x in args.vol_target.split(","):
        x = x.strip().lower()
        vol_targets.append(None if x in ("none", "") else float(x))
    adx_floors = [int(x) for x in args.adx.split(",")]
    turnover_range = None
    if args.turnover_range.strip().lower() not in ("none", ""):
        lo, hi = (float(x) for x in args.turnover_range.split(","))
        turnover_range = (lo, hi)
    sector_name = args.sector_csv.split("/")[-1] if args.sector_csv else None
    sector_rsi_flag = None
    if args.sector_rsi_max is not None and args.sector_csv:
        sector_rsi_flag = pd.Series(
            (rsi14(etf_close) < args.sector_rsi_max).to_numpy(), index=etf_close.index)
    extra_flag = None
    if args.extra_csv:
        ex = pd.read_csv(args.extra_csv, parse_dates=["date"]).sort_values("date")
        ex_close = ex.set_index("date")["close"]
        extra_flag = ex_close > ex_close.rolling(args.extra_ma).mean()

    combos = [(f, s, c, m, v, a) for f, s, m, c, v, a
              in itertools.product(fasts, slows, stops, ceilings, vol_targets, adx_floors) if f < s]
    results = []
    for f, s, c, m, v, a in combos:
        eq, net, pos = run_backtest(df, f, s, c, m, v, a, args.cost, sector_flag, args.vol_floor, args.trend, extra_flag, turnover_range, args.close_vwap, args.vol_ratio_min, args.execution, args.retrace_gap, sector_rsi_flag, args.limit_block)
        st = stats(df, eq, net)
        if st is None:
            continue
        trades = int(pos.diff().abs().fillna(0).sum() / 2)
        results.append({**st, "fast": f, "slow": s, "rsi_ceiling": c, "stop": m,
                        "vol_target": v, "adx_floor": a, "sector": sector_name, "trades": trades})

    res_df = pd.DataFrame(results).sort_values("calmar", ascending=False)
    top = res_df.head(10).to_dict("records")

    def _clean(r):
        if r is None:
            return None
        return {k: (None if pd.isna(v) else float(v)
                    if isinstance(v, (int, float, np.integer, np.floating)) else v)
                for k, v in r.to_dict().items()}

    split = pd.Timestamp(args.split)
    mask = df["date"] < split
    if mask.sum() > 100 and (~mask).sum() > 100:
        is_df = df[mask].reset_index(drop=True)
        oos_df = df[~mask].reset_index(drop=True)
        ranked = []
        for f, s, c, m, v, a in combos:
            eq, net, pos = run_backtest(is_df, f, s, c, m, v, a, args.cost, sector_flag, args.vol_floor, args.trend, extra_flag, turnover_range, args.close_vwap, args.vol_ratio_min, args.execution, args.retrace_gap, sector_rsi_flag, args.limit_block)
            st = stats(is_df, eq, net)
            if st:
                ranked.append({**st, "fast": f, "slow": s, "rsi_ceiling": c, "stop": m,
                               "vol_target": v, "adx_floor": a, "sector": sector_name})
        top_is = sorted(ranked, key=lambda r: r["calmar"], reverse=True)[:5]
        top_is_noextra = sorted(
            (r for r in ranked if r["stop"] is None and r["vol_target"] is None and r["adx_floor"] == 0),
            key=lambda r: r["calmar"], reverse=True)[:3]
        def _oos(cand):
            # 样本外连续运行口径：指标用全历史计算(实盘可获取split前数据)，从split起点归一化，
            # 避免"冷启动"(窗口内重算指标)在前30-40日无信号造成的收益低估与持仓路径偏差。
            eq, net, pos = run_backtest(
                df, cand["fast"], cand["slow"], cand["rsi_ceiling"], cand["stop"],
                cand["vol_target"], cand["adx_floor"], args.cost, sector_flag, args.vol_floor, args.trend, extra_flag, turnover_range, args.close_vwap, args.vol_ratio_min, args.execution, args.retrace_gap, sector_rsi_flag, args.limit_block)
            s, n = eq[~mask], net[~mask]
            base = float(s.iloc[0])
            # 归一化必须用净值+1后的增长率比值：(1+s)/(1+base)-1；
            # 直接用裸净值 s/base-1 在窗口起点净值非0时会把收益夸大(如2017起点-54%时+1677%虚高，正确+529%)
            norm_eq = (1 + s) / (1 + base) - 1 if base > -1 else s
            st = stats(oos_df, norm_eq, n)
            if st:
                return {**st, "fast": cand["fast"], "slow": cand["slow"],
                        "rsi_ceiling": cand["rsi_ceiling"], "stop": cand["stop"],
                        "vol_target": cand["vol_target"], "adx_floor": cand["adx_floor"]}
            return None
        oos_rows = [r for r in (_oos(c) for c in top_is) if r]
        oos_rows_noextra = [r for r in (_oos(c) for c in top_is_noextra) if r]
        bh_ret = oos_df["close"].pct_change()
        bh_eq = (1 + bh_ret).cumprod() - 1
        oos = {"range": [str(split.date()), str(df["date"].iloc[-1].date())],
               "in_sample_top5": [{"fast": r["fast"], "slow": r["slow"],
                                   "rsi_ceiling": r["rsi_ceiling"], "stop": r["stop"],
                                   "vol_target": r["vol_target"], "adx_floor": r["adx_floor"],
                                   "calmar_is": r["calmar"],
                                   "maxdd_is": r["max_drawdown"]} for r in top_is],
               "out_of_sample": oos_rows,
               "no_stop_no_vol_top3_is": [{"fast": r["fast"], "slow": r["slow"],
                                            "rsi_ceiling": r["rsi_ceiling"], "stop": r["stop"],
                                            "vol_target": r["vol_target"], "adx_floor": r["adx_floor"],
                                            "calmar_is": r["calmar"],
                                            "maxdd_is": r["max_drawdown"]} for r in top_is_noextra],
               "no_stop_no_vol_oos": oos_rows_noextra,
               "buy_hold_oos": stats(oos_df, bh_eq, bh_ret)}
    else:
        oos = None

    result = {
        "params": {"fasts": fasts, "slows": slows, "rsi_ceilings": ceilings, "cost": args.cost,
                   "sector": sector_name, "sector_ma": args.sector_ma, "vol_floor": args.vol_floor,
                   "trend": args.trend, "extra": args.extra_csv.split("/")[-1] if args.extra_csv else None,
                   "extra_ma": args.extra_ma, "turnover_range": turnover_range,
                   "close_vwap": args.close_vwap, "vol_ratio_min": args.vol_ratio_min,
                   "execution": args.execution, "retrace_gap": args.retrace_gap,
                   "sector_rsi_max": args.sector_rsi_max},
        "top10_calmar": top,
        "best_return_by_maxdd_bucket": {
            f"le_{int(abs(level)*100)}pct": _clean(next(
                (r for _, r in res_df[res_df["max_drawdown"] >= level].iterrows()), None))
            for level in (-0.30, -0.40, -0.45, -0.50, -0.55, -0.60, -0.65)
        },
        "walk_forward": oos,
        "note": "全样本选参存在过拟合风险；样本外结果仅供稳健性参考。",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
