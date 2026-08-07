#!/usr/bin/env python3
"""滚动 walk-forward：每个折点用"截至当日"的历史数据在594点网格选参，下一段样本外验证。

口径：
  - 选参准则 = 样本内 Calmar 最高（与项目历史一致），平手取总收益高者；
  - IS/OOS 均用 retrace 成交口径、成本0.1%/边；结构过滤(+DI/医药ETF>MA20/创业板>MA30/换手0.8-10%)固定；
  - OOS 收益 = 全样本连续运行下切窗口净收益乘积（正确口径，无归一化近似）；
  - 聚合 = 各折 OOS 净收益按时间拼接，再对聚合路径做 DSR 多重试验校正。
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize import run_backtest, stats  # noqa: E402
from deflated_sharpe import per_period_sharpe, deflated_sharpe_ratio  # noqa: E402


def load_flags(csv, sector_csv, extra_csv, sector_ma, extra_ma):
    df = pd.read_csv(csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    sector_flag = None
    if sector_csv:
        etf = pd.read_csv(sector_csv, parse_dates=["date"]).sort_values("date")
        c = etf.set_index("date")["close"]
        sector_flag = c > c.rolling(sector_ma).mean()
    extra_flag = None
    if extra_csv:
        ex = pd.read_csv(extra_csv, parse_dates=["date"]).sort_values("date")
        c = ex.set_index("date")["close"]
        extra_flag = c > c.rolling(extra_ma).mean()
    return df, sector_flag, extra_flag


def grid(IS, args, sector_flag, extra_flag):
    """返回 [(params, stats, net)]，按 calmar 降序。"""
    rows = []
    for f, s, c, m, v in itertools.product(
            args.fasts, args.slows, args.ceilings, args.stops, args.vols):
        if f >= s:
            continue
        eq, net, pos = run_backtest(
            IS, f, s, c, m, v, 0, args.cost, sector_flag, args.vol_floor, "di",
            extra_flag, args.turnover_range, False, None, args.execution,
            args.retrace_gap, None, args.limit_block)
        st = stats(IS, eq, net)
        if st:
            rows.append(({"fast": f, "slow": s, "rsi_ceiling": c,
                          "stop": m, "vol_target": v}, st, net))
    rows.sort(key=lambda r: (r[1]["calmar"], r[1]["total_return"]), reverse=True)
    return rows


def oos_slice(df, eq, net, pos, start, end):
    m = (df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))
    r = net[m]
    tot = float((1 + r).prod() - 1)
    n = int(m.sum())
    ann = (1 + tot) ** (252.0 / n) - 1 if tot > -1 else -1.0
    g = (1 + r).cumprod()
    mdd = float((g / g.cummax() - 1).min())
    trades = int(pos[m].diff().abs().fillna(0).sum() / 2)
    return {"start": str(df.loc[m, "date"].iloc[0].date()),
            "end": str(df.loc[m, "date"].iloc[-1].date()),
            "days": n, "ret": round(tot, 4), "ann": round(ann, 4),
            "maxdd": round(mdd, 4),
            "calmar": round(ann / abs(mdd), 3) if mdd < 0 else None,
            "trades": trades, "sr_period": round(per_period_sharpe(r.values), 5)}


def main():
    ap = argparse.ArgumentParser(description="滚动walk-forward(逐年重训+样本外)")
    ap.add_argument("csv")
    ap.add_argument("--sector-csv")
    ap.add_argument("--sector-ma", type=int, default=20)
    ap.add_argument("--extra-csv")
    ap.add_argument("--extra-ma", type=int, default=30)
    ap.add_argument("--folds", default="2016-01-01,2017-01-01,2018-01-01,2019-01-01,2020-01-01,2021-01-01,2022-01-01,2023-01-01,2024-01-01,2025-01-01",
                    help="各折点日期(IS=起点~折点, OOS=折点~下折点)")
    ap.add_argument("--is-start", default="2013-01-01")
    ap.add_argument("--fasts", default="3,5,10,20,30")
    ap.add_argument("--slows", default="20,30,60,90,120")
    ap.add_argument("--ceilings", default="none,70,80")
    ap.add_argument("--stops", default="none,3,5")
    ap.add_argument("--vols", default="none,0.25,0.4")
    ap.add_argument("--execution", default="retrace")
    ap.add_argument("--retrace-gap", type=float, default=0.03)
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--vol-floor", type=float, default=0.0)
    ap.add_argument("--limit-block", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    args.fasts = [int(x) for x in args.fasts.split(",")]
    args.slows = [int(x) for x in args.slows.split(",")]
    args.ceilings = [None if x.lower() in ("none", "") else int(x) for x in args.ceilings.split(",")]
    args.stops = [None if x.lower() in ("none", "") else float(x) for x in args.stops.split(",")]
    args.vols = [None if x.lower() in ("none", "") else float(x) for x in args.vols.split(",")]
    args.turnover_range = (0.008, 0.10)
    bounds = [pd.Timestamp(x) for x in args.folds.split(",")]

    df, sector_flag, extra_flag = load_flags(args.csv, args.sector_csv, args.extra_csv,
                                             args.sector_ma, args.extra_ma)
    is_start = pd.Timestamp(args.is_start)
    folds = []
    trial_vars = []
    oos_nets = []
    for i, b in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else df["date"].max() + pd.Timedelta(days=1)
        is_df = df[(df["date"] >= is_start) & (df["date"] < b)].reset_index(drop=True)
        rows = grid(is_df, args, sector_flag, extra_flag)
        if not rows:
            continue
        best, best_st, _ = rows[0]
        srs = np.array([per_period_sharpe(r[2].values) for r in rows])
        srs = srs[np.isfinite(srs)]
        trial_vars.append(float(np.var(srs, ddof=1)) if srs.size > 1 else 0.0)
        eq, net, pos = run_backtest(
            df, best["fast"], best["slow"], best["rsi_ceiling"], best["stop"],
            best["vol_target"], 0, args.cost, sector_flag, args.vol_floor, "di",
            extra_flag, args.turnover_range, False, None, args.execution,
            args.retrace_gap, None, args.limit_block)
        o = oos_slice(df, eq, net, pos, b, end)
        o["is_window"] = f"{args.is_start}~{b.date()}"
        o["selected"] = best
        o["is_calmar"] = best_st["calmar"]
        o["is_ret"] = best_st["total_return"]
        o["is_maxdd"] = best_st["max_drawdown"]
        o["n_trials"] = int(len(srs))
        oos_nets.append(net[(df["date"] >= b) & (df["date"] < end)])
        folds.append(o)
        print(f"[{b.date()}] sel={best['fast']}/{best['slow']} rsi={best['rsi_ceiling']} "
              f"stop={best['stop']} vol={best['vol_target']} IS_calmar={best_st['calmar']} "
              f"OOS_ret={o['ret']} OOS_mdd={o['maxdd']}", flush=True)

    all_net = pd.concat(oos_nets)
    tot = float((1 + all_net).prod() - 1)
    n = int(all_net.shape[0])
    ann = (1 + tot) ** (252.0 / n) - 1 if tot > -1 else -1.0
    g = (1 + all_net).cumprod()
    mdd = float((g / g.cummax() - 1).min())
    sr_var = float(np.mean(trial_vars)) if trial_vars else None
    dsr594 = deflated_sharpe_ratio(all_net.values, n_trials=594, sr_variance=sr_var)
    dsr5940 = deflated_sharpe_ratio(all_net.values, n_trials=5940, sr_variance=sr_var)

    # 固定 D3V 对照（同窗口）
    w_start = bounds[0]
    kw = dict(fast=3, slow=20, rsi_ceiling=80, stop_mult=None, vol_target=0.4, adx_floor=0,
              cost=args.cost, sector_flag=sector_flag, trend_dir="di", extra_flag=extra_flag,
              turnover_range=args.turnover_range, execution=args.execution,
              retrace_gap=args.retrace_gap, limit_block=args.limit_block)
    eq, net, pos = run_backtest(df, **kw)
    fixed = oos_slice(df, eq, net, pos, w_start, df["date"].max() + pd.Timedelta(days=1))
    fixed.pop("selected", None)
    fixed.pop("is_window", None)

    result = {
        "folds": folds,
        "aggregate_wfa": {"ret": round(tot, 4), "ann": round(ann, 4), "maxdd": round(mdd, 4),
                          "calmar": round(ann / abs(mdd), 3) if mdd < 0 else None,
                          "days": n, "trades": sum(f["trades"] for f in folds)},
        "aggregate_dsr": {
            "n_trials_grid": 594, "n_trials_all_folds": 5940,
            "sr_annual": dsr594["observed_sr_annual"],
            "dsr_594": dsr594["deflated_sharpe_ratio"],
            "dsr_5940": dsr5940["deflated_sharpe_ratio"],
            "psr_vs_zero": dsr594["psr_vs_zero"],
            "min_track_record_days_594": dsr594["min_track_record_days"],
            "pooled_trial_sr_var": round(float(sr_var), 6) if sr_var else None},
        "fixed_d3v_same_window": fixed,
        "note": "OOS按全样本连续运行切片(携带折前持仓状态)；聚合DSR的N取单折网格594与全流程5940两种",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
