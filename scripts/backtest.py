#!/usr/bin/env python3
"""点时序回测：均线趋势信号当日收盘计算、次日生效，杜绝未来函数；输出收益与回撤统计。"""
import argparse
import json

import pandas as pd


def max_drawdown(equity):
    growth = 1 + equity
    peak = growth.cummax()
    dd = growth / peak - 1
    return float(dd.min()), int(dd.idxmin())


def main():
    ap = argparse.ArgumentParser(description="MA趋势信号回测(收盘信号次日生效)")
    ap.add_argument("csv", help="download_data.py 输出的CSV")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=60)
    ap.add_argument("--out", help="可选JSON输出路径")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    close = df["close"]
    ret = close.pct_change()

    ma_fast = close.rolling(args.fast).mean()
    ma_slow = close.rolling(args.slow).mean()
    # 当日收盘信号：收盘价 > 快线 且 快线 > 慢线；持仓自次一交易日生效
    signal = ((close > ma_fast) & (ma_fast > ma_slow)).astype(int)
    position = signal.shift(1).fillna(0)
    strat_ret = position * ret

    start_idx = df.index[signal.notna() & signal.eq(1) | signal.notna() & signal.eq(0)].min()
    valid = df.index >= start_idx
    strat_equity = (1 + strat_ret[valid]).cumprod() - 1
    bh_equity = (1 + ret[valid]).cumprod() - 1

    def stats(equity, rets):
        total = float(equity.iloc[-1])
        n = int(equity.shape[0])
        ann = (1 + total) ** (252.0 / n) - 1 if n > 0 and total > -1 else -1.0
        mdd, mdd_idx = max_drawdown(equity)
        growth = 1 + equity
        cur_dd = float(growth.iloc[-1] / growth.cummax().iloc[-1] - 1)
        vol = float(rets.std(ddof=0) * (252 ** 0.5)) if n > 1 else 0.0
        return {
            "total_return": round(total, 4),
            "annual_return": round(ann, 4),
            "max_drawdown": round(mdd, 4),
            "max_drawdown_date": str(df.loc[valid, "date"].iloc[mdd_idx].date()),
            "current_drawdown": round(cur_dd, 4),
            "annual_vol": round(vol, 4),
        }

    trades = []
    entry = None
    for i in df.index[valid]:
        if position[i] == 1 and entry is None:
            entry = i
        elif position[i] == 0 and entry is not None:
            trades.append(float((1 + strat_ret.loc[entry + 1:i + 1]).prod() - 1))
            entry = None
    wins = sum(1 for t in trades if t > 0)

    result = {
        "params": {"fast": args.fast, "slow": args.slow},
        "backtest_range": [str(df.loc[valid, "date"].iloc[0].date()),
                           str(df.loc[valid, "date"].iloc[-1].date())],
        "strategy": stats(strat_equity, strat_ret[valid]),
        "buy_and_hold": stats(bh_equity, ret[valid]),
        "trades": {"count": len(trades), "win_rate": round(wins / len(trades), 4) if trades else None},
        "current_signal": {
            "as_of": str(df["date"].iloc[-1].date()),
            "long": int(position.iloc[-1]),
            "close": float(close.iloc[-1]),
            "ma_fast": round(float(ma_fast.iloc[-1]), 3),
            "ma_slow": round(float(ma_slow.iloc[-1]), 3),
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
