#!/usr/bin/env python3
"""点时序技术指标：全部指标仅使用截至当日的数据滚动计算，杜绝未来函数。"""
import argparse
import json

import numpy as np
import pandas as pd


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def main():
    ap = argparse.ArgumentParser(description="计算点时序技术指标并输出JSON")
    ap.add_argument("csv", help="download_data.py 输出的CSV")
    ap.add_argument("--out", help="可选JSON输出路径，缺省输出到stdout")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"]

    ma = {f"ma{n}": round(close.rolling(n).mean().iloc[-1], 3) for n in (5, 10, 20, 60, 120, 250)}
    dif = ema(close, 12) - ema(close, 26)
    dea = ema(dif, 9)
    macd_hist = (dif - dea) * 2

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    pct = round((last["close"] / prev["close"] - 1) * 100, 2)
    rsi14 = rsi(close).iloc[-1]
    atr14 = atr(df).iloc[-1]
    vol_5avg = df["volume"].rolling(5).mean().iloc[-1]
    vol_20avg = df["volume"].rolling(20).mean().iloc[-1]
    vwap = df["amount"] / df["volume"].replace(0, np.nan)
    turnover = df["turnover"].ffill()
    obv = (df["volume"] * np.sign(df["close"].diff().fillna(0))).cumsum()
    ytd_open = df.loc[df["date"].dt.year == last["date"].year, "close"].iloc[0]
    ytd = close.iloc[-1] / ytd_open - 1

    ma20, ma60 = ma["ma20"], ma["ma60"]
    if close.iloc[-1] > ma20 > ma60:
        trend = "多头排列(close>MA20>MA60)"
    elif close.iloc[-1] < ma20 < ma60:
        trend = "空头排列(close<MA20<MA60)"
    else:
        trend = "震荡交织(均线未同向排列)"
    if rsi14 >= 70:
        zone = "超买区"
    elif rsi14 <= 30:
        zone = "超卖区"
    else:
        zone = "中性区"

    result = {
        "as_of": str(last["date"].date()),
        "rows": int(len(df)),
        "last_close": float(last["close"]),
        "prev_close": float(prev["close"]),
        "day_pct": pct,
        "ma": ma,
        "rsi14": round(float(rsi14), 2),
        "rsi_zone": zone,
        "macd": {
            "dif": round(float(dif.iloc[-1]), 4),
            "dea": round(float(dea.iloc[-1]), 4),
            "hist": round(float(macd_hist.iloc[-1]), 4),
        },
        "atr14": round(float(atr14), 3),
        "range_20d": [float(df["low"].tail(20).min()), float(df["high"].tail(20).max())],
        "range_60d": [float(df["low"].tail(60).min()), float(df["high"].tail(60).max())],
        "volume": {
            "last": float(last["volume"]),
            "amount": float(last["amount"]),
            "vwap": round(float(vwap.iloc[-1]), 3),
            "close_vs_vwap_pct": round((float(last["close"]) / float(vwap.iloc[-1]) - 1) * 100, 2),
            "turnover_pct": round(float(turnover.iloc[-1]) * 100, 3),
            "volume_ratio_20d": round(float(last["volume"] / vol_20avg), 2) if vol_20avg else None,
            "vs_5d_avg": round(float(last["volume"] / vol_5avg), 2) if vol_5avg else None,
            "ma5_volume": round(float(vol_5avg), 0) if not pd.isna(vol_5avg) else None,
            "ma20_volume": round(float(vol_20avg), 0) if not pd.isna(vol_20avg) else None,
            "volume_trend_up": bool(vol_5avg > vol_20avg) if not pd.isna(vol_20avg) else None,
            "obv_above_20d": (bool(obv.iloc[-1] > (m20 := obv.rolling(20).mean().iloc[-1]))
                              if pd.notna(m20 := obv.rolling(20).mean().iloc[-1]) else None),
        },
        "ytd_pct": round(float(ytd) * 100, 2),
        "trend_text": trend,
        "last_5_bars": [
            {
                "date": str(r["date"].date()),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for _, r in df.tail(5).iterrows()
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
