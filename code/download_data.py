#!/usr/bin/env python3
"""下载A股个股全量日线数据（前复权），优先东财、失败回退新浪源。"""
import argparse
import sys
import time

import requests
import akshare as ak


# macOS 系统级代理会被 Python 自动读取且可能处于不可用状态，统一绕过系统代理直连数据源。
def _direct_get(url, **kwargs):
    session = requests.Session()
    session.trust_env = False
    return session.get(url, **kwargs)


requests.get = _direct_get

COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}


def _try_fetch(fetch_fn, label, retries=3, backoff=2.0):
    last_err = None
    for i in range(retries):
        try:
            df = fetch_fn()
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            last_err = e
        if i < retries - 1:
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"{label} 连续 {retries} 次失败: {last_err}")


def main():
    ap = argparse.ArgumentParser(description="下载A股个股全量日线数据(前复权)")
    ap.add_argument("symbol", help="如 sz000710 / sh600000")
    ap.add_argument("out_csv", help="输出CSV路径")
    ap.add_argument("--start", default="19900101", help="起始日期 YYYYMMDD")
    ap.add_argument("--end", default="20991231", help="结束日期 YYYYMMDD")
    args = ap.parse_args()

    code = args.symbol[-6:]
    market = args.symbol[:2].lower()
    if market in ("sz", "sh"):
        sina_symbol = args.symbol
    else:
        sina_symbol = ("sz" if code[0] in ("0", "3") else "sh") + code

    def fetch_em():
        return ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=args.start, end_date=args.end, adjust="qfq",
        )

    def fetch_sina():
        return ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=args.start, end_date=args.end, adjust="qfq",
        )

    df = None
    try:
        df = _try_fetch(fetch_em, "东财")
        print(f"[ok] 东财源 {code} 获取 {len(df)} 行", file=sys.stderr)
    except Exception as e:
        print(f"[warn] 东财源失败，回退新浪: {e}", file=sys.stderr)
    if df is None:
        df = _try_fetch(fetch_sina, "新浪")

    df = df.rename(columns=COLUMN_MAP)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(
        f"[ok] 已写入 {args.out_csv}，{len(df)} 行，"
        f"区间 {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"
    )


if __name__ == "__main__":
    main()
