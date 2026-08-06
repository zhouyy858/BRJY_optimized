# 贝瑞基因 D3V 策略 · 每日信号自动更新

针对 A 股贝瑞基因（000710）的 D3V 趋势策略：每个交易日收盘后增量更新数据、计算信号，并通过 GitHub Actions 自动推送到本仓库。

> ⚠️ 本项目仅供技术研究与学习，**不构成任何投资建议**；策略历史表现不代表未来收益，实盘前请自行评估风险。

## 目录结构

```
.
├── code/                         # 全部 Python 代码
│   ├── daily_signal.py           # 每日信号主脚本：增量更新 6 标的数据 + 计算 D3V 信号
│   ├── download_data.py          # 全量历史日线下载（东财优先，新浪回退，前复权）
│   ├── analyze.py                # 技术指标分析：MA/RSI/MACD/ATR/量能/VWAP 等
│   ├── backtest.py               # 单策略回测（信号当日收盘计算、次日生效）
│   └── optimize.py               # 参数网格优化 + 样本外/扰动/压力校验
├── data/                         # 历史数据与信号输出（脚本自动维护）
│   ├── sz000710_daily_qfq.csv    # 贝瑞基因前复权日线（含换手率）
│   ├── sh512010_daily.csv        # 医药 ETF（板块门控）
│   ├── sh512170_daily.csv        # 医疗 ETF
│   ├── sz399006_daily.csv        # 创业板指
│   ├── sz399001_daily.csv        # 深证成指
│   ├── sh000001_daily.csv        # 上证指数
│   ├── last_signal.json          # 最新信号（含各门控明细）
│   ├── signal_log.csv            # 历史信号流水
│   ├── optimized_params.json     # 当前策略参数
│   └── daily_signal.log          # 运行日志
├── .github/workflows/daily.yml   # GitHub Actions：工作日 17:30（北京时间）自动执行
├── requirements.txt
├── run_daily.sh                  # 本地一键：更新数据 → 算信号 → 提交推送
└── README.md
```

## 策略规则（D3V，2026-08-06 重训定稿）

| 项目 | 规则 |
| --- | --- |
| 进场条件 | close>MA3 且 MA3>MA20 且 RSI<80 且 +DI>-DI 且 医药ETF>MA20 且 创业板>MA30 且 换手率 0.8%~10% |
| 仓位 | min(40% / 20 日年化波动率, 100%)，百分比仓位 |
| 成交假设 | 信号次日开盘成交；大跳空 ≥3% 等回撤（高开不追、低开不杀） |
| 出场 | 任一进场条件失效，次日开盘卖出 |
| 时间因果 | 指标只用当日及以前数据；持仓信号 shift(1) 次日生效，无未来函数 |

### 历史回测（retrace 口径：次日开盘+大跳空≥3%等回撤，成本 0.1%/笔，可复现）

| 区间 | 策略收益 | 最大回撤 | Calmar | 买入持有 |
| --- | --- | --- | --- | --- |
| 2017+ | +530% | -14.3% | 1.55 | -82.9% |
| 2021+ | +269% | -8.9% | 3.10 | — |
| 2013+ | +820% | -14.3% | 1.44 | — |

> 参数经 2015–2018 样本内选择、2019 后样本外验证并做邻域扰动检查；回测未考虑涨跌停无法成交、停牌等现实约束；历史表现不代表未来。

## 快速开始

```bash
pip install -r requirements.txt

# 每日信号（增量更新数据并计算）
python3 code/daily_signal.py

# 只重算信号，不更新数据
python3 code/daily_signal.py --no-update

# 本地一键：更新数据 + 算信号 + 提交推送 GitHub
./run_daily.sh
```

## 复现回测

```bash
python3 code/optimize.py data/sz000710_daily_qfq.csv \
  --fasts 3 --slows 20 --rsi-ceiling 80 --stops none --vol-target 0.4 \
  --adx 0 --sector-csv data/sh512010_daily.csv --sector-ma 20 \
  --extra-csv data/sz399006_daily.csv --extra-ma 30 --trend di \
  --turnover-range 0.008,0.10 --execution retrace --retrace-gap 0.03 \
  --cost 0.001 --split 2017-01-01
```

## 数据说明

- 个股：东财 `stock_zh_a_hist` 优先，失败回退新浪 `stock_zh_a_daily`，前复权，含换手率
- 指数：`index_zh_a_hist` / `stock_zh_index_daily`
- ETF：`fund_etf_hist_sina`
- 更新策略：增量拉取近 15 日及当天，按日期去重合并，避免覆盖历史

## 自动更新

- GitHub Actions：工作日 UTC 09:30（北京时间 17:30）自动运行，提交 `data/` 与 `code/` 变更
- 本地定时（可选）：`crontab -e` 添加 `35 15 * * 1-5 cd /path/to/repo && ./run_daily.sh`

## 免责声明

本项目仅供技术研究，不构成投资建议；策略历史表现不代表未来收益，实盘前请自行评估风险。
