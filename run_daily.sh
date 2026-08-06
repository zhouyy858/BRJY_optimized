#!/usr/bin/env bash
# 每日收盘后运行：更新数据+计算D2V信号+提交推送GitHub
# 本地定时可加 cron：35 15 * * 1-5 cd /Users/mac/代码/贝瑞基因 && ./run_daily.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 code/daily_signal.py
git add data code run_daily.sh README.md requirements.txt
git diff --cached --quiet && exit 0
git commit -m "daily: $(date +%F) 数据与信号更新"
git push
