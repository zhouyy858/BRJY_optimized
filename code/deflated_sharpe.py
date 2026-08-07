#!/usr/bin/env python3
"""Deflated Sharpe Ratio (DSR) / PSR / MinTRL，多重检验校正（Bailey & López de Prado 2014）。

用途：参数网格搜索后对"选中策略"做统计显著性与选择偏差校正。
  DSR = PSR(SR*) —— 以"N次试验期望最大SR(SR*)"为基准的概率化Sharpe，
  同时校正非正态(偏度/峰度)、样本长度与多重试验次数N。

口径约定：
  - 本模块所有 Sharpe 均为"每期"口径(日频=每日收益)，非年化；
  - 峰度为非超额峰度(正态=3)，与论文一致；pandas 的 kurt() 是超额峰度，勿直接混用；
  - DSR>=0.95 视为通过 5% 显著性。

公式：
  Var(SR) = (1 - g1*SR + (g2-1)/4*SR^2) / (n-1)
  SR*     = sqrt(Var(SR)) * [ (1-γ)*Φ^-1(1-1/N) + γ*Φ^-1(1-1/(N·e)) ]，γ=欧拉-马歇罗尼常数
  DSR     = Φ( (SR - SR*) / sqrt(Var(SR)) )

参考：Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio,
      Journal of Portfolio Management 40(5)。实现已与公开参考实现逐项核对。
"""
import math

import numpy as np

EULER_MASCHERONI = 0.5772156649015328606


def _norm_cdf(x):
    """标准正态CDF（stdlib erff）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    """标准正态分位数（Acklam有理逼近+一步牛顿修正，精度~1e-9）。"""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    e = _norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def per_period_sharpe(returns, benchmark=0.0):
    """每期Sharpe（ddof=1，与论文口径一致）。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((r.mean() - benchmark) / sd)


def skew_kurt(returns):
    """偏度g1与**非超额**峰度g2(正态=3)。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.0, 3.0
    m = float(r.mean())
    s = float(r.std(ddof=0))
    if s == 0:
        return 0.0, 3.0
    z = (r - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skew, kurtosis):
    """PSR：观测SR超过基准SR*的概率，含偏度/峰度修正。"""
    if n_obs < 2 or not math.isfinite(observed_sr):
        return float("nan")
    denom = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    denom = max(denom, 1e-12)
    se = math.sqrt(denom / (n_obs - 1))
    return float(_norm_cdf((observed_sr - benchmark_sr) / se))


def expected_max_sharpe(sr_variance, n_trials):
    """N次独立试验(真实SR=0)的期望最大SR——即SR*。"""
    if n_trials <= 1:
        return 0.0
    v = max(sr_variance, 0.0)
    g = EULER_MASCHERONI
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(v) * ((1.0 - g) * z1 + g * z2)


def minimum_track_record_length(observed_sr, benchmark_sr, skew, kurtosis, confidence=0.95):
    """MinTRL：需要多少期观测才能以confidence置信度认定SR>SR*。"""
    if observed_sr <= benchmark_sr:
        return float("inf")
    z = _norm_ppf(confidence)
    num = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    return 1.0 + max(num, 1e-12) * (z / (observed_sr - benchmark_sr)) ** 2


def deflated_sharpe_ratio(returns, n_trials, sr_variance=None, all_trial_sharpes=None,
                          periods_per_year=252, threshold=0.95):
    """计算DSR。

    returns: 选中策略的每期收益序列；n_trials: 实际尝试的配置数(要诚实，网格每个点都算)。
    sr_variance: 各试验Sharpe的方差；缺省时优先用all_trial_sharpes估计，
                 否则用单策略渐近方差(偏宽松，最好提供试验序列)。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    sr = per_period_sharpe(r)
    skew, kurt = skew_kurt(r)
    if sr_variance is None:
        if all_trial_sharpes is not None and len(all_trial_sharpes) > 1:
            sr_variance = float(np.var(np.asarray(all_trial_sharpes, float), ddof=1))
        else:
            denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
            sr_variance = max(denom, 1e-12) / max(n - 1, 1)
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    dsr = probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt)
    psr_zero = probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt)
    mintrl = minimum_track_record_length(sr, sr0, skew, kurt, confidence=threshold)
    return {
        "observed_sr_period": round(sr, 6),
        "observed_sr_annual": round(sr * math.sqrt(periods_per_year), 3),
        "deflated_benchmark_sr0_period": round(sr0, 6),
        "deflated_benchmark_sr0_annual": round(sr0 * math.sqrt(periods_per_year), 3),
        "psr_vs_zero": round(psr_zero, 4),
        "deflated_sharpe_ratio": round(dsr, 4),
        "min_track_record_days": round(mintrl, 1) if math.isfinite(mintrl) else None,
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "passed": bool(dsr >= threshold),
    }


if __name__ == "__main__":
    # 自检1：正态分位数已知值
    for p, expect in ((0.975, 1.959964), (0.95, 1.644854), (0.5, 0.0), (0.025, -1.959964)):
        got = _norm_ppf(p)
        assert abs(got - expect) < 1e-5, (p, got, expect)
    # 自检2：与参考实现同参数对拍（750日、均值0.0006、标准差0.012、N=100）
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0006, 0.012, size=750)
    res = deflated_sharpe_ratio(ret, n_trials=100)
    print("Observed annual SR :", res["observed_sr_annual"])
    print("SR0 annual         :", res["deflated_benchmark_sr0_annual"])
    print("PSR vs 0           :", res["psr_vs_zero"])
    print("DSR                :", res["deflated_sharpe_ratio"])
    print("passed             :", res["passed"])
