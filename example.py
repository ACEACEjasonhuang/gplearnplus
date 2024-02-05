# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :gplearn
# @File     :example.py
# @Date     :2023/3/31 0013 17:37
# @Author   :Junzhe Huang
# @Email    :acejasonhuang@163.com
# @Software :PyCharm
-------------------------------------------------
"""
#####
# 目录
# 1. ALL FUNCTION 全局函数
# 2. TIME SERIES FUNCTION 一般时间序列函数
# 3. TA FUNCTION 技术指标函数
# 4. SECTION FUNCTION 截面函数
# 5. SECTION GROUPBY FUNCTION 截面分类聚合函数
#
#
###
import numpy as np
from typing import Any
import numba as nb
from copy import copy
from numba import jit
from gplearnplus import functions
from functools import wraps
from functions import _groupby


def no_numpy_warning(func):
    @wraps(func)
    def warp(*args, **kwargs):
        with np.errstate(all='ignore'):
            _res = func(*args, **kwargs)
            return _res
    return warp

@nb.jit(nopython=True)
def handle_nan(X):
    # 前值填充
    X = np.copy(X)
    _temp = np.nan
    na_len = 0
    for i in range(len(X)):
        if np.isnan(X[i]):
            X[i] = _temp
            na_len += 1
        else:
            _temp = X[i]
    return X, na_len

#### ALL FUNCTION #####

@jit(nopython=True)
def _combine(X, Y):
    p1 = 15485863
    p2 = 32416190071
    p3 = 100000007
    return np.mod(X * p1 + Y * p2, p3)

combine = functions.make_function(function=_combine, name='combine', arity=2, return_type='category',
                                  param_type=[{'vector': {'category': (None, None)}},
                                              {'vector': {'category': (None, None)}}])

#### TIME SERIES FUNCTION #####

@jit(nopython=True)
def _delay(X, d):
    res = np.empty_like(X)
    res.fill(np.nan)
    end = len(X) - d
    for i in range(d, len(X)):
        res[i] = X[i - d]
    return res

delay = functions.make_function(function=_delay, name='delay', arity=2, function_type='time_series',
                                param_type=[{'vector': {'number': (None, None)}},
                                            {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _delta(X, d):
    res = np.empty_like(X)
    res.fill(np.nan)
    end = len(X) - d
    for i in range(d, len(X)):
        res[i] = X[i] - X[i - d]
    return res

delta = functions.make_function(function=_delta, name='delta', arity=2, function_type='time_series',
                                param_type=[{'vector': {'number': (None, None)}},
                                            {'scalar': {'int':(3, 30)}}])
@jit(nopython=True)
def _ts_min(X, d):
    d = len(X) - 1 if d >= len(X) else d
    shape = (X.size - d + 1, d)
    res = np.empty(X.size, dtype=X.dtype)
    res.fill(np.nan)
    for i in range(len(X) - d + 1):
        res[i + d - 1] = np.nanmin(X[i:i + d])
    return res

ts_min = functions.make_function(function=_ts_min, name='ts_min', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_max(X, d):
    d = len(X) - 1 if d >= len(X) else d
    shape = (X.size - d + 1, d)
    res = np.empty(X.size, dtype=X.dtype)
    res.fill(np.nan)
    for i in range(len(X) - d + 1):
        res[i + d - 1] = np.nanmax(X[i:i + d])
    return res

ts_max = functions.make_function(function=_ts_max, name='ts_max', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_argmax(X, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.empty(len(X), dtype=np.float64)
    res[:d - 1] = np.nan
    for i in range(len(X) - d + 1):
        res[i + d - 1] = np.argmax(X[i:i + d])
    return res

ts_argmax = functions.make_function(function=_ts_argmax, name='ts_argmax', arity=2, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_argmin(X, d):
    n = len(X)
    d = n - 1 if d >= n else d
    res = np.full(n, np.nan)
    for i in range(n - d + 1):
        res[i + d - 1] = np.argmax(X[i:i + d])
    return res
ts_argmin = functions.make_function(function=_ts_argmin, name='ts_argmax', arity=2, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}},
                                                {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_rank(X, d):
    n = len(X)
    d = n - 1 if d >= n else d
    res = np.full(n, np.nan)
    for i in range(n - d + 1):
        rank = np.argsort(X[i:i + d]).argsort()[-1] + 1
        res[i + d - 1] = rank / d
    return res

ts_rank = functions.make_function(function=_ts_rank, name='ts_rank', arity=2, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_sum(X, d):
    n = len(X)
    d = n - 1 if d >= n else d
    res = np.full(n, np.nan)
    cumsum = np.nancumsum(X)
    res[d - 1:n] = cumsum[d - 1:] - cumsum[:-d]
    return res

ts_sum = functions.make_function(function=_ts_sum, name='ts_sum', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_stddev(X, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.empty(len(X))
    res[:] = np.nan
    for i in range(d - 1, len(X)):
        res[i] = np.nanstd(X[i - d + 1:i + 1])
    return res

ts_stddev = functions.make_function(function=_ts_stddev, name='ts_stddev', arity=2, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_corr(X, Y, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.empty(len(X))
    res[:d-1] = np.nan
    for i in range(len(X) - d + 1):
        X_ = X[i:i+d]
        Y_ = Y[i:i+d]
        X_ = X_[~(np.isnan(X_) | np.isnan(Y_))]
        Y_ = Y_[~(np.isnan(X_) | np.isnan(Y_))]
        if len(X_) <= 2:
            res[i+d-1] = np.nan
        else:
            res[i+d-1] = np.corrcoef(X_, Y_)[0][1]
    return res

ts_corr = functions.make_function(function=_ts_corr, name='ts_corr', arity=3, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'vector': {'number': (None, None)}},
                                              {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_mean(X, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.full(len(X), np.nan)
    s = np.sum(X[:d])
    for i in range(d - 1, len(X)):
        res[i] = s / d
        s += X[i + 1] - X[i - d + 1]
    return res

ts_mean = functions.make_function(function=_ts_mean, name='ts_mean', arity=2,
                                  function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _ts_neutralize(X, d):
    N = len(X)
    d = len(X) - 1 if d >= len(X) else d
    mov_mean = np.empty(N - d + 1)
    mov_std = np.empty(N - d + 1)
    res = np.empty(N)

    for i in nb.prange(N - d + 1):
        mov_mean[i] = np.mean(X[i:i + d])
        mov_std[i] = np.sqrt(np.mean((X[i:i + d] - mov_mean[i]) ** 2))
        mov_std[i] = mov_std[i] if mov_std[i] > 0.001 else 0.001

    for i in nb.prange(N):
        if i < d - 1:
            res[i] = np.nan
        else:
            res[i] = (X[i] - mov_mean[i - d + 1]) / mov_std[i - d + 1]

    return res

ts_neutralize = functions.make_function(function=_ts_neutralize, name='ts_neutralize', arity=2,
                                        function_type='time_series',
                                        param_type=[{'vector': {'number': (None, None)}},
                                                    {'scalar': {'int':(3, 30)}}])

@nb.jit(nopython=True)
def _ts_freq(X, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.empty(len(X), dtype=np.float64)
    res[:d - 1] = np.nan
    for i in range(d - 1, len(X)):
        subarr = X[i - d + 1:i + 1]
        res[i] = sum(subarr == X[i])
    return res

ts_freq = functions.make_function(function=_ts_freq, name='ts_freq', arity=2,
                                  function_type='time_series',
                                  param_type=[{'vector': {'category': (None, None)}},
                                              {'scalar': {'int':(3, 30)}}])

#### TIME SERIES TA FUNCTION ####

@nb.jit(nopython=True)
def _EMA(X, d):
    d = len(X) - 1 if d >= len(X) else d
    X, _l = handle_nan(X)
    X = X[_l:]
    if len(X) < d:
        return np.array([np.nan] * (len(X) + _l))
    kt = 2 / (d + 1)
    pre_ma = np.mean(X[:d])
    __res = np.array([np.nan] * (len(X) + _l))
    __res[_l + d - 1] = pre_ma
    for i in range(d, len(X)):
        pre_ma += (X[i] - pre_ma) * kt
        __res[_l + i] = pre_ma
    return __res

EMA = functions.make_function(function=_EMA, name='EMA', arity=2, function_type='time_series',
                              param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _DEMA(X, d):
    d = d if len(X) > 2 * d - 2 else len(X) // 2 - 1
    _ema = _EMA(X, d)
    _eema = _EMA(_ema, d)
    __res = 2 * _ema - _eema
    return __res

DEMA = functions.make_function(function=_DEMA, name='DEMA', arity=2, function_type='time_series',
                               param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _MA(X, d):
    d = len(X) - 1 if d >= len(X) else d
    X, _l = handle_nan(X)
    X = X[_l:]
    if len(X) < d:
        return np.array([np.nan] * (len(X) + _l))
    __res = [np.nan] * (_l + d - 1) + [np.mean(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)

MA = functions.make_function(function=_MA, name='MA', arity=2, function_type='time_series',
                             param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@jit(nopython=True)
def _KAMA(X, d):
    d = len(X) - 1 if d >= len(X) else d
    X, _l = handle_nan(X)
    X = X[_l:]
    if len(X) < d:
        return np.array([np.nan] * (len(X) + _l))
    _af = 2 / (2 + 1)
    _as = 2 / (30 + 1)
    __res = np.array([np.nan] * (len(X) + _l))
    for i in range(d, len(X)):
        period_roc = X[i] - X[i - d]
        sum_roc = np.sum(np.abs(np.diff(X[i - d: i + 1])))
        _er = 1.0 if ((period_roc >= sum_roc) or (sum_roc == 0)) else abs(period_roc / sum_roc)
        _at = (_er * (_af - _as) + _as) ** 2
        __res[_l + i] = _at * X[i] + (1 - _at) * (__res[_l + i - 1] if i != d else X[i - 1])
    return __res

KAMA = functions.make_function(function=_KAMA, name='KAMA', arity=2, function_type='time_series',
                               param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@nb.jit(nopython=True)
def _MIDPOINT(X, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.empty(len(X))
    res[:] = np.nan
    for i in range(d - 1, len(X)):
        res[i] = (np.nanmax(X[i-d+1:i+1]) + np.nanmin(X[i-d+1:i+1])) / 2
    return res

MIDPOINT = functions.make_function(function=_MIDPOINT, name='MIDPOINT', arity=2, function_type='time_series',
                                   param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

@nb.jit(nopython=True)
def _BETA(X, Y, d):
    d = len(X) - 1 if d >= len(X) else d
    res = np.full(len(X), np.nan)
    for i in range(d - 1, len(X)):
        X_slice = X[i - d + 1: i + 1]
        Y_slice = Y[i - d + 1: i + 1]
        X_mean = np.mean(X_slice)
        Y_mean = np.mean(Y_slice)
        numerator = np.sum((X_slice - X_mean) * (Y_slice - Y_mean))
        denominator = np.sum((X_slice - X_mean) ** 2)
        denominator = denominator if denominator > 0.001 else 0.001
        res[i] = numerator / denominator
    return res

BETA = functions.make_function(function=_BETA, name='BETA', arity=3, function_type='time_series',
                               param_type=[{'vector': {'number': (None, None)}},
                                           {'vector': {'number': (None, None)}},
                                           {'scalar': {'int':(3, 30)}}])

@nb.jit(nopython=True)
def _LINEARREG_SLOPE(X, d):
    d = len(X) - 1 if d >= len(X) else d
    Y = np.arange(d)
    res = np.full(len(X), np.nan)
    for i in range(d - 1, len(X)):
        X_slice = X[i - d + 1: i + 1]
        Y_slice = Y[:len(X_slice)]
        X_mean = np.mean(X_slice)
        Y_mean = np.mean(Y_slice)
        numerator = np.sum((X_slice - X_mean) * (Y_slice - Y_mean))
        denominator = np.sum((X_slice - X_mean) ** 2)
        denominator = denominator if denominator > 0.001 else 0.001
        res[i] = numerator / denominator
    return res

LINEARREG_SLOPE = functions.make_function(function=_LINEARREG_SLOPE, name='LINEARREG_SLOPE', arity=2,
                                          function_type='time_series',
                                          param_type=[{'vector': {'number': (None, None)}},
                                                      {'scalar': {'int':(3, 30)}}])

@nb.jit(nopython=True)
def _LINEARREG_ANGLE(X, d):
    d = len(X) - 1 if d >= len(X) else d
    Y = np.arange(d)
    res = np.full(len(X), np.nan)
    for i in range(d - 1, len(X)):
        X_slice = X[i - d + 1: i + 1]
        Y_slice = Y[:len(X_slice)]
        X_mean = np.mean(X_slice)
        Y_mean = np.mean(Y_slice)
        numerator = np.sum((X_slice - X_mean) * (Y_slice - Y_mean))
        denominator = np.sum((X_slice - X_mean) ** 2)
        denominator = denominator if denominator > 0.001 else 0.001
        res[i] = np.arctan(numerator / denominator) * (180.0 / np.pi)
    return res

LINEARREG_ANGLE = functions.make_function(function=_LINEARREG_ANGLE, name='LINEARREG_ANGLE', arity=2,
                                          function_type='time_series',
                                          param_type=[{'vector': {'number': (None, None)}},
                                                      {'scalar': {'int':(3, 30)}}])

@nb.jit(nopython=True)
def _LINEARREG_INTERCEPT(X, d):
    d = len(X) - 1 if d >= len(X) else d
    Y = np.arange(d)
    res = np.full(len(X), np.nan)
    for i in range(d - 1, len(X)):
        X_slice = X[i - d + 1: i + 1]
        Y_slice = Y[:len(X_slice)]
        X_mean = np.mean(X_slice)
        Y_mean = np.mean(Y_slice)
        numerator = np.sum((X_slice - X_mean) * (Y_slice - Y_mean))
        denominator = np.sum((X_slice - X_mean) ** 2)
        denominator = denominator if denominator > 0.001 else 0.001
        _temp = np.arctan(numerator / denominator) * (180.0 / np.pi)
        res[i] = np.sum(X_slice) - _temp * np.sum(Y_slice)
    return res

LINEARREG_INTERCEPT = functions.make_function(function=_LINEARREG_INTERCEPT, name='LINEARREG_INTERCEPT',
                                              arity=2, function_type='time_series',
                                              param_type=[{'vector': {'number': (None, None)}},
                                                          {'scalar': {'int':(3, 30)}}])

#### SECTION FUNCTION ####

@nb.jit(nopython=True)
def _MAX_SECTION(X: np.ndarray) -> np.ndarray:
    return np.full_like(X, np.max(X))

sec_max = functions.make_function(function=_MAX_SECTION, name='sec_max', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])

@nb.jit(nopython=True)
def _MIN_SECTION(X):
    return np.full_like(X, np.min(X))

sec_min = functions.make_function(function=_MIN_SECTION, name='sec_min', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])

@nb.jit(nopython=True)
def _MEAN_SECTION(X):
    return np.full_like(X, np.mean(X))

sec_mean = functions.make_function(function=_MEAN_SECTION, name='sec_mean', arity=1, function_type='section',
                                   param_type=[{'vector': {'number': (None, None)}}])

@nb.jit(nopython=True)
def _MEDIAN_SECTION(X):
    return np.full_like(X, np.median(X))

sec_median = functions.make_function(function=_MEDIAN_SECTION, name='sec_median', arity=1, function_type='section',
                                     param_type=[{'vector': {'number': (None, None)}}])

@nb.jit(nopython=True)
def _STD_SECTION(X):
    return np.full_like(X, np.std(X))

sec_std = functions.make_function(function=_STD_SECTION, name='sec_std', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])

@nb.jit(nopython=True)
def _RANK_SECTION(X):
    idx = np.argsort(X)
    rank = np.empty_like(idx)
    for i in range(len(X)):
        rank[idx[i]] = i
    return rank

sec_rank = functions.make_function(function=_RANK_SECTION, name='sec_rank', arity=1, function_type='section',
                                   param_type=[{'vector': {'number': (None, None)}}])

@nb.jit(nopython=True)
def _NEUTRALIZE_SECTION(X):
    mean = np.mean(X)
    std = np.std(X)
    if std <= 0.001:
        std = 0.001
    return (X - mean) / np.repeat(std, len(X))

sec_neutralize = functions.make_function(function=_NEUTRALIZE_SECTION, name='sec_neutralize', arity=1,
                                         function_type='section', param_type=[{'vector': {'number': (None, None)}}])

@no_numpy_warning
def _FREQ_SECTION(X):
    unique_values, counts = np.unique(X, return_counts=True)
    count_dict = dict(zip(unique_values, counts))
    vectorized_func = np.vectorize(lambda x: count_dict[x])
    return vectorized_func(X)

freq = functions.make_function(function=_FREQ_SECTION, name='freq', arity=1,
                               function_type='section', param_type=[{'vector': {'category': (None, None)}}])

@no_numpy_warning
def _CUT_EQUAL_DISTANCE(X, d):
    '''
    等距分组
    Parameters
    ----------
    X
    d

    Returns
    -------

    '''
    d = len(X) - 1 if d >= len(X) - 1 else d
    bins = [np.min(X) + i * (np.max(X) - np.min(X)) * 1.000001 / d for i in range(d + 1)]
    return np.digitize(X, bins)

cut_equal_distance = functions.make_function(function=_CUT_EQUAL_DISTANCE, name='cut_eq_dist', arity=2,
                                             function_type='section', return_type='category',
                                             param_type=[{'vector': {'number': (None, None)}},
                                                         {'scalar': {'int': (2, 30)}}])

@no_numpy_warning
def _CUT_EQUAL_AMOUNT(X, d):
    X_ = _RANK_SECTION(X)
    return _CUT_EQUAL_DISTANCE(X_, d)

cut_equal_amount = functions.make_function(function=_CUT_EQUAL_AMOUNT, name='cut_eq_amt', arity=2,
                                           function_type='section', return_type='category',
                                           param_type=[{'vector': {'number': (None, None)}},
                                                       {'scalar': {'int': (2, 30)}}])

@no_numpy_warning
def _GROUPBYTHENMAX(gbx, X):
    return _groupby(gbx, _MAX_SECTION, X)

groupby_max = functions.make_function(function=_GROUPBYTHENMAX, name='gb_max', arity=2, function_type='section',
                                      param_type=[{'vector': {'category': (None, None)}},
                                                  {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHENMIN(gbx, X):
    return _groupby(gbx, _MIN_SECTION, X)

groupby_min = functions.make_function(function=_GROUPBYTHENMIN, name='gb_min', arity=2, function_type='section',
                                      param_type=[{'vector': {'category': (None, None)}},
                                                  {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHENMEAN(gbx, X):
    return _groupby(gbx, _MEAN_SECTION, X)
groupby_mean = functions.make_function(function=_GROUPBYTHENMEAN, name='gb_mean', arity=2, function_type='section',
                                       param_type=[{'vector': {'category': (None, None)}},
                                                   {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHENMEDIAN(gbx, X):
    return _groupby(gbx, _MEDIAN_SECTION, X)
groupby_median = functions.make_function(function=_GROUPBYTHENMEDIAN, name='gb_median',
                                         arity=2, function_type='section',
                                         param_type=[{'vector': {'category': (None, None)}},
                                                     {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHENSTD(gbx, X):
    return _groupby(gbx, _STD_SECTION, X)
groupby_std = functions.make_function(function=_GROUPBYTHENSTD, name='gb_std', arity=2, function_type='section',
                                      param_type=[{'vector': {'category': (None, None)}},
                                                  {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHENRANK(gbx, X):
    return _groupby(gbx, _RANK_SECTION, X)
groupby_rank = functions.make_function(function=_GROUPBYTHENRANK, name='gb_rank', arity=2, function_type='section',
                                       param_type=[{'vector': {'category': (None, None)}},
                                                   {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHENNEUTRALIZE(gbx, X):
    return _groupby(gbx, _NEUTRALIZE_SECTION, X)
groupby_neutralize = functions.make_function(function=_GROUPBYTHENNEUTRALIZE, name='gb_neu', arity=2,
                                             function_type='section',
                                             param_type=[{'vector': {'category': (None, None)}},
                                                         {'vector': {'number': (None, None)}}])

@no_numpy_warning
def _GROUPBYTHEN_CUT_EQ_DIST(gbx, X, d):
    return _groupby(gbx, _CUT_EQUAL_DISTANCE, X, d=d)
groupby_cut_equal_distance = functions.make_function(function=_GROUPBYTHEN_CUT_EQ_DIST, name='gb_cut_eq_dist', arity=3,
                                                     function_type='section', return_type='category',
                                                     param_type=[{'vector': {'category': (None, None)}},
                                                                 {'vector': {'number': (None, None)}},
                                                                 {'scalar': {'int': (2, 30)}}])

@no_numpy_warning
def _GROUPBYTHEN_CUT_EQ_AMT(gbx, X, d):
    return _groupby(gbx, _CUT_EQUAL_AMOUNT, X, d=d)
groupby_cut_equal_amount = functions.make_function(function=_GROUPBYTHEN_CUT_EQ_AMT, name='gb_cut_eq_amt', arity=3,
                                                   function_type='section', return_type='category',
                                                   param_type=[{'vector': {'category': (None, None)}},
                                                               {'vector': {'number': (None, None)}},
                                                               {'scalar': {'int': (2, 30)}}])

@no_numpy_warning
def _GROUPBYTHENFREQ(gbx, X):
    return _groupby(gbx, _FREQ_SECTION, X)
groupby_freq = functions.make_function(function=_GROUPBYTHENFREQ, name='gb_freq', arity=2,
                                       function_type='section',
                                       param_type=[{'vector': {'category': (None, None)}},
                                                   {'vector': {'category': (None, None)}}])

__all__ = ['delay', 'delta', 'sec_max', 'sec_min', 'sec_median', 'ts_min', 'ts_max', 'ts_sum', 'ts_corr', 'ts_rank',
           'ts_stddev', 'ts_argmax', 'ts_argmin', 'ts_mean', 'EMA', 'DEMA', 'KAMA', 'MA', 'MIDPOINT',
           'BETA', 'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'LINEARREG_INTERCEPT', 'sec_std', 'sec_rank', 'sec_mean',
           'groupby_std', 'groupby_max', 'groupby_median', 'groupby_mean', 'groupby_rank', 'groupby_min',
           'ts_neutralize', 'sec_neutralize', 'groupby_neutralize', 'cut_equal_amount', 'cut_equal_distance',
           'groupby_cut_equal_amount', 'groupby_freq', 'groupby_cut_equal_distance', 'freq', 'ts_freq']

def test():
    a = np.random.uniform(0.9, 1.1, 30)
    b = np.random.uniform(0.9, 1.1, 30)
    c = np.random.randint(0, 2, size=30)
    print(groupby_cut_equal_distance(c,a,3))


if __name__ == "__main__":
    test()
