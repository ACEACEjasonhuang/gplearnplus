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
import numpy as np
from copy import copy
from gplearnplus import functions
from functools import wraps
from functions import _groupby, _protected_division


def no_numpy_warning(func):
    @wraps(func)
    def warp(*args, **kwargs):
        with np.errstate(all='ignore'):
            _res = func(*args, **kwargs)
            return _res
    return warp


def handle_nan(X):
    X = copy(X)
    _temp = np.nan
    for i, _var in enumerate(X):
        if np.isnan(_var):
            X[i] = _temp
        else:
            _temp = X[i]
    na_len = len(np.where(np.isnan(X))[0])
    return X, na_len


@no_numpy_warning
def _neg(X):
    return -X

@no_numpy_warning
def _combine(X, Y):
    p1 = 15485863
    p2 = 32416190071
    p3 = 100000007
    return np.mod(X * p1 + Y * p2, p3)


@no_numpy_warning
def _delay(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * d + list(X[:-d])
    return np.array(__res)


@no_numpy_warning
def _delta(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * d + list(X[:-d])
    __res = X - __res
    return np.array(__res)


@no_numpy_warning
def _ts_min(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.nanmin(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_max(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.nanmax(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_argmax(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.argmax(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_argmin(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.argmin(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_rank(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [(np.argsort(X[i:i + d])[-1] / d) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_sum(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.nansum(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_stddev(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.nansum(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


def _corrcoef_plus(X, Y):
    assert len(X) == len(Y)
    X_ = X[~(np.isnan(X) | np.isnan(Y))]
    Y_ = Y[~(np.isnan(X) | np.isnan(Y))]
    if len(X_) <= 2:
        return np.nan
    else:
        return np.corrcoef(X_, Y_)[0][1]


@no_numpy_warning
def _ts_corr(X, Y, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    assert len(X) == len(Y)
    __res = [np.nan] * (d - 1) + [_corrcoef_plus(X[i:i + d], Y[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _ts_mean_return(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [np.mean(np.diff(X[i:i + d]) / X[i:i + d - 1]) for i in range(len(X) - d + 1)]
    return np.array(__res)

@no_numpy_warning
def _ts_neutralize(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [_protected_division(X - np.mean(X[i:i + d]), np.repeat(np.std(X[i:i + d])), d)
                                  for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _EMA(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
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
    return np.array(__res)


@no_numpy_warning
def _DEMA(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = d if len(X) > 2 * d - 2 else len(X) // 2 - 1
    _ema = _EMA(X, d)
    _eema = _EMA(_ema, d)
    __res = 2 * _ema - _eema
    return __res


@no_numpy_warning
def _MA(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    X, _l = handle_nan(X)
    X = X[_l:]
    if len(X) < d:
        return np.array([np.nan] * (len(X) + _l))
    __res = [np.nan] * (_l + d - 1) + [np.mean(X[i:i + d]) for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _KAMA(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
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
        sum_roc = sum(abs(np.diff(X[i - d: i + 1])))
        _er = 1.0 if ((period_roc >= sum_roc) or (sum_roc == 0)) else abs(period_roc / sum_roc)
        _at = (_er * (_af - _as) + _as) ** 2
        __res[_l + i] = _at * X[i] + (1 - _at) * (__res[_l + i - 1] if i != d else X[i - 1])
    return np.array(__res)


@no_numpy_warning
def _MIDPONIT(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    __res = [np.nan] * (d - 1) + [(np.nanmax(X[i:i + d]) + np.nanmin(X[i:i + d])) / 2
                                  for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _BETA(X, Y, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    assert len(X) == len(Y)
    X = np.diff(X) / X[:-1]
    Y = np.diff(Y) / Y[:-1]
    __res = [np.nan] * d + [_corrcoef_plus(X[i:i + d], Y[i:i + d])
                            * np.std(Y[i:i + d]) / np.std(X[i:i + d])
                            for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _LINEARREG_SLOPE(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    Y = np.arange(d)
    __res = [np.nan] * (d - 1) + [_corrcoef_plus(X[i:i + d], Y)
                                  * np.std(X[i:i + d]) / np.std(Y)
                                  for i in range(len(X) - d + 1)]
    return np.array(__res)


@no_numpy_warning
def _LINEARREG_ANGLE(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    Y = np.arange(d)
    __res = [np.nan] * (d - 1) + [np.arctan(_corrcoef_plus(X[i:i + d], Y)
                                  * np.std(X[i:i + d]) / np.std(Y)) * (180.0 / np.pi)
                                  for i in range(len(X) - d + 1)]
    return np.array(__res)



@no_numpy_warning
def _LINEARREG_INTERCEPT(X, d):
    # 处理广播情况
    if isinstance(d, (list, np.ndarray)):
        d = d[0]
    d = len(X) - 1 if d >= len(X) else d
    Y = np.arange(d)
    __res = [np.nan] * (d - 1) + [(np.sum(X[i:i + d]) - (_corrcoef_plus(X[i:i + d], Y)
                                  * np.std(X[i:i + d]) / np.std(Y)) * np.sum(Y)) / d
                                  for i in range(len(X) - d + 1)]
    return np.array(__res)

@no_numpy_warning
def _MAX_SECTION(X):
    __res = np.repeat(np.max(X), len(X))
    return __res

@no_numpy_warning
def _MIN_SECTION(X):
    __res = np.repeat(np.min(X), len(X))
    return __res

@no_numpy_warning
def _MEAN_SECTION(X):
    __res = np.repeat(np.min(X), len(X))
    return __res

@no_numpy_warning
def _MEDIAN_SECTION(X):
    __res = np.repeat(np.median(X), len(X))
    return __res

@no_numpy_warning
def _STD_SECTION(X):
    __res = np.repeat(np.std(X), len(X))
    return __res

@no_numpy_warning
def _RANK_SECTION(X):
    # 获取数组的排序索引
    idx = np.argsort(X)
    # 构建等级数组
    rank = np.empty_like(idx)
    rank[idx] = np.arange(len(arr))
    return rank

@no_numpy_warning
def _NEUTRALIZE_SECTION(X):
    return _protected_division(X - np.mean(X), np.repeat(np.std(X), len(X)))

@no_numpy_warning
def _GROUPBYTHENMAX(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _MAX_SECTION, X)

@no_numpy_warning
def _GROUPBYTHENMIN(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _MIN_SECTION, X)

@no_numpy_warning
def _GROUPBYTHENMEAN(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _MEAN_SECTION, X)

@no_numpy_warning
def _GROUPBYTHENMEDIAN(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _MEDIAN_SECTION, X)

@no_numpy_warning
def _GROUPBYTHENSTD(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _STD_SECTION, X)

@no_numpy_warning
def _GROUPBYTHENRANK(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _RANK_SECTION, X)

@no_numpy_warning
def _GROUPBYTHENNEUTRALIZE(X, gbx):
    # 处理广播情况
    return _groupby(gbx, _NEUTRALIZE_SECTION, X)

delay = functions.make_function(function=_delay, name='delay', arity=2, function_type='time_series',
                                param_type=[{'vector': {'number': (None, None), 'category': (None, None)}},
                                            {'scalar': {'int':(3, 30)}}])
delta = functions.make_function(function=_delta, name='delta', arity=2, function_type='time_series',
                                param_type=[{'vector': {'number': (None, None)}},
                                            {'scalar': {'int':(3, 30)}}])
combine = functions.make_function(function=_combine, name='combine', arity=2, return_type='category',
                                  param_type=[{'vector': {'category': (None, None)}},
                                              {'vector': {'category': (None, None)}}])

sec_max = functions.make_function(function=_MAX_SECTION, name='sec_max', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])
sec_min = functions.make_function(function=_MIN_SECTION, name='sec_min', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])
sec_mean = functions.make_function(function=_MEAN_SECTION, name='sec_mean', arity=1, function_type='section',
                                   param_type=[{'vector': {'number': (None, None)}}])
sec_median = functions.make_function(function=_MEDIAN_SECTION, name='sec_median', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])
sec_std = functions.make_function(function=_STD_SECTION, name='sec_std', arity=1, function_type='section',
                                  param_type=[{'vector': {'number': (None, None)}}])
sec_rank = functions.make_function(function=_RANK_SECTION, name='sec_rank', arity=1, function_type='section',
                                   param_type=[{'vector': {'number': (None, None)}}])
sec_neutralize = functions.make_function(function=_NEUTRALIZE_SECTION, name='sec_neutralize', arity=1,
                                         function_type='section', param_type=[{'vector': {'number': (None, None)}}])

groupby_max = functions.make_function(function=_GROUPBYTHENMAX, name='groupby_max', arity=2, function_type='section',
                                      param_type=[{'vector': {'number': (None, None)}},
                                                  {'category': {'number': (None, None)}}])
groupby_min = functions.make_function(function=_GROUPBYTHENMIN, name='groupby_min', arity=2, function_type='section',
                                      param_type=[{'vector': {'number': (None, None)}},
                                                  {'category': {'number': (None, None)}}])
groupby_mean = functions.make_function(function=_GROUPBYTHENMEAN, name='groupby_mean', arity=2, function_type='section',
                                       param_type=[{'vector': {'number': (None, None)}},
                                                   {'category': {'number': (None, None)}}])
groupby_median = functions.make_function(function=_GROUPBYTHENMEDIAN, name='groupby_median', arity=2, function_type='section',
                                         param_type=[{'vector': {'number': (None, None)}},
                                                     {'category': {'number': (None, None)}}])
groupby_std = functions.make_function(function=_GROUPBYTHENSTD, name='groupby_std', arity=2, function_type='section',
                                      param_type=[{'vector': {'number': (None, None)}},
                                                  {'category': {'number': (None, None)}}])
groupby_rank = functions.make_function(function=_GROUPBYTHENRANK, name='groupby_rank', arity=2, function_type='section',
                                       param_type=[{'vector': {'number': (None, None)}},
                                                   {'category': {'number': (None, None)}}])
groupby_neutralize = functions.make_function(function=_GROUPBYTHENNEUTRALIZE, name='groupby_neutralize', arity=2,
                                             function_type='section',
                                             param_type=[{'vector': {'number': (None, None)}},
                                                         {'category': {'number': (None, None)}}])

ts_min = functions.make_function(function=_ts_min, name='ts_min', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_max = functions.make_function(function=_ts_max, name='ts_max', arity=2, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_argmax = functions.make_function(function=_ts_argmax, name='ts_argmax', arity=2, function_type='time_series',
                                     param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_argmin = functions.make_function(function=_ts_argmin, name='ts_argmax', arity=2, function_type='time_series',
                                     param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_rank = functions.make_function(function=_ts_rank, name='ts_rank', arity=2, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_sum = functions.make_function(function=_ts_sum, name='ts_sum', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_stddev = functions.make_function(function=_ts_stddev, name='ts_stddev', arity=2, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_corr = functions.make_function(function=_ts_corr, name='ts_corr', arity=3, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'vector': {'number': (None, None)}},
                                              {'scalar': {'int':(3, 30)}}])
ts_mean_return = functions.make_function(function=_ts_mean_return, name='ts_mean_return', arity=2, function_type='time_series',
                                         param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
ts_neutralize = functions.make_function(function=_ts_neutralize, name='ts_neutralize', arity=2, function_type='time_series',
                                         param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])

EMA = functions.make_function(function=_EMA, name='EMA', arity=2, function_type='time_series',
                              param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
DEMA = functions.make_function(function=_DEMA, name='DEMA', arity=2, function_type='time_series',
                              param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
KAMA = functions.make_function(function=_DEMA, name='DEMA', arity=2, function_type='time_series',
                              param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
MA = functions.make_function(function=_MA, name='MA', arity=2, function_type='time_series',
                              param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
MIDPOINT = functions.make_function(function=_MA, name='MA', arity=2, function_type='time_series',
                              param_type=[{'vector': {'number': (None, None)}}, {'scalar': {'int':(3, 30)}}])
BETA = functions.make_function(function=_BETA, name='BETA', arity=3, function_type='time_series',
                               param_type=[{'vector': {'number': (None, None)}},
                                           {'vector': {'number': (None, None)}},
                                           {'scalar': {'int':(3, 30)}}])
LINEARREG_SLOPE = functions.make_function(function=_LINEARREG_SLOPE, name='LINEARREG_SLOPE', arity=2, function_type='time_series',
                                          param_type=[{'vector': {'number': (None, None)}},
                                                      {'scalar': {'int':(3, 30)}}])
LINEARREG_ANGLE = functions.make_function(function=_LINEARREG_ANGLE, name='LINEARREG_ANGLE', arity=2, function_type='time_series',
                                          param_type=[{'vector': {'number': (None, None)}},
                                                      {'scalar': {'int':(3, 30)}}])
LINEARREG_INTERCEPT = functions.make_function(function=_LINEARREG_INTERCEPT, name='LINEARREG_INTERCEPT', arity=2, function_type='time_series',
                                              param_type=[{'vector': {'number': (None, None)}},
                                                          {'scalar': {'int':(3, 30)}}])
__all__ = ['delay', 'delta', 'sec_max', 'sec_min', 'sec_median', 'ts_min', 'ts_max', 'ts_sum', 'ts_corr', 'ts_rank',
           'ts_stddev', 'ts_argmax', 'ts_argmin', 'ts_mean_return', 'EMA', 'DEMA', 'KAMA', 'MA', 'MIDPOINT',
           'BETA', 'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'LINEARREG_INTERCEPT', 'sec_std', 'sec_rank', 'sec_mean',
           'groupby_std', 'groupby_max', 'groupby_median', 'groupby_mean', 'groupby_rank', 'groupby_min',
           'ts_neutralize']


if __name__ == "__main__":
    a = np.random.uniform(0.9, 1.1, 30)
    b = np.random.uniform(0.9, 1.1, 30)

    a = np.cumprod(a)
    b = np.cumprod(b)
    a[5] = np.nan
    b[6] = np.nan
    print(_EMA(_EMA(a, 10), 10))
