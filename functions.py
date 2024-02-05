# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :gplearnplus 
# @File     :function
# @Date     :2022/12/1 0001 13:46 
# @Author   :Junzhe Huang
# @Email    :acejasonhuang@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import numpy as np
from joblib import wrap_non_picklable_objects

NoneType = type(None)

__all__ = ['make_function', 'raw_function_list']


class _Function(object):
    """
    函数对象，参数至少有一个为向量
    默认函数类型为，all，既可用于时序也可用于截面
    默认返回类型为数值，
    默认输入类型，数值向量或者标量

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    param_type : [{
                  'vector': {'category': (None, None), 'number': (None, None)},
                  'scalar': {'int': (None, None), 'float': (None, None)}
                  },]
    function_type : 'all', 'section', 'time_series‘
    return_type: 'number', 'category'

    """

    def __init__(self, function, name, arity, param_type=None, return_type='number', function_type='all'):
        self.function = function
        self.name = name
        self.arity = arity
        if param_type is None:
            # 默认不接受分类类型
            param_type = arity * [{'vector':{'number': (None, None)},
                                   'scalar': {'int': (None, None), 'float': (None, None)}}]
        else:
            # 防止长度不一
            if len(param_type) != arity:
                raise ValueError(
                    "length of param_type should be equal to arity, it should be {}, not {}".format(arity, len(param_type)))
        self.param_type = param_type
        if (return_type != 'number') and (return_type != 'category'):
            raise ValueError("return_type of function {} should be number or category, NOT {}".format(name, return_type))
        self.return_type = return_type
        self.function_type = function_type

    def __call__(self, *args):
        """
        调用函数特殊处理，
        参数仅接受标量，却传入向量
        则取向量第一个值为标量
        """
        for _param, _param_type in zip(args, self.param_type):
            if len(_param_type) == 1 and 'scalar' in _param_type and isinstance(_param, (list, np.ndarray)):
                _param = _param[0]
        return self.function(*args)

    def add_range(self, const_range):
        # 作用：替换掉参数中没有约束的范围，给所有标量限制范围
        # 若没有const_range, 则表明所有函数不接收常数， 去掉所有的const type
        if const_range is None:
            for i, _dict in enumerate(self.param_type):
                if 'vector' not in _dict:
                    raise ValueError("for None const range, vector type should in all function param")
                if 'scalar' in _dict:
                    self.param_type[i].pop('scalar')
            return
        if not isinstance(const_range, tuple):
            raise ValueError('const_range must be an tuple')
        _min, _max = const_range
        if not isinstance(_min, (int, float)):
            raise ValueError('const_range left must be an int, float')
        if not isinstance(_max, (int, float)):
            raise ValueError('const_range right must be an int, float')
        if _min > _max:
            raise ValueError('const_range left should le right')

        for i, _dict in enumerate(self.param_type):
            if 'scalar' in _dict:
                _scalar_range = _dict['scalar']
                if 'int' in _scalar_range:
                    _l = int(_min) if _scalar_range['int'][0] is None else int(_scalar_range['int'][0])
                    _r = int(_max) if _scalar_range['int'][1] is None else int(_scalar_range['int'][1])
                    self.param_type[i]['scalar']['int'] = (_l, _r)
                if 'float' in _scalar_range:
                    _l = float(_min) if _scalar_range['float'][0] is None else float(_scalar_range['float'][0])
                    _r = float(_max) if _scalar_range['float'][1] is None else float(_scalar_range['float'][1])
                    self.param_type[i]['scalar']['float'] = (_l, _r)

        return

    def is_point_mutation(self, candidate_func):
        # 检验某个待替换函数是否可以替换
        if not isinstance(candidate_func, _Function):
            raise ValueError("wrong type, it should be _Function style")
        # 带替换函数是否与该函数参数长度一致
        if len(candidate_func.param_type) != len(self.param_type):
            return False
        if self.return_type != candidate_func.return_type:
            return False

        # candidate函数的参数必须为待替换函数参数的子集
        # 要求替换和，函数的所有参数仍然合法
        for dict_self, dict_candi in zip(self.param_type, candidate_func.param_type):
            if len(dict_candi) <= len(dict_self):
                return False
            for upper_type in dict_self:
                if upper_type not in dict_candi:
                    return False
                else:
                    for lower_type in dict_self:
                        if lower_type not in dict_candi[upper_type]:
                            return False
                        else:
                            if upper_type == 'scalar':
                                if (dict_candi['scalar'][lower_type][0] > dict_self['scalar'][lower_type][0]) or (
                                        dict_candi['scalar'][lower_type][1] > dict_candi['scalar'][lower_type][1]):
                                    return False
        return True



# warp 用于多进程序列化，会降低进化效率
def make_function(*, function, name, arity, param_type=None, wrap=True, return_type='number', function_type='all'):
    """
       Parameters
       ----------
       function : callable

       name : str

       arity : int

       param_type : [{type: (, ), type: (, )}, ........]

       wrap : bool, optional (default=True)
       """

    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # check out param_type vector > scalar int > float
    if param_type is None:
        param_type = [None] * arity
    if not isinstance(param_type, list):
        raise ValueError('param_type must be list')
    if len(param_type) != arity:
        raise ValueError('len of param_type must be arity')
    # 保证函数中至少有一个向量
    vector_flag = False
    for i, _dict in enumerate(param_type):
        # 转换None type
        # 标记某一个参数是否可接受向量
        non_vector_param = True
        if _dict is None:
            param_type[i] = {'vector': {'category': (None, None), 'number': (None, None)},
                             'scalar': {'int': (None, None), 'float': (None, None)}}
        elif not isinstance(_dict, dict):
            raise ValueError('element in param_type {} must be dict'.format(i + 1))
        if len(_dict) > 2:
            raise ValueError('len of element in param_type {} must be 1, 2'.format(i + 1))
        for upper_type in _dict:
            if upper_type == 'vector':
                if not isinstance(_dict['vector'], dict):
                    raise ValueError('type of element in param_type {} must be {upper_type: {lower_type:( , )}}}'
                                     .format(i + 1))
                if len(_dict['vector']) == 0:
                    raise ValueError('length of upper_type dict in param_type {} should not be 0'.format(i + 1))
                vector_flag = True
                non_vector_param = False
                for lower_type in _dict['vector']:
                    if lower_type not in ['number', 'category']:
                        raise ValueError('key of vector in param_type {} must be number or category'.format(i + 1))
                    param_type[i]['vector'][lower_type] = (None, None)

            elif upper_type == 'scalar':
                if not isinstance(_dict['scalar'], dict):
                    raise ValueError('type of element in param_type {} must be {upper_type: {lower_type:( , )}}}'
                                     .format(i + 1))
                if len(_dict['scalar']) == 0:
                    raise ValueError('length of upper_type dict in param_type {} should not be 0'.format(i + 1))
                for lower_type in _dict['scalar']:
                    if lower_type == 'int':
                        if not isinstance(_dict['scalar']['int'], tuple):
                            raise ValueError('structure of lower_type in param_type {} must be {type: ( , )}}'
                                             .format(i + 1))
                        if len(_dict['scalar']['int']) != 2:
                            raise ValueError("len of lower_type's structure in param_type {} must be 2".format(i + 1))
                        if not isinstance(_dict['scalar']['int'][0], (int, NoneType)):
                            raise ValueError("the first element in lower_type's structure in param_type {} "
                                             "must be None, int or float".format(i + 1))
                        if not isinstance(_dict['scalar']['int'][1], (int, NoneType)):
                            raise ValueError("the second element in lower_type's structure in param_type {} "
                                             "must be None, int or float".format(i + 1))
                        if isinstance(_dict['scalar']['int'][0], int) and isinstance(_dict['scalar']['int'][1], int) \
                                and _dict['scalar']['int'][1] < _dict['scalar']['int'][0]:
                            raise ValueError('the second element should ge the first element in param_type {}'
                                             .format(i + 1))

                    elif lower_type == 'float':
                        if not isinstance(_dict['scalar']['float'], tuple):
                            raise ValueError('structure of lower_type in param_type {} must be {type: ( , )}}'
                                             .format(i + 1))
                        if len(_dict['scalar']['float']) != 2:
                            raise ValueError("len of lower_type's structure in param_type {} must be 2".format(i + 1))
                        if not isinstance(_dict['scalar']['float'][0], (float, int, NoneType)):
                            raise ValueError("the first element in lower_type's structure in param_type {} "
                                             "must be None, int or float".format(i + 1))
                        if not isinstance(_dict['scalar']['float'][1], (float, int, NoneType)):
                            raise ValueError("the second element in lower_type's structure in param_type {} "
                                             "must be None, int or float".format(i + 1))
                        if isinstance(_dict['scalar']['float'][0], (int, float)) and \
                                isinstance(_dict['scalar']['float'][1], (int, float)) \
                                and _dict['scalar']['float'][1] < _dict['scalar']['float'][0]:
                            raise ValueError('the second element should ge the first element in param_type {}'
                                             .format(i + 1))
                    else:
                        raise ValueError('key of scalar in param_type {} must be int or float'.format(i + 1))
            else:
                raise ValueError('key of element in param_type {} must be vector or scalar'.format(i + 1))

    if not vector_flag:
        raise ValueError('there is at least 1 vector in param_type {}'.format(i + 1))

    # Check output shape
    # 生成测试数据
    args = []
    for _dict in param_type:
        if 'vector' in _dict:
            if 'number' in _dict['vector']:
                args.append(np.ones(10))
            else:
                args.append(np.array([1] * 10))
        elif 'scalar' in _dict:
            if 'int' in _dict['scalar']:
                args.append(((0 if _dict['scalar']['int'][1] is None else _dict['scalar']['int'][1]) +
                             (0 if _dict['scalar']['int'][0] is None else _dict['scalar']['int'][0])) // 2)
            else:
                args.append(((0 if _dict['scalar']['float'][1] is None else _dict['scalar']['float'][1]) +
                             (0 if _dict['scalar']['float'][0] is None else _dict['scalar']['float'][0])) // 2)

    try:
        function(*args)
    except (ValueError, TypeError):
        print(args)
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)
    if function(*args).dtype.type is np.float_ and return_type == 'category':
        raise ValueError('the return type should be category not {}'.format(function(*args).dtype.type))
    elif function(*args).dtype not in [np.float, np.int, np.int64] and return_type == 'number':
        raise ValueError('the return type should be category not {}'.format(function(*args).dtype.type))

    # Check closure for zero & negative input arguments
    args2 = []
    args3 = []
    for _dict in param_type:
        if 'vector' in _dict:
            # 兼容category向量
            args2.append(np.zeros(10))
            args3.append(-1 * np.ones(10))
        elif 'scalar' in _dict:
            if 'int' in _dict['scalar']:

                _temp = (((0 if _dict['scalar']['int'][1] is None else _dict['scalar']['int'][1]) +
                          (0 if _dict['scalar']['int'][0] is None else _dict['scalar']['int'][0])) // 2)
                args2.append(_temp)
                args3.append(_temp)
            else:
                _temp = (((0 if _dict['scalar']['float'][1] is None else _dict['scalar']['float'][1]) +
                          (0 if _dict['scalar']['float'][0] is None else _dict['scalar']['float'][0])) // 2)
                args2.append(_temp)
                args3.append(_temp)


    if not np.all(np.isnan(function(*args2)) | np.isfinite(function(*args2))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)

    if not np.all(np.isnan(function(*args3)) | np.isfinite(function(*args3))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)
    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity,
                         param_type=param_type,
                         return_type=return_type,
                         function_type=function_type)
    return _Function(function=function,
                     name=name,
                     arity=arity,
                     param_type=param_type,
                     return_type=return_type,
                     function_type=function_type)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def _groupby(gbx, func, *args, **kwargs):
    indices = np.argsort(gbx)
    gbx_sorted = gbx[indices]
    X = np.column_stack((np.arange(len(gbx)), gbx_sorted, *args))
    splits = np.split(X, np.unique(gbx_sorted, return_index=True)[1][1:])
    result_list = [func(*(split[:, 2:].T), **kwargs) for split in splits]
    result = np.hstack(result_list)
    return result[indices.argsort()]




add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}

raw_function_list = ['add', 'sub', 'mul', 'div', 'sqrt',
                     'sqrt', 'log', 'abs', 'neg', 'inv',
                     'max', 'min', 'sin', 'cos', 'tan']

all_function = raw_function_list.copy()

section_function = []

time_series_function = []

if __name__ == '__main__':
    # def ff(a, b, c):
    #     return a * b + c
    #
    # param_type = [{'vector':{'number': (None, None)}}, {'scalar': {'int':(None, 1)}}, {'scalar': {'float': (-1, None)}}]
    # f_m = make_function(function=ff, name='ff', arity=3, param_type=param_type, wrap=True, return_type='number')
    # f_m.add_range((-1, 1))
    # print(f_m.param_type)
    a = np.array([1, 2, 2, 1, np.nan])
    b = np.array([1, 2, 3, 4, 5])
    print(_groupby(a, max, b))

