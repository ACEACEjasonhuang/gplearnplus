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

__all__ = ['make_function']


class _Function(object):
    """
    函数对象，参数至少有一个为向量

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

    """

    def __init__(self, function, name, arity, param_type=None):
        self.function = function
        self.name = name
        self.arity = arity
        if param_type is None:
            param_type = arity * [{'vector': (None, None), 'int': (None, None), 'float': (None, None)}]
        self.param_type = param_type

    def __call__(self, *args):
        return self.function(*args)

    def add_range(self, const_range):
        # 替换掉参数中没有约束的范围
        # 去掉所有的const type
        if const_range is None:
            for i, _dict in enumerate(self.param_type):
                if 'vector' not in _dict:
                    raise ValueError("for None const range, vector type should in all function param")
                if 'int' in _dict:
                    self.param_type[i].pop('int')
                if 'float' in _dict:
                    self.param_type[i].pop('float')
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
            if 'int' in _dict:
                _l = int(_min) if _dict['int'][0] is None else int(_dict['int'][0])
                _r = int(_max) if _dict['int'][1] is None else int(_dict['int'][1])
                self.param_type[i]['int'] = (_l, _r)
            if 'float' in _dict:
                _l = float(_min) if _dict['float'][0] is None else float(_dict['float'][0])
                _r = float(_max) if _dict['float'][1] is None else float(_dict['float'][1])
                self.param_type[i]['float'] = (_l, _r)

        return

    def is_point_mutation(self, func):
        if not isinstance(func, _Function):
            raise ValueError("wrong type, it should be _Function style")
        if len(func.param_type) != len(self.param_type):
            return False

        for dict_self, dict_func in zip(self.param_type, func.param_type):
            if len(dict_func) <= len(dict_self):
                return False
            for key in dict_self:
                if key not in dict_func:
                    return False
                else:
                    if key != 'vector':
                        if (dict_func[key][0] >  dict_self[key][0]) or (dict_func[key][1] >  dict_self[key][1]):
                            return False

        return True








# warp 用于多进程序列化，会降低进化效率
def make_function(*, function, name, arity, param_type=None, wrap=True):
    """
       Parameters
       ----------
       function : callable

       name : str

       arity : int

       param_type : [{type: (, ), type: (,)}, ........]

       wrap : bool, optional (default=True)
       """

    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # check out param_type vector > int > float
    if param_type is None:
        param_type = [None] * arity
    if not isinstance(param_type, list):
        raise ValueError('param_type must be list')
    if len(param_type) != arity:
        raise ValueError('len of param_type must be arity')


    vector_flag = False
    for i, _dict in enumerate(param_type):
        # 转换None type
        non_vector_param = True
        if _dict is None:
            param_type[i] = {'vector': (None, None), 'int': (None, None), 'float': (None, None)}
        elif not isinstance(_dict, dict):
            raise ValueError('element in param_type must be dict')
        if len(_dict) > 3:
            raise ValueError('len of element in param_type must be 1, 2, 3')
        for _key in _dict:
            if _key == 'vector':
                if not isinstance(_dict['vector'], tuple):
                    raise ValueError('type of element in param_type must be {type: ( , )}}')
                param_type[i]['vector'] = (None, None)
                vector_flag = True
                non_vector_param = False
            elif _key == 'int':
                if not isinstance(_dict['int'], tuple):
                    raise ValueError('type of element in param_type must be {type: ( , )}}')
                if len(_dict['int']) != 2:
                    raise ValueError('len of dict_element in param_type must be 2')
                if not isinstance(_dict['int'][0], (int, NoneType)):
                    raise ValueError('type of dict_first_element in param_type must be None, int or float')
                if not isinstance(_dict['int'][1], (int, NoneType)):
                    raise ValueError('type of dict_second_element in param_type must be None, int or float')
                if isinstance(_dict['int'][0], int) and isinstance(_dict['int'][1], int) \
                        and _dict['int'][1] < _dict['int'][0]:
                    raise ValueError('dict_second_element should ge dict_first_element')

            elif _key == 'float':
                if not isinstance(_dict['float'], tuple):
                    raise ValueError('type of element in param_type must be {type: ( , )}}')
                if len(_dict['float']) != 2:
                    raise ValueError('len of dict_element in param_type must be 2')
                if not isinstance(_dict['float'][0], (float, int, NoneType)):
                    raise ValueError('type of dict_first_element in param_type must be None, int or float')
                if not isinstance(_dict['float'][1], (float, int, NoneType)):
                    raise ValueError('type of dict_second_element in param_type must be None, int or float')
                if isinstance(_dict['float'][0], (int, float)) and isinstance(_dict['float'][1], (int, float)) \
                        and _dict['float'][1] < _dict['float'][0]:
                    raise ValueError('dict_second_element should ge dict_first_element')
            else:
                raise ValueError('key of element in param_type must be vector, int or float')

    if not vector_flag:
        raise ValueError('there is at least  1 vector in param_type')

    # Check output shape
    args = []
    for _dict in param_type:
        if 'vector' in _dict:
            args.append(np.ones(10))
        elif 'int' in _dict:
            args.append(((0 if _dict['int'][1] is None else _dict['int'][1]) +
                         (0 if _dict['int'][0] is None else _dict['int'][0])) // 2)
        else:
            args.append(((0 if _dict['float'][1] is None else _dict['float'][1]) +
                         (0 if _dict['float'][0] is None else _dict['float'][0])) // 2)

    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args2 = []
    args3 = []
    for _dict in param_type:
        if 'vector' in _dict:
            args2.append(np.zeros(10))
            args3.append(-1 * np.ones(10))
        elif 'int' in _dict:
            _temp = (((0 if _dict['int'][1] is None else _dict['int'][1]) +
                      (0 if _dict['int'][0] is None else _dict['int'][0])) // 2)
            args2.append(_temp)
            args3.append(_temp)
        else:
            _temp = (((0 if _dict['float'][1] is None else _dict['float'][1]) +
                      (0 if _dict['float'][0] is None else _dict['float'][0])) // 2)
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
                         param_type=param_type)
    return _Function(function=function,
                     name=name,
                     arity=arity,
                     param_type=param_type)



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


if __name__ == '__main__':
    def ff(a, b, c):
        return a * b + c

    param_type = [{'vector':(None, None)}, {'int':(None, 1)}, {'float': (-1, None)}]
    f_m = make_function(function=ff, name='ff', arity=3, param_type=param_type, wrap=True)
    f_m.add_range((-1, 1))
    print(f_m.param_type)

