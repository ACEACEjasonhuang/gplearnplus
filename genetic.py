# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :gplearnplus 
# @File     :genetic
# @Date     :2022/12/5 0005 4:23 
# @Author   :Junzhe Huang
# @Email    :acejasonhuang@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder

from ._program import _Program
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .utils import _partition_estimators
from .utils import check_random_state

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max

# 并行实现子树交叉，变异
def _parallel_evolve(n_programs, parents, X, y, security_data, time_series_data, sample_weight, seeds, params):
    """

    Parameters
    ----------
    n_programs: 遗传代数
    parents：父辈个体集合
    X：原始特征
    y：预测label
    security_data：个体标记， 时序数据为none
    time_series_data：时间标记， 界面数据为none
    sample_weight：抽样比例
    seeds：随机种子
    params：参数

    Returns
    -------

    """

    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    data_type = params['data_type']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    cat_var_number = params['cat_var_number']

    max_samples = int(max_samples * n_samples)

    def _tournament():
        # 从所有父代中随机选择tournament_size个，取其中最优个体子代
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           data_type=date_type,
                           feature_names=feature_names,
                           random_state=random_state,
                           cat_var_number = cat_var_number,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

        programs.append(program)

    return programs


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 tolerable_corr=0.0,
                 class_weight=None,
                 feature_names=None,
                 time_series_index=None,
                 security_index=None,
                 category_features=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 data_type='section',
                 random_state=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.category_features = category_features
        self.time_series_index = time_series_index
        self.security_index = security_index
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.data_type = data_type
        self.tolerable_corr = tolerable_corr

    # 打印训练日志
    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     remaining_time))

    # fit 的时候考虑时序问题
    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # 检查数据类型
        if self.data_type not in ('section', 'time_series', 'panel'):
            raise ValueError('Valid data_type methods include '
                             '"section", "time_series" and "panel". Given %s.'
                             % self.data_type)

        # 检查数据结构
        # 若含有security或者timeindex 必须为DataFrame
        if self.security_index is not None or self.time_series_index is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError('with security ot time index, data structure should be DataFrame')

        # 检查时间index和个股index， 对于截面，时序和面板数据分别检查
        security_data = None
        time_series_data = None
        if self.data_type == 'section':
            if self.time_series_index is not None:
                raise ValueError('For Section Data, time_series_index should be None')
            if self.security_index is not None:
                # 在index和columns中寻找security_index
                if self.security_index not in X.columns and self.security_index not in X.index.name:
                    raise ValueError('Can not fund security_index {} in both columns and index'
                                     .format(self.security_index))
                elif self.security_index in X.columns:
                    X.set_index(self.security_index, inplace=True)

                # 判断是否有重复个股
                if len(X[self.security_index].unique()) < len(X[self.security_index]):
                    raise ValueError('For Section Data, security data should be unique')

                security_data = X.index.values

        elif self.data_type == 'time_series':
            if self.time_series_index is None:
                raise ValueError('For time_series Data, time_series_index should NOT be None')
            if self.security_index is not None:
                raise ValueError('For time_series Data, security_index should be None')
            if self.time_series_index not in X.columns and self.time_series_index not in X.index.name:
                raise ValueError('Can not fund time_series_index {} in both columns and index'
                                 .format(self.time_series_index))
            elif self.time_series_index in X.columns:
                X.set_index(self.time_series_index, inplace=True)

            # 判断是否有重复时间
            if len(X[self.time_series_index].unique()) < len(X[self.time_series_index]):
                raise ValueError('For time_series Data, time_series data should be unique')

            X_combine = X.copy()
            X_combine['_label'] = y
            X_combine.sort_index(inplace=True)
            X, y = X_combine.loc[:, self.feature_names], X_combine.loc[:, '_label']
            time_series_data = X.index.values

        else:
            if self.time_series_index is None:
                raise ValueError('For panel Data, time_series_index should NOT be None')
            if self.security_index is None:
                raise ValueError('For panel Data, security_index should NOT be None')

            # security time_series 进入index
            if self.time_series_index not in X.columns and self.time_series_index not in X.index.name:
                raise ValueError('Can not fund time_series_index {} in both columns and index'
                                 .format(self.time_series_index))
            elif self.security_index not in X.columns and self.security_index not in X.index.name:
                raise ValueError('Can not fund security_index {} in both columns and index'
                                 .format(self.security_index))
            elif self.time_series_index in X.columns and self.security_index in X.columns:
                X.set_index([self.time_series_index, self.security_index], inplace=True)
            elif self.time_series_index in X.columns:
                X.set_index(self.security_index, inplace=True, append=True)
            elif self.security_index in X.columns:
                X.set_index(self.time_series_index, inplace=True, append=True)

            # 判断没有重复
            if len(X.index) != len(X.index.drop_duplicates()):
                raise ValueError('For time_series Data, time_series data should be unique')


            X_combine = X.copy()
            X_combine['_label'] = y
            X_combine.sort_index(inplace=True)
            X, y = X_combine.loc[:, self.feature_names], X_combine.loc[:, '_label']
            time_series_data = X.index.get_level_values(self.time_series_index).values
            security_data = X.index.get_level_values(self.security_index).values

        # 检查category_features是否与全包含在feature_names中
        # 当存在分类数据时，输入数据类型必须为pd。DataFrame
        if self.category_features is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError('while there are category_features in X, X must be pd.DataFrame')
            if not isinstance(self.category_features, list):
                raise ValueError('category_features must be list')
            for cat_feature in self.category_features:
                if cat_feature not in self.feature_names:
                    raise ValueError('Valid category_feature {} , not in feature_names'.format(cat_feature))
            # 处理分类数据，转换为整型
            label_encoder = LabelEncoder()
            X[self.category_features] = X[self.category_features].apply(label_encoder.fit_transform)
            # 重构顺序，将分类类型放在前面
            self.feature_names = \
                [self.category_features + [_col for _col in self.feature_names if _col not in self.category_features]]
            X = X[self.feature_names]

        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # 检查数据内容
        if isinstance(self, ClassifierMixin):
            # 验证y是否为分类数据， X， y强转ndarray
            # todo 分类场景的处理有待优化，暂时不处理
            X, y = self._validate_data(X, y, y_numeric=False)
            check_classification_targets(y)

            if self.class_weight:
                if sample_weight is None:
                    sample_weight = 1.
                # modify the sample weights with the corresponding class weight
                sample_weight = (sample_weight *
                                 compute_sample_weight(self.class_weight, y))

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            self.n_classes_ = len(self.classes_)

        else:
            # 验证y是否为数值数据， X， y强转ndarray
            X, y = self._validate_data(X, y, y_numeric=True)

        # check hall_of_fame and n_components ,if have
        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        # 检查feature_names是否与n_features_in_一致
        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        # 检查const_range
        if not ((isinstance(self.const_range, tuple) and
                 len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        # 检查function, 稍作修改， 结合const_range到range里面, 并区分number func 和 cat function
        # 存放不同类型的函数（分类和数值）
        self._function_dict = {'number': [], 'category': []}
        # 检验是否存在接受分类变量参数的函数
        _cat_func_flag = False
        for function in self.function_set:
            # 类型检验
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                function = deepcopy(_function_map[function])
                self._function_dict['number'].append(function)
            elif isinstance(function, _Function):
                function = deepcopy(function)
                # 添加常数范围
                function.add_range(self.const_range)
                # 检验是否有仅接收分类变量的函数
                if not _cat_func_flag:
                    for _param in function.param_type:
                        if len(_param) == 1 and 'vector' in _param and \
                                len(_param['vector']) == 1 and 'category' in _param['vector']:
                            _cat_func_flag = True
                if function.return_type == 'number':
                    self._function_dict['number'].append(function)
                else:
                    self._function_dict['category'].append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))

        # number类型函数必须有
        if len(self._function_dict['number']) == 0:
            raise ValueError('No valid functions found in `function_set`.')

        # 当存在只接受分类变量参数的函数时（如groupby），category变量不能为空
        if _cat_func_flag and len(self.category_features) == 0:
            raise ValueError('There no category var in input features, but there are functions only get category param')

        # 点变异记录函数参数个数， 需要在点变异中再考察参数类型
        self._arities = {'number': {}, 'category': {}}
        for _type in ['number', 'category']:
            for function in self._function_dict[_type]:
                arity = function.arity
                self._arities[_type][arity] = self._arities[_type].get(arity, [])
                self._arities[_type][arity].append(function)

        # 检查fitness
        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        # 检查概率参数
        # todo 增加交叉变异方法后需要修改此处
        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)
        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        # 检查初始化模式
        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        # 检查初始化深度
        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        # 初始化transformer函数
        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_dict'] = self._function_dict
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs
        params['cat_var_number'] = len(self.category_features)

        # 清空_program
        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):
            start_time = time()

            if gen == 0:
                parents = None
            else:
                try:
                    parents = self._programs[gen - 1]
                except:
                    print(len(self._programs))
                    print(gen)

                    exit()
            # Parallel loop
            # 将population_size分配给n_job个进程
            n_jobs, n_programs, starts = _partition_estimators(self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          security_data,
                                          time_series_data,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            # 惩罚系数
            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # 去除没有进入下一代的父辈种群
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # 在low_memory的情况下，去除所有
                self._programs[gen - 1] = None

            # 记录运行细节
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # 是否进入停止条件
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        # 特征工程专属模块
        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            # 找出适应度最优的hall_of_fame个进入fitness
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                # 去除hall_of_fame - n_components个高度相关特征
                # 找到相关系数矩阵中相关系数绝对值最大的两个特征，删去其中fitness较低的那个
                # 相关性低于某一阈值时按照fitness筛选（gplearnplus新增）
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                if max(correlations) < self.tolerable_corr:
                    break
                indices = list(range(len(components)))
            # 余下的选出最优的self.n_components个
            components = components[:self.n_components]
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self


class SymbolicRegressor(BaseSymbolic, RegressorMixin):
    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)

        return y


class SymbolicClassifier(BaseSymbolic, ClassifierMixin):
    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer='sigmoid',
                 metric='log loss',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicClassifier, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            transformer=transformer,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            class_weight=class_weight,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def _more_tags(self):
        return {'binary_only': True}

    def predict_proba(self, X):
        # 输出概率 只支持二分类
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        scores = self._program.execute(X)
        proba = self._transformer(scores)
        proba = np.vstack([1 - proba, proba]).T
        return proba

    def predict(self, X):
        # 输出预测结果
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


class SymbolicTransformer(BaseSymbolic, TransformerMixin):
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='pearson',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicTransformer, self).__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __len__(self):
        """Overloads `len` output to be the number of fitted components."""
        if not hasattr(self, '_best_programs'):
            return 0
        return self.n_components

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self._best_programs[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self])
        return output.replace("',", ",\n").replace("'", "")

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }

    def transform(self, X):
        # 将X转换成以及训练好的特征
        if not hasattr(self, '_best_programs'):
            raise NotFittedError('SymbolicTransformer not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        X_new = np.array([gp.execute(X) for gp in self._best_programs]).T

        return X_new

    def fit_transform(self, X, y, sample_weight=None):
        # 训练之后转换
        return self.fit(X, y, sample_weight).transform(X)

