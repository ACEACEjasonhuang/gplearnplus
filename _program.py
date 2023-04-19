# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :gplearnplus 
# @File     :_program
# @Date     :2022/12/1 0001 13:37 
# @Author   :Junzhe Huang
# @Email    :acejasonhuang@163.com
# @Software :PyCharm
-------------------------------------------------
"""
from copy import copy, deepcopy
import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function, _groupby
from .utils import check_random_state


class _Program(object):
    """

    修改：
    arities

    """

    def __init__(self,
                 function_dict,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 data_type,
                 cat_var_number,
                 security_data=None,
                 time_series_data=None,
                 transformer=None,
                 feature_names=None,
                 program=None):

        self.function_dict = function_dict
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.data_type = data_type
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.cat_func_number = cat_var_number
        self.security_data = security_data
        self.time_series_data = time_series_data

        self.num_func_number = self.function_dict['number']
        self.cat_func_number = self.function_dict['category']

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """
        参数中无program 初始化方法
        # v1.55 修改数的生成逻辑
        :param random_state: RandomState 对象， 随机数生成器
        :return: list,
        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        # 因子返回类型必须为数值类型
        function = random_state.randint(len(self.function_dict['number']))
        function = self.function_dict['number'][function]

        program = [function]
        terminal_stack = [deepcopy(function.param_type)]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + self.num_func_number + self.cat_func_number
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if not isinstance(terminal_stack[-1], list):
                raise ValueError("element in terminal_stack should be list")
            if not isinstance(terminal_stack[-1][0], dict):
                raise ValueError("element in terminal_stack'element should be dict")
            # 插入函数的情况
            if ('vector' in terminal_stack[-1][0]) and (depth < max_depth) \
                    and (method == 'full' or choice <= (self.num_func_number + self.cat_func_number)):
                # 必须可接收向量且深度未满
                _choice = random_state.randint(self.cat_func_number + self.num_func_number)
                if 'number' in terminal_stack[-1][0]['vector'] and 'category' in terminal_stack[-1][0]['vector']:
                    key = 'number' if _choice < self.num_func_number else 'category'
                else:
                    key = 'number' if 'number' in terminal_stack[-1][0]['vector'] else 'category'
                function = self.function_dict[key][_choice %
                                                   (self.num_func_number if key == 'number' else self.cat_func_number)]
                program.append(function)
                terminal_stack.append(deepcopy(function.param_type))
            else:
                # 插入变量或者常量
                terminal = random_state.randint(self.n_features + 1)
                # 特殊情况调整
                if terminal == self.n_features and \
                        ((self.const_range is None) or \
                        (('int' or 'float') not in terminal_stack[-1][0])):
                    # 只能插入向量的情况
                    if 'vector' not in terminal_stack[-1][0]:
                        raise ValueError('Error param type {}'.format(terminal_stack[-1][0]))

                    terminal = random_state.randint(self.n_features)
                elif ('vector' not in terminal_stack[-1][0]):
                    # 只能插入常量的情况
                    terminal = self.n_features

                if terminal < self.n_features:
                    # 插入变量
                    if 'number' in terminal_stack[-1][0]['vector'] and 'category' in terminal_stack[-1][0][
                        'vector']:
                        key = 'category' if terminal < self.cat_func_number else 'number'
                    else:
                        key = 'number' if 'number' in terminal_stack[-1][0]['vector'] else 'category'
                    if self.cat_func_number == 0 and key == 'category':
                        raise ValueError("There no category var in input features, but it need")
                    candicate_var = (terminal % self.cat_func_number) if key == 'category' else \
                            ((terminal % (self.n_features - self.cat_func_number) + self.cat_func_number))
                    program.append(str(candicate_var))
                else:
                    # 插入常量量
                    if 'float' in terminal_stack[-1][0]:
                        terminal = random_state.uniform(*terminal_stack[-1][0]['float'])
                    elif 'int' in terminal_stack[-1][0]:
                        terminal = random_state.randint(*terminal_stack[-1][0]['int'])
                    else:
                        raise ValueError('Error param type {}'.format(terminal_stack[-1][0]))
                    program.append(terminal)

                terminal_stack[-1].pop(0)
                while len(terminal_stack[-1]) == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1].pop(0)
        # We should never get here
        return None

    # 检查函数是否可用，不包括类型检查
    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    # 打印树
    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, str):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[int(node)]
                elif isinstance(node, int):
                    output += '%d' % node
                elif isinstance(node, float):
                    output += '%.3f' % node
                else:
                    raise ValueError('Error param type {}'.format(node))
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    # 可视化整个树
    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'

                if isinstance(node, str):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[int(node)]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                elif isinstance(node, int):
                    output += ('%d [label="%d", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                elif isinstance(node, int):
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                else:
                    raise ValueError('Error param type {}'.format(node))

                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    # 计算树的深度
    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    # 计算公式中函数和变量的数量
    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    # 计算参数X的函数结果
    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        # 常数
        if isinstance(node, (float, int)):
            return np.repeat(node, X.shape[0])
        # 变量
        if isinstance(node, str):
            return X[:, int(node)]

        apply_stack = []
        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, (float, int))
                             else (X[:, int(t)] if isinstance(t, str)
                             else t) for t in apply_stack[-1][1:]]
                # 对于时序和截面函数加入管道
                if self.data_type == 'panel' and function.function_type == 'section':
                    intermediate_result = _groupby(self.time_series_data, function, *terminals)
                elif self.data_type == 'panel' and function.function_type == 'time_series':
                    intermediate_result = _groupby(self.security_data, function, *terminals)
                else:
                    intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    # 选择部分样本
    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    # 或许衡量模型适应度的指标
    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    # 原始适应度
    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    # todo 引入非线性适应度
    # 惩罚后适应度 对函数长度进行惩罚
    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    # 此处做了修改，不会选到常数
    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # 90%选到函数，10%选取向量叶子节点
        probs = np.array([0.9 if isinstance(node, _Function) else (0.1 if isinstance(node, str) else 0.0)
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    # 此处不会交换常数
    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    # 此处不会选择常数
    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    # 将子树的子树变上提，简化公式
    # 由于子树不会选到常数，故符合条件
    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    # 点变异完全修改
    # 要求函数满足is_point_mutation条件
    # 由于无法得知范围，常数不变异
    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]
        tag = np.array([True] * len(mutate))
        for i, node in enumerate(mutate):
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement_list = [func_ for func_ in self.arities[arity] if program[node].is_point_mutation(func_)]
                if len(replacement_list) == 0:
                    # 没有满足条件的变异
                    tag[i] = False
                    continue
                replacement = random_state.randint(len(replacement_list))
                replacement = replacement_list[replacement]
                program[node] = replacement
            elif isinstance(program[node], str):
                # We've got a terminal, add a const or variable
                terminal = random_state.randint(self.n_features)
                program[node] = str(terminal)
            else:
                # 常数不发生变异
                tag[i] = False
        if len(mutate):
            mutate = mutate[tag]
        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
