# -*- coding: utf-8 -*-
# @Time     : 2024/2/7
# @Author   : Junzhe Huang
# @Email    : huangjz01@igoldenbeta.com
# @File     : data_trans
# @Software : gplearnplus
import pandas as pd


# todo 移植部分
def data_transform(X, y, data_type, number_feature_list, category_feature_list=None,
                   security_index=None, time_series_index=None):
    # 检查数据类型
    if data_type not in ('section', 'time_series', 'panel'):
        raise ValueError('Valid data_type methods include '
                         '"section", "time_series" and "panel". Given %s.'
                         % data_type)

    # X必须为pd.DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError('Data structure must be DataFrame')

    # 验证y的长度是否与X相同
    if len(X) != len(y):
        raise ValueError('X and y must have same length')

    # 检查column 是否包含category_feature_list 和 number_feature_list
    # 将category_feature_list 调整至前 number_feature_list 调整至后
    # 找出X的columns与category_feature_list的交集列表
    if category_feature_list is not None:
        if not isinstance(category_feature_list, list):
            raise ValueError('category_feature_list must be list')
        category_feature_list_inX = [col for col in X.columns if col not in category_feature_list]
    else:
        category_feature_list_inX = []
    # 找出X的columns与number_feature_list的交集列表
    if not isinstance(number_feature_list, list):
        raise ValueError('number_feature_list must be list')
    number_feature_list_inX = [col for col in X.columns if col not in number_feature_list]
    # 重构顺序，将分类类型放在前面, 并把第一列设为常数1，column为 const_1
    X['const_1'] = 1
    feature_names = category_feature_list_inX + number_feature_list_inX
    X_trans = X[['const_1'] + feature_names].copy()

    # 若存在security_index和time_series_index，插入X_trans最后，默认先插入security_index再插入time_series_index
    if security_index is not None:
        # 若security_index在X的columns中，或者为X.index，将其插入到X_trans最后
        if security_index in X.columns:
            X_trans[security_index] = X[security_index]
        elif X.index.name == security_index:
            X_trans[security_index] = X.index.get_level_values(security_index)
        else:
            # 若security_index不在X_trans的columns中，也不再index中，报错
            raise ValueError('Can not fund security_index {} in both columns and index'
                             .format(security_index))
   if time_series_index is not None:
        # 若time_series_index在X的columns
        if time_series_index in X.columns:
            X_trans[time_series_index] = X[time_series_index]
        elif X.index.name == time_series_index:



    # 检查时间index和个股index， 对于截面，时序和面板数据分别检查
    if data_type == 'section':
        if time_series_index is not None:
            raise ValueError('For Section Data, time_series_index should be None')
        if security_index is not None:
            # 在index和columns中寻找security_index
            # 判断是否有重复个股
            if len(X[security_index].unique()) < len(X[security_index]):
                raise ValueError('For Section Data, security data should be unique')
    elif data_type == 'time_series':
        if security_index is not None:
            raise ValueError('For time_series Data, security_index should be None')
        if time_series_index is not None:
            # 在index和columns中寻找time_series_index
            if time_series_index not in X.columns and \
                    (X.index.name is None or time_series_index not in X.index.name):
                raise ValueError('Can not fund time_series_index {} in both columns and index'
                                 .format(time_series_index))
            elif time_series_index in X.columns:
                X.set_index(time_series_index, inplace=True)
            # 判断是否有重复时间
            if len(X.index.drop_duplicates()) < len(X):
                raise ValueError('For time_series Data, time_series data should be unique')
            X_combine = X.copy()
            X_combine['_label'] = y.values if isinstance(y, pd.Series) else y
            X_combine.sort_index(inplace=True)
            X, y = X_combine.loc[:, self.feature_names], X_combine.loc[:, '_label']
            # debug

            time_series_data = X.index.values

    else:
        if self.time_series_index is None:
            raise ValueError('For panel Data, time_series_index should NOT be None')
        if self.security_index is None:
            raise ValueError('For panel Data, security_index should NOT be None')

        # security time_series 进入index
        if self.time_series_index not in X.columns and \
                (X.index.name is None or self.time_series_index not in X.index.name):
            raise ValueError('Can not fund time_series_index {} in both columns and index'
                             .format(self.time_series_index))
        elif self.security_index not in X.columns and \
                (X.index.name is None or self.security_index not in X.index.name):
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
        X_combine['_label'] = y.values if isinstance(y, pd.Series) else y
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