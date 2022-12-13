# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :gplearnplus 
# @File     :test
# @Date     :2022/12/13 0013 2:05 
# @Author   :Junzhe Huang
# @Email    :acejasonhuang@163.com
# @Software :PyCharm
-------------------------------------------------
"""
from .genetic import SymbolicTransformer
import numpy as np

if __name__ == "__main__":
    m1 = SymbolicTransformer(verbose=1, generations=3)
    m1.fit(np.random.rand(10, 5), np.random.rand(10))