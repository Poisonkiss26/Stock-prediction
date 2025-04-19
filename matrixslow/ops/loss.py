#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2020-04-03 19:40:23
@LastEditTime: 2020-04-20 16:57:33
@LastEditors: chenzhen
@Description:
'''
# -*- coding: utf-8 -*-
"""
Created on Wed July  9 15:13:01 2019

@author: chenzhen
"""

import numpy as np

from ..core import Node
from ..ops import SoftMax


class LossFunction(Node):
    '''
    定义损失函数抽象类
    '''
    pass


class LogLoss(LossFunction):

    def compute(self):
        assert len(self.parents) == 1

        x = self.parents[0].value

        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))

        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFunction):
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            jacobi = (prob - self.parents[1].value).T
            # print(np.linalg.norm(jacobi), end='\t')
            return jacobi
        else:
            return (-np.log(prob)).T


class FocusCrossEntropyWithSoftMax(LossFunction):
    def __init__(self, *parent, **kargs):
        LossFunction.__init__(self, *parent, **kargs)
        self.gamma = kargs.get('gamma')

    def compute(self):
        prob = np.array(SoftMax.softmax(self.parents[0].value))
        label = np.array(self.parents[1].value)
        self.value = np.mat(
            -np.sum((1 - prob) ** self.gamma * np.log(prob) * label))

    def get_jacobi(self, parent):
        p = np.array(SoftMax.softmax(self.parents[0].value))
        label = np.array(self.parents[1].value)
        ind = np.where(label == 1)[0]
        ps = p[ind]
        gamma = self.gamma
        if parent is self.parents[0]:
            jacobi = p * np.sum(-gamma * (1 - p) ** (gamma - 1) * np.log(p) * p * label + (1 - p) ** gamma * label)
            jacobi[ind] += gamma * (1 - ps) ** (gamma - 1) * np.log(ps) * ps - (1 - ps) ** gamma
            return np.mat(jacobi.T)
        else:
            raise 'You\'re not expected to get any jacobi matrix for your label. Please put output variable onto ' \
                  'first parameter and put label onto second variable!'


class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


class MeanSquaredErrorLoss(LossFunction):
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape() == self.parents[1].shape()
        self.value = np.mat(
            np.sum(np.power(self.parents[0].value - self.parents[1].value, 2)) / self.parents[0].dimension()).reshape(
            (1, 1))

    def get_jacobi(self, parent):
        # 要求Y_hat是其第一个父节点，标签是第二个父节点
        assert len(self.parents) == 2 and parent is self.parents[0]
        return np.mat(2 * (self.parents[0].value.flatten() - self.parents[1].value.flatten())) / parent.dimension()


class MAELoss(LossFunction):
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape() == self.parents[1].shape()
        self.value = np.mat(
            np.sum(np.abs(self.parents[0].value - self.parents[1].value)) / self.parents[0].dimension()).reshape(
            (1, 1))

    def get_jacobi(self, parent):
        # 要求Y_hat是其第一个父节点，标签是第二个父节点
        assert len(self.parents) == 2 and parent is self.parents[0]
        jacobi = np.zeros((1, self.parents[0].dimension())).T
        jacobi[self.value < 0] = -1
        jacobi[self.value > 0] = 1
        jacobi = np.mat(jacobi.T / parent.dimension())
        # print(jacobi)
        return jacobi
