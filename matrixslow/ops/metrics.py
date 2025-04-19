# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:34:46 CST 2019

@author: chenzhen
"""
import numpy as np
import abc
from ..core import Node


class Metrics(Node):
    '''
    评估指标算子抽象基类
    '''

    def __init__(self, *parents, **kargs):
        # 默认情况下，metrics节点不需要保存
        kargs['need_save'] = kargs.get('need_save', False)
        # self.threshold = kargs.get('threshold', 0.5)
        Node.__init__(self, *parents, **kargs)
        self.allow_compute = False
        # 初始化节点
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    def allow(self):  # 允许计算评估指标。在训练时设置参数allow_compute为False，不计算评估指标
        self.allow_compute = True

    @abc.abstractmethod
    def init(self):
        # 如何初始化节点由具体子类实现
        pass

    @staticmethod
    def prob_to_label(prob, thresholds=0.5):
        if prob.shape[0] > 1:
            # 如果是多分类，预测类别为概率最大的类别
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            # 否则以thresholds为概率阈值判断类别
            labels = np.where(prob < thresholds, 0, 1)

        return labels

    def get_jacobi(self):

        # 对于评估指标节点，计算雅可比无意义
        raise NotImplementedError()

    def value_str(self):
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class Evaluate(Metrics):
    def __init__(self, *parents, **kargs):
        self.threshold = kargs.get('threshold', 0.5)
        Metrics.__init__(self, *parents, **kargs)
        # self.init()

    def init(self, **kwargs):
        pred = np.argmax(self.parents[0].value, axis=0).flatten()
        label = np.argmax(self.parents[1].value, axis=0).flatten()
        assert label.shape[0] == pred.shape[0]
        category = np.unique(np.array(label))
        confusion_matrix = np.zeros((category.shape[0], category.shape[0]), dtype=np.int64)
        for i in range(category.shape[0]):
            for j in range(category.shape[0]):
                true_cate = category[i]
                pred_cate = category[j]
                confusion_matrix[true_cate, pred_cate] = np.sum(np.bitwise_and(label == true_cate, pred == pred_cate))
        self.confusion_matrix = confusion_matrix

    def compute(self):
        self.value = {'Accuracy': np.round(self.get_accuracy(), 4),
                      'Recall': np.round(self.get_recall(), 4),
                      'Precision:': np.round(self.get_precision(), 4)}

    def get_accuracy(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    def get_recall(self):
        if self.confusion_matrix.shape[0] <= 2:  # 若问题为2分类，只需要计算accuracy就行了
            return 'Accuracy'
        else:
            confusion_matrix = self.confusion_matrix
            recall = []
            for i in range(confusion_matrix.shape[0]):
                recall.append(confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]))
            return recall

    def get_precision(self):
        if self.confusion_matrix.shape[0] <= 2:
            return 'Accuracy'
        else:
            confusion_matrix = self.confusion_matrix
            precision = []
            for i in range(confusion_matrix.shape[0]):
                precision.append(confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]))
            return precision

    # '''
    # 评估节点
    # 为ConfusionMatrix的子类
    # 根据父类初始化生成的confusion_matrix计算各类评价指标(适用于多分类)
    # '''
    #
    # def __init__(self, *parents, **kargs):
    #     ConfusionMatrix.__init__(self, *parents, **kargs)
    #
    # def compute(self):
    #     self.value = {'Accuracy': self.get_accuracy(),
    #                   'Recall': self.get_recall(),
    #                   'Precision:': self.get_precision()}
    #
    # def get_accuracy(self):
    #     return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)
    #
    # def get_recall(self):
    #     if self.confusion_matrix.shape[0] <= 2:  # 若问题为2分类，只需要计算accuracy就行了
    #         return 'Accuracy'
    #     else:
    #         confusion_matrix = self.confusion_matrix
    #         recall = []
    #         for i in range(confusion_matrix.shape[0]):
    #             recall.append(confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]))
    #         return recall
    #
    # def get_precision(self):
    #     if self.confusion_matrix.shape[0] <= 2:
    #         return 'Accuracy'
    #     else:
    #         confusion_matrix = self.confusion_matrix
    #         precision = []
    #         for i in range(confusion_matrix.shape[0]):
    #             precision.append(confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]))
    #         return precision


# class Accuracy(Metrics):
#     '''
#     正确率节点
#     '''
#
#     def __init__(self, *parents, **kargs):
#         Metrics.__init__(self, *parents, **kargs)
#
#     def init(self):
#         self.correct_num = 0
#         self.total_num = 0
#
#     def compute(self):
#         '''
#         计算Accrucy: (TP + TN) / TOTAL
#         这里假设第一个父节点是预测值（概率），第二个父节点是标签
#         '''
#
#         pred = Metrics.prob_to_label(self.parents[0].value)
#         gt = self.parents[1].value
#         assert len(pred) == len(gt)
#         if pred.shape[0] > 1:
#             self.correct_num += np.sum(np.multiply(pred, gt))
#             self.total_num += pred.shape[1]
#         else:
#             self.correct_num += np.sum(pred == gt)
#             self.total_num += len(pred)
#         self.value = 0
#         if self.total_num != 0:
#             self.value = float(self.correct_num) / self.total_num
#
#
# class Precision(Metrics):
#     '''
#     查准率节点
#     '''
#
#     def __init__(self, *parents, **kargs):
#         Metrics.__init__(self, *parents, **kargs)
#
#     def init(self):
#         self.true_pos_num = 0
#         self.pred_pos_num = 0
#
#     def compute(self):
#         '''
#         计算Precision： TP / (TP + FP)
#         '''
#         assert self.parents[0].value.shape[1] == 1
#
#         pred = Metrics.prob_to_label(self.parents[0].value)
#         gt = self.parents[1].value
#         self.pred_pos_num += np.sum(pred == 1)
#         self.true_pos_num += np.sum(pred == gt and pred == 1)
#         self.value = 0
#         if self.pred_pos_num != 0:
#             self.value = float(self.true_pos_num) / self.pred_pos_num
#
#
# class Recall(Metrics):
#     '''
#     查全率节点
#     '''
#
#     def __init__(self, *parents, **kargs):
#         Metrics.__init__(self, *parents, **kargs)
#
#     def init(self):
#         self.gt_pos_num = 0
#         self.true_pos_num = 0
#
#     def compute(self):
#         '''
#         计算Recall： TP / (TP + FN)
#         '''
#         assert self.parents[0].value.shape[1] == 1
#
#         pred = Metrics.prob_to_label(self.parents[0].value)
#         gt = self.parents[1].value
#
#         self.gt_pos_num += np.sum(gt == 1)
#         self.true_pos_num += np.sum(pred == gt and pred == 1)
#
#         self.value = 0
#         if self.gt_pos_num != 0:
#             self.value = float(self.true_pos_num) / self.gt_pos_num


class ROC(Metrics):
    '''
    ROC曲线
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.count = 100
        self.gt_pos_num = 0
        self.gt_neg_num = 0
        self.true_pos_num = np.array([0] * self.count)
        self.false_pos_num = np.array([0] * self.count)
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def compute(self):

        prob = self.parents[0].value
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt == 1)
        self.gt_neg_num += np.sum(gt == -1)

        # 最小值0.01，最大值0.99，步长0.01，生成99个阈值
        thresholds = list(np.arange(0.01, 1.00, 0.01))

        # 分别使用多个阈值产生类别预测，与标签比较
        for index in range(0, len(thresholds)):
            pred = Metrics.prob_to_label(prob, thresholds[index])
            self.true_pos_num[index] += np.sum(pred == gt and pred == 1)
            self.false_pos_num[index] += np.sum(pred != gt and pred == 1)

        # 分别计算TPR和FPR
        if self.gt_pos_num != 0 and self.gt_neg_num != 0:
            self.tpr = self.true_pos_num / self.gt_pos_num
            self.fpr = self.false_pos_num / self.gt_neg_num

    def value_str(self):
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.ylim(0, 1)
        # plt.xlim(0, 1)
        # plt.plot(self.fpr, self.tpr)
        # plt.show()
        return ''


class ROC_AUC(Metrics):
    '''
    ROC AUC
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.gt_pos_preds = []
        self.gt_neg_preds = []

    def compute(self):

        prob = self.parents[0].value
        gt = self.parents[1].value

        # 简单起见，假设只有一个元素
        if gt[0, 0] == 1:
            self.gt_pos_preds.append(prob)
        else:
            self.gt_neg_preds.append(prob)

        self.total = len(self.gt_pos_preds) * len(self.gt_neg_preds)

    def value_str(self):
        count = 0

        # 遍历m x n个样本对，计算正类概率大于负类概率的数量
        for gt_pos_pred in self.gt_pos_preds:
            for gt_neg_pred in self.gt_neg_preds:
                if gt_pos_pred > gt_neg_pred:
                    count += 1

        # 使用这个数量，除以m x n
        self.value = float(count) / self.total
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class F1Score(Metrics):
    '''
    F1 Score算子

    '''

    def __init__(self, *parents, **kargs):
        '''
        F1Score算子
        '''
        Metrics.__init__(self, *parents, **kargs)
        self.true_pos_num = 0
        self.pred_pos_num = 0
        self.gt_pos_num = 0

    def compute(self):
        '''
        计算f1-score: (2 * pre * recall) / (pre + recall)
        '''

        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt)
        self.pred_pos_num += np.sum(pred)
        self.true_pos_num += np.multiply(pred, gt).sum()
        self.value = 0
        pre_score = 0
        recall_score = 0

        if self.pred_pos_num != 0:
            pre_score = float(self.true_pos_num) / self.pred_pos_num

        if self.gt_pos_num != 0:
            recall_score = float(self.true_pos_num) / self.gt_pos_num

        self.value = 0
        if pre_score + recall_score != 0:
            self.value = 2 * \
                         np.multiply(pre_score, recall_score) / \
                         (pre_score + recall_score)
