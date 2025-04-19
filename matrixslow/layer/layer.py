# -*- coding: utf-8 -*-
from ..core import *
from ..ops import *


def conv(feature_maps, input_shape, kernels, kernel_shape, activation, init=None):
    """
    :param feature_maps: 数组，包含多个输入特征图，它们应该是值为同形状的矩阵的节点
    :param input_shape: tuple ，包含输入特征图的形状（宽和高）
    :param kernels: 整数，卷积层的卷积核数量
    :param kernel_shape: tuple ，卷积核的形状（宽和高）
    :param activation: 激活函数类型
    :return: 数组，包含多个输出特征图，它们是值为同形状的矩阵的节点
    """
    # 与输入同形状的全 1 矩阵
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    for i in range(kernels):

        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=init, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)

        channles = Add(*channels)
        bias = ScalarMultiply(Variable((1, 1), init=True, trainable=True), ones)
        affine = Add(channles, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernels
    return outputs


def conv_FFT(feature_maps, input_shape, kernels, kernel_shape, activation, init=None):
    """
    :param feature_maps: 数组，包含多个输入特征图，它们应该是值为同形状的矩阵的节点
    :param input_shape: tuple ，包含输入特征图的形状（宽和高）
    :param kernels: 整数，卷积层的卷积核数量
    :param kernel_shape: tuple ，卷积核的形状（宽和高）
    :param activation: 激活函数类型
    :return: 数组，包含多个输出特征图，它们是值为同形状的矩阵的节点
    """
    # 与输入同形状的全 1 矩阵
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    for i in range(kernels):

        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=init, trainable=True)
            conv = Convolve_FFT(fm, kernel)
            channels.append(conv)

        channles = Add(*channels)
        bias = ScalarMultiply(Variable((1, 1), init=True, trainable=True), ones)
        affine = Add(channles, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernels
    return outputs


def pooling(feature_maps, kernel_shape, stride):
    """
    :param feature_maps: 数组，包含多个输入特征图，它们应该是值为同形状的矩阵的节点
    :param kernel_shape: tuple ，池化核（窗口）的形状（宽和高）
    :param stride: tuple ，包含横向和纵向步幅
    :return: 数组，包含多个输出特征图，它们是值为同形状的矩阵的节点
    """
    outputs = []
    for fm in feature_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))

    return outputs


def fc(input, input_size, size, activation, regularization=None, weight_init=None, bias_init=None,
       print_gradient=None):
    """
    :param regularization:
    :param input: 输入向量
    :param input_size: 输入向量的维度
    :param size: 神经元个数，即输出个数（输出向量的维度）
    :param activation: 激活函数类型
    :return: 输出向量
    """
    weights = Variable((size, input_size), init=weight_init, trainable=True, print_gradient=print_gradient)
    bias = Variable((size, 1), init=bias_init, trainable=True, print_gradient=print_gradient)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine


def LSTM(x, *input, **kwargs):
    '''
    X：当前输入节点
    C_0：上一个记忆节点
    H_0：
    Wf Bf：遗忘门权重
    Wi Bi：输入门权重
    Wo Bo：输出门权重
    Wc Bc：
    C_1：输出的记忆细胞
    H_1：
    '''
    id = kwargs.get('id', 'default')
    n = kwargs.get('n', 10)
    l = kwargs.get('l', 16)
    X = x
    H_0, C_0 = input
    concat = Concat(H_0, X, axis=0, name=f'LSTM_concat_{id}')

    Wf = Variable((n, n + l), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Wf_{id}')
    Bf = Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Bf_{id}')

    Wi = Variable((n, n + l), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Wi_{id}')
    Bi = Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Bi_{id}')

    Wc = Variable((n, n + l), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Wc_{id}')
    Bc = Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Bc_{id}')

    Wo = Variable((n, n + l), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Wo_{id}')
    Bo = Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_Bo_{id}')

    gate_forget = Logistic(Add(MatMul(Wf, concat), Bf), name=f'LSTM_gateforget_{id}')
    gate_input = Logistic(Add(MatMul(Wi, concat), Bi), name=f'LSTM_gateinput_{id}')
    C_hat = Logistic(Add(MatMul(Wc, concat), Bc), name=f'LSTM_C-hat_{id}')
    gate_output = Logistic(Add(MatMul(Wo, concat), Bo), name=f'LSTM_gateoutput_{id}')

    multiply_C0_forget = Multiply(C_0, gate_forget, name=f'LSTM_multiply_C0_forget_{id}')
    multiply_input_Chat = Multiply(gate_input, C_hat, name=f'LSTM_multiply_gateinput_Chat_{id}')

    add = Add(multiply_C0_forget, multiply_input_Chat, name=f'LSTM_add_{id}')
    C_1 = add
    tanh = Tanh(add, name=f'LSTM_tanh_{id}')
    H_1 = Multiply(tanh, gate_output, name=f'LSTM_h_{id}')
    return H_1, C_1
