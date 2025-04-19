# -*- coding: utf-8 -*-

"""
Created on Wed Jun  5 15:23:01 2019

@author: zhangjuefei
"""
import numpy as np

from ..core import Node


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
           filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    '''
    定义操作符抽象类
    '''
    pass


class Add(Operator):
    """
    （多个）矩阵加法
    """

    def compute(self):
        # assert len(self.parents) == 2 and self.parents[0].shape() == self.parents[1].shape()
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension()))  # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵


class MatMul(Operator):
    """
    矩阵乘法
    """

    def compute(self):
        assert len(self.parents) == 2
        assert self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """

        # 很神秘，靠注释说不明白了
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Logistic(Operator):
    """
    对向量的分量施加Logistic函数
    """

    def compute(self):
        x = self.parents[0].value
        # 对父节点的每个分量施加Logistic
        self.value = np.mat(
            1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class ReLU(Operator):
    """
    对矩阵的元素施加ReLU函数
    """

    nslope = 0.1  # 负半轴的斜率

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value)
        )

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, self.nslope))


class Tanh(Operator):
    '''
    Tanh函数
    '''

    def compute(self):
        self.value = np.mat(np.tanh(self.parents[0].value))

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        diag = (1 - np.power(self.value, 2)).A1
        return np.mat(np.diag(diag))


class SoftMax(Operator):
    """
    SoftMax函数
    """

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，
        训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class Reshape(Operator):
    """
    改变父节点的值（矩阵）的形状
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2

    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))


class Multiply(Operator):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Convolve(Operator):
    """
    以第二个父节点的值为滤波器，对第一个父节点的值做二维离散卷积
    """

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)

        self.padded = None

    def compute(self):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data

        self.value = np.mat(np.zeros((w, h)))

        # 二维离散卷积
        for i in np.arange(hkw, hkw + w):
            for j in np.arange(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(
                    np.multiply(self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh], kernel))

    def get_jacobi(self, parent):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)


class Convolve_FFT(Operator):
    """
    以第二个父节点的值为滤波器，对第一个父节点的值做二维离散卷积
    """

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)

        self.padded = None

    def compute(self):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        img_fft = np.fft.fft2(data)
        kernel_fft = np.fft.fft2(kernel, s=data.shape)
        img_conv_fft = np.fft.ifft2(img_fft * kernel_fft)
        img_conv = np.abs(img_conv_fft)
        self.value = img_conv

    def get_jacobi(self, parent):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)


class Haar(Operator):
    def compute(self):
        assert len(self.parents) == 1
        src = np.array(self.parents[0].value)
        kernel1 = np.array([[1, 1], [1, 1]]) / 4
        kernel2 = np.array([[1, -1], [1, -1]]) / 4
        kernel3 = np.array([[1, 1], [-1, -1]]) / 4
        kernel4 = np.array([[1, -1], [-1, 1]]) / 4
        kernels = [kernel1, kernel2, kernel3, kernel4]
        dst = []
        for kernel in kernels:
            img_fft = np.fft.fft2(src)
            # print(img_fft.shape)
            kernel_fft = np.fft.fft2(kernel, s=src.shape)
            # print(kernel_fft.shape)

            conv_fft1 = np.fft.ifft2(img_fft * kernel_fft)
            dst.append(np.abs(conv_fft1)[1::2, 1::2])
        self.dst = dst
        self.value = np.mat(np.vstack((np.hstack((dst[0], dst[1])), np.hstack((dst[2], dst[3])))))

    def get_jacobi(self, parent):
        img = np.array(self.parents[0].value)
        r, c = img.shape
        step = 2
        ind_list_r = np.arange(0, r, step)
        ind_list_c = np.arange(0, c, step)
        ksize_r, ksize_c = (2, 2)
        ind_list_r = ind_list_r[(ind_list_r + ksize_r) <= r]
        ind_list_c = ind_list_c[(ind_list_c + ksize_c) <= c]
        ind_list_r = np.tile(ind_list_r, (ksize_r, 1)).T
        ind_list_c = np.tile(ind_list_c, (ksize_c, 1)).T
        ind_list_r += np.arange(ksize_r)
        ind_list_c += np.arange(ksize_c)
        piece = img[ind_list_r]
        piece = piece[:, :, ind_list_c]
        piece = np.swapaxes(piece, 1, 2)
        res = self.dst
        ind = np.argwhere(np.bitwise_or(piece, piece == 0))[:, -2:]
        mask = np.arange(ind.shape[0]) // (ksize_c * ksize_r)
        ind[:, 0] += (mask // ind_list_c.shape[0]) * step
        ind[:, 1] += (mask % ind_list_c.shape[0]) * step

        jacobi_c_ind = np.ravel_multi_index(ind.T, (r, c))
        jacobi_c_ind = jacobi_c_ind.reshape((-1, ksize_c * ksize_r))
        jacobi1 = np.zeros((res[0].shape[0] * res[0].shape[1], img.shape[0] * img.shape[1]), dtype=np.float16)
        jacobi2 = np.zeros((res[0].shape[0] * res[0].shape[1], img.shape[0] * img.shape[1]), dtype=np.float16)
        jacobi3 = np.zeros((res[0].shape[0] * res[0].shape[1], img.shape[0] * img.shape[1]), dtype=np.float16)
        jacobi4 = np.zeros((res[0].shape[0] * res[0].shape[1], img.shape[0] * img.shape[1]), dtype=np.float16)
        jacobi1[
            np.tile(np.expand_dims(np.arange(res[0].shape[0] * res[0].shape[1]), axis=1),
                    (1, 4)), jacobi_c_ind.tolist()] = np.array(
            [0.25, 0.25, 0.25, 0.25])
        jacobi2[
            np.tile(np.expand_dims(np.arange(res[0].shape[0] * res[0].shape[1]), axis=1),
                    (1, 4)), jacobi_c_ind.tolist()] = np.array(
            [0.25, -0.25, 0.25, -0.25])
        jacobi3[
            np.tile(np.expand_dims(np.arange(res[0].shape[0] * res[0].shape[1]), axis=1),
                    (1, 4)), jacobi_c_ind.tolist()] = np.array(
            [0.25, 0.25, -0.25, -0.25])
        jacobi4[
            np.tile(np.expand_dims(np.arange(res[0].shape[0] * res[0].shape[1]), axis=1),
                    (1, 4)), jacobi_c_ind.tolist()] = np.array(
            [0.25, -0.25, -0.25, 0.25])

        jacobi = np.zeros((self.dimension(), self.parents[0].dimension()), dtype=np.float16)
        middle_ind = self.dimension() / 2
        assert middle_ind % 1 == 0
        middle_ind = int(middle_ind)
        jacobi_index = np.arange(0, middle_ind)
        jacobi[:middle_ind][jacobi_index % (res[0].shape[1] * 2) // res[0].shape[1] == 0] = jacobi1
        jacobi[:middle_ind][jacobi_index % (res[0].shape[1] * 2) // res[0].shape[1] == 1] = jacobi2
        jacobi[middle_ind:][jacobi_index % (res[0].shape[1] * 2) // res[0].shape[1] == 0] = jacobi3
        jacobi[middle_ind:][jacobi_index % (res[0].shape[1] * 2) // res[0].shape[1] == 1] = jacobi4
        return np.mat(jacobi)


class MaxPooling(Operator):
    """
    最大值池化
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.stride = kargs.get('stride')
        assert self.stride is not None
        self.stride = tuple(self.stride)
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert self.size is not None
        self.size = tuple(self.size)
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self):
        data = self.parents[0].value  # 输入特征图
        w, h = data.shape  # 输入特征图的宽和高
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(
                    np.max(window)
                )

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):

        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


class MaxPooling2d(Operator):
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.step = kargs.get('step', 2)
        self.size = kargs.get('ksize', (2, 2))
        self.size = self.size if len(self.size) == 2 else (2, 2)
        self.step = self.step if isinstance(self.step, int) else self.size[0]

    def compute(self):
        img = np.array(self.parents[0].value)
        r, c = img.shape
        step = self.step
        ind_list_r = np.arange(0, r, step)
        ind_list_c = np.arange(0, c, step)
        ksize_r, ksize_c = self.size
        ind_list_r = ind_list_r[(ind_list_r + ksize_r) <= r]
        ind_list_c = ind_list_c[(ind_list_c + ksize_c) <= c]
        ind_list_r = np.tile(ind_list_r, (ksize_r, 1)).T
        ind_list_c = np.tile(ind_list_c, (ksize_c, 1)).T
        ind_list_r += np.arange(ksize_r)
        ind_list_c += np.arange(ksize_c)
        piece = img[ind_list_r]
        piece = piece[:, :, ind_list_c]
        piece = np.swapaxes(piece, 1, 2)
        res = np.max(np.max(piece, axis=3), axis=2)
        self.value = res
        self.piece = piece
        self.ind_list_c = ind_list_c
        self.ind_list_r = ind_list_r

    def get_jacobi(self, parent):
        piece = self.piece
        step = self.step
        ksize_r, ksize_c = self.size
        r, c = self.parents[0].shape()
        ind = np.array(np.unravel_index(np.argmax(piece.reshape((piece.shape[0], piece.shape[1], 1, -1), ), axis=3),
                                        (ksize_r, ksize_c)))
        ind = np.swapaxes(ind, 0, 1)
        ind = np.swapaxes(ind, 1, 2).squeeze()
        ind = np.ravel(ind).reshape((-1, 2))
        mask = np.arange(ind.shape[0])
        ind_list_c = self.ind_list_c
        ind[:, 0] += mask // (ind_list_c.shape[0]) * step
        ind[:, 1] += mask % (ind_list_c.shape[0]) * step
        jacobi_c_ind = np.ravel_multi_index(ind.T, (r, c))
        jacobi_r_ind = np.arange(self.dimension())
        jacobi = np.zeros((jacobi_r_ind.shape[0], self.parents[0].dimension()))
        jacobi[jacobi_r_ind, jacobi_c_ind] = 1
        return np.mat(jacobi)


class AveragePooling2d(Operator):
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.step = kargs.get('step', 2)
        self.size = kargs.get('ksize', (2, 2))
        self.size = self.size if len(self.size) == 2 else (2, 2)
        self.step = self.step if isinstance(self.step, int) else self.size[0]

    def compute(self):
        img = np.array(self.parents[0].value)
        step = self.step
        size = self.size
        r, c = img.shape
        ind_list_r = np.arange(0, r, step)
        ind_list_c = np.arange(0, c, step)
        ksize_r, ksize_c = size
        ind_list_r = ind_list_r[(ind_list_r + ksize_r) <= r]
        ind_list_c = ind_list_c[(ind_list_c + ksize_c) <= c]
        ind_list_r = np.tile(ind_list_r, (ksize_r, 1)).T
        ind_list_c = np.tile(ind_list_c, (ksize_c, 1)).T
        ind_list_r += np.arange(ksize_r)
        ind_list_c += np.arange(ksize_c)
        piece = img[ind_list_r]
        piece = piece[:, :, ind_list_c]
        piece = np.swapaxes(piece, 1, 2)
        self.piece = piece
        self.value = np.mat(np.average(piece.reshape((piece.shape[0], piece.shape[1], -1)), axis=2))
        self.ind_list_c = ind_list_c

    def get_jacobi(self, parent):
        piece = self.piece
        ksize_r, ksize_c = self.size
        step = self.step
        ind_list_c = self.ind_list_c
        r, c = self.parents[0].shape()
        ind = np.argwhere(np.bitwise_or(piece, piece == 0))[:, -2:]
        mask = np.arange(ind.shape[0]) // (ksize_c * ksize_r)
        ind[:, 0] += (mask // ind_list_c.shape[0]) * step
        ind[:, 1] += (mask % ind_list_c.shape[0]) * step
        jacobi_c_ind = np.ravel_multi_index(ind.T, (r, c))
        jacobi = np.zeros((self.dimension(), r * c))
        jacobi[mask, jacobi_c_ind] = 1 / (ksize_c * ksize_r)
        return np.mat(jacobi)


class Concat(Operator):
    """
    将多个父节点的值连接成向量
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        # self.axis = kargs.get('axis', 0)
        # assert self.axis == 0 or self.axis == 1
        # print(self.parents[0].shape())
        # print(self.parents[1].shape())

    def compute(self):
        assert len(self.parents) > 0

        # 将所有父节点矩阵展平并连接成一个向量dim=(n,1)

        # print(self.name)
        # a=np.mat(np.concatenate([p.value.flatten() for p in self.parents], axis=0))
        self.value = np.mat(np.concatenate([p.value for p in self.parents], axis=0))

    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]  # 各个父节点的元素数量
        pos = self.parents.index(parent)  # 当前是第几个父节点
        dimension = parent.dimension()  # 当前父节点的元素数量

        assert dimension == dimensions[pos]

        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension,
        0:dimension] = np.eye(dimension)

        return np.mat(jacobi)


class ScalarMultiply(Operator):
    """
    用标量（1x1矩阵）数乘一个矩阵
    """

    def compute(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]


class Step(Operator):

    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi(self, parent):
        np.mat(np.eye(self.dimension()))
        return np.zeros(np.where(self.parents[0].value.A1 >= 0.0, 0.0, -1.0))


class Welding(Operator):

    def compute(self):
        assert len(self.parents) == 1 and self.parents[0] is not None
        self.value = self.parents[0].value

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

    def weld(self, node):
        """
        将本节点焊接到输入节点上
        """

        # 首先与之前的父节点断开

        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)

        self.parents.clear()

        # 与输入节点焊接
        self.parents.append(node)
        node.children.append(self)


class L2Normsum(Operator):
    def compute(self):
        assert len(self.parents) == 1
        self.value = np.mat(np.linalg.norm(self.parents[0].value) ** 2)

    def get_jacobi(self, parent):
        assert len(self.parents) == 1
        return np.mat(2 * self.parents[0].value.A1)


class Broadcast(Operator):
    def __init__(self, dim, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.dim = dim

    def compute(self):
        assert len(self.parents) == 1 and len(self.dim) == 2
        self.value = np.mat(np.broadcast_to(self.parents[0].value, self.dim))

    def get_jacobi(self, parent):
        assert parent in self.parents
        if self.parents[0].shape() == (1, 1):
            return np.mat(np.ones((1, self.dimension())))
