import matrixslow as ms
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def train(lr=0.00015, days=7, epochs=30, batch_size=4):
    data_csv = pd.read_csv('Daily_ZX.csv')
    data = data_csv.to_numpy()[:, 2:].astype(np.float64)
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # 特征标准化

    train_data = data[:4000, :]
    test_data = data[4000:, :]
    np.save('./data/train_data.npy', train_data)
    np.save('./data/test_data.npy', test_data)

    # 构造计算图，输入节点数为days，输出一个节点
    feature_num = 9
    n = 10
    inputs = [ms.core.Variable(dim=(feature_num, 1), init=False, trainable=False, name=f'input_vec_{i}') for i in
              range(days)]  # 输入节点列表

    last_step = None
    for ind, input_vec in enumerate(inputs):
        if last_step is None:
            h = ms.Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_h_input_{ind}')
            c = ms.Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_C_{ind}')
        h, c = ms.layer.LSTM(input_vec, h, c, id=ind, n=n,
                             l=feature_num)  # id表示LSTM层所在的层数；n表示记忆节点H和节点C的维度，l表示输入节点x的维度
        last_step = h

    fc1 = ms.layer.fc(last_step, n, 40, "ReLU")  # 记忆节点到输出节点的全连接层
    output = ms.layer.fc(fc1, 40, 1, "ReLU")
    label = ms.core.Variable((1, 1), trainable=False)
    loss = ms.ops.loss.MeanSquaredErrorLoss(output, label)  # 训练使用的loss函数为MSE
    test_loss = ms.ops.MAELoss(output, label)  # 测试使用的loss函数为MAE，MAE能直观体现预测和标签的差异
    learning_rate = lr
    optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

    test_loss_list_eachepoch = []
    train_loss_list_eachepoch = []

    for epoch in range(epochs):
        loss_one_iter = []
        for itr_count in range(500):  # 每个epoch进行1000次minibatch梯度下降
            pos = np.random.randint(len(train_data) - 1 - days - batch_size)
            # 单次梯度下降过程
            loss_list = []
            for batch_k in range(batch_size):
                # 提取连续的时间序列，赋给输入节点
                start = pos + batch_k
                seq = data[start: start + days]
                for j, x in enumerate(inputs):
                    x.set_value(np.mat(np.expand_dims(seq[j].T, axis=1)))
                # 给标签节点赋值
                target_ch = data[start + days, 0]
                label.set_value(np.mat(target_ch))
                optimizer.one_step()
                loss_list.append(loss.value[0, 0])
            loss_one_iter.append(np.mean(loss_list))
            print(f'epoch:{epoch}\t\titer_count:{itr_count}\t\tMSELoss:{round(loss.value[0, 0], 9)}', end='\n')
            optimizer.update()

        train_loss_list_eachepoch.append(np.mean(loss_one_iter))

        test_loss_list = []
        # 每过一个epoch测试一次,在1000个验证样本中取500个
        print('testing...')
        for itr_count in range(500):
            print(f'{itr_count}/500', end='')
            pos = np.random.randint(len(test_data) - 1 - days)
            seq = test_data[pos: pos + days]
            for j, x in enumerate(inputs):
                x.set_value(np.mat(np.expand_dims(seq[j].T, axis=1)))
            label.set_value(np.mat(test_data[pos + days, 0]))
            test_loss.forward()
            test_loss_list.append(test_loss.value)
            print('\r', end='')
        print(
            f'epoch:{epoch}\ttesting loss using MAE\texpectation:{np.round(np.mean(test_loss_list), 8)}\tvariance:{np.round(np.var(test_loss_list), 8)}')
        test_loss_list_eachepoch.append(np.round(np.mean(test_loss_list), 8))
        saver = ms.trainer.Saver(root_dir=f'./model/epoch{epoch}')  # 每个epoch保存模型
        saver.save(graph=ms.default_graph)

    min_loss, best_epoch = np.min(test_loss_list_eachepoch), np.argmin(test_loss_list_eachepoch)
    with open('result/training_result.txt', 'w') as f:
        f.write(f'The best model for validation is epoch{best_epoch}.\nvalidation_loss = {min_loss}')

    # 画出测试loss收敛曲线
    plt.figure()
    plt.plot(test_loss_list_eachepoch)
    plt.plot(train_loss_list_eachepoch)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['validation loss', 'training loss'])
    plt.show()
    print('Done.')


def test(best_epoch, days=7):  # 取所有的验证样本进行验证
    test_data = np.load('./data/test_data.npy')
    feature_num = 9
    n = 10
    inputs = [ms.core.Variable(dim=(feature_num, 1), init=False, trainable=False, name=f'input_vec_{i}') for i in
              range(days)]  # 输入节点列表

    last_step = None
    for ind, input_vec in enumerate(inputs):
        if last_step is None:
            h = ms.Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_h_input_{ind}')
            c = ms.Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name=f'LSTM_C_{ind}')
        h, c = ms.layer.LSTM(input_vec, h, c, id=ind, n=n,
                             l=feature_num)  # id表示LSTM层所在的层数；n表示记忆节点H和节点C的维度;l表示输入节点x的维度
        last_step = h

    fc1 = ms.layer.fc(last_step, n, 40, "ReLU")  # 记忆节点到输出节点的全连接层
    output = ms.layer.fc(fc1, 40, 1, "ReLU")
    label = ms.core.Variable((1, 1), trainable=False)
    test_loss = ms.ops.MAELoss(output, label)  # 测试使用的loss函数为MAE，因为在这里MAE能直观体现预测和标签的差异

    saver = ms.trainer.Saver(f'./model/epoch{best_epoch}/')
    saver.load(model_file_name='model.json', weights_file_name='weights.npz')
    test_loss_list = []

    '''
    在测试集中选取长度为200的连续序列查看标准化后的输出结果
    '''
    length = 200
    pos = np.random.randint(len(test_data) - 1 - 2 * days - length)
    real = test_data[pos:pos + length, 0]
    pred = []
    print('Preparing prediction curve...')
    for i in range(length):
        print(f'{i}/{length}', end='')
        seq = test_data[pos - days + i: pos + i]
        for j, x in enumerate(inputs):
            x.set_value(np.mat(np.expand_dims(seq[j].T, axis=1)))
        label.set_value(np.mat(test_data[pos + days, 0]))
        test_loss.forward()
        pred.append(output.value[0, 0])
        print('\r', end='')
    pred = np.array(pred)
    plt.figure()
    plt.plot(real)
    plt.plot(pred)
    plt.title(f'real opening price compared with prediction (predict by {days} days)')
    plt.legend(['real', 'prediction'])
    plt.savefig(f'images/real opening price compared with prediction (predict by {days} days).png')
    plt.show()

    print('Testing...')
    for itr_count in range(1000 - days):
        print(f'{itr_count}/{1000 - days + 1}', end=' ')
        pos = itr_count
        seq = test_data[pos: pos + days]
        for j, x in enumerate(inputs):
            x.set_value(np.mat(np.expand_dims(seq[j].T, axis=1)))

        label.set_value(np.mat(test_data[pos + days, 0]))
        test_loss.forward()
        test_loss_list.append(test_loss.value)
        print('\r', end='')
    test_result = f'validation loss using MAE\nexpectation:{np.round(np.mean(test_loss_list), 8)}\nvariance:{np.round(np.var(test_loss_list), 8)}'
    with open('./result/validation_result.txt', 'w') as f:
        f.write(test_result)
    print('Done.')


def train_MLP(lr=0.0001, days=7, epochs=30, batch_size=4):
    data_csv = pd.read_csv('Daily_ZX.csv')
    data = data_csv.to_numpy()[:, 2:].astype(np.float64)
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))  # 特征标准化

    train_data = data[:4000, :]
    test_data = data[4000:, :]
    np.save('./data/train_data.npy', train_data)
    np.save('./data/test_data.npy', test_data)

    # 构造MLP，输入维度为9*days，输出维度为1
    feature_num = 9
    n = 128
    input = ms.Variable((feature_num * days, 1), init=False, trainable=False, name='input')
    W1 = ms.Variable((n, feature_num * days), init=('Gaussian', 0, 0.005), trainable=True, name='W1')
    b1 = ms.Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name='b1')

    W2 = ms.Variable((n, n), init=('Gaussian', 0, 0.005), trainable=True, name='W2')
    b2 = ms.Variable((n, 1), init=('Gaussian', 0, 0.005), trainable=True, name='b2')

    W3 = ms.Variable((1, n), init=('Gaussian', 0, 0.005), trainable=True, name='W3')
    b3 = ms.Variable((1, 1), init=('Gaussian', 0, 0.005), trainable=True, name='b3')

    fc1 = ms.ops.ReLU(ms.ops.Add(ms.ops.MatMul(W1, input), b1), name='ReLU1')
    fc2 = ms.ops.ReLU(ms.ops.Add(ms.ops.MatMul(W2, fc1), b2), name='ReLU2')
    output = ms.ops.Tanh(ms.ops.Add(ms.ops.MatMul(W3, fc2), b3), name='output')
    label = ms.core.node.Variable((1, 1), trainable=False)
    loss = ms.ops.Add(ms.ops.loss.MAELoss(output, label), ms.ops.loss.MeanSquaredErrorLoss(output, label))
    test_loss = ms.ops.MAELoss(output, label)  # 测试使用的loss函数为MAE
    learning_rate = lr
    optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

    test_loss_list_eachepoch = []
    train_loss_list_eachepoch = []

    for epoch in range(epochs):
        loss_one_iter = []
        for itr_count in range(1000):  # 每个epoch进行1000次minibatch梯度下降
            pos = np.random.randint(len(train_data) - 1 - days - batch_size)
            # 单次梯度下降过程
            loss_list = []
            for batch_k in range(batch_size):
                # 提取连续的时间序列，赋给输入节点
                start = pos + batch_k
                seq = data[start: start + days]
                input.set_value(np.mat(seq.ravel()).T)
                # 给标签节点赋值
                target_ch = data[start + days, 0]
                label.set_value(np.mat(target_ch))
                optimizer.one_step()
                loss_list.append(loss.value[0, 0])
            loss_one_iter.append(np.mean(loss_list))
            print('\r', end='')
            print(f'epoch:{epoch}\t\titer_count:{itr_count}\t\tLoss:{round(loss.value[0, 0], 9)}', end='')
            optimizer.update()

        train_loss_list_eachepoch.append(np.mean(loss_one_iter))

        test_loss_list = []
        # 每过一个epoch测试一次,在1000个验证样本中取500个
        print('\ntesting...')
        for itr_count in range(500):
            print(f'{itr_count}/500', end='')
            pos = np.random.randint(len(test_data) - 1 - days)
            seq = test_data[pos: pos + days]
            input.set_value(np.mat(seq.ravel()).T)
            label.set_value(np.mat(test_data[pos + days, 0]))
            test_loss.forward()
            test_loss_list.append(test_loss.value)
            print('\r', end='')
        print(
            f'epoch:{epoch}\ttesting loss\texpectation:{np.round(np.mean(test_loss_list), 8)}\tvariance:{np.round(np.var(test_loss_list), 8)}')
        test_loss_list_eachepoch.append(np.round(np.mean(test_loss_list), 8))
        saver = ms.trainer.Saver(root_dir=f'./model_MLP/epoch{epoch}')  # 每个epoch保存模型
        saver.save(graph=ms.default_graph)

    min_loss, best_epoch = np.min(test_loss_list_eachepoch), np.argmin(test_loss_list_eachepoch)
    with open('result/training_result_MLP.txt', 'w') as f:
        f.write(f'The best model for validation is epoch{best_epoch}.\nvalidation_loss = {min_loss}')

    # 画出测试loss收敛曲线
    plt.figure()
    plt.plot(test_loss_list_eachepoch)
    plt.plot(train_loss_list_eachepoch)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['validation loss', 'training loss'])
    plt.show()
    print('Done.')
