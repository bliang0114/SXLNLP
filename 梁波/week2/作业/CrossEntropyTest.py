import numpy as np
import torch
import matplotlib.pyplot as plt

"""
任务描述：
    已知x1,x2,x3；
    1. 若 x1 > x2 > x3, y = 2
    2. 若 x1 < x2 < x3, y = 1
    3. 其他情况, y = 0
"""

# 根据输入 x1,x2,x3 定义输入维度为 3
in_size = 3
# 根据 y = 0, 1, 2 定义输出维度为 3
out_size = 3

def build_example():
    """
    随机构建单个样本
    """
    data = np.random.rand(in_size)  # 随机构建一个三维向量
    if data[0] > data[1] > data[2]:
        return data, 2
    elif data[0] < data[1] < data[2]:
        return data, 1
    else:
        return data, 0


def build_dataset(sample_num):
    """
    构建样本数据集
    Args:
        sample_num: 样本数量

    Returns: 样本数据集

    """
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_example()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


class CrossEntropyModel(torch.nn.Module):
    """
    定义基于交叉熵的模型
    """

    def __init__(self):
        super(CrossEntropyModel, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)  # 线性层
        self.activation = torch.relu  # 激活函数
        self.loss = torch.nn.functional.cross_entropy  # 交叉熵损失函数

    def forward(self, x, y=None):
        """
        定义前向传播过程
        """
        y_pred = self.linear(x)  # 通过线性层，获取预测值
        y_pred = self.activation(y_pred) # 使用了激活函数，反而误差更大？
        if y is not None:
            # print("y_pred:", y_pred.shape, "\ny_true:", y.shape)
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    learning_rate = 0.001  # 学习率

    model = CrossEntropyModel()  # 模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    train_x, train_y = build_dataset(train_sample)
    log = []
    for epoch in range(epoch_num):
        model.train()   # 训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果的正确率
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.pt")  # 保存模型
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def evaluate(model):
    '''将模型设置为评估模式（evaluation mode）。在评估模式下，模型的权重不会被更新，而且不会记录梯度'''
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def predict(model_path, input_vec):
    model = CrossEntropyModel()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # print(vec)
        # print(res)
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果

def test():
    test_vec = [[1, 2, 3],
                [3, 2, 1],
                [1, 3, 2],
                [1, 1, 1]]
    predict("model.pt", test_vec)


if __name__ == '__main__':
    # print(build_dataset(10))
    main()
    test()
