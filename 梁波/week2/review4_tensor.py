import numpy as np

"""
张量
    将3个2*2的矩阵排列在一起，就可以称为一个3*2*2的张量
    张量是神经网络的训练中最为常见的数据形式
"""

def generate_tensor():
    """
    创建张量
    """
    tensor1 = np.array([
        [[1,2],[1,2]],[[1, 2], [1, 2]],[[1, 2], [1, 2]],
    ])
    print(tensor1, tensor1.shape, type(tensor1))
    tensor2 = np.ndarray((3, 2, 2), dtype=int)
    print(tensor2, tensor2.shape, type(tensor2))
    tensor3 = np.random.rand(3, 2, 2)
    print(tensor3, "\n随机3*2*2张量,[0,1)", tensor3.shape, type(tensor3))
    tensor4 = np.random.randint(1, 10, (3, 2, 2))
    print(tensor4, "\n随机3*2*2张量, [1, 10)", tensor4.shape, type(tensor4))
    print(tensor4.shape, "\n获取张量的维度")


def tensor_operation():
    """
    张量运算
    """
    tensor1 = np.random.randint(1,10, (2, 2, 2))
    print(tensor1, "\n随机3*2*2张量, [1, 10)", tensor1.shape, type(tensor1))
    # tensor2 = tensor1.transpose(1, 0, 2)
    # print(tensor2, "\n张量转置", tensor2.shape)
    # tensor3 = tensor1.transpose(2, 1, 0)
    # print(tensor3, "\n张量转置", tensor3.shape)
    tensor4 = tensor1.view()
    print(tensor4, "\n张量view", tensor4.shape)
    tensor1 = np.random.randint(1, 10, (2, 2, 2))
    print(tensor1, "\n随机3*2*2张量, [1, 10)", tensor1.shape, type(tensor1))
    print(tensor4)
    # TODO: transpose 和 view


if __name__ == '__main__':
    # generate_tensor()
    tensor_operation()