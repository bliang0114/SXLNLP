import numpy as np

"""
numpy 常用api
"""

scalar = np.random.randint(1, 10)
print(scalar)
vector = np.random.randint(1, 10, 5)
print(vector)
matrix = np.random.randint(-10, 1, (3, 4))
print(matrix)
tensor = np.random.randint(1, 10, (3, 4, 2))
print(tensor)

# dim
# scalar.dim()
# vector.dim()
# matrix.dim()
# np.dim()

# shape
print(matrix.shape)

# size
print(matrix.size)

# dtype
print(matrix.dtype)

# arange
print(np.arange(1, 10))

# ones
print(np.ones((3,4), dtype=int))
# zeros
print(np.zeros((3,4), dtype=int))

# full
print(np.full((3,4), 100))

# eye
print(np.eye(3, dtype=int))

# linspace
print(np.linspace(1, 10, 5, dtype=int))

# abs fabs 绝对值
print(np.abs(matrix), "绝对值")

# sqrt 求平方
print(np.sqrt(25), "平方")

# square
print(np.square(5), "平方")

# log log2 log10
print(np.log2(8))
print(np.log(9))
print(np.log10(100))
# cell 向上取整，向下取整
print(np.ceil(5.2))
print(np.floor(5.2))
# rint 四舍五入
print(np.rint(5.2))
# modf 浮点数拆分
print(np.modf(5.2))
# sin cos tan
print(np.sin(0.5), np.cos(0.5), np.tan(0.5))
# exp 指数值
print(np.exp(0.5))
# sign 获取符号值 -1 ：- ；0 ：0 ；1 ：+
print(np.sign(0))
# sum
print(np.sum(matrix))
# mean 计算期望
print(np.mean(matrix))
# average 计算加权平均值
print(np.average(matrix))
# std 计算标准差
print(np.std(matrix))
# min max
print(np.min(matrix), np.max(matrix))
# ptp 计算极差,最大值与最小值的差
print(np.ptp(matrix))
# median 计算中位数
print(np.median(matrix))
# reshape 改变形状
print(matrix.reshape((2, 6)))
# resize
matrix.resize((2, 6))
print(matrix)
# swapaxes 交换维度
print(matrix.swapaxes(0, 1))
# flatten 展平
print(matrix.flatten())
# seed
for i in range(5):
    np.random.seed(4) # 只作用一次
    print(np.random.rand())