import numpy as np
import matplotlib.pyplot as plt

# 输入数据
# 4个数据分别对应0-0异或，0-1异或，1-0异或，1-1异或
X = np.array([  # 第一列都为偏置值
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# 标签，分别对应4种异或情况的结果
# 注意：这里我们使用 -1 作为负标签
T = np.array([
    [-1],
    [1],
    [1],
    [-1]
])

# 权值初始化，3行1列
# np.random.random 可以生成 0到1 的随机数
W = np.random.random([3, 1])

# 学习率设置
lr = 0.1

# 神经网络输出
Y = 0


# 更新一次权值
def train():
    global X, W, Y, lr  # 使用全局变量
    Y = np.dot(X, W)  # 计算网路预测值
    delat_W = lr * (X.T.dot(T - Y)) / X.shape[0]  # 计算权值的改变
    W = W + delat_W  # 更新权值


# 训练100次
for i in range(100):
    train()  # 每次循环更新一次权值

# 画图
# 正样本
x1 = [0, 1]
y1 = [1, 0]

# 负样本
x2 = [0, 1]
y2 = [0, 1]

# 计算分界线的斜率以及截距
k = - W[1] / W[2]
d = - W[0] / W[2]

xdata = (-2, 3)  # 设定两个点
plt.plot(xdata, xdata * k + d, 'r')  # 通过两点来确定一条直线，用红色的线画出分解线
plt.scatter(x1, y1, c='b')  # 用蓝色的点画出正样本
plt.scatter(x2, y2, c='y')  # 用黄色的点画出负样本
plt.show()
