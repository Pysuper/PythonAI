import numpy as np
import matplotlib.pyplot as plt

# 定义输入数据
X = np.array([  # 四行三列
    [1, 3, 3],
    [1, 4, 3],
    [1, 1, 1],
    [1, 2, 1]
])

# 定义标签
T = np.array([
    [1],  # 四行一列
    [1],
    [-1],
    [-1]
])

# 权值初始化
W = np.random.random([3, 1])  # 生成3行一列的（左右侧是多少个）

# 设置学习率
lr = 0.1

# 神经网络输出
Y = 0


def train():
    """更新权值"""
    global X, Y, W, T, lr  # 作为全局变量
    Y = np.sign(np.dot(X, W))  # 同时计算四个数据的预测值
    E = T - Y  # 得到的四个标签与预测值的误差值E(4,1)
    delta_W = lr * (X.T.dot(E) / X.shape[0])  # 计算权值的变化
    W = W + delta_W  # 更新权值


# 训练模型
for i in range(100):
    train()  # 更新权值
    print("epoch:", i + 1)  # 当前训练次数
    print("weights:", W)  # 当前的权值
    Y = np.sign(np.dot(X, W))  # 计算当前输出
    if (Y == T).all():  # all() 表示Y中所有值跟T中的所有值都对应相等，才为真
        print("Finished")
        break  # 跳出循环

# 最后结果画图
# 正样本xy坐标
x1 = [3, 4]
y1 = [3, 3]

# 负样本xy坐标
x2 = [1, 2]
y2 = [1, 1]

# 分类边界线
# w0 + w1*x1 + w2*x2 =>> 分类边界线 0
# 定义分类边界线的斜率换个截距
k = -W[1] / W[2]
d = -W[0] / W[2]
# 设定两个点
xdata = (0, 5)
# 通过来确定一条直线，用红色的线画出分界线
plt.plot(xdata, xdata * k + d, 'r')
# 用蓝色的线画正样本
plt.scatter(x1, y1, c='b')
# 用黄色的点画负样本
plt.scatter(x2, y2, c='y')
plt.show()
