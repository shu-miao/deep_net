'''
添加层级
'''

import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs,np.array(weights).T) + biases # 第一层的计算
print('layer1_outputs:\n',layer1_outputs)

layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T) + biases2 # 第一层的输出就是第二层的输入
print('layer2_outputs:\n',layer2_outputs)


'''
nnfs库用于创建非线性数据
'''
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# 利用nnfs确保实验的重复性
nnfs.init() # nnfs.init()执行：1.将随机种子置为0；2.创建一个float32数据类型的默认值，并覆盖numpy的原始点积函数

X,y = spiral_data(samples=100,classes=3) # spiral_data创建一个数据集，samples为数据量，classes为类别数量
# X是一个二维数组，形状为（samples*classes，2），表示每个样本的特征（坐标）
# X[:,0]表示第一列，每个样本的x坐标；X[:,1]表示第二列，每个样本的y坐标
# y是这个数据集的类别数量，是一个一维数组，长度为samples*classes

plt.scatter(X[:, 0],X[:, 1])
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()


'''
密集层（全连接）（FC）类
'''
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        '''
        :param n_inputs: 输入层的神经元数量，也就是前一层（或输入数据）的特征数量
        :param n_neurons: 当前层的神经元数量
        '''
        # 随机初始化权重，生成的权重矩阵形状为（n_inputs,n_neurons），每个值从正态分布中随机抽取并乘以0.01
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        # 偏置初始化为形状为（1,n_neurons）的全零矩阵
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        '''
        前向传播方法
        :param inputs: 该层的输入
        :return: 全连接层的输出
        '''
        self.output = np.dot(inputs,self.weights) + self.biases # 输入*权重+偏置

# X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,3) # 输入数据特征数量为2，dense1层的神经元为3
dense1.forward(X)
print(dense1.output[:5])