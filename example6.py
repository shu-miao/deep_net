'''
优化
'''

'''
神经网络的输出实际上是置信度，对正确答案的更多置信度是更好的
因此，应努力增加正确的置信度并减少错误放置的置信度
'''

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
import numpy as np

# 全连接（FC）层
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

# ReLU激活函数层
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

# Softmax激活函数层
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) # 获取未归一化的值，减去最大值防止溢出异常，由于指数函数的特性，这个操作不会影响最终的输出
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True) # 对每个样本进行归一化
        self.output = probabilities # 传递输出

# 创建损失类
class Loss:
    # 需要子类继承并重写forward方法实现
    def calculate(self,output,y):
        '''
        :param output: 模型输出的置信度
        :param y: 真实标签/目标值
        :return: 平均损失
        '''
        sample_losses = self.forward(output,y) # 计算每个样本的损失
        data_loss = np.mean(sample_losses)
        return data_loss

# 子类继承实现分类交叉熵损失类
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred) # 获取批量样本数
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7) # 数值稳定处理

        # 处理不同格式的真实标签
        if len(y_true.shape) == 1: # 稀疏标签，如[2,5,3]
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2: # one-hot编码标签，如[[1,0,0],[0,1,0]]
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences) # 计算负对数（负对数似然）
        return negative_log_likelihoods

# 创建一个简单的数据集
nnfs.init() # 保持操作可复制性
X,y = vertical_data(samples=100,classes=3) # 创建一个简单的数据集
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap='brg') # 绘图
plt.show()

# 创建模型
dense1 = Layer_Dense(2,3) # 全连接层
activation1 = Activation_ReLU() # ReLU激活函数层
dense2 = Layer_Dense(3,3) # 全连接层
activation2 = Activation_Softmax() # Softmax激活函数层
loss_function = Loss_CategoricalCrossentropy() # 损失函数层

# 跟踪最佳损耗和相关权重
lowest_loss = 9999999 # 初始化一个很大的损失值，目的是在训练中找到更低的损失
best_dense1_weights = dense1.weights.copy() # 使用copy方法创建dense1层的权重的“分身”用以跟踪权重的变化
best_dense1_biases = dense1.biases.copy() # 跟踪dense1层的偏置的变化
best_dense2_weights = dense2.weights.copy() # 跟踪dense2层的权重的变化
best_dense2_biases = dense2.biases.copy() # 跟踪dense2层的偏置的变化

# 多次迭代
for iteration in range(10000):
    # 为权重和偏置值选择随机值，且维度与该层维度保持一致
    dense1.weights = 0.05 * np.random.randn(2,3)
    dense1.biases = 0.05 * np.random.randn(1,3)
    dense2.weights = 0.05 * np.random.randn(3,3)
    dense2.biases = 0.05 * np.random.randn(1,3)

    dense1.forward(X) # 数据集进入dense1层的前向传播
    activation1.forward(dense1.output) # 进入activation1（ReLU）层的前向传播
    dense2.forward(activation1.output) # 进入dense2层的前向传播
    activation2.forward(dense2.output) # 进入activation2（Softmax）层的前向传播

    loss = loss_function.calculate(activation2.output,y) # 计算损失

    # 从activation2层的输出和目标y计算准确率
    predictions = np.argmax(activation2.output,axis=1) # 获取预测结果
    accuracy = np.mean(predictions==y) # 计算准确率

    # 如果当前损失小于最低损失
    if loss < lowest_loss:
        print('纯随机迭代，找到一组新的权重，iteration:',iteration,'loss:',loss,'accuracy:',accuracy)
        # 跟踪最新的权重和偏置
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss # 更新最低损失
    '''
    观察结果发现损失确实是在持续下降，不过幅度不大，且准确率并没有持续提升，
    每次权重和偏置的更新都是随机选择，这会导致迭代时间过长，训练过程全靠运气，
    由此可见，无目的随机更新不是一个可靠的方法，
    一个可行的方法是，部分应用随机选择的值来优化参数，而不是每次都完全依赖随机选择，
    不在每次迭代中都随机选择值来设置权重和偏置，而是应用这些值的一部分到参数上，
    通过这种方法，权重将根据当前最低损失的结果进行更新，而不是无目的地随机更新，减小了随机性带来的噪声
    另外，如果当前的调整减小了损失， 则将其设置为新的调整起点，
    如果由于调整导致损失增加，则回到之前的点
    如此，便将无目的的更新改为有目的迭代（梯度下降），可大幅加速模型的收敛 
    '''

# 创建模型
dense1 = Layer_Dense(2,3) # 全连接层
activation1 = Activation_ReLU() # ReLU激活函数层
dense2 = Layer_Dense(3,3) # 全连接层
activation2 = Activation_Softmax() # Softmax激活函数层
loss_function = Loss_CategoricalCrossentropy() # 损失函数层

# 跟踪最佳损耗和相关权重
lowest_loss = 9999999 # 初始化一个很大的损失值，目的是在训练中找到更低的损失
best_dense1_weights = dense1.weights.copy() # 使用copy方法创建dense1层的权重的“分身”用以跟踪权重的变化
best_dense1_biases = dense1.biases.copy() # 跟踪dense1层的偏置的变化
best_dense2_weights = dense2.weights.copy() # 跟踪dense2层的权重的变化
best_dense2_biases = dense2.biases.copy() # 跟踪dense2层的偏置的变化

# 迭代多次
for iteration in range(10000):
    # 使用累加一个随机值，对权重和偏置进行小幅调整，而不是直接选择一个随机值
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)  # 数据集进入dense1层的前向传播
    activation1.forward(dense1.output)  # 进入activation1（ReLU）层的前向传播
    dense2.forward(activation1.output)  # 进入dense2层的前向传播
    activation2.forward(dense2.output)  # 进入activation2（Softmax）层的前向传播

    loss = loss_function.calculate(activation2.output, y)  # 计算损失

    # 从activation2层的输出和目标y计算准确率
    predictions = np.argmax(activation2.output, axis=1)  # 获取预测结果
    accuracy = np.mean(predictions == y)  # 计算准确率

    # 如果当前损失小于最低损失
    if loss < lowest_loss:
        print('梯度下降迭代，找到一组新的权重，iteration:',iteration,'loss:',loss,'accuracy:',accuracy)
        # 跟踪最新的权重和偏置
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss  # 更新最低损失
    # 如果当前损失不小于最低损失
    else:
        # 回退调整
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense2_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

'''
使用螺旋数据集
'''

from nnfs.datasets import spiral_data
X,y = spiral_data(samples=100,classes=3)
# 创建模型
dense1 = Layer_Dense(2,3) # 全连接层
activation1 = Activation_ReLU() # ReLU激活函数层
dense2 = Layer_Dense(3,3) # 全连接层
activation2 = Activation_Softmax() # Softmax激活函数层
loss_function = Loss_CategoricalCrossentropy() # 损失函数层

# 跟踪最佳损耗和相关权重
lowest_loss = 9999999 # 初始化一个很大的损失值，目的是在训练中找到更低的损失
best_dense1_weights = dense1.weights.copy() # 使用copy方法创建dense1层的权重的“分身”用以跟踪权重的变化
best_dense1_biases = dense1.biases.copy() # 跟踪dense1层的偏置的变化
best_dense2_weights = dense2.weights.copy() # 跟踪dense2层的权重的变化
best_dense2_biases = dense2.biases.copy() # 跟踪dense2层的偏置的变化

# 迭代多次
for iteration in range(10000):
    # 使用累加一个随机值，对权重和偏置进行小幅调整，而不是直接选择一个随机值
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)  # 数据集进入dense1层的前向传播
    activation1.forward(dense1.output)  # 进入activation1（ReLU）层的前向传播
    dense2.forward(activation1.output)  # 进入dense2层的前向传播
    activation2.forward(dense2.output)  # 进入activation2（Softmax）层的前向传播

    loss = loss_function.calculate(activation2.output, y)  # 计算损失

    # 从activation2层的输出和目标y计算准确率
    predictions = np.argmax(activation2.output, axis=1)  # 获取预测结果
    accuracy = np.mean(predictions == y)  # 计算准确率

    # 如果当前损失小于最低损失
    if loss < lowest_loss:
        print('梯度下降迭代，使用螺旋数据集，找到一组新的权重，iteration:',iteration,'loss:',loss,'accuracy:',accuracy)
        # 跟踪最新的权重和偏置
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss  # 更新最低损失
    # 如果当前损失不小于最低损失
    else:
        # 回退调整
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense2_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
    '''
    可以发现使用螺旋数据集后，训练几乎毫无进展，损失略有减少，精度略高于初始值
    造成这种现象的原因是数据复杂度的增加，以及”损失的局部最小值“
    '''