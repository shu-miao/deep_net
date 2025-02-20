'''
激活函数
'''


'''
ReLU激活函数
'''
inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)
print(output)

# 简化过程
inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

for i in inputs:
    output.append(max(0,i))
print(output)

# 利用numpy简化
import numpy as np

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = np.maximum(0,inputs)
print(output)
print('Type of output:\n',type(output))

# 整流线性激活类
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

'''
将该函数应用于密集层的输出
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

X,y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense1.forward(X)
activation1.forward(dense1.output)
print('应用激活函数之前\n',dense1.output[:5])
print('应用激活函数之后\n',activation1.output[:5])


'''
Softmax激活函数
'''

layer_outputs = [4.8,1.21,2.385]
E = 2.71828182846 # 自然对数
exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)
print('指数值：\n',exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print('归一化的指数值：\n',norm_values) # 归一化：使用当前值除以总值，结果一定处在[0,1]之间
print('归一化值的总值：\n',sum(norm_values))

# 使用numpy的做法
layer_outputs = [4.8,1.21,2.385]
exp_values = np.exp(layer_outputs)
print('指数值：\n',exp_values)
norm_values = exp_values/np.sum(exp_values)
print('归一化的指数值：\n',norm_values)
print('归一化后的总值：\n',np.sum(norm_values))


'''
示例：轴如何影响numpy的求和
'''
layer_outputs = np.array([[4.8,1.21,2.385],
                          [8.9,-1.81,0.2],
                          [1.41,1.051,0.026]])
print('不使用axis参数的求和：\n',np.sum(layer_outputs))
print('使用默认值为None的axis参数的求和：\n',np.sum(layer_outputs,axis=None)) # 若没有指定轴，sum只是将所有维度的所有值求和
print('使用axis值为0的求和，即对所有列求和：\n',np.sum(layer_outputs,axis=0))

# 原始的对行求和
for i in layer_outputs:
    print(sum(i))
print('使用axis值为1的求和，即对所有行求和：\n',np.sum(layer_outputs,axis=1))
print('使用axis值为1的求和，即对所有行求和，同时保持维度不变：\n',np.sum(layer_outputs,axis=1,keepdims=True)) # 使用keepdims=True参数保持输出的维度和输入的维度相同，输入是一个（3，3）的矩阵，则输出是（3，1）的矩阵


'''
死神经元和爆炸值
'''
print(np.exp(1))
print(np.exp(10))
print(np.exp(100))
# print(np.exp(1000)) # 输入为1000时，就足以引起溢出异常
print(np.exp(-np.inf), np.exp(0)) # 当输入为负无穷时，指数函数趋向0；当输入为0时，指数函数趋向1


'''
Softmax类
'''
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) # 获取未归一化的值，减去最大值防止溢出异常，由于指数函数的特性，这个操作不会影响最终的输出
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True) # 对每个样本进行归一化
        self.output = probabilities # 传递输出

softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print(softmax.output)
softmax.forward([[-2,-1,0]])
print(softmax.output)

softmax.forward([[0.5,1,1.5]]) # 将输入全除以2
print(softmax.output) # 输出置信度由于指数化的非线性特性而发生了变化


'''
FC+ReLU+Softmax
'''
dense1 = Layer_Dense(2,3) # 第一层全连接
activation1 = Activation_ReLU() # ReLU激活函数层
dense2 = Layer_Dense(3,3) # 第二层全连接，输入为ReLU层的输出
activation2 = Activation_Softmax() # Softmax激活函数层

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

