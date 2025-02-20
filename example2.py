'''
神经元的进阶，神经元组成的层
'''

l = [1,5,6,2] # 简单的列表（向量）

lol = [[1,5,6,2],
       [3,2,1,3]] # 列表中的列表（矩阵）

lolol = [ [[1,5,6,2], # 列表中的列表中的列表（三维矩阵），形状为（3，2，4）
           [3,2,1,3]],
          [[5,2,1,2],
           [6,4,8,4]],
          [[2,8,5,3],
           [1,1,9,4]]]

another_list_of_lists = [[4,2,3], # 不规则的列表中的列表
                         [5,1]]

list_matrix_array = [[4,2], # 形状为（3，2）的矩阵
                     [5,1],
                     [8,2]]

a = [1,2,3]
b = [2,3,4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] # 点积计算
print('dot_product:\n',dot_product)

import numpy as np
# 借助numpy的能力完成神经元的计算
inputs = [1.0,2.0,3.0,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2.0

outputs = np.dot(weights,inputs) + bias # 输入乘以权重加上偏置
print('outputs:\n',outputs)

inputs = [1.0,2.0,3.0,2.5]
weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
biases = [2.0,3.0,0.5]

layer_outputs = np.dot(weights,inputs) + biases # （1，4）的输入向量和（3，4）的权重矩阵及（1，3）的偏置矩阵
print('layer_outputs:\n',layer_outputs)
print('Type of layer_outputs:\n',type(layer_outputs))


'''
向量到矩阵
'''
a = [1,2,3]
b = np.array([a]) # 使用numpy将向量转换为矩阵，[1,2,3]-->[[1,2,3]]
print('B:\n',b)
print('Type of B:\n',type(b))

# 另一种方法
b = np.expand_dims(np.array(a),axis=0) # np.expand_dims方法在第0轴上扩展数组的维度
print('B:\n',b)
print('Type of B:\n',type(b))


'''
矩阵的转置
'''
a = [1,2,3]
b = [2,3,4]
a = np.array([a])
b_T = np.array([b]).T # .T用于矩阵的转置
c = np.dot(a,b_T) # 点积
print('c:\n',c)


'''
使用numpy进行层级的计算
'''
inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2.0,3.0,0.5]

layer_outputs = np.dot(inputs,np.array(weights).T) + biases # 将形状为（3，4）的权重矩阵转置为（4，3）再与形状为（3，4）的输入进行点积计算
print('layer_outputs:\n',layer_outputs)
print('Type of layer_outputs:\n',type(layer_outputs))
