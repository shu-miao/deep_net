'''
单个神经元的计算方式，输入乘以权重加上偏置
'''

inputs = [1,2,3] # 输入
weights = [0.2,0.8,-0.5] # 权重
bias = 2 # 偏置
output = (inputs[0]*weights[0]+
          inputs[1]*weights[1]+
          inputs[2]*weights[2]+bias)
print('output:\n',output)

inputs = [1.0,2.0,3.0,2.5] # 输入
weights = [0.2,0.8,-0.5,1.0] # 权重
bias = 2.0 # 偏置
output = (inputs[0]*weights[0]+
          inputs[1]*weights[1]+
          inputs[2]*weights[2]+
          inputs[3]*weights[3]+bias)
print('output:\n',output)

'''
一层神经元，每个神经元都有各自的权重和偏置
'''

inputs = [1,2,3,2.5] # 输入

weigths1 = [0.2,0.8,-0.5,1] # 权重1
weigths2 = [0.5,-0.91,0.26,-0.5] # 权重2
weigths3 = [-0.26,-0.27,0.17,0.87] # 权重3

bias1 = 2 # 偏置1
bias2 = 3 # 偏置2
bias3 = 0.5 # 偏置3

outputs = [
    # 第一个神经元
    inputs[0] * weigths1[0] +
    inputs[1] * weigths1[1] +
    inputs[2] * weigths1[2] +
    inputs[3] * weigths1[3] + bias1,
    # 第二个神经元
    inputs[0] * weigths2[0] +
    inputs[1] * weigths2[1] +
    inputs[2] * weigths2[2] +
    inputs[3] * weigths2[3] + bias2,
    # 第三个神经元
    inputs[0] * weigths3[0] +
    inputs[1] * weigths3[1] +
    inputs[2] * weigths3[2] +
    inputs[3] * weigths3[3] + bias3,
]

print('outputs:\n',outputs)


# 使用循环来扩展并动态处理输入和层的大小
inputs = [1,2,3,2.5]
# 将原本分开的权重变量写成一个权重列表
weights = [
    [0.2,0.8,-0.5,1],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-0.27,0.17,0.87]
]
biases = [2,3,0.5] # 偏置列表

layer_outputs = []

# 使用zip()方法通过neuron_weights和neuron_bias遍历weights和biases两个列表
for neuron_weights,neuron_bias in zip(weights,biases):
    neuron_output = 0 # 初始化神经元的输出为0
    for n_input,weight in zip(inputs,neuron_weights): # 遍历输入向量和权重向量得到神经元的输入和权重
        neuron_output = neuron_output + n_input*weight # 输入乘以权重
    neuron_output = neuron_output + neuron_bias # 加上偏置
    layer_outputs.append(neuron_output) # 将神经元的输出添加到层的输出列表

print('layer_outputs:\n',layer_outputs)
print(list(zip(weights, biases)))
print(list(zip(weights, biases))[0])
print(type(list(zip(weights, biases))[0]))


