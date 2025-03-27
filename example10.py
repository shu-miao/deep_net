'''
优化器
'''

'''
在实现了反向传播之后，需要考虑的是如何利用梯度来减少损失的度量。
在example9的示例中，减去了每个权重和偏置的梯度的一部分，用来减少神经元激活函数的输出，
这种方法是一种被广泛使用的优化器，称为随机梯度下降SGD
在使用随机梯度下降时，会选择一个学习率，然后从实际参数值中减去学习率*参数梯度，
比如学习率为1.0，那么就会从参数中减去完整的参数梯度
'''

import numpy as np
from nnfs.datasets import spiral_data
import nnfs


# 网络层类
class Layer_Dense:
    def __init__(self,inputs,neurons):
        self.weights = 0.01 * np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

# ReLU激活函数层
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax激活函数层
class Activation_Softmax:
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_dvalues.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

# 损失函数层
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # 前向传播
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # 反向传播
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]  # 使用单位矩阵（从左上到右下的对角线上的值为1，其余为0）创建one_hot编码

        self.dinputs = -y_true / dvalues  # 分类交叉熵损失导数的计算：负的真实向量除以预测值向量
        self.dinputs = self.dinputs / samples

# Softmax和损失函数层
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        # 初始化方法
        self.activation = Activation_Softmax() # 创建Softmax激活函数的实例
        self.loss = Loss_CategoricalCrossentropy() # 创建一个分类交叉熵损失函数的实例

    def forward(self,inputs,y_true):
        # 前向传播
        self.activation.forward(inputs) # 调用Softmax激活函数的前向传播方法，计算并存储输出
        self.output = self.activation.output # 保存激活函数的输出，通常是每个类别的概率
        return self.loss.calculate(self.output,y_true) # 使用分类交叉熵损失计算并返回损失值

    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues) # 样本数量，等于dvalue（收到后续层的梯度）的长度
        if len(y_true.shape) == 2: # 检查y_true是否是独热编码格式
            y_true = np.argmax(y_true,axis=1) # 如果是，使用np.argmax将其转换为整数标签
        self.dinputs = dvalues.copy() # 复制梯度值以便进行修改
        self.dinputs[range(samples),y_true] -= 1 # 根据真实标签更新梯度，针对每个样本的正确类别减去1
        self.dinputs = self.dinputs / samples # 对梯度进行归一化，确保计算的梯度是平均值

# 优化器类
class Optimizer_SGD:
    def __init__(self,learning_rate=1.0):
        # 初始化学习率
        self.learning_rate = learning_rate
    def update_params(self,layer):
        # 更新参数
        layer.weights += -self.learning_rate * layer.dweights # 权重-学习率*权重梯度
        layer.biases += -self.learning_rate * layer.dbiases # 偏置-学习率*偏置梯度


# 实例化
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()

# 前向传播
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output,y)
print('loss:',loss)
predictions = np.argmax(loss_activation.output,axis=1)
if len(y.shape) == 2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(predictions==y)
print('acc:',accuracy)

# 反向传播
loss_activation.backward(loss_activation.output,y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# 使用优化器更新权重和偏置
optimizer.update_params(dense1)
optimizer.update_params(dense2)


# 将上述流程进行循环迭代
# 实例化
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)

    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print(f'epoch:{epoch},' +
              f'acc:{accuracy:.3f},' +
              f'loss:{loss:.3f}')

    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # 更新权重和偏置
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    '''
    这部分输出的准确率大约是0.8左右，且随着轮次的增加，损失和准确率并没有持续下降，
    从这个现象可以合理假设当前的学习率太高，导致模型陷入了局部最小值
    '''

'''
进一步，讨论学习率对优化器和损失函数的影响，
目前为止，已经得到了模型及其损失函数对所有参数的梯度，后续引入学习率来将这个梯度的一部分应用到参数上以降低损失值，
通常不会直接应用负梯度（即负的导数或偏导，当它们为负时，原函数呈下降趋势），因为函数最陡下降的方向会持续变化，而且这些值通常对于模型的有效改进来说太大了，
正确的做法是小步调整，计算梯度，通过负梯度的一部分更新参数，并在循环中重复这个过程，
但小步调整也有可能过小，导致模型陷入局部最小值，从而学习停滞
'''

# 将SGD优化器的学习率设置为0.85再次尝试
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
learning_rate = .85
optimizer = Optimizer_SGD(learning_rate=learning_rate)
# 迭代
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
        accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f}')

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    '''
    这部分的输出准确率仍然是0.8左右，损失也没有变得更好，
    另外，较低的损失并不总是与更高的准确性相关联，
    优化器的任务是减少损失，而不是直接提高准确性，
    损失是所有样本的平均值，这些样本的损失中的一些可能会显著下降，而其他的一些可能只是略微上升，同时将它们的预测从正确类别改为错误类别，这会导致总体上较低的平均损失，但也会导致更多预测不正确的样本，这同时会降低准确性
    这次输出的准确率较低的可能原因是模型找到了另一个局部最小值，
    不同的学习率并没有显示出这个学习率越低越好，
    通常的做法是从较大的学习率开始，并随时间/步骤逐渐降低学习率，
    于是便引入了学习率衰减，以保持初始更新的大幅度并在训练期间探索各种学习率
    '''


'''
学习率的衰减的概念是：
从较大的学习率开始，然后在训练过程中逐渐减小它，做法不止一种
其中一种是根据跨越多个训练周期的损失来降低学习率，例如，如果损失开始趋于平稳/达到平台期或开始跳跃大幅度变化，可以逻辑的编程这种行为，或者简单的追踪随时间的损失并在适当时手动降低学习率，
另一种是设置一个衰减率，它会稳定地按每个批次或周期降低学习率
'''

# 设置一个衰减率来衰减学习率
starting_learning_rate = 1. # 初始学习率
learning_rate_decay = 0.1 # 衰减率
step =  1 # 训练步数
learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step)) # 衰减后的学习率，随着步数增加，学习率逐渐减小
print(learning_rate)

# 在一个循环中模拟学习率的衰减
for step in range(20):
    learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step))
    print(learning_rate) # 输出的学习率随步数的增加而衰减


# 更新SGD优化器类,加入学习率衰减策略
class Optimizer_SGD:
    def __init__(self,learning_rate=1.0,decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay # 衰减率
        self.iterations = 0 # 迭代步数

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1 # 迭代步数加1

# 使用0.01（1e-2）的衰减率再次训练模型
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
decay = 1e-2 # 0.01的衰减率
optimizer = Optimizer_SGD(decay=decay)

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)

    # 每100步打印信息
    if not epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')

    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # 更新权重和偏置
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    这次输出的信息仍然不理想，合理的假设是因为学习率衰减得太快，使模型再次陷入了另一个局部最小值
    '''

'''
尝试将衰减率设为一个更小的值来稍微慢一些地进行衰减
'''
# 使用0.001（1e-3）的衰减率再次训练模型
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
decay = 1e-3 # 0.001的衰减率
optimizer = Optimizer_SGD(decay=decay)

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)

    # 每100步打印信息
    if not epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')

    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # 更新权重和偏置
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    理论上还有可能找到使结果更好的参数，通过调整初始学习率和衰减率，有可能使模型表现得更好
    '''


'''
引入学习率的随机梯度下降理论上可以做的很好，但仍然只是一种基本的优化方法，它只遵循梯度，没有任何可能帮助模型找到损失函数全局最小值的额外逻辑，
基于这个现象，引入动量来改良SGD优化器
动量会创建一个在一定更新次数上的梯度滚动平均值，并在每一步使用这个平均值和独特的梯度。
这个动量可以类比物理世界的动量来理解，当一个小球沿着山坡向下滚动（梯度下降），即使它遇到了一个小坑或者小山丘（局部最小值），惯性（动量）也会让它直接通过，向着山丘的底部滚动（全局最小值）
动量的计算过程是将前一次的更新值乘以动量因子，并减去当前梯度乘以学习率的值，从而平滑更新并加速收敛
'''

# 加入动量的SGD优化类
class Optimizer_SGD:
    def __init__(self,learning_rate=1.0,decay=0.,momentum=0.):
        self.learning_rate = learning_rate # 学习率
        self.current_learning_rate = learning_rate # 每次更新的学习率
        self.decay = decay # 衰减率
        self.iterations = 0 # 迭代步数
        self.momentum = momentum # 动量

    def pre_update_params(self):
        if self.decay:
            # 使用衰减率调整学习率
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
        if self.momentum:
            # 使用动量更新参数
            if not hasattr(layer,'weight_momentums'):
                # 如果layer层中还没有动量变量，则初始化为与权重和偏置形状相同的零矩阵
                layer.weight_momentums = np.zeros_like(layer.weights) # 权重动量
                layer.bias_momentums = np.zeros_like(layer.biases) # 偏置动量

            # 权重的更新，使用前一次的更新值乘以权重动量因子，再减去当前权重梯度乘以学习率的值
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # 偏置的更新，使用前一次的更新值乘以偏置动量因子，再减去当前偏置梯度乘以学习率的值
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # 不使用动量，则按照之前的更新策略
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1 # 步数加1

# 使用学习率为1，衰减率为0.001，动量为0.5，进行训练
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=1e-3,momentum=0.5)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    这部分输出可以看到准确率稳定上升，损失稳定下降，是目前最好的输出
    '''

# 观察动量对模型训练的影响，尝试将动量设置为0.9
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=1e-3,momentum=0.9)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    这部分输出的准确率达到0.9，损失下降到0.1，是目前最佳结果
    '''


'''
携带衰减率和动量的随机梯度优化器SGD已经完成，
接下来讨论自适应梯度优化器AdaGrad
自适应梯度AdaGrad为每个参数引入了独立的学习率，而不是全局共享学习率。
其核心思想是对特征的更新进行归一化。在训练过程中，某些权重可能会显著上升，而其他权重变化较小，而通常权重之间的差异不宜过大，
AdaGrad通过保留先前更新的历史记录来实现参数更新的归一化，即：如果参数更新的总和（无论是正还是负）越大，那么后续训练中进行的更新就越小，
这使得更新频率较低的参数能够跟上变化，从而更有效的利用更多的神经元进行训练

AdaGrad的计算:
cache += parm_gradient ** 2
parm_updates = learning_rate * parm_gradient / (sqrt(cache) + eps)
缓存cache保存了梯度平方的历史，它是一个累加的平方值，而参数更新parm_updates是学习率乘以梯度然后除以缓存的平方根加上一个e值的函数，
由于缓存的不断增加，这种除法操作可能会导致学习停滞，因为随着时间推移，更新变得越来越小，这也是AdaGrad优化器并不被广泛应用的原因，
e是一个超参数（预训练时的控制参数设置），用于防止分母为0的情况。e的值通常是一个很小的数，例如1e-7
'''

# 创建AdaGrad优化器类
class Optimizer_Adagrad:
    def __init__(self,learning_rate=1.,decay=0.,epsilon=1e-7):
        # 初始化
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay # 衰减率
        self.iterations = 0
        self.epsilon = epsilon # e值，用来防止除零错误

    def pre_update_params(self):
        # 衰减率
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            # 若layer层中没有参数缓存，则创建和权重/偏置形状相同的零矩阵
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # 权重和偏置的AdaGrad计算
        # 缓存等于累加的梯度的平方
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        # 学习率乘以梯度然后除以缓存的平方根再加上e的值，添加负号是为了朝着损失减小的方向进行调整
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        # 步数加1
        self.iterations += 1

# 使用AdaGrad优化器进行训练
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adagrad(decay=1e-4)

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    # 计算准确率，打印参数
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)
    if not  epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')
    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward((loss_activation.dinputs))
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # 参数更新
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    这部分输出的表现不如带衰减率和动量的SGD优化器效果好
    '''


'''
接下来讨论均方根传播优化器（RMSProp）
与AdaGrad类似，均方根传播RMSProp为每个参数计算自适应学习率，但计算方式有所不同

RMSProp的计算：
cache += rho * cache + (1 - rho) * gradient ** 2
parm_updates = learning_rate * parm_gradient / (sqrt(cache) + eps)
这种方法类似于带动量的SGD优化器和AdaGrad的缓存机制。RMSProp添加了一种类似于动量的机制，同时还引入了每个参数的自适应学习率，使学习率的变化更加平滑。
这有助于保持变化的全局方向，同时减缓方向的改变。
与AdaGrad持续将梯度平方累加到缓存当中，RMSProp使用缓存的移动平均值，
每次更新缓存时，保留一部分旧缓存，并用新梯度平方的一部分进行更新，
通过这种方式，缓存内容随时间和数据变化移动，避免学习过程停滞，
在这种优化器中，每个参数的学习率可以根据最近的更新和当前的梯度而上升或下降。
rho是衰减率，RMSProp以与AdaGrad相同的方式应用缓存
'''

# 创建均方根传播RMSProp优化器
class Optimizer_RMSprop:
    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,rho=0.9):
        # 初始化
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # 计算缓存
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        # 更新参数
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        # 步数加1
        self.iterations += 1

# 使用RMSProp优化器进行训练
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_RMSprop(decay=1e-4)

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    # 计算准确率，打印参数
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)
    if not  epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')
    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward((loss_activation.dinputs))
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # 参数更新
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    这部分输出接近带动量的SGD，但不是特别好
    '''


'''
接下来讨论自适应动量优化器（Adam）
自适应动量优化器Adam是当前最广泛使用的优化器，它建立在RMSProp的基础上，并重新引入了SGD的动量概念，
这就使得优化器并不直接应用当前的梯度，而是像带动量的SGD优化器一样应用动量，然后结合RMSProp的缓存机制，为每个权重应用自适应学习率
另外，Adam优化器额外添加了一个偏差校正机制（不要与偏置混淆），偏差校正机制应用于缓存和动量，用于补偿初始零值在训练的最初阶段尚未“热启动”的情况。
为了实现这一校正，动量和缓存都需要除以1减一个beta值（应用于动量或衰减率或缓存的初期分数）的步数step次方（1-beta^step）
这个表达式在初期阶段为一个较小的分数，并随着训练的进行逐渐接近1
对这个方法的解释：用一个小于1的分数除法会使它们的值增加数倍，从而显著加速训练的初始阶段，直到经过多次初始步骤后动量和缓存逐渐“热启动”，
这个偏差校正系数在训练过程中会趋向于1，使参数更新在后期训练阶段恢复到通常的值。
为了获得参数更新，将缩放后的动量除以缩放后的平方根缓存
'''

# 创建RMSProp优化器
class Optimizer_Adam:
    # 初始化超参数
    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        '''
        :param learning_rate: 初始学习率
        :param decay: 衰减率
        :param epsilon: 小值e
        :param beta_1: 一阶动量的指数衰减率
        :param beta_2: 二阶动量的指数衰减率
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        # 更新学习率
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            # 如果layer没有缓存数据，则初始化
            layer.weight_momentums = np.zeros_like(layer.weights) # 权重动量
            layer.weight_cache = np.zeros_like(layer.weights) # 权重缓存
            layer.bias_momentums = np.zeros_like(layer.biases) # 偏置动量
            layer.bias_cache = np.zeros_like(layer.biases) # 偏置缓存

        # 计算动量
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # 动量修正
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # 计算缓存
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # 缓存修正
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # 参数更新
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        # 步数加1
        self.iterations += 1

# 使用Adam优化器进行训练
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.02,decay=1e-5)

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    # 计算准确率，打印参数
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)
    if not  epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')
    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward((loss_activation.dinputs))
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # 参数更新
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    这部分输出准确率0.9，损失0.2
    '''

# 调整学习率和衰减率再次训练
X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05,decay=5e-7)

for epoch in range(10001):
    # 前向传播
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output,axis=1)
    # 计算准确率，打印参数
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)
    if not  epoch % 100:
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{loss:.3f},'+
              f'lr:{optimizer.current_learning_rate}')
    # 反向传播
    loss_activation.backward(loss_activation.output,y)
    dense2.backward((loss_activation.dinputs))
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # 参数更新
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    准确率略有提升，损失略有下降
    '''
