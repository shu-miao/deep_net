'''
Dropout
'''

'''
除了L1和L2正则化外，常见的还有一种被称为dropout的方法
在网络中添加一个dropout层，这种类型的层会禁用一些神经元，其他神经元保持不变，
其理念与正则化类似，是为了防止神经网络对任何一个神经元的依赖过高，或在某个特定实例中完全依赖某个神经元，
dropout还能帮助解决共适应现象，共适应是指神经元依赖其他神经元的输出值，而不能独立学习底层函数，

dropout函数在每次前向传播中随机禁用一定比例的神经元，这迫使网络必须学会在仅剩一部分随机选择的神经元的情况下依然进行准确的预测，

另外，dropout既不会减少使用的神经元数量，也不会让训练过程在禁用一半神经元时快一倍，因为dropout层实际上并没有真正禁用神经元，而是将它们的输出置为0。
'''

import random
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# dropout示例
# dropout_rate = 0.5
# example_output = [0.27,-1.03,0.67,0.99,0.05,-0.37,-2.01,1.13,0.07,0.73]
# while True:
#     index = random.randint(0,len(example_output) - 1)
#     example_output[index] = 0
#     dropped_out = 0
#     for value in example_output:
#         if value == 0:
#             dropped_out += 1
#
#     if dropped_out / len(example_output) >= dropout_rate:
#         break
#
# print(example_output)


# 创建dropout层
class Layer_Dropout:
    def __init__(self,rate):
        '''
        :param rate: 传入超参数，丢弃的比例
        '''
        self.rate = 1 - rate

    def forward(self,inputs):
        self.inputs = inputs # 输入数据
        self.binary_mask = np.random.binomial(1,self.rate,size=inputs.shape) / self.rate # 是通过二项分布生成的随机掩码，决定哪些神经元将被保留
        self.output = inputs * self.binary_mask # 得到输出

    def backward(self,dvalues):
        self.dinputs = dvalues * self.binary_mask # dropout层的梯度，只有保留的神经元才有梯度


# 全连接层
class Layer_Dense:
    def __init__(self,inputs,neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0,bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))
        # 引入正则化强度超参数，也就是lambda值
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        # 反向传播
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)

        # L1正则化相对于权重
        if self.weight_regularizer_l1 > 0:
            # dL1是L1正则化对权重的梯度：当权重为正时，梯度为1；当权重为负时，梯度为-1
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1 # 权重的梯度加上L1正则化乘以正则化强度超参数的值
        # L2正则化相对于权重
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights # 通过对L2正则化函数求导可得，L2正则化对权重的梯度是2倍的正则化强度超参数乘以权重

        # L1正则化相对于偏置
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2正则化相对于偏置
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues,self.weights.T)


# 损失函数层
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    # 正则化仅仅是一个与数据损失值相加的惩罚项，加和后得到最终的损失值
    def regularization_loss(self,layer):
        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss


# 分类交叉熵损失函数层
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


X,y = spiral_data(samples=1000,classes=3)
dense1 = Layer_Dense(2,64,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05,decay=5e-5)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output,y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test,y_test = spiral_data(samples=100,classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output,y_test)
predictions = np.argmax(loss_activation.output,axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test,axis=1)
accuracy = np.mean(predictions==y_test)
print(f'测试集, acc: {accuracy:.3f}, loss: {loss:.3f}')

'''
应用了dropout层后，在训练集大小1000，全连接层神经元数512，dropout率0.1的设置下，
模型的准确率和损失大幅下降，但验证集准确率比训练集更高，且损失较训练集也更低，
随后将层大小改为512，结果较好，但比无dropout模型稍差，
另外，加入了dropout层的模型的验证集精度大于训练集，进一步证明之前发生了过拟合现象
'''