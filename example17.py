'''
回归
'''

'''
目前讨论的模型都是分类模型，如果希望模型预测明天的气温或者房价，则需要输出更加精细的结果，
也意味着需要一种新的方法来衡量损失，并且需要一个新的输出层函数，
于是就引入了回归这一概念
'''

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
import numpy as np

nnfs.init()

# 针对回归模型的新的数据
# X,y = sine_data() # 正弦数据
# plt.plot(X,y)
# plt.show() # 绘出的图像是一个正弦曲线

'''
线性激活，由于不再使用分类标签，而是要预测一个标量数值，因此将对输出层使用线性激活函数
该函数不会修改输入，而是将其直接传递到输出：y=x，在反向传播中导数为1
'''
# 线性激活函数
# 这段代码看上去比较冗余，但这样写是为了增加完整性和清晰性
class Activation_Linear:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = inputs # 不修改输入，直接输出
    def backward(self,dvalues):
        self.dinputs = dvalues.copy() # 反向传播应用链式法则，使用前一层梯度乘以本层导数1

'''
接下来讨论损失函数，
由于不再使用分类标签，因此无法计算交叉熵，
回归任务中计算误差的两种主要方法是均方误差和平均绝对误差
'''

# 均方误差将预测值和真实值之间的差异取平方（对于多个回归输出，每个输出都会计算差异），然后对这些平方值求平均
# 损失函数主类
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

# 构建均方误差损失函数
class Loss_MeanSquaredError(Loss):
    def forward(self,y_pred,y_true):
        # 前向传播
        sample_losses = np.mean((y_true - y_pred)**2,axis=-1) # 预测值和真实值的差值取平方，在求平均
        return sample_losses
    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples # 计算均方误差的导数


# 平均绝对误差损失将预测值和真实值之间的差异取绝对值，然后对这些绝对值求平均
# 构建平均绝对误差损失函数
class Loss_MeanAbsoluteError(Loss):
    def forward(self,y_pred,y_true):
        # 前向传播
        sample_losses = np.mean(np.abs(y_true - y_pred),axis=-1) # 预测值和真实值取绝对差异，然后对这些绝对值求平均
        return sample_losses
    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) /outputs
        self.dinputs = self.dinputs / samples

'''
回归精度
所谓回归精度，就是计算模型的准确率的方法
在分类模型中，使用交叉熵可以计算预测与真实目标值相等的情况数，并将其除以样本数量，从而衡量模型的准确率
然而，对于回归模型，模型中的每个输出神经元都是单独的输出，这与分类器不同（分类器的所有输出会共同贡献于一个预测结果），
另外，回归模型的预测值是浮点数，因此无法简单的检查输出值是否与真实目标值完全相等，比如真实值为100，预测值为100.01，也会被认为是不相等，但实际上它们的差值很小，
目前对于回归任务，没有完美的方法来衡量准确率。但为了直观展示性能，还是需要一个准确率度量

定义一个准确率指标，引入一个“精度”，使用中需要引入一个限制值，使用预测值和真实值的标准差除以这个限制值得到精度
accuracy_precision = np.std(y) / 250 # 标准差除以限制值，这里定义限制值为250
然后将这个精度值用作回归输出的“容差范围”，在比较真实值和预测值的准确性时起到缓冲作用，
比较预测值和真实值的差异的绝对值是否小于这个精度值
predictions = activation2.output
accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
'''


'''
新的激活函数、损失函数和计算精度的方法讨论完成，
根据这些方法创建模型
'''
# 完整的代码
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

# Sigmoid激活函数层
class Activation_Sigmoid:
    def forward(self,inputs):
        # 前向传播
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs)) # σ(x) = 1/(1+e^(-x))
    def backward(self,dvalues):
        # 反向传播
        self.dinputs = dvalues * (1 - self.output) * self.output # σ(x)(1-σ(x))

# 线性激活函数
class Activation_Linear:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = inputs # 不修改输入，直接输出
    def backward(self,dvalues):
        self.dinputs = dvalues.copy() # 反向传播应用链式法则，使用前一层梯度乘以本层导数1

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

# 二元交叉熵损失函数层
class Loss_BinaryCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        # 前向传播
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7) # 防止除零错误
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) # 计算损失值
        sample_losses = np.mean(sample_losses,axis=1) # 求平均值
        return sample_losses
    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues,1e-7,1-1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples # 计算导数

# 构建均方误差损失函数
class Loss_MeanSquaredError(Loss):
    def forward(self,y_pred,y_true):
        # 前向传播
        sample_losses = np.mean((y_true - y_pred)**2,axis=-1) # 预测值和真实值的差值取平方，在求平均
        return sample_losses
    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples # 计算均方误差的导数

# 构建平均绝对误差损失函数
class Loss_MeanAbsoluteError(Loss):
    def forward(self,y_pred,y_true):
        # 前向传播
        sample_losses = np.mean(np.abs(y_true - y_pred),axis=-1) # 预测值和真实值取绝对差异，然后对这些绝对值求平均
        return sample_losses
    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) /outputs
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

# 创建模型
X,y = sine_data() # 创建正弦数据
dense1 = Layer_Dense(1,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,1)
activation2 = Activation_Linear() # 线性激活
loss_function = Loss_MeanSquaredError() # 均方误差损失函数
optimizer = Optimizer_Adam()
accuracy_precision = np.std(y) / 250 # 精度

# 迭代训练
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output,y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = activation2.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_function.backward(activation2.output,y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    训练效果非常差，最后一轮输出：
    epoch: 10000, acc: 0.004, loss: 0.145 (data_loss: 0.145, reg_loss: 0.000), lr: 0.001
    '''

# 添加绘制测试数据的功能，可视化模型的预测值
X_test,y_test = sine_data() # 测试数据
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
# 绘制
plt.plot(X_test,y_test)
plt.plot(X_test,activation2.output)
plt.show()
'''
分析绘出的图像，观察模型的输出可以判断模型试图通过两次转折来拟合这个正弦曲线，这显然是不够的，
模型需要进行更多的“转折”，这个转折是非线性激活函数层给予模型的能力，这意味着模型需要更多的层
'''

# 创建一个新的模型，在原来的基础上增加一层
X,y = sine_data()
dense1 = Layer_Dense(1,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64,1)
activation3 = Activation_Linear() # 线性激活
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam()
accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output,y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_function.backward(activation3.output,y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# 再次绘制测试数据和模型的输出
X_test,y_test = sine_data()
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
plt.plot(X_test,y_test)
plt.plot(X_test,activation3.output)
plt.show()
'''
这次拟合的效果很不错，模型最后一轮的输出为
epoch: 10000, acc: 0.982, loss: 0.000 (data_loss: 0.000, reg_loss: 0.000), lr: 0.001

另外，调试模型通常是一个相当困难的任务，调试时还应该考虑权重初始化、学习率及其更新策略对模型的影响。
'''

