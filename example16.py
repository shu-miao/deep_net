'''
二元Logistic回归
'''

'''
目前已经完成了一个模型的创建和训练，以及为了解决过拟合问题为模型新增的结构，已经得到了一个比较完整的模型，
但当前模型是一个比较原始的分类模型，它输出的是一个概率分布，其中所有值表示特定类别为正确类别的置信水平，这些置信度之和为1，
为了使模型可以用于解决更多问题，接下来讨论另一种输出层选项，其中每个神经元分别代表两个类别，0表示其中一个类别，1表示另一个类别，
这类型输出层的模型被称为二元逻辑回归。
一个模型可以有多个二元输出神经元，
例如，一个模型可能有两个二元输出神经元，其中一个可以区分“人/非人”，另一个神经元可以区分“室内/室外”。

二元逻辑回归是一种回归类型的算法，它的不同之处在于使用sigmoid激活函数最为输出层，而不是softmax，
并使用二元交叉熵而不是分类交叉熵来计算损失
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Sigmoid激活函数
# 公式为σ(x) = 1/(1+e^(-x))
# sigmoid激活函数能够将负无穷到正无穷的输出范围压缩到0和1之间，0和1的界限分别代表两种可能的类别
# sigmoid激活函数的导数
# 推导可得导数式为：σ(x)(1-σ(x))
# 确定了sigmoid激活函数和其导数，也就确定了它的前向传播和反向传播过程
# 构建Sigmoid激活函数类
class Activation_Sigmoid:
    def forward(self,inputs):
        # 前向传播
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs)) # σ(x) = 1/(1+e^(-x))
    def backward(self,dvalues):
        # 反向传播
        self.dinputs = dvalues * (1 - self.output) * self.output # σ(x)(1-σ(x))


# 二元交叉熵损失函数
# 计算二元交叉熵损失函数可以类比分类交叉熵损失中的负对数概念（负对数似然）
# 不同的是二元交叉熵损失函数不会仅针对目标类别进行计算，
# 而是分别对每个神经元的正确类别和错误类别的对数似然进行求和，
# 由于类别值仅为0或1，可以将错误类别简化为1-正确类别，这相当于对值进行取反
# 然后再计算正确类别和错误类别的负对数似然，并将它们相加

# 二元交叉熵损失函数用于二分类问题，能够衡量模型预测与真实标签之间的差异
# 当真实标签为1时，损失函数关注的是预测为1的概率，当真实标签为0时，损失函数关注的是预测为0的概率

# 二元交叉熵损失函数的导数
# 推导时可参照分类交叉熵的推导

# 构建二元交叉熵损失类
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


# 之前的代码
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


# 利用新的输出层创建新的模型
# 创建数据集
X,y = spiral_data(100,2)
y = y.reshape(-1,1) # 重塑标签

dense1 = Layer_Dense(2,64,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,1) # 对于二分类任务，dense2应该只有一个输出，它的输出正好表示2个类别（0或1）
activation2 = Activation_Sigmoid() # 应用sigmoid激活函数

loss_function = Loss_BinaryCrossentropy() # 二元交叉熵损失函数
optimizer = Optimizer_Adam(decay=5e-7)

# 迭代训练
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output,y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)
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

# 测试
X_test,y_test = spiral_data(samples=100,classes=2)
y_test = y_test.reshape(-1,1) # 无论对训练数据进行了怎样的预处理操作，对于测试数据也要进行相同的操作
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output,y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)
print(f'测试集, acc: {accuracy:.3f}, loss: {loss:.3f}')