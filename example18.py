'''
模型对象
'''

'''
目前已经构建了一个包含了全连接层、dropout层、激活函数层、损失函数层和优化器等结构的模型，
它可以完整的执行前向传播和反向传播的流程，并在训练中加入了正则、精度计算等方法来提升模型的表现，
同时尝试了修改输出层激活函数和损失函数等，使模型能够解决二元逻辑回归和回归类问题，
这个模型已经比较完整，但在每次使用时重新组装各个网络层太麻烦，代码篇幅也不好控制，
于是考虑创建一个模型类，将模型本身转化为一个对象

以使用正弦数据的回归模型为例
'''

# 之前的代码
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data,spiral_data
import numpy as np
import sys

nnfs.init()

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


# 创建Adam优化器
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


# 定义数据
X, y = sine_data()

# 开始制作模型类
class Model:
    def __init__(self):
        # 创建一个用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)

# model = Model() # 实例化
# model.add(Layer_Dense(1,64)) # 第一层全连接层
# model.add(Activation_ReLU()) # 第一层ReLU()激活函数层
# model.add(Layer_Dense(64,64)) # 第二层全连接层
# model.add(Activation_ReLU()) # 第二层ReLU()激活函数层
# model.add(Layer_Dense(64,1)) # 第三层全连接层
# model.add(Activation_Linear()) # 第三层线性激活

# print(model.layers) # 查看这个模型
# [<__main__.Layer_Dense object at 0x0000026955FF6C70>, <__main__.Activation_ReLU object at 0x0000026977269C70>, <__main__.Layer_Dense object at 0x0000026955FDEE20>, <__main__.Activation_ReLU object at 0x0000026955FDEE50>, <__main__.Layer_Dense object at 0x00000269771833D0>, <__main__.Activation_Linear object at 0x0000026977183250>]


# 除了添加层，模型还需要进行损失函数和优化器的设置，为此创建一个set方法
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer

# 在之前的步骤中加入损失函数和优化器
# model = Model() # 实例化
# model.add(Layer_Dense(1,64)) # 第一层全连接层
# model.add(Activation_ReLU()) # 第一层ReLU()激活函数层
# model.add(Layer_Dense(64,64)) # 第二层全连接层
# model.add(Activation_ReLU()) # 第二层ReLU()激活函数层
# model.add(Layer_Dense(64,1)) # 第三层全连接层
# model.add(Activation_Linear()) # 第三层线性激活
# model.set(
#     loss=Loss_MeanSquaredError(), # 均方误差损失函数
#     optimizer=Optimizer_Adam(learning_rate=0.005,decay=1e-3), # Adam优化器
# )


# 设置好模型的层、损失函数和优化器后，下一步就是训练
# 因此需要添加一个train方法，先将其作为一个占位符，后续再进行填充
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
    def train(self,X,y,*,epochs=1,print_every=1):
        # 训练模型，先将其作为一个占位符，后续再进行填充
        for epoch in range(1,epochs+1):
            pass

# 在之前的步骤中加入模型训练方法的调用
# model = Model() # 实例化
# model.add(Layer_Dense(1,64)) # 第一层全连接层
# model.add(Activation_ReLU()) # 第一层ReLU()激活函数层
# model.add(Layer_Dense(64,64)) # 第二层全连接层
# model.add(Activation_ReLU()) # 第二层ReLU()激活函数层
# model.add(Layer_Dense(64,1)) # 第三层全连接层
# model.add(Activation_Linear()) # 第三层线性激活
# model.set(
#     loss=Loss_MeanSquaredError(), # 均方误差损失函数
#     optimizer=Optimizer_Adam(learning_rate=0.005,decay=1e-3), # Adam优化器
# )
# model.train(X,y,epochs=10000,print_every=100) # 迭代次数为10000，每100步打印一次消息


# 要进行训练，需要执行前向传播，在对象中执行前向传播比之前的方式稍微复杂一些，
# 因为这个操作需要在层的循环中完成，并且需要知道前一层的输出以正确的传递数据，
# 然而查询前一层的一个问题是，第一层没有“前一层”，
# 所以，通常的一个做法是创建一个“输入层”，这一层被认为是神经网络中的一层，但没有与之相关的权重和偏置，
# “输入层”仅包含训练数据，只在循环迭代时将其作为第一层的“前一层”
# 创建一个新类，称为Layer_Input
class Layer_Input:
    def forward(self,inputs):
        # 前向传播
        self.output = inputs
    # 这里不用创建反向传播方法，因为这一层没有权重和偏置

# 接下来需要在模型中的每一层设置其“前一层”和“后一层”的属性，新增一个finalize方法
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
    def train(self,X,y,*,epochs=1,print_every=1):
        # 训练模型，先将其作为一个占位符，后续再进行填充
        for epoch in range(1,epochs+1):
            pass
    def finalize(self):
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算

# 模型进行前向传播的准备工作已经就绪，接下来再在模型类中添加一个forward方法用于执行前向传播
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
    def train(self,X,y,*,epochs=1,print_every=1):
        # 训练模型，先将其作为一个占位符，后续再进行填充
        for epoch in range(1,epochs+1):
            pass
    def finalize(self):
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output

# 将模型的前向传播方法forward补充到train方法中
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
    def train(self,X,y,*,epochs=1,print_every=1):
        # 训练模型，先将其作为一个占位符，后续再进行填充
        for epoch in range(1,epochs+1):
            output = self.forward(X) # 执行前向传播
            print(output)
            sys.exit()
    def finalize(self):
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output

# 尝试调用模型进行训练
# model = Model() # 实例化
# model.add(Layer_Dense(1,64)) # 第一层全连接层
# model.add(Activation_ReLU()) # 第一层ReLU()激活函数层
# model.add(Layer_Dense(64,64)) # 第二层全连接层
# model.add(Activation_ReLU()) # 第二层ReLU()激活函数层
# model.add(Layer_Dense(64,1)) # 第三层全连接层
# model.add(Activation_Linear()) # 第三层线性激活
# model.set(
#     loss=Loss_MeanSquaredError(), # 均方误差损失函数
#     optimizer=Optimizer_Adam(learning_rate=0.005,decay=1e-3), # Adam优化器
# )
# model.finalize() # 模型的初始化设置，执行前向传播的准备工作
# model.train(X,y,epochs=10000,print_every=100) # 迭代次数为10000，每100步打印一次消息


# 目前已经完成了前向传播，接下来需要计算损失和准确率并进行反向传播
# 在此之前，需要知道那些层是“可更新的”，也就是说这些层具有可以调整的权重和偏置
# 修改Model类中的finalize方法，使其可以检查层是否有weight属性（只需要检查weight就够了，因为权重和偏置同时出现）
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
    def train(self,X,y,*,epochs=1,print_every=1):
        # 训练模型，先将其作为一个占位符，后续再进行填充
        for epoch in range(1,epochs+1):
            output = self.forward(X) # 执行前向传播
            print(output)
            sys.exit()
    def finalize(self):
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        self.trainable_layers = []
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
            if hasattr(self.layers[i],'weights'):
                # 检查层中是否包含weights属性，如果有，将其添加到可训练层列表中
                self.trainable_layers.append(self.layers[i])
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output

# 为了计算整个模型的正则化损失，和配合模型类做出的修改，
# 需要对loss损失函数父类进行调整
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y) # 计算损失函数层的直接损失
        data_loss = np.mean(sample_losses) # 求平均损失
        return data_loss,self.regularization_loss() # 返回平均损失和正则化损失
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers # 设置可训练层
    def regularization_loss(self):
        # 计算正则化损失
        regularization_loss = 0 # 默认值为0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

# 由于Loss父类的改变，均方误差损失函数需要重新继承Loss
class Loss_MeanSquaredError(Loss):
    def forward(self,y_pred,y_true):
        # 前向传播
        sample_losses = np.mean((y_true - y_pred)**2,axis=1) # 预测值和真实值的差值取平方，在求平均
        return sample_losses
    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples # 计算均方误差的导数

# 计算准确率需要模型的预测结果，之前的模型在预测时需要使用不同的代码
# 例如对于softmax分类器，使用的是np.argmax()，而对于回归模型，输出层使用线性激活函数，预测结果直接为输出值
# 为此对不同的模型，需要一个预测方法，该方法能够为模型选择合适的预测方式
# 做法是在每个激活函数类中添加一个prediction方法
# Softmax激活函数层
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_dvalues.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)

# Sigmoid激活函数层
class Activation_Sigmoid:
    def forward(self,inputs):
        # 前向传播
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs)) # σ(x) = 1/(1+e^(-x))
    def backward(self,dvalues):
        # 反向传播
        self.dinputs = dvalues * (1 - self.output) * self.output # σ(x)(1-σ(x))
    def predictions(self,outputs):
        return (outputs > 0.5) * 1

# 线性激活函数
class Activation_Linear:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = inputs # 不修改输入，直接输出
    def backward(self,dvalues):
        self.dinputs = dvalues.copy() # 反向传播应用链式法则，使用前一层梯度乘以本层导数1
    def predictions(self,outputs):
        return outputs

# ReLU激活函数层
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self,outputs):
        return outputs

# 完善Model类，在finalize方法中为最终层的激活函数设置一个引用
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
    def train(self,X,y,*,epochs=1,print_every=1):
        # 训练模型，先将其作为一个占位符，后续再进行填充
        for epoch in range(1,epochs+1):
            output = self.forward(X) # 执行前向传播
            print(output)
            sys.exit()
    def finalize(self):
        # 完成模型的设置和初始化
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        self.trainable_layers = []
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
                self.output_layer_activation = self.layers[i] # 将最后一层激活函数层赋值给output_layer_activation，后续再进行引用
            if hasattr(self.layers[i],'weights'):
                # 检查层中是否包含weights属性，如果有，将其添加到可训练层列表中
                self.trainable_layers.append(self.layers[i])
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output

# 和不同的预测方法一样，为了计算不同模型的准确率需要使用不同的计算方法
# 为了代码的清晰性，首先创建一个通用的Accuracy类
class Accuracy:
    # 准确率计算父类
    def calculate(self,predictions,y):
        # 获取比较结果
        comparisons = self.compare(predictions,y)
        # 计算准确率
        accuracy = np.mean(comparisons)
        return accuracy

# 继承Accuracy，实现一个用于回归模型的准确性计算类
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None # 计算准确性时的精度（即允许的误差范围）
    def init(self,y,reinit=False):
        if self.precision is None or reinit:
            # 如果precision为None或者reinit为True，则重新初始化
            self.precision = np.std(y) / 250 # 计算精度
    def compare(self,predictions,y):
        # 比较预测值和真实值，判断预测是否在允许的误差范围内
        return np.absolute(predictions - y) < self.precision # 返回一个布尔数组，每个元素表示对应的预测值是否在容差范围内

# 完善Model模型类，
# 在set方法中设置准确率对象，
# 并将损失和准确率的计算添加到train方法中
# 在finalize方法中将可训练层信息传递给损失对象，以便在计算正则化损失时使用
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer,accuracy):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    def train(self,X,y,*,epochs=1,print_every=1):
        self.accuracy.init(y)
        for epoch in range(1,epochs+1):
            output = self.forward(X) # 执行前向传播
            data_loss,regularization_loss = self.loss.calculate(output,y) # 计算直接损失和正则化损失
            loss = data_loss + regularization_loss # 加和得到最终的损失
            predictions = self.output_layer_activation.predictions(output) # 预测方法
            accuracy = self.accuracy.calculate(predictions,y) # 计算准确率
    def finalize(self):
        # 完成模型的设置和初始化
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        self.trainable_layers = []
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
                self.output_layer_activation = self.layers[i] # 将最后一层激活函数层赋值给output_layer_activation，后续再进行引用
            if hasattr(self.layers[i],'weights'):
                # 检查层中是否包含weights属性，如果有，将其添加到可训练层列表中
                self.trainable_layers.append(self.layers[i])
            self.loss.remember_trainable_layers(self.trainable_layers) # 将可训练层信息传递给loss对象
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output


# 目前模型类已经完成了完整的前向传播并计算了损失和准确率
# 下一步是反向传播
# 完善Model类，加入反向传播backward方法，在train方法的末尾继续backward方法，之后再使用优化器优化参数，最后使用print_every参数打印信息
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer,accuracy):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    def train(self,X,y,*,epochs=1,print_every=1):
        self.accuracy.init(y)
        for epoch in range(1,epochs+1):
            output = self.forward(X) # 执行前向传播
            data_loss,regularization_loss = self.loss.calculate(output,y) # 计算直接损失和正则化损失
            loss = data_loss + regularization_loss # 加和得到最终的损失
            predictions = self.output_layer_activation.predictions(output) # 预测方法
            accuracy = self.accuracy.calculate(predictions,y) # 计算准确率

            self.backward(output,y) # 执行反向传播
            # 优化器操作
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                # 打印信息
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
    def finalize(self):
        # 完成模型的设置和初始化
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        self.trainable_layers = []
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
                self.output_layer_activation = self.layers[i] # 将最后一层激活函数层赋值给output_layer_activation，后续再进行引用
            if hasattr(self.layers[i],'weights'):
                # 检查层中是否包含weights属性，如果有，将其添加到可训练层列表中
                self.trainable_layers.append(self.layers[i])
            self.loss.remember_trainable_layers(self.trainable_layers) # 将可训练层信息传递给loss对象
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output
    def backward(self,output,y):
        self.loss.backward(output,y) # 从末端的损失函数层开始进行反向传播
        for layer in reversed(self.layers):
            # 以相反的顺序遍历所有层
            # 调用每一层的backward反向传播方法，传入参数为上一层的dinputs
            layer.backward(layer.next.dinputs)

# 尝试将精度类对象传入模型（回归问题），进行模型训练
model = Model() # 实例化
model.add(Layer_Dense(1,64)) # 第一层全连接层
model.add(Activation_ReLU()) # 第一层ReLU()激活函数层
model.add(Layer_Dense(64,64)) # 第二层全连接层
model.add(Activation_ReLU()) # 第二层ReLU()激活函数层
model.add(Layer_Dense(64,1)) # 第三层全连接层
model.add(Activation_Linear()) # 第三层线性激活
model.set(
    loss=Loss_MeanSquaredError(), # 均方误差损失函数
    optimizer=Optimizer_Adam(learning_rate=0.005,decay=1e-3), # Adam优化器
    accuracy=Accuracy_Regression() # 准确率计算方法
)
model.finalize() # 模型的初始化设置，执行前向传播的准备工作
model.train(X,y,epochs=10000,print_every=100) # 迭代次数为10000，每100步打印一次消息


# 目前已经可以通过Model类创建模型，并且在回归模型的创建和训练上表现不错
# 继续进行完善，使其支持全新的模型，比如二元逻辑回归
# 创建分类准确率计算方法
class Accuracy_Categorical(Accuracy):
    def init(self,y):
        pass
    def compare(self,predictions,y):
        if len(y.shape) == 2:
            y = np.argmax(y,axis=1)
        return predictions == y # 返回一个布尔列表，表示分类正确的标签

# 接下来添加使用验证数据对模型进行验证的能力
# 验证只需执行前向传播并计算损失（仅数据损失）
# 修改Loss损失函数父类的calculate方法，使其能够计算验证损失
class Loss:
    def calculate(self,output,y,*,include_regularization=False):
        sample_losses = self.forward(output,y) # 计算损失函数层的直接损失
        data_loss = np.mean(sample_losses) # 求平均损失
        if not include_regularization:
            # include_regularization为“包含正则化”标识，当其为False时，不进行正则化计算，直接返回数据损失
            return data_loss
        return data_loss,self.regularization_loss() # 返回平均损失和正则化损失
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers # 设置可训练层
    def regularization_loss(self):
        # 计算正则化损失
        regularization_loss = 0 # 默认值为0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

# 由于Loss的改变，二元交叉熵损失函数层需要重新继承Loss
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

# 由于Loss的改变,分类交叉熵损失函数层需要重新继承Loss
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

# 由于Loss类中include_regularization（是否包含正则化）参数的加入，Model类的train方法也需要更改
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer,accuracy):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    def train(self,X,y,*,epochs=1,print_every=1,validation_data=None):
        self.accuracy.init(y)
        for epoch in range(1,epochs+1):
            output = self.forward(X) # 执行前向传播
            data_loss,regularization_loss = self.loss.calculate(output,y,include_regularization=True) # 设置include_regularization为True，计算直接损失和正则化损失
            loss = data_loss + regularization_loss # 加和得到最终的损失
            predictions = self.output_layer_activation.predictions(output) # 预测方法
            accuracy = self.accuracy.calculate(predictions,y) # 计算准确率

            self.backward(output,y) # 执行反向传播
            # 优化器操作
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                # 打印信息
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
        if validation_data is not None:
            # 如果存在验证数据集，则进行验证
            # 对模型的验证，只需要执行前向传播和损失函数的计算（仅计算直接损失，不引入正则化）
            X_val,y_val = validation_data
            output = self.forward(X_val)
            loss = self.loss.calculate(output,y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions,y_val)
            # 打印信息
            print(f'validation,'+
                  f'acc:{accuracy:.3f},'+
                  f'loss:{loss:.3f}')
    def finalize(self):
        # 完成模型的设置和初始化
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        self.trainable_layers = []
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
                self.output_layer_activation = self.layers[i] # 将最后一层激活函数层赋值给output_layer_activation，后续再进行引用
            if hasattr(self.layers[i],'weights'):
                # 检查层中是否包含weights属性，如果有，将其添加到可训练层列表中
                self.trainable_layers.append(self.layers[i])
            self.loss.remember_trainable_layers(self.trainable_layers) # 将可训练层信息传递给loss对象
    def forward(self,X):
        # 模型的前向传播
        self.input_layer.forward(X) # 将输入数据传入输入层
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output)
        return layer.output
    def backward(self,output,y):
        self.loss.backward(output,y) # 从末端的损失函数层开始进行反向传播
        for layer in reversed(self.layers):
            # 以相反的顺序遍历所有层
            # 调用每一层的backward反向传播方法，传入参数为上一层的dinputs
            layer.backward(layer.next.dinputs)

# 创建模型并进行训练
X,y = spiral_data(samples=100,classes=2)
X_test,y_test = spiral_data(samples=100,classes=2)
y = y.reshape(-1,1)
y_test = y_test.reshape(-1,1)
model = Model()
model.add(Layer_Dense(2,64,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Sigmoid())
model.set(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-7),
    accuracy=Accuracy_Categorical()
)
model.finalize()
model.train(X,y,validation_data=(X_test,y_test),epochs=10000,print_every=100)


# 目前实现了包含验证过程的模型类，且兼容二元逻辑回归问题
# 接下来引入dropout，在模型中引入dropout，需要确保在验证和预测时不使用dropout
# 然而在模型中使用一个通用方法同时执行训练和验证的前向传播，因此需要引入一个参数，用来告知各层是否进行dropout

# 在所有层和激活函数类的前向传播中添加一个布尔参数training，即只在模型训练时使用dropout，验证和预测时不使用
# 全连接层的前向传播中添加一个布尔参数training
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

    def forward(self,inputs,training):
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

# ReLU激活函数层的前向传播中添加一个布尔参数training
class Activation_ReLU:
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self,outputs):
        return outputs

# Softmax激活函数层的前向传播中添加一个布尔参数training
class Activation_Softmax:
    def forward(self,inputs,training):
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
    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)

# Sigmoid激活函数层的前向传播中添加一个布尔参数training
class Activation_Sigmoid:
    def forward(self,inputs,training):
        # 前向传播
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs)) # σ(x) = 1/(1+e^(-x))
    def backward(self,dvalues):
        # 反向传播
        self.dinputs = dvalues * (1 - self.output) * self.output # σ(x)(1-σ(x))
    def predictions(self,outputs):
        return (outputs > 0.5) * 1

# 线性激活函数的前向传播中添加一个布尔参数training
class Activation_Linear:
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = inputs # 不修改输入，直接输出
    def backward(self,dvalues):
        self.dinputs = dvalues.copy() # 反向传播应用链式法则，使用前一层梯度乘以本层导数1
    def predictions(self,outputs):
        return outputs

# 输入层的前向传播中添加一个布尔参数training
class Layer_Input:
    # 前向传播
    def forward(self, inputs, training):
        self.output = inputs

# 添加完成后修改dropout层
class Layer_Dropout:
    def __init__(self,rate):
        '''
        :param rate: 传入超参数，丢弃的比例
        '''
        self.rate = 1 - rate

    def forward(self,inputs,training):
        self.inputs = inputs # 输入数据
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1,self.rate,size=inputs.shape) / self.rate # 是通过二项分布生成的随机掩码，决定哪些神经元将被保留
        self.output = inputs * self.binary_mask # 得到输出

    def backward(self,dvalues):
        self.dinputs = dvalues * self.binary_mask # dropout层的梯度，只有保留的神经元才有梯度

# 在Model类的forward方法中添加training参数，并调用各层的forward方法传递该参数的值
# 并更新train方法，在调用forward方法时，设置training值
# 加入对Softmax激活和交叉熵损失的联合对象的处理
class Model:
    def __init__(self):
        # 用于存储网络层的列表
        self.layers = []
        self.softmax_classifier_output = None
    def add(self,layer):
        # 向模型中添加层
        self.layers.append(layer)
    def set(self,*,loss,optimizer,accuracy):
        # 设置损失函数和优化器，*表示后续的参数（例子中也就是loss和optimizer）为关键字参数
        # 由于这些参数没有默认值，因此它们是必需的关键字参数，也就是说必须通过名称和值的形式传递，从而使代码更加易读
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    def train(self,X,y,*,epochs=1,print_every=1,validation_data=None):
        self.accuracy.init(y)
        for epoch in range(1,epochs+1):
            output = self.forward(X,training=True) # 执行前向传播，训练时training=True
            data_loss,regularization_loss = self.loss.calculate(output,y,include_regularization=True) # 设置include_regularization为True，计算直接损失和正则化损失
            loss = data_loss + regularization_loss # 加和得到最终的损失
            predictions = self.output_layer_activation.predictions(output) # 预测方法
            accuracy = self.accuracy.calculate(predictions,y) # 计算准确率

            self.backward(output,y) # 执行反向传播
            # 优化器操作
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                # 打印信息
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
        if validation_data is not None:
            # 如果存在验证数据集，则进行验证
            # 对模型的验证，只需要执行前向传播和损失函数的计算（仅计算直接损失，不引入正则化）
            X_val,y_val = validation_data
            output = self.forward(X_val,training=False) # 验证时training=False
            loss = self.loss.calculate(output,y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions,y_val)
            # 打印信息
            print(f'validation,'+
                  f'acc:{accuracy:.3f},'+
                  f'loss:{loss:.3f}')
    def finalize(self):
        # 完成模型的设置和初始化
        # 为模型中的每一层设置其“前一层”和“后一层”的属性
        self.input_layer = Layer_Input() # 将定义的“输入层”传递给input_layer属性，用于接收输入数据
        layer_count = len(self.layers) # 获取模型的隐藏层的数量
        self.trainable_layers = []
        for i in range(layer_count):
            # 遍历所有的隐藏层
            if i == 0:
                # 如果当前是第一层（隐藏层的第一层，索引为0）
                self.layers[i].prev = self.input_layer # 将prev属性（前一层）设置为输入层input_layer
                self.layers[i].next = self.layers[i+1] # 将next（后一层）设置为列表中下一层
            elif i < layer_count - 1:
                # 如果当前是中间层（隐藏层中的既不是第一层也不是最后一层的层）
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.layers[i+1] # 将next设置为后一层
            else:
                # 如果当前层是最后一层
                self.layers[i].prev = self.layers[i-1] # 将prev设置为前一层
                self.layers[i].next = self.loss # 将next设置为损失函数对象，最后一层的输出直接传递给损失函数进行计算
                self.output_layer_activation = self.layers[i] # 将最后一层激活函数层赋值给output_layer_activation，后续再进行引用
            if hasattr(self.layers[i],'weights'):
                # 检查层中是否包含weights属性，如果有，将其添加到可训练层列表中
                self.trainable_layers.append(self.layers[i])
            self.loss.remember_trainable_layers(self.trainable_layers) # 将可训练层信息传递给loss对象
        if isinstance(self.layers[-1],Activation_Softmax) and isinstance(self.loss,Loss_CategoricalCrossentropy):
            # 如果最后一层是Softmax激活函数，且损失函数是交叉熵
            # 创建联合激活和损失对象（优化计算效率）
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
    def forward(self,X,training):
        # 模型的前向传播
        self.input_layer.forward(X,training) # 将输入数据传入输入层，传递training值
        for layer in self.layers:
            # 遍历每一层，执行前向传播
            layer.forward(layer.prev.output,training) # 传递training值
        return layer.output
    def backward(self,output,y):
        # 如果使用的是softmax分类器
        if self.softmax_classifier_output is not None:
            # 首先调用softmax分类器与损失函数结合的对象的反向传播方法
            # 这会设置dinputs属性
            self.softmax_classifier_output.backward(output,y)

            # 因为不会单独调用最后一次（softmax激活层）的反向传播方法
            # 因为使用了结合激活函数和损失函数的对象
            # 所有需要手动设置最后一层的dinputs
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # 按逆序遍历所有层（除了最后一层），并调用它们的反向传播方法
            # 将dinputs作为参数传递
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # 如果没有softmax分类器
        self.loss.backward(output,y) # 从末端的损失函数层开始进行反向传播
        for layer in reversed(self.layers):
            # 以相反的顺序遍历所有层
            # 调用每一层的backward反向传播方法，传入参数为上一层的dinputs
            layer.backward(layer.next.dinputs)


# 经过上述修改，代码中不再需要Activation_Softmax_Loss_CategoricalCrossentropy类的初始化器和前向传播方法，可以考虑将其移除，仅保留反向传播
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs / samples


# 使用dropout测试更新后的Model对象
X,y = spiral_data(samples=1000,classes=3)
X_test,y_test = spiral_data(samples=100,classes=3)
model = Model()
model.add(Layer_Dense(2,512,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512,3))
model.add(Activation_Softmax())
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05,decay=5e-5),
    accuracy=Accuracy_Categorical()
)
model.finalize()
model.train(X,y,validation_data=(X_test,y_test),epochs=10000,print_every=100)
'''
目前已经将之前讨论的创建模型的方法都集中到了Model类中，可以通过这个类搭建模型
'''