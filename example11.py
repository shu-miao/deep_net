'''
使用样本外的数据进行测试
'''

'''
通过改进优化器，模型的准确率逐渐提高，损失逐渐下降，并且梯度下降和参数更新可以按照一定规则进行，
接下来希望这个神经网络可以学习数据集中的规则的表示，并利用这种表示来预测额外生成数据的类别。
神经网络的复杂性既是它最大的优势，也是它最大的挑战，
因为具有大量可调参数，神经网络非常擅长拟合数据，这既是天赋，也是诅咒，一个好的模型应该在这之间找到一个良好的平衡，
如果神经元足够多，一个模型可以轻松记住一个数据集；然而，神经元太少则无法对数据进行泛化，
这就说明了解决解决问题不能仅仅依赖增加神经元数量或使用更大的模型。
'''

'''
目前尚不确定当前模型表现出的较高准确率是由于有效的学习了数据集还是过拟合了数据，
过拟合实际上只是对数据的记忆，而不对其进行任何理解，
所以，应该使用模型从未见过的数据来进行测试，也就是使用样本外的数据进行测试
'''

# 之前的代码
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

# 创建测试数据集
X_test,y_test = spiral_data(samples=100,classes=3)
# 测试数据集的前向传播
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output,y_test)

# 预测
predictions = np.argmax(loss_activation.output,axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test,axis=1)
accuracy = np.mean(predictions==y_test)
print(f'验证，acc:{accuracy:.3f},loss:{loss:.3f}')

'''
验证准确率为0.8，损失为0.8，
这个结果不算太差，但相比训练时出现的0.9的准确率和0.1的损失相比仍有差距
这表明模型出现了过拟合现象
'''