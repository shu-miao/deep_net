'''
真实数据集
'''

'''
目前使用的数据集均用于示例，并没有太多的实际意义
为了真实考量这个模型是否可以胜任真实任务，引入Fashion MNIST数据集
Fashion MNIST数据集包含60,000个训练样本和10,000个测试样本，
这些样本是28x28像素的图像，涵盖了10种不同的服装类别，例如鞋子、靴子、衬衫、包等。

另外，这个例子中使用的数据集图像是以.png格式保存的图像，
通常，对于图像来说，使用无损压缩是最好的，因为有损压缩（例如JPEG）会通过更改图像数据对图像造成影响
'''

import nnfs
import numpy as np
from zipfile import ZipFile
import os
import urllib
import urllib.request
import cv2
import matplotlib.pyplot as plt

# 在开始之前，即将提到的相关的代码：
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

class Accuracy:
    # 准确率计算父类
    def calculate(self,predictions,y):
        # 获取比较结果
        comparisons = self.compare(predictions,y)
        # 计算准确率
        accuracy = np.mean(comparisons)
        return accuracy

# 数据准备，只需执行一次
# URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# FILE = 'fashion_mnist_images.zip'
# FOLDER = 'fashion_mnist_images'
# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} and saving as {FILE}...')
#     urllib.request.urlretrieve(URL,FILE)
# print('Unzipping images...')
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)
# print('Done!')

# 数据加载
# labels = os.listdir('fashion_mnist_images/train') # 子目录本身就是标签
# print(labels)
# files = os.listdir('fashion_mnist_images/train/0') # 查看一个类别
# print(len(files)) # 6000，有6000个类别0的样本，总共有60000个样本，每个类别6000个
# 每个类别样本数相同，这个数据集是平衡的，如果数据不平衡，神经网络可能会倾向于预测包含最多图像的类别

# image_data = cv2.imread('fashion_mnist_images/train/7/0002.png',cv2.IMREAD_UNCHANGED) # 加载图像数据
# np.set_printoptions(linewidth=200) # 设置打印长度，每行200个字符
# print(image_data) # 可以看出来这是一双鞋

# plt.imshow(image_data,cmap='gray') # 使用matplotlib查看
# plt.show() # 确实是一双鞋

# 创建一个数据载入函数，遍历所有的样本，将其添加到输入数据（X）和目标列表（y）中
def load_mnist_dataset(dataset,path):
    labels = os.listdir(os.path.join(path,dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image = cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X),np.array(y).astype('uint8') # 返回数据列表，和标签列表（标签设置为整数）
# 创建一个数据返回函数，创建和返回训练集和测试集
def create_data_mnist(path):
    X,y = load_mnist_dataset('train',path)
    X_test,y_test = load_mnist_dataset('test',path)
    return X,y,X_test,y_test

# 加载数据
# X,y,X_test,y_test = create_data_mnist('fashion_mnist_images')


# 数据预处理，注意预处理方法需要处理所有的数据，包括用于测试和推理的数据
# 最大像素值为255，为了将数据缩放到-1到1区间，将每个像素值减去最大像素值的一半，再除以这个半值，255/2=127.5
# X = (X.astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# print(X.min(),X.max()) # 检查数据是否被成功缩放，输出-1.0 1.0
# 检查输入数据的形状
# print(X.shape) # (60000,28,28)
# 全连接网络输入的是一维的样本数据，不能直接处理二维的图像数据
# 所以需要对这个图像数组进行展平
# X = X.reshape(X.shape[0],-1) # -1作为第二维度意味着希望将所有样本数据放入这个单一维度中，形成一维数组
# X_test = X_test.reshape(X_test.shape[0],-1)


# 数据洗牌
# 按照原本的顺序会导致模型训练是总是围绕某个局部最小值，所以需要进行打乱
# 为了确保样本数据X和目标列表y同步打乱，方法是获取所有的键（样本列表和目标列表的索引），然后对这些键进行打乱
# keys = np.array(range(X.shape[0])) # 获取键
# print(keys[:10]) # [0 1 2 3 4 5 6 7 8 9]
# nnfs.init()
# np.random.shuffle(keys) # 打乱
# print(keys[:10]) # [ 3048 19563 58303  8870 40228 31488 21860 56864   845 25770]
# 这个key键充当样本列表和目标列表的索引
# X = X[keys]
# y = y[keys]
# 它们确实被成功打乱
# print(X[:10])
# print(y[:10])


# 批次
# 批次是数据的一个固定大小的切片，对于较大的数据集，通常需要将其分为多个批次传入模型进行训练
# 这个批次需要在一定程度上代表整个数据集，所以它不能太小
# 太大的批次会导致训练缓慢，所以它不能太大
# 通常将批次设置为32到128个样本
# 假设选择批次大小为128，并进行10轮训练，这意味着在每一轮训练中，都会遍历数据集，每次拟合128个样本来训练模型
# 每次训练的批次被称为一个步骤，样本数量除以批次大小可得步骤的数量
'''
到目前为止，对于损失，模型计算的是整个数据集每次拟合的损失，
由于批次的引入，自然需要计算批次统计和轮次统计，
对于总体的损失和准确率，则希望计算样本级别的平均值
所以，在每轮结束时累积所有批次的损失总和和样本数量
'''

# 完善Loss类中的calculate方法
class Loss:
    def calculate(self,output,y,*,include_regularization=False):
        sample_losses = self.forward(output,y) # 计算损失函数层的直接损失
        data_loss = np.mean(sample_losses) # 求平均损失
        self.accumulated_sum += np.sum(sample_losses) # 存储损失值的累计和，将当前批次的总损失加到之前的累积损失中
        self.accumulated_count += len(sample_losses) # 存储样本的累计数量，将当前批次的样本数量加到之前的累积样本数量中
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
    def calculate_accumulated(self,*,include_regularization=False):
        # 计算累积损失
        data_loss = self.accumulated_sum / self.accumulated_count # 计算平均损失
        if not include_regularization:
            # 如果不需要正则化，直接返回数据损失
            return data_loss
        # 返回数据损失和正则化损失
        return data_loss,self.regularization_loss()
    def new_pass(self):
        # 在新一轮训练中重置总和和计数值
        self.accumulated_sum = 0
        self.accumulated_count = 0

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

class Activation_Softmax_Loss_CategoricalCrossentropy():
    # def __init__(self):
    #     # 初始化方法
    #     self.activation = Activation_Softmax() # 创建Softmax激活函数的实例
    #     self.loss = Loss_CategoricalCrossentropy() # 创建一个分类交叉熵损失函数的实例
    #
    # def forward(self,inputs,y_true):
    #     # 前向传播
    #     self.activation.forward(inputs) # 调用Softmax激活函数的前向传播方法，计算并存储输出
    #     self.output = self.activation.output # 保存激活函数的输出，通常是每个类别的概率
    #     return self.loss.calculate(self.output,y_true) # 使用分类交叉熵损失计算并返回损失值

    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues) # 样本数量，等于dvalue（收到后续层的梯度）的长度
        if len(y_true.shape) == 2: # 检查y_true是否是独热编码格式
            y_true = np.argmax(y_true,axis=1) # 如果是，使用np.argmax将其转换为整数标签
        self.dinputs = dvalues.copy() # 复制梯度值以便进行修改
        self.dinputs[range(samples),y_true] -= 1 # 根据真实标签更新梯度，针对每个样本的正确类别减去1
        self.dinputs = self.dinputs / samples # 对梯度进行归一化，确保计算的梯度是平均值


# 类似的在Accuracy类中实现相同的功能
class Accuracy:
    # 准确率计算父类
    def calculate(self,predictions,y):
        # 获取比较结果
        comparisons = self.compare(predictions,y)
        # 计算准确率
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    def calculate_accumulated(self):
        # 返回累计的准确率
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    def new_pass(self):
        # 每轮训练开始时重置计数
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Categorical(Accuracy):
    def init(self,y):
        pass
    def compare(self,predictions,y):
        if len(y.shape) == 2:
            y = np.argmax(y,axis=1)
        return predictions == y # 返回一个布尔列表，表示分类正确的标签

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

# 完善Model类中的train方法以适配使用批次训练
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
    def train(self,X,y,*,epochs=1,batch_size=None,print_every=1,validation_data=None):
        self.accuracy.init(y) # 初始化准确率计算对象
        train_steps = 1 # 步数
        if validation_data is not None:
            # 如果提供了验证数据，则初始化验证步骤validation_steps为1，解包验证数据为X_val,y_yal
            validation_steps =  1
            X_val,y_yal = validation_data
        if batch_size is not None:
            # 如果设置了批次大小
            train_steps = len(X) // batch_size # 计算训练所需的步数，总样本数除以批次大小
            if train_steps * batch_size < len(X):
                # 如果数据不能被批量大小整除，则增加一个步骤以覆盖剩余数据
                train_steps += 1
            if validation_data is not None:
                # 如果提供了验证数据
                validation_steps = len(X_val) // batch_size # 计算验证所需的步数
                if validation_steps * batch_size < len(X_val):
                    # 同样，如果数据不能被批量大小整除，则增加一个步骤以覆盖剩余数据
                    validation_steps += 1
        for epoch in range(1,epochs+1):
            # 训练主循环
            print(f'epoch:{epoch}') # 打印当前轮次
            self.loss.new_pass() # 重置损失的累计值
            self.accuracy.new_pass() # 重置准确率的累计值
            for step in range(train_steps):
                # 遍历当前轮次中的每个训练步骤
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                output = self.forward(batch_X,training=True) # 将当前批次的数据输入模型进行前向传播，计算输出
                data_loss,regularization_loss = self.loss.calculate(output,batch_y,include_regularization=True) # 计算数据损失和正则化损失
                loss = data_loss + regularization_loss # 加和得到总损失
                predictions = self.output_layer_activation.predictions(output) # 获取预测结果
                accuracy = self.accuracy.calculate(predictions,batch_y) # 使用预测结果和真实标签计算当前批次的准确率
                self.backward(output,batch_y) # 执行反向传播，计算梯度
                self.optimizer.pre_update_params() # 优化器前置操作
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer) # 对可训练的层进行逐层更新
                self.optimizer.post_update_params() # 优化器后置操作
                if not step % print_every or step == train_steps - 1:
                    # 每print_every步打印一次信息，或在最后一步打印信息
                    print(f'step:{step},'+
                          f'acc:{accuracy:.3f},'+
                          f'loss:{loss:.3f}('+
                          f'data_loss:{data_loss:.3f},'+
                          f'reg_loss:{regularization_loss:.3f}),'+
                          f'lr:{self.optimizer.current_learning_rate}')
            epoch_data_loss,epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss # 轮次损失
            epoch_accuracy = self.accuracy.calculate_accumulated() # 轮次准确率、
            # 每一轮次打印该轮次信息
            print(f'training,'+
                  f'acc:{epoch_accuracy:.3f},'+
                  f'loss:{epoch_loss:.3f}('+
                  f'data_loss:{epoch_data_loss:.3f},'+
                  f'reg_loss:{epoch_regularization_loss:.3f}),'+
                  f'lr:{self.optimizer.current_learning_rate}')
            if validation_data is not None:
                self.loss.new_pass() # 重置验证损失
                self.accuracy.new_pass() # 重置验证准确率
                for step in range(validation_steps):
                    # 遍历验证数据
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_yal
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_yal[step*batch_size:(step+1)*batch_size]
                    output = self.forward(batch_X,training=False) # 执行前向传播，计算输出
                    self.loss.calculate(output,batch_y) # 计算损失
                    predictions = self.output_layer_activation.predictions(output) # 预测
                    self.accuracy.calculate(predictions,batch_y) # 计算准确率
                validation_loss = self.loss.calculate_accumulated() # 验证损失
                validation_accuracy = self.accuracy.calculate_accumulated()
                print(f'validation,'+
                      f'acc:{validation_accuracy:.3f},'+
                      f'loss:{validation_loss:.3f}')
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

'''
在开始测试使用批次进行模型训练前，有必要重新梳理所有的代码避免出现错误
'''

# 载入数据
def load_mnist_dataset(dataset,path):
    labels = os.listdir(os.path.join(path,dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image = cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X),np.array(y).astype('uint8') # 返回数据列表，和标签列表（标签设置为整数）
# 创建一个数据返回函数，创建和返回训练集和测试集
def create_data_mnist(path):
    X,y = load_mnist_dataset('train',path)
    X_test,y_test = load_mnist_dataset('test',path)
    return X,y,X_test,y_test

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

class Layer_Input:
    # 前向传播
    def forward(self, inputs, training):
        self.output = inputs

class Activation_ReLU:
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self,outputs):
        return outputs

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

class Activation_Linear:
    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = inputs # 不修改输入，直接输出
    def backward(self,dvalues):
        self.dinputs = dvalues.copy() # 反向传播应用链式法则，使用前一层梯度乘以本层导数1
    def predictions(self,outputs):
        return outputs

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

class Loss:
    def calculate(self,output,y,*,include_regularization=False):
        sample_losses = self.forward(output,y) # 计算损失函数层的直接损失
        data_loss = np.mean(sample_losses) # 求平均损失
        self.accumulated_sum += np.sum(sample_losses) # 存储损失值的累计和，将当前批次的总损失加到之前的累积损失中
        self.accumulated_count += len(sample_losses) # 存储样本的累计数量，将当前批次的样本数量加到之前的累积样本数量中
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
    def calculate_accumulated(self,*,include_regularization=False):
        # 计算累积损失
        data_loss = self.accumulated_sum / self.accumulated_count # 计算平均损失
        if not include_regularization:
            # 如果不需要正则化，直接返回数据损失
            return data_loss
        # 返回数据损失和正则化损失
        return data_loss,self.regularization_loss()
    def new_pass(self):
        # 在新一轮训练中重置总和和计数值
        self.accumulated_sum = 0
        self.accumulated_count = 0

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

class Activation_Softmax_Loss_CategoricalCrossentropy():
    # def __init__(self):
    #     # 初始化方法
    #     self.activation = Activation_Softmax() # 创建Softmax激活函数的实例
    #     self.loss = Loss_CategoricalCrossentropy() # 创建一个分类交叉熵损失函数的实例
    #
    # def forward(self,inputs,y_true):
    #     # 前向传播
    #     self.activation.forward(inputs) # 调用Softmax激活函数的前向传播方法，计算并存储输出
    #     self.output = self.activation.output # 保存激活函数的输出，通常是每个类别的概率
    #     return self.loss.calculate(self.output,y_true) # 使用分类交叉熵损失计算并返回损失值

    def backward(self,dvalues,y_true):
        # 反向传播
        samples = len(dvalues) # 样本数量，等于dvalue（收到后续层的梯度）的长度
        if len(y_true.shape) == 2: # 检查y_true是否是独热编码格式
            y_true = np.argmax(y_true,axis=1) # 如果是，使用np.argmax将其转换为整数标签
        self.dinputs = dvalues.copy() # 复制梯度值以便进行修改
        self.dinputs[range(samples),y_true] -= 1 # 根据真实标签更新梯度，针对每个样本的正确类别减去1
        self.dinputs = self.dinputs / samples # 对梯度进行归一化，确保计算的梯度是平均值

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

class Accuracy:
    # 准确率计算父类
    def calculate(self,predictions,y):
        # 获取比较结果
        comparisons = self.compare(predictions,y)
        # 计算准确率
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    def calculate_accumulated(self):
        # 返回累计的准确率
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    def new_pass(self):
        # 每轮训练开始时重置计数
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Categorical(Accuracy):
    def init(self,y):
        pass
    def compare(self,predictions,y):
        if len(y.shape) == 2:
            y = np.argmax(y,axis=1)
        return predictions == y # 返回一个布尔列表，表示分类正确的标签

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
    def train(self,X,y,*,epochs=1,batch_size=None,print_every=1,validation_data=None):
        self.accuracy.init(y) # 初始化准确率计算对象
        train_steps = 1 # 步数
        if validation_data is not None:
            # 如果提供了验证数据，则初始化验证步骤validation_steps为1，解包验证数据为X_val,y_yal
            validation_steps =  1
            X_val,y_yal = validation_data
        if batch_size is not None:
            # 如果设置了批次大小
            train_steps = len(X) // batch_size # 计算训练所需的步数，总样本数除以批次大小
            if train_steps * batch_size < len(X):
                # 如果数据不能被批量大小整除，则增加一个步骤以覆盖剩余数据
                train_steps += 1
            if validation_data is not None:
                # 如果提供了验证数据
                validation_steps = len(X_val) // batch_size # 计算验证所需的步数
                if validation_steps * batch_size < len(X_val):
                    # 同样，如果数据不能被批量大小整除，则增加一个步骤以覆盖剩余数据
                    validation_steps += 1
        for epoch in range(1,epochs+1):
            # 训练主循环
            print(f'epoch:{epoch}') # 打印当前轮次
            self.loss.new_pass() # 重置损失的累计值
            self.accuracy.new_pass() # 重置准确率的累计值
            for step in range(train_steps):
                # 遍历当前轮次中的每个训练步骤
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                output = self.forward(batch_X,training=True) # 将当前批次的数据输入模型进行前向传播，计算输出
                data_loss,regularization_loss = self.loss.calculate(output,batch_y,include_regularization=True) # 计算数据损失和正则化损失
                loss = data_loss + regularization_loss # 加和得到总损失
                predictions = self.output_layer_activation.predictions(output) # 获取预测结果
                accuracy = self.accuracy.calculate(predictions,batch_y) # 使用预测结果和真实标签计算当前批次的准确率
                self.backward(output,batch_y) # 执行反向传播，计算梯度
                self.optimizer.pre_update_params() # 优化器前置操作
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer) # 对可训练的层进行逐层更新
                self.optimizer.post_update_params() # 优化器后置操作
                if not step % print_every or step == train_steps - 1:
                    # 每print_every步打印一次信息，或在最后一步打印信息
                    print(f'step:{step},'+
                          f'acc:{accuracy:.3f},'+
                          f'loss:{loss:.3f}('+
                          f'data_loss:{data_loss:.3f},'+
                          f'reg_loss:{regularization_loss:.3f}),'+
                          f'lr:{self.optimizer.current_learning_rate}')
            epoch_data_loss,epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss # 轮次损失
            epoch_accuracy = self.accuracy.calculate_accumulated() # 轮次准确率、
            # 每一轮次打印该轮次信息
            print(f'training,'+
                  f'acc:{epoch_accuracy:.3f},'+
                  f'loss:{epoch_loss:.3f}('+
                  f'data_loss:{epoch_data_loss:.3f},'+
                  f'reg_loss:{epoch_regularization_loss:.3f}),'+
                  f'lr:{self.optimizer.current_learning_rate}')
            if validation_data is not None:
                self.loss.new_pass() # 重置验证损失
                self.accuracy.new_pass() # 重置验证准确率
                for step in range(validation_steps):
                    # 遍历验证数据
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_yal
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_yal[step*batch_size:(step+1)*batch_size]
                    output = self.forward(batch_X,training=False) # 执行前向传播，计算输出
                    self.loss.calculate(output,batch_y) # 计算损失
                    predictions = self.output_layer_activation.predictions(output) # 预测
                    self.accuracy.calculate(predictions,batch_y) # 计算准确率
                validation_loss = self.loss.calculate_accumulated() # 验证损失
                validation_accuracy = self.accuracy.calculate_accumulated()
                print(f'validation,'+
                      f'acc:{validation_accuracy:.3f},'+
                      f'loss:{validation_loss:.3f}')
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

# 使用批次进行训练
X,y,X_test,y_test = create_data_mnist('fashion_mnist_images')

# 洗牌
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# 缩放样本至-1到1
X = (X.reshape(X.shape[0],-1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0],-1).astype(np.float32) - 127.5) / 127.5

# 构建模型
model = Model()
# 添加层
model.add(Layer_Dense(X.shape[1],128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128,128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128,10))
model.add(Activation_Softmax())
# 设置损失、优化器、精度
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# 模型初始化
model.finalize()

model.train(X,y,validation_data=(X_test,y_test),epochs=5,batch_size=128,print_every=100)
# training,acc:0.869,loss:0.359(data_loss:0.359,reg_loss:0.000),lr:0.0008950948800572861
# validation,acc:0.857,loss:0.399