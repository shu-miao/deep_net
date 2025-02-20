'''
损失函数：计算网络误差
'''

import math

softmax_output = [0.7,0.1,0.2] # softmax层的输出
target_output = [1,0,0] # one-hot目标向量类别标签

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print('loss:\n',loss)

'''
one-hot向量中，0乘以任何数都为0，所以只用取值为1的标签进行计算，过程可以简化为
'''
softmax_output = [0.7,0.1,0.2]
loss - -math.log(softmax_output[0])
print('loss:\n',loss)

# log函数的性质
import math
print(math.log(1.))
print(math.log(0.95))
print(math.log(0.9))
print(math.log(0.8))
print('...')
print(math.log(0.2))
print(math.log(0.1))
print(math.log(0.05))
print(math.log(0.01))

# 使用numpy库
import numpy as np
b = 5.2
print(np.log(b)) # 1.6486586255873816
print(math.e ** 1.6486586255873816) # 5.199999999999999
# 这种差异是由python中的浮点数精度引起的

# 假设神经网络在三个类别之间进行分类，并且神经网络以三批的方式进行分类，在通过softmax激活函数处理一批三个样本和三个类别之后
# 网络的输出层产生如下：
softmax_outputs = [[0.7,0.1,0.2],
                  [0.1,0.5,0.4],
                  [0.02,0.9,0.08]]
class_targets = [0,1,1]
for targ_idx,distribution in zip(class_targets,softmax_outputs):
    print(targ_idx,distribution)
    print(distribution[targ_idx])
    print('=====================')

# 使用numpy简化
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = [0,1,1]
class_targets_value = softmax_outputs[[0,1,2],class_targets]
print(class_targets_value)

# 加入索引
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = [0,1,1]
class_targets_value = softmax_outputs[range(len(softmax_outputs)),class_targets]
print(class_targets_value)

# 应用负对数
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = [0,1,1]
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)),class_targets])
print(neg_log)

# 计算每批的平均损失值，以便了解模型在训练中的表现
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = [0,1,1]
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)),class_targets])
average_loss = np.mean(neg_log)
print(average_loss)

# 对于多维的输入，将置信度与目标相乘，除了正确标签的值外，将所有值归零
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = np.array([[1,0,0],
                          [0,1,0],
                          [0,1,0]])
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)),class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )
# 损失
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)


'''
Softmax输出，也就是这个损失函数的输入，是一系列置信度，也就是0-1的数字。
模型可能会对某个标签有完全的信心，使得所有剩余的置信度为零。
同样，模型也可能将完全的信心分配给一个非目标值。
接下来，尝试计算这个置信度为0的损失
'''
# loss_0 = -np.log(0)
# RuntimeWarning: divide by zero encountered in log
# log（0）的值为负无穷，此时程序会报除零错误，且单个值为无穷会导致该列表均值也为无穷，对模型的后续处理也会造成影响

# 为了解决这个问题，给置信度加上一个很小的值，防止它为零，例如1e-7
result = -np.log(1e-7)
print(result)

# 但这样处理会出现新的问题，当置信度为1时
result = -np.log(1+1e-7)
print(result) # -9.999999505838704e-08
# 此时损失为负数，这显然是不合理的
# 于是从两端同时剪切值，当置信度为0时，置为1e-7，当置信度为1时，置为1-1e-7
result = -np.log(1-1e-7)
print(result)

# 使用numpy实现
y_pred = np.array([0.1,0.2,0.3,np.inf,-np.inf])
y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
print(y_pred_clipped) # 该方法可以对数值数组执行剪切，因此我们可以直接将其应用于预测值，并将其保存为一个单独的数组


'''
创建分类交叉熵损失类
'''
class Loss:
    # 需要子类继承并重写forward方法实现
    def calculate(self,output,y):
        '''
        :param output: 模型输出的置信度
        :param y: 真实标签/目标值
        :return: 平均损失
        '''
        sample_losses = self.forward(output,y) # 计算每个样本的损失
        data_loss = np.mean(sample_losses)
        return data_loss

# 子类继承
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred) # 获取批量样本数
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7) # 数值稳定处理

        # 处理不同格式的真实标签
        if len(y_true.shape) == 1: # 稀疏标签，如[2,5,3]
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2: # one-hot编码标签，如[[1,0,0],[0,1,0]]
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences) # 计算负对数（负对数似然）
        return negative_log_likelihoods

softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = np.array([0,1,1])
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs,class_targets)
print(loss)