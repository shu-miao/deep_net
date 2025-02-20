'''
反向传播
'''
import numpy as np
from timeit import timeit
import nnfs
from nnfs.datasets import spiral_data

# 从一个简化的前向传递开始，只涉及一个神经元，来简化解释
# 从单个神经元的ReLU函数进行反向传播，就好像打算最小化这个神经元的输出一样
# 最小化一个神经元的输出除了练习以外没有其他意义，只是为了利用链式法则以及导数和偏导数来计算每个变量对ReLU激活输出的影响

# 使用一个具有三个输入的神经元作为示例，这意味这它有三个权重和一个偏置
x = [1.0,-2.0,3.0] # 输入
w = [-3.0,-1.0,2.0] # 权重
b = 1.0 # 偏置

x0w0 = x[0] * w[0] # ①输入乘以权重
# 对x1、w1、x2、w2重复操作
x1w1 = x[1] * w[1]
x2w2 = x[2] * w[2]
print(x0w0,x1w1,x2w2,b)

z = x0w0 + x1w1 + x2w2 + b # ②对所有带有偏差的加权输入进行求和，z是神经元直接的输出（未经过激活函数）
print(z)

y = max(z,0) # ③再在神经元的输出上应用ReLU激活函数
print('y:',y)

# noinspection SpellCheckingInspection
'''
通过单个神经元和ReLU激活函数的完整的前向传递，①②③链式函数构成一个大函数
这个大函数接受输入值(x)、权重(w)、偏置(b)作为输入，并输出y
这个大函数可以表示为ReLU(x0w0+x1w1+x2w2+b)
重写为y = ReLU(sum(mul(x0,w0),mul(x1,w1),mul(x2,w2),b))
求取偏导：∂[ReLU(sum(mul(x0,w0),mul(x1,w1),mul(x2,w2),b))]/∂x = (dReLU()/dsum())*(∂sum()/∂mul(x0,w0))*(∂mul(x0,w0)/∂x0)
这个方程表明求取偏导需要计算所有原子操作的导数和偏导数，并将它们相乘，以获得x0对输出的影响，并将被用来更新这些权重和偏置
关于输入的导数（即梯度）被用来通过将它们传递给链中的前一个函数，以链接更多的层。
'''

'''
在反向传递过程中，计算损失函数的导数，并使用它与输出层激活函数的导数相乘，然后使用这个结果与输出层的导数相乘，
以此类推，通过所有隐藏层和激活函数。
在这些层内，与权重和偏置相关的导数将形成用来更新权重和偏置的梯度，与输入相关的导数将形成与前一层链接的梯度。
这一层可以计算其权重和偏置对损失的影响，并在输入上进一步反向传播梯度
'''


drelu_dsum = (1. if z > 0 else 0.) # ReLU函数相对于z的导数，也就是对ReLU(z)求导，z就是神经元未经过激活函数的直接输出
print('First drelu_dsum:',drelu_dsum) # drelu_dsum = ∂ReLU()/∂sum()

# 假设神经元从下一层接收到一个梯度为1的输入
dvalue = 1.0
drelu_dsum = dvalue * drelu_dsum # 将∂ReLU()/∂sum()与从下一层接收到的导数相乘
print('Second drelu_dsum:',drelu_dsum)

# 神经网络中向后传播时，在执行激活函数前，立即出现的函数时加权输入和偏置的求和，也就是说激活函数的下一层子函数是加权输入和偏置的求和sum
# 那么使用链式法则，则需要将ReLU函数（外部函数）的导数与sum函数（子函数）的导数相乘
# 对ReLU(sum(mul(x0,w0),mul(x1,w1),mul(x2,w2),b))求偏导有
# drelu_dx0w0:ReLU相对于第一个加权输入w0x0的偏导数
# drelu_dx1w1:ReLU相对于第二个加权输入w1x1的偏导数
# drelu_dx2w2:ReLU相对于第三个加权输入w2x2的偏导数
# drelu_db:ReLU相对于偏置b的偏导数

dsum_dx0w0 = 1 # 对于求和操作的偏导数总是1，f(x,y,z,b) = x + y + z + b
drelu_dx0w0 = drelu_dsum * dsum_dx0w0 # 链式法则
print('drelu_dx0w0:',drelu_dx0w0)
# 其他偏导数
dsum_dx1w1 = 1
drelu_dx1w1 = drelu_dsum * dsum_dx1w1
print('drelu_dx1w1:',drelu_dx1w1)
dsum_dx2w2 = 1
drelu_dx2w2 = drelu_dsum * dsum_dx2w2
print('drelu_dx2w2:',drelu_dx2w2)
dsum_db = 1
drelu_db = drelu_dsum * dsum_db
print('drelu_db:',drelu_db)

# 接着对更深一层的函数应用链式法则进行求导，也就是sum函数内部的mul函数
# 写成mul(x0,w0),mul(x1,w1),mul(x2,w2)
# 这是三个二元函数，分别对x0,w0,x1,w1,x2,w2求偏导
# 而mul是加权输入操作，其计算是输出等于输入乘以权重，f(x) = wx，将权重也视为自变量得到二元函数f(w,x) = wx
# 那么无论它是否加上偏置，它相对于输入x的导数应该等于权重w，相对于权重w的导数应该等于输入x
dmul_dx0 = w[0] # 相对于输入x0的偏导等于对应权重w0
drelu_dx0 = drelu_dx0w0 * dmul_dx0 # 链式法则
# 其他偏导数
dmul_dx1 = w[1]
drelu_dx1 = drelu_dx1w1 * dmul_dx1
dmul_dx2 = w[2]
drelu_dx2 = drelu_dx2w2 * dmul_dx2
dmul_dw0 = x[0]
drelu_dw0 = drelu_dx0w0 * dmul_dw0
dmul_dw1 = x[1]
drelu_dw1 = drelu_dx1w1 * dmul_dw1
dmul_dw2 = x[2]
drelu_dw2 = drelu_dx2w2 * dmul_dw2
print("drelu_dx0: %.1f, drelu_dw0: %.1f." % (drelu_dx0, drelu_dw0))
print("drelu_dx1: %.1f, drelu_dw1: %.1f." % (drelu_dx1, drelu_dw1))
print("drelu_dx2: %.1f, drelu_dw2: %.1f." % (drelu_dx2, drelu_dw2))

# 代码计算十分清晰明了，但过于臃肿
# 简化代码（实际上就是合并）
'''
对于drelu_dx0 = drelu_dx0w0 * dmul_dx0,其中dmul_dx0 = w[0],那么drelu_dx0 = drelu_dx0w0 * w[0]
对于drelu_dx0 = drelu_dx0w0 * w[0],其中drelu_dx0w0 = drelu_dsum * dsum_dx0w0,那么drelu_dx0 = drelu_dsum * dsum_dx0w0 * w[0]
对于drelu_dx0 = drelu_dsum * dsum_dx0w0 * w[0],其中dsum_dx0w0 = 1,那么drelu_dx0 = drelu_dsum * 1 * w[0] = drelu_dsum * w[0]
对于drelu_dx0 = drelu_dsum * 1 * w[0] = drelu_dsum * w[0],其中drelu_dsum = dvalue * (1. if z > 0 else 0.),那么drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
最终结论drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
'''

# 函数链上的偏导已经全部求出，使用这些导数组成梯度
dx = [drelu_dx0,drelu_dx1,drelu_dx2]
dw = [drelu_dw0,drelu_dw1,drelu_dw2]
db = drelu_db
# 在这个单个神经元的例子中，dx并不需要，因为当前例子研究的是权重w对于输出的影响
print('梯度dw:',dw)

print('当前的权重和偏置:',w,b)
# 将一部分梯度应用到这些值上,使用梯度稍微改变了权重和偏置
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]

b += -0.001 * db
print('应用梯度后的权重和偏置:',w,b)

# 执行一次前向传递查看这次使用梯度做出的调整对输出的影响
x0w0 = x[0] * w[0]
x1w1 = x[1] * w[1]
x2w2 = x[2] * w[2]
z = x0w0 + x1w1 + x2w2 + b
y = max(z,0)
print('应用梯度后的输出:',y) # 应用梯度后的输出: 5.985
# 这个调整顺利的使神经元的输出减小了（在真实的神经网络中，减少神经元的输出是没有意义的）
# 以单个神经元的练习已经顺利完成，这个例子实际想展示的是如何使用导数、偏导数和链式法则智能地减少链式函数的值

'''
将上述单个神经元的例子推广到样本列表及整个神经元层

对于单个神经元的反向传播，它接收一个代表下一层的单一的导数以应用链式法则，
而在全连接层中，单个神经元连接到下一层的所有神经元，即下一层的所有神经元都接收这个神经元的输出，
那么在反向传播中，下一层的每个神经元将返回其函数相对于这个神经元输出的偏导数，这个神经元接收由这些导数组成的向量
为了继续反向传播，需要将这个向量求和为一个值

再进一步，用一层神经元替换当前的这一个单一神经元，一层的神经元将输出一个值向量并且层中的每个神经元都连接到下一层的所有神经元，
那么在反向传播中，当前层的每个神经元都会接收一个偏导数向量，那么对于这一层神经元，它将收到一个以偏导数向量组成的列表或者2D数组，
分析这个二维数组，它的每一行都是一个偏导数向量，这个向量的元素数量与前一层的神经元数量有关，实际上就是与当前层的输出（下一层的输入）有关，这个二维数组实际上是一个形状为下一层神经元数量a行*上一层神经元数量b列，
也就是该数组的列数等于输入维度也等于上一层神经元个数，行数等于下一层神经元个数
同单个神经元的处理，需要进行求和，不同的是，对于这个二维的数组，考虑到每个神经元将输出相对与其所有输入的偏导数的梯度，当前层的所有神经元都需要形成一个梯度向量
由上述对这个二维数组的分析可以得知，应当沿输入求和，也就是对列求和

更进一步，为了计算相对于输入的偏导数，则需要权重（相对于输入的偏导数等于相关的权重），即相对于所有输入的偏导数数组等于权重数组，
由于权重数组是转置的，则需要对其行而不是列进行求和，
应用链式法则，需要将它们乘以后续函数的梯度

注意：反向传播中的“下一层”是模型创建顺序中的前一层
'''

# 从下一层传来的梯度，设置为全1
dvalues = np.array([[1.,1.,1.]])

# 这次示例使用：三组权重（每组神经元(4个神经元)一个），四个输入（因此有四个权重），用于清晰求和过程
# 保留权重转置
weights = np.array([[0.2,0.8,-0.5,1],
                    [0.5,-0.91,0.26,-0.5],
                    [-0.26,-0.27,0.17,0.87]]).T
# 这个权重经过转置实际上等于[[0.2,0.5,-0.26],[0.8,-0.91,-0.27],[-0.5,0.26,0.17],[1,-0.5,0.87]]

# 对给定权重求和并乘以对应权重
dx0 = sum(weights[0])*dvalues[0]
dx1 = sum(weights[1])*dvalues[0]
dx2 = sum(weights[2])*dvalues[0]
dx3 = sum(weights[3])*dvalues[0]

# 神经元函数相对于输入的梯度
dinputs = np.array([dx0,dx1,dx2,dx3])
print(dinputs)

# 从numpy的角度简化dx0到dx3的计算
dx0 = sum(weights[0]*dvalues[0])
dx1 = sum(weights[1]*dvalues[0])
dx2 = sum(weights[2]*dvalues[0])
dx3 = sum(weights[3]*dvalues[0])
dinputs = np.array([dx0,dx1,dx2,dx3])
print(dinputs)

# 从单个样本推广到一批样本
# 从下一层传递来的梯度，为了这个例子的目的（降低输出），使用一系列递增的梯度值作为示例
dvalues = np.array([[1.,1.,1.],
                    [2.,2.,2.],
                    [3.,3.,3.]])

weights = np.array([[0.2,0.8,-0.5,1],
                    [0.5,-0.91,0.26,-0.5],
                    [-0.26,-0.27,0.17,0.87]]).T

# 对给定输入的权重求和，并将其乘以下一层传递来的梯度，使用np.dot简化计算
dinputs = np.dot(dvalues, weights.T)
print(dinputs)

'''
计算相对于输入的梯度已经完成，接下来讨论相对于权重的梯度，
在计算相对于输入的梯度中，匹配输入的形状进行计算，而在计算相对与权重的梯度时，匹配的是权重的形状，
由于相对于权重的导数等于输入，权重被转置，所以对应的需要转置输入以获得相对于权重的神经元的导数，
之后再将这些转置输入作为点积的第一个参数（即点积计算中将通过输入乘以行），其中每行包含所有样本的给定输入的数据与dvalues的列相乘（使用行乘是因为它已经被转置），
这些列与所有样本的单个神经元的输出相关，所有结果将是一个具有权重形状的数组，
这个数组的内容是输入相对于该权重的梯度，并且这个梯度是与批次中所有的样本的传入梯度相乘得到的
'''
dvalues = np.array([[1.,1.,1.],
                    [2.,2.,2.],
                    [3.,3.,3.]])
inputs = np.array([[1,2,3,2.5],
                   [2.,5.,-1.,2],
                   [-1.5,2.7,3.3,-0.8]])

# 使用从下一次传来的梯度与输入相乘，得到相对于权重的梯度
dweights = np.dot(inputs.T,dvalues)
print(dweights)

'''
相对于输入的梯度、相对于权重的梯度都已经得到，接下来讨论相对于偏置的梯度，
对应偏置，其导数来自求和操作并且总是等于1，然后乘以传入的梯度以应用链式法则，
由于梯度是梯度的列表（每个神经元对所有样本的梯度向量），只需按列将它们与神经元求和
'''
dvalues = np.array([[1.,1.,1.],
                    [2.,2.,2.],
                    [3.,3.,3.]])
biases = np.array([[2,3,0.5]])
dbiases = np.sum(dvalues,axis=0,keepdims=True) # keepdims保持梯度作为行向量
print(dbiases)

'''
接下来讨论ReLU函数的导数，
ReLU函数：如果输入大于0，则等于1；否则等于0。
在前向传递过程中，层通过ReLU函数激活函数传递其输出，
对于反向传递，ReLU接收一个与输入数据（即经过ReLU函数处理的输出）形状相同的的梯度，
ReLU函数的导数将形成一个形状相同的数组，当相关输入大于0时填充1，否则填充0，
之后应用链式法则，将这个数组与后续函数的梯度相乘
'''
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]]) # 为经过激活函数的神经元的直接输出
dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
drelu = np.zeros_like(z) # 创建一个与z形状相同的0数组
drelu[z>0] = 1 # 当相关输入大于0时填充1
print(drelu)
# 链式法则
drelu *= dvalues
print(drelu)

# 由于ReLU函数导数数组填充有1，这些1不会改变与它们相乘的值，而0会使乘数值为0
# 于是可以取后续函数的梯度，并将所有对应于ReLU输入且小于等于0的值设为0
# 那么可以简化操作
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

drelu = dvalues.copy() # 复制dvalues确保在计算ReLU导数时不会修改它
drelu[z <= 0] = 0 # 将所有对应于ReLU输入且小于等于0的值设为0
print(drelu)

'''
将单个神经元的前向和后向传播与全层和基于批处理的部分导数结合起来，
再次最小化ReLU的输出
'''
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]]) # 传递过来的权重
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]]) # 输入
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T # 权重
biases = np.array([[2, 3, 0.5]]) # 偏置

# 前向传播
layer_outputs = np.dot(inputs,weights) + biases # 输入乘以权重加上偏置
relu_outputs = np.maximum(0,layer_outputs) # ReLU激活函数

# 反向传播
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0  # 激活函数的导数（相对于输入值）
dinputs = np.dot(drelu,weights.T) # 相对于输入的梯度
dweights = np.dot(inputs.T,drelu) # 相对于权重的梯度
dbiases = np.sum(drelu,axis=0,keepdims=True) # 相对于偏置的梯度

# 更新权重和偏置
weights += -0.001 * dweights
biases += -0.001 * dbiases
print(weights)
print(biases)


'''
在全连接层类中更新反向传播方法
'''

class Layer_Dense:
    def __init__(self,inputs,neurons):
        # 初始化权重和偏置
        self.weights = 0.01 * np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))

    def forward(self,inputs):
        # 前向传播
        self.inputs = inputs # 由于在反向传播中需要用到输入，所以用一个对象属性保存输入
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues) # 前一层传来的梯度乘以输入计算相对于权重的梯度
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True) # 计算相对于偏置的梯度
        self.dinputs = np.dot(dvalues,self.weights.T) # 计算相对于输入的梯度


class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs # 保存输入
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 # 计算ReLU的导数


'''
分类交叉熵损失导数
'''

class Loss:
    # 损失函数父类
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    # 基础父类，追加forward方法
    def forward(self,y_pred,y_true):
    # 前向传播
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self,dvalues,y_true):
    # 反向传播
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] # 使用单位矩阵（从左上到右下的对角线上的值为1，其余为0）创建one_hot编码

        self.dinputs = -y_true / dvalues # 分类交叉熵损失导数的计算：负的真实向量除以预测值向量
        self.dinputs = self.dinputs / samples


'''
Softmax激活函数导数
'''
class Activation_Softmax:
    def forward(self,inputs):
        # 前向传播
        self.inputs = inputs # 保存输入

        # 计算未归一化的概率值
        # 使用np.exp计算输入的指数值，为了数据的稳定性，减去每一行的最大值
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        # 对每个样本进行归一化，使得输出值变成概率分布
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)

        # 保存输出值
        self.output = probabilities
    def backward(self,dvalues):
        # 反向传播

        # 创建与dvalues形状相同的未初始化数组，用于储存梯度
        self.dinputs = np.empty_like(dvalues)
        # 遍历每个样本的输出和对应的梯度
        for index, (single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            # 将单个样本的输出展平为列向量
            single_output = single_output.reshape(-1,1)
            # 计算该样本输出的雅可比矩阵
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            # 计算该样本的梯度（雅可比矩阵与梯度相乘）
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

'''
常见的分类交叉熵损失和Softmax激活函数导数
'''

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

'''
分别使用上述两种解决方案测试计算梯度
dvalues1：使用Activation_Softmax_Loss_CategoricalCrossentropy
dvalues2：使用Activation_Softmax和Loss_CategoricalCrossentropy
'''
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('梯度：组合损失和激活函数:')
print(dvalues1)
print('梯度：分离损失函数和激活函数:')
print(dvalues2)


'''
完整的反向传播测试
'''
# 生成数据集
X,y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3 ) # 创建一个全连接层
activation1 = Activation_ReLU() # ReLU函数层
dense2 = Layer_Dense(3,3) # 第二层全连接层
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy() # Softmax和损失函数层

dense1.forward(X) # 第一层全连接的前向传播
activation1.forward(dense1.output) # ReLU函数层的前向传播
dense2.forward(activation1.output) # 第二层的前向传播
loss = loss_activation.forward(dense2.output,y) # Softmax和损失函数层的前向传播
print(loss_activation.output[:5])
print('loss:',loss)

predictions = np.argmax(loss_activation.output,axis=1) # 返回最大预测值的索引
if len(y.shape) == 2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(predictions==y) # 计算准确率
print('acc:',accuracy)

loss_activation.backward(loss_activation.output,y) # 损失函数的反向传播
dense2.backward(loss_activation.dinputs) # 第二层全连接的反向传播
activation1.backward(dense2.dinputs) # ReLU函数层的反向传播
dense1.backward(activation1.dinputs) # 第一层的反向传播

# 输出梯度
print('第一层相对于权重的梯度',dense1.dweights)
print('第一层相对于偏置的梯度',dense1.dbiases)
print('第二层相对于权重的梯度',dense2.dweights)
print('第二层相对于偏置的梯度',dense2.dbiases)
