'''
梯度下降/导数
'''

'''
随机搜索最佳权重和偏置的方法是错误的，原因是权重和偏置的可能组合是无限的，因此需要找到更科学的更新权重和偏置的方法
权重和偏置如何影响总体损失的函数并不一定是线性的，为了知道如何调整权重和偏置，需要先知道它们对损失的影响
虽然权重和偏置影响了损失，但损失函数中并不包含权重和偏置，损失函数的输入是模型输出的置信度，这个置信度被神经元的权重和偏置影响
因此，即使是从模型的输出计算损失，神经元的权重和偏置也直接影响了这个损失
换言之，权重和偏置同损失之间，是一个可以使用复合函数描述的关系
为了解释这种关系，将引入偏导数、梯度、梯度下降和反向传播来详细描述这一过程
'''

import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单函数y=2x
def f(x):
    return 2*x

x = np.array(range(5))
y = f(x)

print(x) # [0 1 2 3 4]
print(y) # [0 2 4 6 8]

plt.plot(x,y)
plt.show()
# 描述x对y的影响：y是x的两倍，即y=2x的斜率为2
print((y[1]-y[0])/(x[1]-x[0])) # 斜率的计算方法，y的变化量除以x的变化量

# 对于一个非线性函数
def f(x):
    return 2*x**2
p2_delta = 0.0001 # 定义一个特别小的数，用来取微
x1 = 1
x2 = x1 + p2_delta
y1 = f(x1)
y2 = f(x2)
approximate_derivative = (y2-y1)/(x2-x1) # 计算曲线的瞬时斜率，y的变化量除以x的变化量
print(approximate_derivative)

def f(x):
    return 2*x**2
x = np.arange(0,5,0.001)
y = f(x)
plt.plot(x,y)
plt.show() # 通过绘制一小段一小段切线的方法，绘制该曲线

# 通过x和y的值，确定切线截距以绘制切线
def f(x):
    return 2*x**2

x = np.arange(0,5,0.001)
y = f(x)
plt.plot(x,y)
p2_delta = 0.0001
x1 = 2
x2 = x1+p2_delta
y1 = f(x1)
y2 = f(x2)
print((x1,y1),(x2,y2))
approximate_derivative = (y2-y1)/(x2-x1)
b = y2 - approximate_derivative*x2 # 确定截距

def tangent_line(x):
    return approximate_derivative*x + b

to_plot = [x1-0.9,x1,x1+0.9]
plt.plot(to_plot,[tangent_line(i) for i in to_plot])
print('f(x) 的近似导数',f'当 x = {x1} 时，导数为 {approximate_derivative}。')
plt.show() # 绘制曲线和x=2时的近似切线

# 绘制多条切线
def f(x):
    return 2*x**2
x = np.array(np.arange(0,5,0.001))
y = f(x)
plt.plot(x,y)
colors = ['k','g','r','b','c']

def approximate_tangent_line(x,approximate_derivative):
    return (approximate_derivative*x) + b

for i in range(5):
    p2_delta = 0.001
    x1 = i
    x2 = i + p2_delta
    y1 = f(x1)
    y2 = f(x2)

    print((x1,y1),(x2,y2))
    approximate_derivative = (y2-y1)/(x2-x1)
    b = y2-(approximate_derivative*x2)
    to_plot = [x1-0.9,x1,x1+0.9]
    plt.scatter(x1,y1,c=colors[i])
    plt.plot([point for point in to_plot],[approximate_tangent_line(point,approximate_derivative) for point in to_plot],c=colors[i])
    print('f(x) 的近似导数',f'当 x = {x1} 时，导数为 {approximate_derivative}。')
plt.show()

'''
对于这类简单函数，f(x)=2x^2，通过近似导数（即切线的斜率），并没有产生较大的误差，就描述了x对y的影响
但在神经网络中实际使用的函数并不简单，损失函数包含了所有的层、权重和偏置，这是一个在多个维度上运作的极为庞大的函数
在这个函数上进行数值微分计算导数需要对单个参数更新进行多次前向传递。
先执行前向传递作为参考，然后通过增量值更新单个参数，并再次通过模型执行前向传递以查看损失值的变化，
接下来计算导数并恢复为这次计算所做的参数更改。
必须对每个权重和偏置以及每个样本重复这一过程，这将非常耗时，这种方法实际上是强行计算单个参数输入的函数的导数（斜率）的方法
为了利用这种能力计算损失函数在每个权重和偏置点的斜率，需要将单变量推广到多变量，将导数推广到偏导数
'''