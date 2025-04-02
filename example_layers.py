import numpy as np

class AttentionLayer:
    def __init__(self,inputs,output_dim,num_heads=1,
                 weight_regularizer_l1=0,weight_regularizer_l2=0):
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # 权重初始化
        self.weights_q = 0.01 * np.random.randn(inputs,output_dim)
        self.weights_k = 0.01 * np.random.randn(inputs,output_dim)
        self.weights_v = 0.01 * np.random.randn(inputs,output_dim)
        self.biases = np.zeros((1,output_dim))

        # 正则化参数
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2

    def forward(self,inputs):
        # 计算Q，K，V
        self.q = np.dot(inputs,self.weights_q)
        self.k = np.dot(inputs,self.weights_k)
        self.v = np.dot(inputs,self.weights_v)

        # 分头
        self.q = self.q.reshape(-1,self.num_heads,self.head_dim)
        self.k = self.k.reshape(-1,self.num_heads,self.head_dim)
        self.v = self.v.reshape(-1,self.num_heads,self.head_dim)

        # 计算注意力分数
        scores = np.dot(self.q,self.k.transpose(0,2,1)) / np.sqrt(self.head_dim)
        self.attention_weights = self.softmax(scores)

        # 加权和
        self.output = np.dot(self.attention_weights,self.v).reshape(-1,self.output_dim) + self.biases

    def softmax(self,x):
        exp_x = np.exp(x - np.max(x,axis=-1,keepdims=True))
        return exp_x / np.sum(exp_x,axis=-1,keepdims=True)

    def backward(self,dvalues):
        # 反向传播
        self.dvalues_v = np.dot(self.attention_weights.transpose(0,2,1),dvalues)

        # 计算梯度
        self.dweights_v = np.dot(self.q.transpose(0,2,1).reshape(-1,self.head_dim),dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)

        # L2正则化
        if self.weight_regularizer_l2 > 0:
            self.dweights_v += 2 * self.weight_regularizer_l2 * self.weights_v

        # 计算dvalues
        self.dinputs = np.dot(dvalues,self.weights_v.T)

    def get_parameters(self):
        # 获取权重和偏置的方法
        return self.weights_q, self.weights_k, self.weights_v, self.biases