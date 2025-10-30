import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import load_mnist
from collections import OrderedDict

# 출력층의 활성화 함수로 softmax를 사용
def softmax(x):
    # max 값을 빼줌으로써 overflow 방지
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    # Broadcasting 위해 차원을 유지해서 sum
    y = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return y

# Loss Function으로 CEE를 사용
def cross_entropy_error(y, t):
    # 1차원인 경우 Batch 처리를 위해 사이즈를 (1, size)로 변경
    if (y.ndim == 1):
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # y와 t가 size가 동일하면 one-hot vector이므로 추출을 위해서 label의 인덱스로 변경
    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# Affine 계층
# Y = XW + b 연산
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
 
    def backward(self, dout):
        # dx = dL/dX = dL/dOut * W.T
        dx = np.dot(dout, self.W.T)
        # dW = dL/dW = x.T * dL/dOut
        self.dW = np.dot(self.x.T, dout)
        # db = dL/db = sum(dL/dOut)
        self.db = np.sum(dout, axis=0)
        return dx

# ReLU 계층
# 활성화 함수로 ReLU 함수를 사용
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        # 여러 계층이 재사용함으로 원본 데이터 유지하도록 copy
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 이전에 forward에서 미리 계산해놓은 mask 적용
        dout[self.mask] = 0
        dx = dout
        return dx

# 출력층 계층으로 Softmax와 CEE를 합쳐서 연산
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None # 이 때 정답 레이블은 One-Hot vector
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    # 일반적으로 최종 loss부터 backdrop 계산을 시작함으로 dout의 기본 값을 1로 설정
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # CEE와 softmax 계층을 거치고 나온 grad: (y - T) / N
        dx = (self.y - self.t) / batch_size
        return dx


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.last_layer = SoftmaxWithLoss()

    # 순전파 결과를 출력합니다.
    # Predict는 출력층에 softmax를 사용하는 것이 아닌 항등 함수를 적용합니다.
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # predict에서 나온 값을 통해 softmax를 적용하고 cee로 loss를 구해줍니다.
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # x에 대해서 pred을 구하고 실제 레이블과 비교하여 정확도를 비교합니다.
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 순전파로 loss를 구한 후 역전파를 통해 각 layer들의 W, b에 대한 grad를 구해줍니다.
    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backdrop
        dout = 1
        dout = self.last_layer.backward(dout)

        # 계층이 반대에서 시작함으로 reverse를 해주고 계층을 통과시킵니다.
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # Hyper Parameters
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.3

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # Batch 단위로 처리할 무작위 훈련 데이터를 선정합니다.
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 각 Parameter에 대한 Gradient를 구해줍니다.
        grads = network.gradient(x_batch, t_batch)

        # 구한 Gradient를 바탕으로 학습을 진행합니다.
        for key in {'W1', 'b1', 'W2', 'b2'}:
            network.params[key] -= learning_rate * grads[key]

        # loss를 구한 후 loss history에 추가합니다.
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1 epoch마다 정확도를 기록합니다.
        if (i % iter_per_epoch == 0):
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f} | loss: {train_loss_list[i]:.4f}")