import numpy as np

class MLP:
    def __init__(self, input_shape, hidden_shape=2, output_shape=1, learning_rate=1):
        # 初始化權重和偏移量
        self.W1 = np.random.randn(input_shape, hidden_shape)
        self.b1 = np.zeros((1, hidden_shape))
        self.W2 = np.random.randn(hidden_shape, output_shape)
        self.b2 = np.zeros((1, output_shape))
        
        # 初始化學習率
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, y):
        return y * (1 - y)
    
    def forward(self, x):
        # 前向傳播：計算每層的輸出
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def loss_function(self, y_hat, y):
        # 計算均方誤差 (MSE) 損失
        return np.mean(0.5 * (y - y_hat) ** 2)
    
    def backward(self, x, y):
        # 計算梯度並更新權重和偏移量
        m = x.shape[0]
        delta2 = (self.a2 - y) * self.sigmoid_derivative(self.a2)
        dW2 = (1 / m) * np.dot(self.a1.T, delta2)
        db2 = (1 / m) * np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1 / m) * np.dot(x.T, delta1)
        db1 = (1 / m) * np.sum(delta1, axis=0, keepdims=True)
        
        # 更新權重與偏移量
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def predict(self, x):
        # 預測時直接進行前向傳播
        y = self.forward(x) > 0.5
        return y.astype(int)
    
def training(model, x_train, y_train, epochs=100):
    for epoch in range(epochs):
        y_hat = model.forward(x_train)
        loss = model.loss_function(y_hat, y_train)
        model.backward(x_train, y_train)
        if epoch % 1000 == 0:    
            print(f'Epoch {epoch}, Loss: {loss:.5f}')
    print('訓練完成!')

# XOR 取代 AND 作為訓練數據
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train_XOR = np.array([[0], [1], [1], [0]])

# 建立 MLP 模型
model = MLP(input_shape=x_train.shape[1])

# 訓練模型
training(model, x_train, y_train_XOR, epochs=10000)

# 模型訓練後預測結果
print("\n模型訓練後預測結果:")
pred = model.predict(x_train)
for x, y in zip(x_train, pred):
    print(f'輸入: {x}, 預測輸出: {y}')
