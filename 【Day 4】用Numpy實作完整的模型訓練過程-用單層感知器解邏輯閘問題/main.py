import numpy as np

class Perceptron:
    def __init__(self, input_shape, bias=0, learning_rate=0.1):
        # 初始化權重
        self.weights = np.random.randn(input_shape)
        
        # 初始化偏移量
        self.bias = bias
        
        # 初始化學習率
        self.learning_rate = learning_rate
    
    def forward(self, x):
        # 前向傳播公式 wx+b
        z = np.dot(x, self.weights) + self.bias
        # 階梯函數轉換結果
        y_hat = self.step_function(z)
        return y_hat
        
    def step_function(self, z):
        return (z >= 0).astype(int)
    
    def loss_function(self, y_hat, y):
        # MSE計算損失值
        return 0.5 * (y - y_hat) ** 2
    
    def backward(self, x, y, y_hat):
        # 計算梯度
        grad = (y - y_hat)
        
        # 優化器更新參數
        self.weights += self.learning_rate * grad * x
        self.bias += self.learning_rate * grad
    
    def predict(self, x):
        # 預測時直接調用訓練好的前向傳播函數
        return self.forward(x)
        

def training(model, x_train, y_train, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(x_train, y_train):
            y_hat = model.forward(x)
            loss = model.loss_function(y_hat, y)
            total_loss += loss
            model.backward(x, y, y_hat)
        print(f'Epoch {epoch}, Loss: {total_loss:.5f}')
    print('訓練完成!')
    
    

# Training data for the AND function
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train_AND = np.array([0, 0, 0, 1])

# 建立單層感知器模型
model = Perceptron(input_shape=x_train.shape[1], learning_rate=0.1)

# 暫存訓練前的權重
init_weights = model.weights
init_bias = model.bias

# 訓練模型
training(model, x_train, y_train_AND, epochs=20)
training_weights =  model.weights
training_bias = model.bias

# 模型訓練前
print("模型訓練前預測結果:")
for x in x_train:
    print(f'輸入: {x}, 預測輸出: {model.predict(x)}')
    
# 測試模型預測結果
print("\n模型訓練後預測結果:")
for x in x_train:
    print(f'輸入: {x}, 預測輸出: {model.predict(x)}')