import numpy as np
from neuron import Neuron
from neuron_network import NeuronNetwork

import matplotlib.pyplot as plt

# 测试XOR问题
def test_xor() -> None:
    # 准备XOR数据
    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [9, 9]

    ])
    y = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
    
    # 创建神经网络: 2个输入, 4个隐藏神经元, 1个输出
    nn = NeuronNetwork([2, 64, 16, 1], activation="sigmoid", output_activation="linear")
    
    # 训练网络
    epochs = 20000
    learning_rate = 0.0001
    decay_rate = 0.000000001 # 学习率衰减率
    losses = nn.train(X, y, epochs=epochs, learning_rate=learning_rate, decay_rate=decay_rate, patience=0, verbose=True)
    
    # 测试网络
    predictions = nn.predict(X)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i]}, Predicted: {predictions[i][0]:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('xor_training_loss.png')
    plt.show()

if __name__ == "__main__":
    test_xor()