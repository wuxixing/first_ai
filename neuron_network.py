from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from neuron import Neuron

class NeuronNetwork:
    def __init__(self, layer_sizes: List[int], activation: str = "sigmoid") -> None:
        """
        初始化神经网络
        
        参数:
            layer_sizes: 每层神经元数量的列表(包括输入层、隐藏层和输出层)
            activation: 激活函数类型
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.layers = []
        
        # 构建神经网络的各层
        for i in range(1, len(layer_sizes)):
            layer = []
            # 在当前层中创建指定数量的神经元
            for _ in range(layer_sizes[i]):
                # 每个神经元接收前一层所有神经元的输出作为输入
                neuron = Neuron(layer_sizes[i-1], activation)
                layer.append(neuron)
            self.layers.append(layer)
    
    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        前向传播过程
        
        参数:
            inputs: 网络的输入数据
            
        返回:
            网络的输出结果
        """
        current_inputs = inputs
        
        # 逐层计算
        for layer in self.layers:
            layer_outputs = []
            # 计算当前层每个神经元的输出
            for neuron in layer:
                output = neuron.forward(current_inputs)
                layer_outputs.append(output)
            # 当前层的输出作为下一层的输入
            current_inputs = np.array(layer_outputs)
            
        return current_inputs
    
    def backward(self, expected_output: NDArray[np.float64], 
                 learning_rate: float = 0.01) -> np.float64:
        """
        反向传播过程
        
        参数:
            expected_output: 期望的输出值
            learning_rate: 学习率
            
        返回:
            当前批次的损失值
        """
        # 获取网络的实际输出
        output_layer = self.layers[-1]
        actual_outputs = np.array([neuron.output for neuron in output_layer])
        
        # 计算输出层的误差(损失)
        errors = expected_output - actual_outputs
        loss:np.float64 = np.float64(np.mean(np.square(errors)))  # 均方误差

        # 计算MSE的梯度（均方误差(MSE)的梯度应该是 -2 * errors）
        gradient_errors = -2 * errors / len(errors) 

        # 从输出层开始反向传播
        layer_errors = gradient_errors
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            next_layer_errors = np.zeros(self.layer_sizes[i])
            
            # 更新当前层中每个神经元的权重
            for j, neuron in enumerate(layer):
                # 当前神经元的误差梯度
                error_gradient = layer_errors[j]
                # 反向传播并获取传递给前一层的误差
                prev_errors = neuron.backward(error_gradient, learning_rate)
                # 累加传递给前一层的误差
                next_layer_errors += prev_errors
                
            # 更新误差,准备处理前一层
            layer_errors = next_layer_errors
            
        return loss
    
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], 
              epochs: int = 1000, learning_rate: float = 0.01, 
              verbose: bool = True) -> List[float]:
        """
        训练神经网络
        
        参数:
            X: 训练数据
            y: 目标值
            epochs: 训练轮数
            learning_rate: 学习率
            verbose: 是否打印训练进度
            
        返回:
            每个epoch的损失列表
        """
        losses = []
        
        for epoch in range(epochs):
            total_loss:float = 0.0
            
            # 遍历每个训练样本
            for i in range(len(X)):
                # 前向传播
                self.forward(X[i])
                
                # 反向传播
                if y.ndim == 1:  # 如果y是一维数组
                    target = np.array([y[i]])
                else:
                    target = y[i]
                
                loss = self.backward(target, learning_rate)
                total_loss += loss
                
            # 计算平均损失
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            # 打印训练进度
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                
        return losses
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        使用训练好的网络进行预测
        
        参数:
            X: 输入数据
            
        返回:
            预测结果
        """
        predictions = []
        for x in X:
            output = self.forward(x)
            predictions.append(output)
        return np.array(predictions)