from typing import Optional
import numpy as np
from numpy.typing import NDArray

class Neuron:
    weights: NDArray[np.float64]
    bias: float
    activation: str
    inputs: Optional[np.ndarray]
    output: NDArray[np.float64]
    
    def __init__(self, input_size: int, activation: str = "sigmoid") -> None:
        """
        初始化神经元 
        
        参数:
            input_size: 输入特征的数量
            activation: 激活函数类型 ('sigmoid', 'tanh', 'relu')
        """
        # 随机初始化权重和偏置
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = np.random.randn() * 0.01
        self.activation = activation
        
        # 存储前向传播的中间值，用于反向传播
        self.inputs = None
        self.output = None
        
    def _sigmoid(self, x: np.ndarray) -> NDArray[np.float64]:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> NDArray[np.float64]:
        """Sigmoid函数的导数"""
        return x * (1 - x)
    
    def _tanh(self, x: np.ndarray) -> NDArray[np.float64]:
        """Tanh激活函数"""
        return np.tanh(x)
    
    def _tanh_derivative(self, x: np.ndarray) -> NDArray[np.float64]:
        """Tanh函数的导数"""
        return 1 - np.power(x, 2)
    
    def _relu(self, x: np.ndarray) -> NDArray[np.float64]:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> NDArray[np.float64]:
        """ReLU函数的导数"""
        return np.where(x > 0, 1, 0)
    
    def _linear(self, x: np.ndarray) -> NDArray[np.float64]:
        return x
    
    def _linear_derivative(self, x: np.ndarray) -> NDArray[np.float64]:
        return np.ones_like(x)
    
    def activate(self, x: np.ndarray) -> NDArray[np.float64]:
        """应用激活函数"""
        if self.activation == "sigmoid":
            return self._sigmoid(x)
        elif self.activation == "tanh":
            return self._tanh(x)
        elif self.activation == "relu":
            return self._relu(x)
        elif self.activation == "linear":
            return self._linear(x)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def activate_derivative(self, x: np.ndarray) -> NDArray[np.float64]:
        """应用激活函数的导数"""
        if self.activation == "sigmoid":
            return self._sigmoid_derivative(x)
        elif self.activation == "tanh":
            return self._tanh_derivative(x)
        elif self.activation == "relu":
            return self._relu_derivative(x)
        elif self.activation == "linear":
            return self._linear_derivative(x)
        else:
            raise ValueError(f"不支持的激活函数导数: {self.activation}")
    
    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        前向传播
        
        参数:
            inputs: 输入特征向量
            
        返回:
            神经元的输出
        """
        self.inputs = np.array(inputs)
        # 计算加权和
        weighted_sum: np.ndarray = np.dot(self.weights, self.inputs) + self.bias
        # 应用激活函数
        self.output = self.activate(weighted_sum)
        return self.output
    
    def backward(self, error_gradient: float, learning_rate: float = 0.01) -> NDArray[np.float64]:
        """
        反向传播
        
        参数:
            error_gradient: 输出的误差梯度
            learning_rate: 学习率
            
        返回:
            传递给前一层的误差梯度
        """
        # 计算输出对加权和的梯度
        delta: NDArray[np.float64] = error_gradient * self.activate_derivative(self.output)
        
        # 计算权重梯度并更新权重
        weight_gradients: NDArray[np.float64] = delta * self.inputs
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * delta
        
        # 计算并返回传递给前一层的误差梯度
        return delta * self.weights