#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def artificial_neuron():
    """人工神经元：线性组合 + 激活函数"""
    print("=== Artificial Neuron ===")

    x = np.array([1.0, 0.5, -0.3])
    w = np.array([0.4, -0.2, 0.8])
    b = 0.5

    z = np.dot(x, w) + b
    output_linear = z
    output_sigmoid = 1 / (1 + np.exp(-z))
    output_relu = max(0, z)

    print(f"输入 x: {x}")
    print(f"权重 w: {w}")
    print(f"偏置 b: {b}")
    print(f"线性输出: {output_linear:.4f}")
    print(f"Sigmoid输出: {output_sigmoid:.4f}")
    print(f"ReLU输出: {output_relu:.4f}")

def activation_functions():
    """激活函数对比"""
    print("\n=== Activation Functions ===")

    x = np.linspace(-4, 4, 100)

    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e^-x)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    axes[0, 1].plot(x, tanh, 'r-', linewidth=2)
    axes[0, 1].set_title('Tanh: tanh(x)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    axes[1, 0].plot(x, relu, 'g-', linewidth=2)
    axes[1, 0].set_title('ReLU: max(0, x)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    axes[1, 1].plot(x, leaky_relu, 'm-', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU: max(0.01x, x)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

    plt.tight_layout()
    plt.show()

    test_val = 2.0
    print(f"\n当x={test_val}时:")
    print(f"  Sigmoid: {1/(1+np.exp(-test_val)):.4f}")
    print(f"  Tanh: {np.tanh(test_val):.4f}")
    print(f"  ReLU: {max(0, test_val):.4f}")
    print(f"  Leaky ReLU: {max(0.01*test_val, test_val):.4f}")

def forward_propagation():
    """前向传播：输入 → 隐藏层 → 输出"""
    print("\n=== Forward Propagation ===")

    np.random.seed(42)

    X = np.array([[0.5, 0.8]])

    W1 = np.array([[0.1, 0.3, -0.5],
                   [0.2, -0.4, 0.6]])
    b1 = np.array([[0.1, 0.2, -0.1]])

    W2 = np.array([[0.4],
                   [-0.3],
                   [0.5]])
    b2 = np.array([[0.2]])

    print("网络结构: 2 → 3 → 1")
    print(f"输入: {X[0]}")

    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    print(f"\n隐藏层:")
    print(f"  z1 = XW1 + b1 = {z1[0]}")
    print(f"  a1 = tanh(z1) = {a1[0]}")

    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))
    print(f"\n输出层:")
    print(f"  z2 = a1W2 + b2 = {z2[0, 0]:.4f}")
    print(f"  输出 = sigmoid(z2) = {a2[0, 0]:.4f}")

def backward_propagation():
    """反向传播：计算梯度"""
    print("\n=== Backward Propagation ===")

    np.random.seed(42)

    X = np.array([[0.5, 0.8]])
    y = np.array([[1.0]])

    W1 = np.random.randn(2, 3) * 0.5
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.5
    b2 = np.zeros((1, 1))

    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))

    loss = -y * np.log(a2) - (1-y) * np.log(1-a2)
    print(f"预测值: {a2[0,0]:.4f}")
    print(f"真实值: {y[0,0]:.4f}")
    print(f"损失: {loss[0,0]:.4f}")

    dL_da2 = (a2 - y) / (a2 * (1 - a2))
    da2_dz2 = a2 * (1 - a2)
    dz2 = dL_da2 * da2_dz2

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * (1 - a1**2)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    print(f"\n梯度:")
    print(f"  dW2 shape: {dW2.shape}")
    print(f"  dW1 shape: {dW1.shape}")
    print(f"  最大梯度值: {max(np.max(np.abs(dW1)), np.max(np.abs(dW2))):.4f}")

def optimizers_demo():
    """优化器对比：SGD、Momentum、Adam"""
    print("\n=== 优化器 ===")

    def f(x):
        return x**2

    def df(x):
        return 2*x

    x0 = 10.0
    steps = 30

    x_sgd = x0
    history_sgd = [x_sgd]
    lr = 0.1
    for _ in range(steps):
        x_sgd = x_sgd - lr * df(x_sgd)
        history_sgd.append(x_sgd)

    x_momentum = x0
    v = 0
    history_momentum = [x_momentum]
    lr = 0.1
    beta = 0.9
    for _ in range(steps):
        v = beta * v - lr * df(x_momentum)
        x_momentum = x_momentum + v
        history_momentum.append(x_momentum)

    x_adam = x0
    m, v = 0, 0
    history_adam = [x_adam]
    lr = 1.0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    for t in range(1, steps + 1):
        grad = df(x_adam)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x_adam = x_adam - lr * m_hat / (np.sqrt(v_hat) + eps)
        history_adam.append(x_adam)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_sgd, 'b-o', label='SGD', markersize=3)
    plt.plot(history_momentum, 'r-s', label='Momentum', markersize=3)
    plt.plot(history_adam, 'g-^', label='Adam', markersize=3)
    plt.xlabel('Steps')
    plt.ylabel('x')
    plt.title('Optimization Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('symlog')

    plt.subplot(1, 2, 2)
    plt.plot([f(x) for x in history_sgd], 'b-', label='SGD')
    plt.plot([f(x) for x in history_momentum], 'r-', label='Momentum')
    plt.plot([f(x) for x in history_adam], 'g-', label='Adam')
    plt.xlabel('Steps')
    plt.ylabel('f(x) = x²')
    plt.title('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    print(f"Final Value (Target=0):")
    print(f"  SGD: {history_sgd[-1]:.6f}")
    print(f"  Momentum: {history_momentum[-1]:.6f}")
    print(f"  Adam: {history_adam[-1]:.6f}")

def xor_network():
    """XOR问题：非线性分类"""
    print("\n=== XOR Neural Network ===")

    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("XOR真值表:")
    for i in range(len(X)):
        print(f"  {X[i]} → {y[i,0]}")

    W1 = np.random.randn(2, 4) * 2
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 2
    b2 = np.zeros((1, 1))

    lr = 0.5
    epochs = 500
    losses = []

    for epoch in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = 1 / (1 + np.exp(-z2))

        loss = np.mean((a2 - y)**2)
        losses.append(loss)

        dz2 = 2 * (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    predictions = 1 / (1 + np.exp(-z2))

    print(f"\n训练后预测:")
    for i in range(len(X)):
        print(f"  {X[i]} → {predictions[i,0]:.3f} (目标: {y[i,0]})")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                         np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    z1_grid = grid @ W1 + b1
    a1_grid = np.tanh(z1_grid)
    z2_grid = a1_grid @ W2 + b2
    pred_grid = 1 / (1 + np.exp(-z2_grid))
    pred_grid = pred_grid.reshape(xx.shape)

    plt.contourf(xx, yy, pred_grid, levels=20, cmap='RdBu_r', alpha=0.7)
    plt.colorbar(label='Predicted Value')

    colors = ['red' if y[i,0] == 0 else 'blue' for i in range(len(y))]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)

    for i in range(len(X)):
        plt.annotate(f'{y[i,0]}', (X[i,0], X[i,1]),
                    ha='center', va='center', fontsize=12, color='white')

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('XOR决策边界')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def learning_rate_schedules():
    """学习率调度策略"""
    print("\n=== 学习率调度 ===")

    epochs = 100

    constant = [0.1] * epochs

    exponential = [0.1 * 0.95**i for i in range(epochs)]

    step_decay = []
    for i in range(epochs):
        if i < 30:
            step_decay.append(0.1)
        elif i < 60:
            step_decay.append(0.01)
        else:
            step_decay.append(0.001)

    cosine = [0.05 * (1 + np.cos(np.pi * i / epochs)) for i in range(epochs)]

    warmup = []
    warmup_epochs = 10
    for i in range(epochs):
        if i < warmup_epochs:
            warmup.append(0.1 * (i + 1) / warmup_epochs)
        else:
            warmup.append(0.1 * 0.95**(i - warmup_epochs))

    plt.figure(figsize=(12, 8))

    strategies = [
        (constant, 'Constant', 'b-'),
        (exponential, 'Exponential Decay', 'r-'),
        (step_decay, 'Step Decay', 'g-'),
        (cosine, 'Cosine Decay', 'm-'),
        (warmup, 'Warmup+Decay', 'c-')
    ]

    for i, (schedule, name, style) in enumerate(strategies, 1):
        plt.subplot(2, 3, i)
        plt.plot(schedule, style, linewidth=2)
        plt.title(name)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        if name in ['Exponential Decay', 'Step Decay', 'Warmup+Decay']:
            plt.yscale('log')

    plt.subplot(2, 3, 6)
    for schedule, name, style in strategies:
        plt.plot(schedule, style, label=name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('All Strategies Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    print("Learning Rate Scheduling Strategies:")
    print("  1. 常数: 保持不变")
    print("  2. Exponential Decay: lr = lr₀ × decay^epoch")
    print("  3. Step Decay: 在特定epoch降低")
    print("  4. Cosine Decay: 周期性变化")
    print("  5. 预热: 先增后减")

def main():
    artificial_neuron()
    activation_functions()
    forward_propagation()
    backward_propagation()
    optimizers_demo()
    xor_network()
    learning_rate_schedules()

if __name__ == "__main__":
    main()