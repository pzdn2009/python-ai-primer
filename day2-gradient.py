#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def basic_gradient_descent():
    """基础梯度下降示例：求解 f(x)=x^2 的最小值"""
    print("=== 基础梯度下降 ===")
    x = 10      # 初始位置
    lr = 0.1    # 学习率
    history = []

    for i in range(20):
        grad = 2*x  # f'(x) = 2x
        x = x - lr*grad
        history.append(x)
        print(f"迭代{i}: x={x:.4f}, f(x)={x**2:.4f}")

    return history

def learning_rate_comparison():
    """比较不同学习率的效果"""
    print("\n=== 学习率对比 ===")
    learning_rates = [0.01, 0.1, 0.5, 0.9]

    plt.figure(figsize=(12, 8))

    for idx, lr in enumerate(learning_rates):
        x = 10
        history = [x]

        for i in range(30):
            grad = 2*x
            x = x - lr*grad
            history.append(x)

        plt.subplot(2, 2, idx+1)
        plt.plot(history, 'o-')
        plt.title(f'学习率 lr={lr}')
        plt.xlabel('迭代次数')
        plt.ylabel('x值')
        plt.grid(True)

        print(f"lr={lr}: 最终x={x:.6f}, f(x)={x**2:.6f}")

    plt.tight_layout()
    plt.show()

def gradient_descent_2d():
    """二维梯度下降：求解 f(x,y) = x^2 + y^2 的最小值"""
    print("\n=== 二维梯度下降 ===")
    x, y = 5, 3  # 初始位置
    lr = 0.1

    path_x, path_y = [x], [y]

    for i in range(20):
        grad_x = 2*x  # ∂f/∂x = 2x
        grad_y = 2*y  # ∂f/∂y = 2y

        x = x - lr*grad_x
        y = y - lr*grad_y

        path_x.append(x)
        path_y.append(y)

        if i < 5 or i >= 15:  # 只打印前5次和后5次
            print(f"迭代{i}: x={x:.4f}, y={y:.4f}, f(x,y)={x**2+y**2:.4f}")

    # 可视化
    fig = plt.figure(figsize=(10, 5))

    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(-6, 6, 50)
    Y = np.linspace(-6, 6, 50)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = X_grid**2 + Y_grid**2

    ax1.plot_surface(X_grid, Y_grid, Z, alpha=0.3, cmap='viridis')
    ax1.plot(path_x, path_y, [x**2 + y**2 for x, y in zip(path_x, path_y)],
             'ro-', markersize=5, linewidth=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('梯度下降路径（3D）')

    # 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X_grid, Y_grid, Z, levels=20, alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path_x, path_y, 'ro-', markersize=5, linewidth=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('梯度下降路径（等高线）')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def adaptive_learning_rate():
    """自适应学习率示例"""
    print("\n=== 自适应学习率 ===")
    x = 10
    lr = 0.5  # 初始学习率
    decay = 0.95  # 衰减率

    for i in range(30):
        grad = 2*x
        x = x - lr*grad
        lr = lr * decay  # 学习率衰减

        if i < 5 or i >= 25:
            print(f"迭代{i}: x={x:.6f}, f(x)={x**2:.6f}, lr={lr:.4f}")

def momentum_gradient_descent():
    """带动量的梯度下降"""
    print("\n=== 带动量的梯度下降 ===")
    x = 10
    lr = 0.1
    momentum = 0.9
    velocity = 0

    for i in range(20):
        grad = 2*x
        velocity = momentum * velocity - lr * grad  # 动量更新
        x = x + velocity

        if i < 5 or i >= 15:
            print(f"迭代{i}: x={x:.4f}, f(x)={x**2:.4f}, velocity={velocity:.4f}")

def rosenbrock_function():
    """Rosenbrock函数优化（非凸函数）"""
    print("\n=== Rosenbrock函数优化 ===")
    print("f(x,y) = (1-x)^2 + 100(y-x^2)^2")

    x, y = -1.5, 1.5  # 初始位置
    lr = 0.001

    path_x, path_y = [x], [y]

    for i in range(5000):
        # Rosenbrock函数的梯度
        grad_x = -2*(1-x) - 400*x*(y-x**2)
        grad_y = 200*(y-x**2)

        x = x - lr*grad_x
        y = y - lr*grad_y

        if i % 1000 == 0:
            f_val = (1-x)**2 + 100*(y-x**2)**2
            print(f"迭代{i}: x={x:.4f}, y={y:.4f}, f(x,y)={f_val:.4f}")

        if i < 100 or i % 50 == 0:  # 记录路径用于可视化
            path_x.append(x)
            path_y.append(y)

    # 最终结果
    f_val = (1-x)**2 + 100*(y-x**2)**2
    print(f"最终: x={x:.4f}, y={y:.4f}, f(x,y)={f_val:.6f}")
    print(f"理论最优解: x=1, y=1, f(x,y)=0")

    # 可视化
    X = np.linspace(-2, 2, 100)
    Y = np.linspace(-1, 3, 100)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = (1-X_grid)**2 + 100*(Y_grid-X_grid**2)**2

    plt.figure(figsize=(10, 8))
    plt.contour(X_grid, Y_grid, np.log(Z+1), levels=30, alpha=0.6)
    plt.plot(path_x, path_y, 'ro-', markersize=3, linewidth=1)
    plt.plot(1, 1, 'g*', markersize=15, label='最优解(1,1)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rosenbrock函数梯度下降路径')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 基础梯度下降
    basic_gradient_descent()

    # 学习率对比
    learning_rate_comparison()

    # 二维梯度下降
    gradient_descent_2d()

    # 自适应学习率
    adaptive_learning_rate()

    # 带动量的梯度下降
    momentum_gradient_descent()

    # 复杂函数优化
    rosenbrock_function()

if __name__ == "__main__":
    main()