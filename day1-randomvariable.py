#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dice_simulation():
    # https://numpy.org/doc/2.1/reference/random/generated/numpy.random.randint.html
    dice = np.random.randint(1, 7, 20)
    print("掷骰子20次:", dice)
    
    # 统计每个点数的频率
    unique_values, counts = np.unique(dice, return_counts=True)
    
    # 创建完整的频率表（1-6）
    frequency = np.zeros(6, dtype=int)
    for value, count in zip(unique_values, counts):
        frequency[value - 1] = count
    
    print("\n点数频率统计:")
    for i in range(6):
        print(f"点数 {i + 1}: {frequency[i]} 次")

def normal_distribution():
    x = np.random.normal(loc=0, scale=1, size=1000)  # 标准正态分布
    plt.hist(x, bins=30, density=True)
    plt.title("正态分布示例")
    plt.show()

def conditional_probability():
    # 咳嗽和流感的条件概率
    data = pd.DataFrame({"咳嗽":[1,0,1,1,0,0],"流感":[1,0,1,0,0,0]})
    p_B = data["咳嗽"].mean()
    p_A_and_B = ((data["咳嗽"]==1)&(data["流感"]==1)).mean()
    print("P(流感|咳嗽) =", p_A_and_B/p_B)
    
    print("\n数据表:")
    print(data)
    print(f"\nP(咳嗽) = {p_B:.3f}")
    print(f"P(咳嗽且流感) = {p_A_and_B:.3f}")
    print(f"P(流感|咳嗽) = {p_A_and_B/p_B:.3f}")

def homework_exercise():
    # 练习：100个学生，40个做作业，20个做作业且考试合格
    total_students = 100
    homework_students = 40  # 做作业的学生
    homework_and_pass = 20  # 做作业且考试合格的学生

    # P(合格|做作业) = P(做作业且合格) / P(做作业)
    p_homework = homework_students / total_students
    p_homework_and_pass = homework_and_pass / total_students
    p_pass_given_homework = p_homework_and_pass / p_homework

    print(f"总学生数: {total_students}")
    print(f"做作业的学生: {homework_students}")
    print(f"做作业且考试合格的学生: {homework_and_pass}")
    print(f"\nP(做作业) = {homework_students}/{total_students} = {p_homework:.2f}")
    print(f"P(做作业且合格) = {homework_and_pass}/{total_students} = {p_homework_and_pass:.2f}")
    print(f"P(合格|做作业) = {p_homework_and_pass:.2f}/{p_homework:.2f} = {p_pass_given_homework:.2f}")

def bayes_disease_test():
    # 贝叶斯定理：疾病检测
    p_disease = 0.01  # 患病率 1%
    p_pos_given_disease = 0.9  # 患病时检测为阳性的概率
    p_pos_given_no = 0.1  # 未患病时检测为阳性的概率（假阳性率）

    # 全概率公式计算检测为阳性的总概率
    p_pos = p_pos_given_disease * p_disease + p_pos_given_no * (1 - p_disease)

    # 贝叶斯定理计算检测为阳性时患病的概率
    p_disease_given_pos = p_pos_given_disease * p_disease / p_pos

    print(f"患病率: {p_disease:.1%}")
    print(f"测试准确率（敏感性）: {p_pos_given_disease:.1%}")
    print(f"假阳性率: {p_pos_given_no:.1%}")
    print(f"\n检测为阳性的总概率: {p_pos:.4f}")
    print(f"测阳性得病概率: {p_disease_given_pos:.4f} ({p_disease_given_pos:.2%})")

def main():
    print("=== 掷骰子模拟 ===")
    dice_simulation()

    print("\n=== 正态分布可视化 ===")
    normal_distribution()

    print("\n=== 条件概率：咳嗽与流感 ===")
    conditional_probability()

    print("\n=== 练习：作业与考试合格 ===")
    homework_exercise()

    print("\n=== 贝叶斯定理：疾病检测 ===")
    bayes_disease_test()

if __name__ == "__main__":
    main()