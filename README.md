# 作业6 深入理解DQN算法

### 作业目标
理解DQN算法原理

选取gym环境中除Mountain Car、CartPole、FrozenLake之外的某个环境，分析它的状态空间、动作空间的维数、参数及取值范围。参考DQN实现代码训练一个智能体，要求训练能持续进行，即：可以在上次训练基础上对模型继续进行训练而不是一次性完成训练。测试智能体的性能，计算平均回报。

## 状态空间、动作空间和奖励值分析

本次实验选择的是OpenAI Gym中的“Acrobot-v1”环境。Acrobot是一个双连杆摆的控制问题，其目标是通过施加力矩使得摆的末端达到一定的高度。智能体需要在每个时间步选择合适的动作，以尽快实现这个目标。

### 状态空间
Acrobot的状态空间由六个连续变量组成，这些变量共同描述了双连杆摆的当前状态：

1. 连杆1的角度（theta1）：表示第一个连杆相对于竖直方向的角度，取值范围为 [-π, π]。
2. 连杆2的角度（theta2）：表示第二个连杆相对于第一个连杆的角度，取值范围为 [-π, π]。
3. 连杆1的角速度（theta1_dot）：取值范围为 [-4π, 4π]。
4. 连杆2的角速度（theta2_dot）：取值范围为 [-9π, 9π]。
5. 连杆1的角余弦值（cos(theta1)）：取值范围为 [-1, 1]。
6. 连杆2的角余弦值（cos(theta2)）：取值范围为 [-1, 1]。

### 动作空间
Acrobot的动作空间是离散的，包括以下三个动作：

1. 向左施加力矩（Action 0）：在摆上施加一个固定大小的力矩，使其向左旋转。
2. 不施加力矩（Action 1）：不在摆上施加力矩，摆仅受重力影响。
3. 向右施加力矩（Action 2）：在摆上施加一个固定大小的力矩，使其向右旋转。

这些动作允许智能体通过控制力矩的方向和大小来影响双连杆摆的运动，从而实现目标。

### 奖励值
在Acrobot环境中，奖励值的设置比较简单：

每个时间步的奖励值为-1。无论智能体采取什么动作，每个时间步都会获得一个固定的负奖励。这种设计鼓励智能体尽快达到目标状态，以最小化时间步数。

终止条件：当摆的末端超过给定的高度时，episode结束。如果智能体在500个时间步内未能达到目标高度，episode同样会结束。

## 代码使用

### 训练

参数设置

> -e 训练的episode
>
> -c 是否继续训练
>
> -cfile 继续训练的模型地址

示例：

从头开始训练 10000 episodes
> python train.py -e 10000

目前每200episodes保存一个模型，存放在model/xxx.pth

### 测试

与训练类似，-t为选取特定的episodes进行评估
示例：
> python test.py -t 200 600 1000