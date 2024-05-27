import argparse
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from DQN import DQNAgent, QNetwork, device  # 确保从 train.py 导入正确的模块和类

def evaluate(agent, env, n_episodes=100):
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps=0.0)  # 在评估时不使用epsilon贪婪策略
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            score += reward
            if done:
                break
        scores.append(score)
        print(f'\rEpisode {i_episode}\tScore: {score}', end="")
    average_score = np.mean(scores)
    return scores, average_score

def main(args):
    env = gym.make('Acrobot-v1')
    train_times = args.train_times  # 训练次数列表
    eval_episodes = args.episodes  # 每次评估的episode数

    for train_time in train_times:
        # 存放模型的地址
        checkpoint_path = f'Code/model/checkpoint_{train_time}.pth'
        agent = DQNAgent(state_size=6, action_size=3, seed=0)
        agent.load(checkpoint_path)  # 加载训练好的模型

        print(f'\n正在评估 {train_time} 次数')
        scores, average_score = evaluate(agent, env, n_episodes=eval_episodes)
        print(f'{train_time}次数的平均回报为: {average_score:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN evaluation script")
    parser.add_argument('-e', '--episodes', type=int, default=100, help='Number of episodes to evaluate the agent')
    parser.add_argument('-t', '--train_times', type=int, nargs='+', required=True, help='List of training times to evaluate the agent')
    args = parser.parse_args()
    
    main(args)
