from DQN import QNetwork, ReplayBuffer, DQNAgent
from collections import deque, namedtuple
import numpy as np
import gym
import argparse

def main(args):
    if args.continue_training:
        scores = dqn(n_episodes=args.episodes, checkpoint_file=args.checkpoint)
    else:
        scores = dqn(n_episodes=args.episodes)
    plot_scores(scores)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, checkpoint_file=None):
    env = gym.make('Acrobot-v1')
    agent = DQNAgent(state_size=6, action_size=3, seed=0)
    
    if checkpoint_file:
        agent.load(checkpoint_file)
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        print(f'\rEpisode {i_episode}\tScore: {score}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        # 每200步保存一个模型
        if i_episode % 200 == 0:
            print(f'\rEpisode {i_episode}\tScore: {score}\tAverage Score: {np.mean(scores_window):.2f}')
            agent.save(f'Code/model/checkpoint_{i_episode}.pth')
    
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN training script")
    parser.add_argument('-e', '--episodes', type=int, default=2000, help='Number of episodes to train the agent')
    parser.add_argument('-c', '--continue_training', action='store_true', help='Continue training from a checkpoint')
    parser.add_argument('-cfile', '--checkpoint', type=str, help='Path to checkpoint file to continue training from')
    args = parser.parse_args()

    main(args)

