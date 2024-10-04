import gym
from kuhn_poker_env import KuhnPokerEnv

env = KuhnPokerEnv(n_players=3)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random agent for testing
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        print(f"Game over! Reward: {reward}")
