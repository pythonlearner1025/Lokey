import gym
from kuhn_poker_env import KuhnPokerEnv

env = KuhnPokerEnv(n_players=3)

obs = env.reset()
done = False

# actions = [1, 2, 0, 2, 1]
# a_index = 0

while not done:
    action = env.action_space.sample()  # Random agent for testing
    # action = actions[a_index]
    # a_index += 1
    print("action:", action)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        print(f"Game over! Reward: {reward}")
