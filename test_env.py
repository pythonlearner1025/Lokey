import gym
from kuhn_poker_env import KuhnPokerEnv
from copy import deepcopy
from itertools import combinations

env = KuhnPokerEnv(n_players=3)

obs = env.reset()
done = False

manual_player = input("Yes/No: Use Manual Actions \n")

# actions = [1, 2, 0, 2, 1]
# a_index = 0

while not done:
    print("ENV Hands: ", env.hands)
    if manual_player.lower().strip() == "yes":
        action = int(input("Action: "))
    else:
        action = env.action_space.sample()  # Random agent for testing
    # action = actions[a_index]
    # a_index += 1
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        print(f"Game over! Reward: {reward}")
