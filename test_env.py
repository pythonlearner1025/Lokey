import gym
from kuhn_poker_env import KuhnPokerEnv
from copy import deepcopy
from itertools import combinations

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

def sample_action(player_turn, env):
    pass

def cfr(player, env):
    if env.is_done(player):
        return env.player_payoffs(player)
    elif env.current_player == player:
        actions = env.available_actions(player)
        for action in actions:
            newenv = deepcopy(env)
            newenv.step(action)
            v = cfr(player, newenv)
        
        # calculate regrets & advantage

        # add to dict
    else:
        sampled_action = sample_action(env.turn(), env)
        env.step(sampled_action)
        return cfr(player, env)

CFR_ITERATIONS = 1000
n_players = 3

# cfr update step
# card assignment

for i in range(CFR_ITERATIONS):
   # assign cards 
    for cards in combinations([0,1,2,3,4], 3):
        env = KuhnPokerEnv(n_players=3, cards)
        iter_dict = {}
        for player in range(n_players):
            cfr(player, env, iter_dict)
