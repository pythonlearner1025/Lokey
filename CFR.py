from copy import deepcopy
import numpy as np
from kuhn_poker_env import KuhnPokerEnv
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl


def key(player, info_state):
    return f'{player}:{info_state}'

class KuhnPokerCFR:
    def __init__(self, n_players=3):
        self.n_players = n_players
        self.ACTION_SPACE = 3  # fold, call, raise = 3 actions
        # Regrets for each information set and action
        self.regret_sum = defaultdict(lambda: np.zeros(self.ACTION_SPACE))
        # Sum of strategies for averaging
        self.strategy_sum = defaultdict(lambda: np.zeros(self.ACTION_SPACE))
        self._root_info = None
    
    def get_strategy(self, player, info_state, realization_weight):
        """Compute strategy based on regrets, normalize to a probability distribution."""
        regrets = self.regret_sum[key(player, info_state)]
        assert regrets.shape == (self.ACTION_SPACE,)
        # Only positive regrets contribute
        positive_regrets = np.maximum(regrets, 0)
        total_regret = positive_regrets.sum()
        if total_regret > 0:
            strategy = positive_regrets / total_regret
        else:
            strategy = np.ones(self.ACTION_SPACE) / \
                self.ACTION_SPACE  # Equal probability
        self.strategy_sum[key(player, info_state)] = self.strategy_sum[key(player, info_state)] + \
            realization_weight * strategy
        return strategy

    def cfr(self, history, realization_weights, current_player, env):
        """CFR recursive function."""
        # For first CFR iteration, set root infoset
        if self._root_info is None:
            self._root_info = self._get_info_state(env)

        if env._is_done():
            return env._get_reward()  # Terminal payoff

        # A unique identifier for the information set
        info_state = self._get_info_state(env)
        strategy = self.get_strategy(
            current_player,
            info_state, 
            realization_weights[current_player]
        )

        # Initialize utility and regrets for each action
        action_utils = np.zeros(self.ACTION_SPACE)
        node_utility = 0.0

        # Loop over each action to calculate the counterfactual regret
        for action in range(self.ACTION_SPACE):
            # Copy environment to simulate action
            env_copy = deepcopy(env)
            _, reward, done, _ = env_copy.step(action)

            # Calculate the recursive CFR utility
            new_realization_weights = realization_weights.copy()
            new_realization_weights[current_player] *= strategy[action]
            #print(f'act: {action}')
            #print(f'rwd: {reward}')
            util = (
                reward if done else 
                self.cfr(history + [action], new_realization_weights,
                         (current_player + 1) % self.n_players, env_copy)
            )
            #print(util)
            action_utils[action] = util
            node_utility += strategy[action] * action_utils[action]

        #print('stuff;')        
        #print(f'player: {current_player}')
        #print(strategy.shape)
        #print(strategy[action])
        #print(action_utils[action])

        for action in range(self.ACTION_SPACE):
            # print("ENTIERING")
            regret = action_utils[action] - node_utility
            # self.regret_sum[info_state][action] += realization_weights[current_player] * regret
            # print(self.regret_sum.get(info_state, "NONEXISTENT"))
            # print(info_state)
            # self.regret_sum.get(info_state, np.zeros(self.ACTION_SPACE))[
            #     action] += realization_weights[current_player] * regret
            self.regret_sum[key(current_player, info_state)][action] = self.regret_sum[key(current_player, info_state)][action] + \
                realization_weights[current_player] * regret
            # print(realization_weights[current_player] * regret)
            # print(self.regret_sum.get(info_state,
            #   "REALLY SHOULD BE HERE?????--------------------"))


        return node_utility

    def _get_info_state(self, env):
        """Construct a unique information state identifier based on game history."""
        return f"{env.hands}-{env.bets}-{env.folded}-{env.current_player}"

    def train(self, iterations=500):
        for _ in tqdm(range(iterations)):
            for player in range(self.n_players):
                env = KuhnPokerEnv()
                env.reset(current_player=player)
                # Initial realization weights for all players
                realization_weights = [1] * self.n_players
                self.cfr([], realization_weights, player, env)

    def get_average_strategy(self, info_state, player):
        """Extract the average strategy for a given information state."""
        strategy_sum = self.strategy_sum[key(player, info_state)]
        normalizing_sum = strategy_sum.sum()
        if normalizing_sum > 0:
            return strategy_sum / normalizing_sum
        return np.ones(3) / 3

n_players = 3
cfr = KuhnPokerCFR(n_players=n_players)
cfr.train()

# print("------------------------------")
# print(cfr.regret_sum)
# print("------------------------------")

info_state = cfr._root_info
print("Root Information State (Default Strategy)")
print(info_state)
for p in range(n_players):
    print(cfr.get_average_strategy(info_state=info_state, player=p))
    print(cfr.get_average_strategy(info_state=info_state[:-1] + "1", player=p))
    print(cfr.get_average_strategy(info_state=info_state[:-1] + "2", player=p))
    print()
    print()
    print(cfr.get_average_strategy(
        info_state="[4, 3, 2]-[1, 1, 1]-[False, False, False]-0",
        player=p))
    print(cfr.get_average_strategy(
        info_state="[4, 3, 2]-[1, 1, 1]-[False, False, False]-1",
        player=p))
    print(cfr.get_average_strategy(
        info_state="[4, 3, 2]-[1, 1, 1]-[False, False, False]-2",
        player=p))
    print(cfr.get_average_strategy(
        info_state="[4, 3, 1]-[1, 1, 1]-[False, False, False]-2",
        player=p))
    print()
    print()
    print(cfr.get_average_strategy(
        info_state="[4, 3, 2]-[1, 2, 3]-[False, False, False]-0",
        player=p))
    print(cfr.get_average_strategy(
        info_state="[4, 3, 2]-[1, 2, 3]-[True, False, False]-1",
        player=p))
    print(cfr.get_average_strategy(
        info_state="[4, 3, 2]-[3, 2, 3]-[False, False, False]-1",
        player=p))

    with open("game_data/regrets.pkl", "wb") as file:
        pkl.dump(dict(cfr.regret_sum), file)

    with open("game_data/strategy.pkl", "wb") as file:
        pkl.dump(dict(cfr.strategy_sum), file)

    # TODO: Note
    # can now collapse these values into each player's "ideal" policy based on any point in the game:
    # Current card, current bet, other player bets, fold status
    # TODO: Need a test bed to play singular games and see rewards etc
