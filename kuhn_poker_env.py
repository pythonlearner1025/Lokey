import gym
from gym import spaces
import numpy as np

class Player:
    def __init__(self, idx, hand):
        self.idx = idx
        self.hand = hand

class KuhnPokerEnv(gym.Env):
    """Multiplayer Kuhn Poker Environment"""

    def __init__(self, n_players=3):
        super(KuhnPokerEnv, self).__init__()

        self.n_players = n_players

        # Kuhn poker cards (10-A)
        self.deck = [0, 1, 2, 3, 4]
        self.folded = [False] * n_players
        self.current_player = 0

        # Define action space (0 = fold, 1 = check/call, 2 = bet)
        self.action_space = spaces.Discrete(3)

        # Define observation space (hand plus betting/folding status)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(n_players + 2,), dtype=np.int32)

        # Initialize the game state
        self.reset()
    
    def assign_hand(self, player, card):
        self.players[player].hand = card

    def reset(self):
        """Reset the environment to start a new game."""
        np.random.shuffle(self.deck)
        self.hands = self.deck[:self.n_players]
        # Track the current bet for each player
        self.bets = [0] * self.n_players
        self.folded = [False] * self.n_players  # Track if a player has folded
        self.pot = 0
        self.current_player = 0
        return self._get_observation()

    def step(self, action):
        """Apply an action and move the game state forward."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if action == 2:  # Player bets
            self.bets[self.current_player] += 1
            self.pot += 1
        elif action == 0:
            self.folded[self.current_player] = True

        # Update player turn
        self._next_player()

        done = self._is_done()  # Check if the game is done
        reward = self._get_reward() if done else 0

        return self._get_observation(), reward, done, {}

    def _next_player(self):
        """
        Move to the next player who hasn't folded.
        """
        while True:
            self.current_player = (self.current_player + 1) % self.n_players
            if not self.folded[self.current_player]:
                break

    def render(self, mode='human'):
        """Render the current state of the game."""
        print(f"Player hands: {self.hands}")
        print(f"Current player: {self.current_player}")
        print(f"Current bets: {self.bets}")
        print(f"Pot: {self.pot}")

    def _get_observation(self):
        """Return the observation for the current player."""
        # Example: player's hand and betting history and folding
        # return np.array([self.hands[self.current_player]] + self.bets)
        return np.array([self.hands[self.current_player]] + self.bets + [int(self.folded[self.current_player])])

    def _is_done(self):
        """Check if the game is over (e.g., all players have taken their turn)."""
        # return sum(self.bets) >= self.n_players  # Simplistic end condition
        # Game ends if only one player remains or if all players have bet or folded
        active_players = sum(not f for f in self.folded)
        return active_players == 1 or sum(self.bets) >= self.n_players

    def _get_reward(self):
        """Calculate the reward at the end of the game."""
        # Compare hands at the end and determine the winner
        active_players = [i for i in range(
            self.n_players) if not self.folded[i]]

        if len(active_players) == 1:
            return self.pot if self.current_player == active_players[0] else 0

        best_hand = max(self.hands[i] for i in active_players)
        winning_player = self.hands.index(best_hand)
        rewards = [-self.pot / self.n_players] * self.n_players
        rewards[winning_player] += self.pot  # Winner gets the pot
        return rewards[self.current_player]
