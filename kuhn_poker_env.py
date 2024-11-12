import gym
from gym import spaces
import numpy as np


class Player:
    def __init__(self, idx, hand):
        self.idx = idx
        self.hand = hand


class KuhnPokerEnv(gym.Env):
    """Multiplayer Kuhn Poker Environment"""
    # TODO:

    def __init__(self, cards=None, n_players=3):
        super(KuhnPokerEnv, self).__init__()

        self.n_players = n_players

        # Kuhn poker cards (10-A)
        if not cards:
            self._cfr = False
            self.deck = [0, 1, 2, 3, 4]
        else:
            self._cfr = True
            self.deck = cards
        self.folded = [False] * n_players
        self.raised = [False] * n_players
        self.current_player = 0
        # true for players who still need to act
        self.players_to_act = [True] * n_players

        # Define action space (0 = fold, 1 = check/call, 2 = raise)
        self.action_space = spaces.Discrete(3)

        # Define observation space (hand plus betting/folding status)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(n_players + 2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def assign_hand(self, player, card):
        self.players[player].hand = card

    def assign_cards(self, cards):  # [0,1,2], [2,1,3], etc.
        self.hands = cards

    def assign_player_card(self, player_idx, card):
        self.hands[player_idx] = card

    def reset(self, current_player=0):
        """Reset the environment to start a new game."""
        if not self._cfr:
            np.random.shuffle(self.deck)
        self.hands = self.deck[:self.n_players]
        # Track the current bet for each player
        # Each bet starts at 1 as a requirement for game ante
        self.bets = [1] * self.n_players
        self.max_bet = 1
        self.raised = [False] * self.n_players  # Track if a player has raised
        self.folded = [False] * self.n_players  # Track if a player has folded
        self.players_to_act = [True] * self.n_players
        self.pot = sum(self.bets)
        self.current_player = current_player
        return self._get_observation()

    def step(self, action):
        """Apply an action and move the game state forward."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if action == 0:  # Fold
            self.folded[self.current_player] = True
        elif action == 1:  # Call
            self.bets[self.current_player] = self.max_bet
        elif action == 2:  # Raising
            if self.raised[self.current_player]:
                # if we raised already, "strong" action just represents a call
                # TODO: unknown if this is how we want to represent the game,
                #       or if we want the players to check themselves
                self.bets[self.current_player] = self.max_bet
            else:
                self.max_bet += 1
                self.raised[self.current_player] = True
                self.bets[self.current_player] = self.max_bet
                self.players_to_act = [not p for p in self.folded]
                self.players_to_act[self.current_player] = False
                self.pot = sum(self.bets)

        # after current action, player no longer gets to play
        self.players_to_act[self.current_player] = False

        # Update player turn
        self._next_player()

        done = self._is_done()  # Check if the game is done
        reward = self._get_reward() if done else 0

        return self._get_observation(), reward, done, {}

    def _next_player(self):
        """
        Move to the next player who hasn't folded.
        """
        if not any(self.players_to_act):
            return
        while True:
            self.current_player = (self.current_player + 1) % self.n_players
            if not self.folded[self.current_player] and self.players_to_act[self.current_player]:
                break

    def render(self, mode='human'):
        """Render the current state of the game."""
        print(f"Player hands: {self.hands}")
        print(f"Current player: {self.current_player}")
        print(f"Current bets: {self.bets}")
        print(f"Fold Status: {self.folded}")
        print(f"Players Acting: {self.players_to_act}")
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
        # active_players = sum(not f for f in self.folded)
        active_players = [not f for f in self.folded]
        return sum(active_players) == 1 or not any(self.players_to_act)

    def _get_reward(self):
        """Calculate the reward at the end of the game."""
        # Compare hands at the end and determine the winner
        active_players = [i for i in range(
            self.n_players) if not self.folded[i]]

        if len(active_players) == 1:
            return self.pot if self.current_player == active_players[0] else 0

        best_hand = max(self.hands[i] for i in active_players)
        winning_player = self.hands.index(best_hand)
        # rewards = [-self.pot / self.n_players] * self.n_players
        rewards = [-bet for bet in self.bets]
        self.pot = sum(self.bets)
        rewards[winning_player] += self.pot  # Winner gets the pot
        # TODO: Eventually we can consider adding pot sharing
        return rewards[self.current_player]
