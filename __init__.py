from gym.envs.registration import register
from kuhn_poker_env import KuhnPokerEnv

register(
    id='KuhnPoker-v0',
    entry_point='Lokey:KuhnPokerEnv',
)
