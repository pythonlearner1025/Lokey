import os
import sys

# Add mlx-examples/llms to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "mlx-examples/llms"))

from mlx_lm import load, generate
from kuhn_poker_env import KuhnPokerEnv
from prompts import ser_llm_input, get_preferences
import mlx.core as mx
import mlx.optimizers.optimizers as optim

class Agent:
    def __init__(self, llm, position):
        model, tokenizer = llm
        self.model = model
        self.tokenizer = tokenizer
        self.position = position
        
    def act(self, game_state, private_messages):
        # Serialize state + messages into context
        context = ser_llm_input(game_state, private_messages)
        messages = [{"role": "user", "content": context}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Get LLM generation
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt,
            max_tokens=512
        )
        return response

def generate_comparison_samples(agent: Agent, buffer, **kwargs):
    def sample_with_temp(state, msg):
        # Generate two different outputs for same input with different temps
        return {
            'high_temp': agent.act(state, msg, **kwargs),
            'low_temp': agent.act(state, msg, **kwargs)
        }
    
    samples = []
    for state, msg in buffer:
        responses = sample_with_temp(state, msg)
        samples.append({
            'state': state,
            'response_a': responses['high_temp'],
            'response_b': responses['low_temp']
        })
    return samples

def train_with_dpo(agent: Agent, ref_model, buffer, num_steps):
    optimizer = optim.Adam(agent.model.parameters())
    
    for step in range(num_steps):
        # Generate samples
        samples = generate_comparison_samples(agent, buffer)
        
        # Get preferences
        preferences = get_preferences(samples)
        
        # TODO
        # save the preferences in a dataset.
        
        # Train step
        # use the siLLM library

def test_model():
    model, tokenizer = load(
        #'Qwen/Qwen2-7B-Instruct-MLX',
        'Qwen/Qwen2.5-1.5B-Instruct',
        tokenizer_config={"eos_token": "<|im_end|>", "trust_remote_code": True},
    )
    print('Qwen/Qwen2.5-1.5B-Instruct')

    print(dir(model))

    prompt = "You are playing heads up poker. You have the hand AA. The pot has 10bb. The opponent bets 30bb. It is your turn - the turn is preflop. What is your action?"

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    response = generate(model, tokenizer, prompt=prompt, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)

def train():
    env = KuhnPokerEnv(n_players=3)
    obs = env.reset()
    print(env.ser())


if __name__ == '__main__':
    '''
    from datasets import load_dataset
    dataset = load_dataset("argilla/dpo-mix-7k")
    os.makedirs('data')
    dataset.save_to_disk("./data/dpo-mix-7k")

    from huggingface_hub import snapshot_download

    # Download model to default cache directory
    model_path = snapshot_download(repo_id='Qwen/Qwen2-1.5B-Instruct-MLX',)
    '''
    test_model()

#    train()